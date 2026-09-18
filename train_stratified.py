"""
Run this from the ROOT of the cloned kla-restoration repo in Colab
(it imports from src.*, same as train.py does).

Setup in Colab first:
    !git clone https://github.com/Nag123-tel/kla-restoration.git
    %cd kla-restoration
    !pip install -r requirements.txt -q

Then copy this file into the repo root (or just paste this cell) and run:
    !python train_stratified.py \
        --noisy_dir /content/semicon_train_data_extracted/semicon_train_data/semicon_train_data/NoisyLR \
        --gt_dir /content/semicon_train_data_extracted/semicon_train_data/semicon_train_data/GT \
        --manifest /content/results/val_split_FINAL.json \
        --scale 2 --patch_size 64 --batch_size 16 --epochs 100 --model_size tiny \
        --ckpt_dir weights --results_dir results

This is a thin wrapper around train.py's own logic -- everything (model,
losses, logging, checkpointing) is identical. The ONLY change is how the
train/val split is built: instead of train.py's random_split, we use the
stratified train_files/val_files lists from the clustering manifest, so
validation reflects the true content mix (fiber mesh, wires, dense particles,
porous/foam, faceted grain, etc.) instead of a lucky/unlucky random draw.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from src.dataset import PairedRestorationDataset
from src.losses import CombinedLoss
from src.metrics import psnr, ssim
from src.model import build_model
from losses_intensity import IntensityWeightedLoss


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--noisy_dir", type=str, required=True)
    p.add_argument("--gt_dir", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True,
                    help="Path to val_split_FINAL.json (or similar) with train_files/val_files stems.")
    p.add_argument("--synth_from_gt_dir", type=str, default=None)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--in_ch", type=int, default=1)
    p.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small"])
    p.add_argument("--pretrained_checkpoint", type=str, default=None,
                    help="Path to an existing checkpoint (.pt) to initialize weights from.")
    p.add_argument("--use_vst", action="store_true",
                    help="Enable the signed-log1p variance-stabilizing transform on the "
                         "network's internal input path (targets the confirmed multiplicative "
                         "noise). NOTE: changes what `intro` sees, so a --pretrained_checkpoint "
                         "trained without this flag is being fine-tuned into a new input "
                         "distribution, not just having its weights refined -- expect to watch "
                         "the first few epochs closely.")
    p.add_argument("--vst_k", type=float, default=4.0,
                    help="Compression strength for --use_vst (higher = more compression at "
                         "large |x|; the transform is slope-1 near 0 regardless of k).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_ssim", type=float, default=0.2)
    p.add_argument("--loss_type", type=str, default="combined",
                    choices=["combined", "intensity_weighted"],
                    help="'combined' = original repo loss (unchanged). "
                         "'intensity_weighted' = Charbonnier reweighted by inverse "
                         "local GT intensity, targeting the confirmed multiplicative/"
                         "speckle noise (corr=0.864 between GT intensity and noise level).")
    p.add_argument("--intensity_eps", type=float, default=0.05,
                    help="Only used when --loss_type intensity_weighted. Floor added to "
                         "GT intensity before inverting, bounds max per-pixel weight.")
    p.add_argument("--oversample_categories", type=str, default=None,
                    help="Comma-separated category names (from the manifest's category_names) "
                         "to oversample during training, e.g. "
                         "'sparse_particles_on_substrate,fiber_mesh'. Leave unset for plain "
                         "uniform random sampling (original behavior).")
    p.add_argument("--oversample_factor", type=float, default=3.0,
                    help="Sampling weight multiplier applied to --oversample_categories "
                         "relative to all other training samples (weight 1.0).")
    p.add_argument("--rotate_augment", action="store_true",
                    help="Apply random 90/180/270-degree rotation to each training pair, "
                         "in addition to whatever crop/flip augmentation the dataset already "
                         "does. Valid for SEM imagery (no canonical 'up', unlike natural "
                         "photos). Training only -- does not affect inference/throughput.")
    p.add_argument("--calibrated_synth", action="store_true",
                    help="If --synth_from_gt_dir is set, use a calibrated speckle/gaussian "
                         "noise range (speckle U(0.15,0.20), gaussian U(0.04,0.05), forced "
                         "downsample-first ordering) instead of degrade()'s defaults, which "
                         "measurement showed correlate poorly with real data's measured "
                         "noise-vs-intensity relationship (corr=0.864). Default off = uses "
                         "the repo's original degrade() unchanged.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_dir", type=str, default="weights")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=10)
    args = p.parse_args()
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            p.set_defaults(**{k: v})
        args = p.parse_args()
    return args


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rotate_augment_batch(noisy, gt):
    """Apply an independent random k*90-degree rotation to each sample in the
    batch, identically to its noisy input and GT target so they stay spatially
    aligned. Safe for square patches (both NoisyLR and GT patches are square
    here, so rotation doesn't change tensor shape)."""
    B = noisy.shape[0]
    noisy_out = torch.empty_like(noisy)
    gt_out = torch.empty_like(gt)
    for i in range(B):
        k = torch.randint(0, 4, (1,)).item()
        noisy_out[i] = torch.rot90(noisy[i], k, dims=[-2, -1])
        gt_out[i] = torch.rot90(gt[i], k, dims=[-2, -1])
    return noisy_out, gt_out


def apply_calibrated_synthetic_degradation():
    """Monkey-patch src.dataset's reference to degrade() with a calibrated
    version, matching the measured real noise statistics (corr=0.864 between
    GT intensity and noise level) far more closely than the repo's default
    degrade() (measured corr as low as 0.33-0.44 depending on op order,
    vs 0.916 with this calibration). Must patch src.dataset.degrade (the
    name bound via `from src.degradations import degrade` in that module),
    not src.degradations.degrade, since rebinding the latter after dataset.py
    already imported it would have no effect.
    """
    import random as _random
    import src.dataset as _dataset_module
    from src.degradations import add_speckle_noise, add_gaussian_noise, downsample

    def calibrated_degrade(clean, scale=2, **kwargs):
        speckle_level = _random.uniform(0.15, 0.20)
        gaussian_sigma = _random.uniform(0.04, 0.05)
        out = downsample(clean, scale)          # downsample-first ordering, matches
        out = add_speckle_noise(out, speckle_level)  # measured real data best of the
        out = add_gaussian_noise(out, gaussian_sigma)  # 3 orderings tested (corr=0.916)
        return out

    _dataset_module.degrade = calibrated_degrade
    print("Calibrated synthetic degradation ACTIVE: speckle~U(0.15,0.20), "
          "gaussian~U(0.04,0.05), forced downsample-first order "
          "(measured corr=0.916 vs real 0.864, vs repo default corr=0.33-0.44)")


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnrs, ssims = [], []
    for noisy, gt in loader:
        noisy, gt = noisy.to(device), gt.to(device)
        pred = model(noisy).clamp(0, 1)
        for i in range(pred.shape[0]):
            psnrs.append(psnr(pred[i], gt[i]))
            ssims.append(ssim(pred[i], gt[i]))
    model.train()
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)


def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)
    print("Config:", vars(args))

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_stratified"
    config_path = os.path.join(args.results_dir, f"train_config_{run_id}.json")
    with open(config_path, "w") as f:
        json.dump({
            **vars(args),
            "run_id": run_id,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "split_method": "stratified_by_cluster_manifest",
        }, f, indent=2)
    print(f"Saved run config to: {config_path}")

    log_path = os.path.join(args.results_dir, f"train_log_{run_id}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "lr", "val_psnr", "val_ssim", "epoch_time_seconds"])
    print(f"Logging per-epoch metrics to: {log_path}")

    # ---------- Load full dataset (same as train.py) ----------
    if args.calibrated_synth and args.synth_from_gt_dir:
        apply_calibrated_synthetic_degradation()
    elif args.calibrated_synth and not args.synth_from_gt_dir:
        print("WARNING: --calibrated_synth was set but --synth_from_gt_dir is empty, "
              "so there's nothing for it to affect.")

    full_ds = PairedRestorationDataset(
        noisy_dir=args.noisy_dir,
        gt_dir=args.gt_dir,
        synth_from_gt_dir=args.synth_from_gt_dir,
        patch_size=args.patch_size,
        scale=args.scale,
        train=True,
    )
    real_ids = full_ds.real_ids  # list of filename stems, index-aligned with dataset indices

    # ---------- Build stratified split from manifest instead of random_split ----------
    with open(args.manifest) as f:
        manifest = json.load(f)
    train_stems = set(manifest["train_files"])
    val_stems = set(manifest["val_files"])

    stem_to_idx = {stem: i for i, stem in enumerate(real_ids)}

    missing_train = train_stems - set(real_ids)
    missing_val = val_stems - set(real_ids)
    if missing_train or missing_val:
        print(f"WARNING: {len(missing_train)} train stems and {len(missing_val)} val stems "
              f"from the manifest were not found in this dataset's real_ids. "
              f"Check that --noisy_dir/--gt_dir match what the manifest was built from.")

    train_stems_sorted = sorted(train_stems)  # deterministic order, needed to align sample weights below
    train_stems_present = [s for s in train_stems_sorted if s in stem_to_idx]
    train_indices = [stem_to_idx[s] for s in train_stems_present]
    val_indices = [stem_to_idx[s] for s in val_stems if s in stem_to_idx]

    # If synthetic pairs are being mixed in (indices beyond len(real_ids)), they go to train only
    synth_indices = []
    if len(full_ds) > len(real_ids):
        synth_indices = list(range(len(real_ids), len(full_ds)))
        train_indices = train_indices + synth_indices
        print(f"Added {len(synth_indices)} synthetic-pair indices to training set.")

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    n_train, n_val = len(train_ds), len(val_ds)
    print(f"Train samples: {n_train} | Val samples: {n_val} (stratified from manifest)")

    val_manifest_path = os.path.join(args.results_dir, f"val_split_{run_id}.json")
    with open(val_manifest_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "source_manifest": args.manifest,
            "num_val_files": len(val_indices),
            "val_file_stems": sorted(val_stems & set(real_ids)),
            "noisy_dir": args.noisy_dir,
            "gt_dir": args.gt_dir,
        }, f, indent=2)
    print(f"Saved validation file list to: {val_manifest_path}")

    if args.oversample_categories:
        target_categories = {c.strip() for c in args.oversample_categories.split(",") if c.strip()}
        tags = manifest.get("tags", {})
        category_names = manifest.get("category_names", {})

        def stem_category_name(stem):
            cid = tags.get(stem)
            return category_names.get(cid, cid)

        sample_weights = []
        cat_counts = defaultdict(int)
        for stem in train_stems_present:
            cat_name = stem_category_name(stem)
            w = args.oversample_factor if cat_name in target_categories else 1.0
            sample_weights.append(w)
            cat_counts[cat_name] += 1
        sample_weights.extend([1.0] * len(synth_indices))  # synthetic pairs: no boost, no category info

        unmatched = target_categories - set(cat_counts.keys())
        if unmatched:
            print(f"WARNING: --oversample_categories contains names not found in this manifest's "
                  f"category_names: {unmatched}. Check spelling against the manifest.")

        total_weight = sum(sample_weights)
        print(f"\nOversampling categories {sorted(target_categories)} at {args.oversample_factor}x weight.")
        print("Effective per-epoch sampling distribution (expected, with replacement):")
        for cat_name, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
            w = args.oversample_factor if cat_name in target_categories else 1.0
            expected_frac = (n * w) / total_weight
            print(f"  {cat_name:<32} raw n={n:>5} -> expected {expected_frac*100:5.1f}% of each epoch"
                  f"{'  <- boosted' if cat_name in target_categories else ''}")

        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                   num_workers=args.num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers) if n_val > 0 else None

    model = build_model(in_ch=args.in_ch, scale=args.scale, size=args.model_size,
                         use_vst=args.use_vst, vst_k=args.vst_k).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: NAFNetSR ({args.model_size}), params={n_params:,}, "
          f"use_vst={args.use_vst}" + (f" (k={args.vst_k})" if args.use_vst else ""))

    if args.pretrained_checkpoint:
        pretrained = torch.load(args.pretrained_checkpoint, map_location=device)
        model.load_state_dict(pretrained['model_state_dict'])
        print(f"Loaded pretrained weights from {args.pretrained_checkpoint} "
              f"(epoch {pretrained.get('epoch', '?')}) -- FINE-TUNING.")
    else:
        print("No --pretrained_checkpoint -- training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.loss_type == "intensity_weighted":
        loss_fn = IntensityWeightedLoss(weight_ssim=args.weight_ssim, intensity_eps=args.intensity_eps)
        print(f"Using IntensityWeightedLoss (intensity_eps={args.intensity_eps}, "
              f"weight_ssim={args.weight_ssim})")
    else:
        loss_fn = CombinedLoss(channels=args.in_ch, weight_ssim=args.weight_ssim)
        print(f"Using original CombinedLoss (weight_ssim={args.weight_ssim})")

    if args.rotate_augment:
        print("Rotation augmentation ENABLED (random 90/180/270 deg per training sample)")

    best_psnr = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            if args.rotate_augment:
                noisy, gt = rotate_augment_batch(noisy, gt)
            optimizer.zero_grad()
            pred = model(noisy)
            loss, parts = loss_fn(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)
        scheduler.step()

        epoch_loss = running_loss / n_train
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | loss: {epoch_loss:.5f} | "
              f"lr: {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

        val_psnr_val, val_ssim_val = "", ""
        if val_loader is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            val_psnr, val_ssim = validate(model, val_loader, device)
            val_psnr_val, val_ssim_val = val_psnr, val_ssim
            print(f"  [val] PSNR: {val_psnr:.3f} dB | SSIM: {val_ssim:.4f}")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                }, os.path.join(args.ckpt_dir, "best.pt"))
                print(f"  -> saved new best checkpoint (PSNR={val_psnr:.3f} dB)")

        log_writer.writerow([epoch, epoch_loss, scheduler.get_last_lr()[0], val_psnr_val, val_ssim_val, dt])
        log_file.flush()

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
            }, os.path.join(args.ckpt_dir, f"epoch{epoch}.pt"))

    log_file.close()
    print("Training complete. Best val PSNR:", best_psnr)
    print(f"Full training log saved to: {log_path}")


if __name__ == "__main__":
    main()
