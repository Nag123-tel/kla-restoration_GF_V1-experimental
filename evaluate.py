"""
Full-resolution evaluation: computes PSNR/SSIM/LPIPS for the trained model
AND a bicubic-only baseline, on FULL images (not training crops) -- matching
KLA's requirements to report metrics at full resolution and compare against
at least one baseline.

Also exports the best-N and worst-N restorations (by PSNR) as .npy triplets
(noisy, restored, gt) for the "successful and failed cases" deliverable.

Usage:
    python evaluate.py \
        --noisy_dir data/val/NoisyLR --gt_dir data/val/GT \
        --checkpoint weights/best.pt \
        --results_dir results

IMPORTANT: point --noisy_dir/--gt_dir at a held-out validation set that was
NOT used for training (see train.py's --val_split, or keep a manually
separated validation folder) -- this script assumes whatever you pass it is
your clean evaluation set and does not re-split anything itself.
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics import psnr, ssim, LPIPSMetric, _LPIPS_AVAILABLE
from src.model import build_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy_dir", type=str, required=True)
    p.add_argument("--gt_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_examples", type=int, default=3,
                    help="How many best/worst examples (by PSNR) to export.")
    p.add_argument("--use_lpips", action="store_true", default=True)
    p.add_argument("--no_lpips", dest="use_lpips", action="store_false",
                    help="Skip LPIPS (faster, no extra dependency needed).")
    return p.parse_args()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("args", {})
    model = build_model(
        in_ch=cfg.get("in_ch", 1),
        scale=cfg.get("scale", 2),
        size=cfg.get("model_size", "tiny"),
        use_vst=cfg.get("use_vst", False),
        vst_k=cfg.get("vst_k", 4.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def load_npy(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[0] not in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def main():
    args = get_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)
    examples_dir = os.path.join(args.results_dir, "example_restorations")
    os.makedirs(examples_dir, exist_ok=True)

    print("Evaluation device:", device)
    model, model_cfg = load_model(args.checkpoint, device)
    scale = model_cfg.get("scale", 2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({n_params:,} params), scale={scale}")

    lpips_metric = None
    if args.use_lpips:
        if not _LPIPS_AVAILABLE:
            print("[WARN] lpips package not installed (pip install lpips) -- skipping LPIPS.")
        else:
            lpips_metric = LPIPSMetric(net="alex", device=device)

    noisy_paths = {Path(p).stem: p for p in glob.glob(os.path.join(args.noisy_dir, "*.npy"))}
    gt_paths = {Path(p).stem: p for p in glob.glob(os.path.join(args.gt_dir, "*.npy"))}
    stems = sorted(set(noisy_paths) & set(gt_paths))
    assert len(stems) > 0, "No matched NoisyLR/GT pairs found."
    print(f"Evaluating on {len(stems)} full-resolution image(s).")

    per_image = []
    t_start = time.time()

    with torch.no_grad():
        for stem in stems:
            noisy = load_npy(noisy_paths[stem]).unsqueeze(0).to(device)
            gt = load_npy(gt_paths[stem]).unsqueeze(0).to(device)

            pred = model(noisy).clamp(0, 1)
            baseline = F.interpolate(noisy, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)

            model_psnr = psnr(pred[0], gt[0])
            model_ssim = ssim(pred[0], gt[0])
            base_psnr = psnr(baseline[0], gt[0])
            base_ssim = ssim(baseline[0], gt[0])

            model_lpips = lpips_metric(pred[0], gt[0]) if lpips_metric else None
            base_lpips = lpips_metric(baseline[0], gt[0]) if lpips_metric else None

            per_image.append({
                "stem": stem,
                "model_psnr": model_psnr, "model_ssim": model_ssim, "model_lpips": model_lpips,
                "baseline_psnr": base_psnr, "baseline_ssim": base_ssim, "baseline_lpips": base_lpips,
                "noisy_path": noisy_paths[stem], "gt_path": gt_paths[stem],
                "pred": pred[0].cpu().numpy(), "baseline_pred": baseline[0].cpu().numpy(),
            })

    t_eval = time.time() - t_start

    # --- Aggregate summary ---
    def avg(key):
        vals = [r[key] for r in per_image if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "num_images": len(stems),
        "model": {
            "psnr_mean": avg("model_psnr"), "ssim_mean": avg("model_ssim"), "lpips_mean": avg("model_lpips"),
        },
        "baseline_bicubic": {
            "psnr_mean": avg("baseline_psnr"), "ssim_mean": avg("baseline_ssim"), "lpips_mean": avg("baseline_lpips"),
        },
        "checkpoint": args.checkpoint,
        "model_params": n_params,
        "scale": scale,
        "eval_wall_clock_seconds": t_eval,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    print("\n" + "=" * 60)
    print(f"MODEL    -> PSNR: {summary['model']['psnr_mean']:.3f} dB | "
          f"SSIM: {summary['model']['ssim_mean']:.4f} | "
          f"LPIPS: {summary['model']['lpips_mean']:.4f}" if summary['model']['lpips_mean'] is not None
          else f"MODEL    -> PSNR: {summary['model']['psnr_mean']:.3f} dB | SSIM: {summary['model']['ssim_mean']:.4f}")
    print(f"BASELINE -> PSNR: {summary['baseline_bicubic']['psnr_mean']:.3f} dB | "
          f"SSIM: {summary['baseline_bicubic']['ssim_mean']:.4f} | "
          f"LPIPS: {summary['baseline_bicubic']['lpips_mean']:.4f}" if summary['baseline_bicubic']['lpips_mean'] is not None
          else f"BASELINE -> PSNR: {summary['baseline_bicubic']['psnr_mean']:.3f} dB | SSIM: {summary['baseline_bicubic']['ssim_mean']:.4f}")
    print("=" * 60)

    summary_path = os.path.join(args.results_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    per_image_path = os.path.join(args.results_dir, "metrics_per_image.csv")
    import csv
    with open(per_image_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "model_psnr", "model_ssim", "model_lpips",
                          "baseline_psnr", "baseline_ssim", "baseline_lpips"])
        for r in per_image:
            writer.writerow([r["stem"], r["model_psnr"], r["model_ssim"], r["model_lpips"],
                              r["baseline_psnr"], r["baseline_ssim"], r["baseline_lpips"]])
    print(f"Per-image metrics saved to: {per_image_path}")

    # --- Export best-N and worst-N examples (successful and failed cases) ---
    per_image_sorted = sorted(per_image, key=lambda r: r["model_psnr"])
    worst = per_image_sorted[:args.num_examples]
    best = per_image_sorted[-args.num_examples:]

    for tag, group in [("worst", worst), ("best", best)]:
        for r in group:
            out_dir = os.path.join(examples_dir, tag)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"{r['stem']}_noisy.npy"), np.load(r["noisy_path"]))
            np.save(os.path.join(out_dir, f"{r['stem']}_restored.npy"), r["pred"][0] if r["pred"].shape[0] == 1 else r["pred"])
            np.save(os.path.join(out_dir, f"{r['stem']}_baseline.npy"), r["baseline_pred"][0] if r["baseline_pred"].shape[0] == 1 else r["baseline_pred"])
            np.save(os.path.join(out_dir, f"{r['stem']}_gt.npy"), np.load(r["gt_path"]))

    print(f"Exported {len(worst)} worst-case and {len(best)} best-case example(s) to: {examples_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
