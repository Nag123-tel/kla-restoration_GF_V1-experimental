"""
Standalone inference script for the KLA Hackathon 2026 submission.

Mandatory behavior (per problem statement Section 4.C):
  - accepts an input-directory argument and an output-directory argument
  - loads every degraded image, restores it, saves each output to output dir
  - preserves file naming (same stem, .npy in -> .npy out)
  - supports NVIDIA GPU execution, batches when memory permits
  - does not require editing source code / paths
  - does not clip or renormalize outputs unless done inside this pipeline
    ( if you want clipped outputs for visual sanity checks, use --clip)

Usage:
    python inference.py --input_dir /path/to/NoisyLR --output_dir /path/to/restored \
        --checkpoint weights/best.pt

Timing: end-to-end runtime (disk read, preprocessing, H2D transfer, model
forward, D2H transfer, postprocessing, disk write) is measured and printed,
matching KLA's stated runtime definition.
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.model import build_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True, help="Directory of degraded .npy NoisyLR images.")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write restored .npy images to.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint from train.py.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or None to auto-detect.")
    p.add_argument("--clip", action="store_true", help="Clip outputs to [0,1] before saving (off by default).")
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
    return model


def load_npy_batch(paths):
    """Loads a list of .npy files into a single (B,C,H,W) float32 tensor.
    Assumes all images in a batch share the same shape (true for a fixed
    NoisyLR resolution such as 128x128 or 256x256 per KLA spec)."""
    arrs = []
    for p in paths:
        a = np.load(p).astype(np.float32)
        if a.ndim == 2:
            a = a[None, ...]
        elif a.ndim == 3 and a.shape[0] not in (1, 3):
            a = np.transpose(a, (2, 0, 1))
        arrs.append(a)
    return torch.from_numpy(np.stack(arrs, axis=0))


def main():
    args = get_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inference device:", device)
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA version: {torch.version.cuda}")

    os.makedirs(args.output_dir, exist_ok=True)

    t_load_model_start = time.time()
    model = load_model(args.checkpoint, device)
    t_load_model = time.time() - t_load_model_start
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({n_params:,} params) in {t_load_model:.2f}s")

    input_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    assert len(input_paths) > 0, f"No .npy files found in {args.input_dir}"
    print(f"Found {len(input_paths)} input images.")

    # Group by shape so we can batch safely (handles mixed 256x256/512x512 evaluation sets).
    by_shape = {}
    for p in input_paths:
        shape = np.load(p, mmap_mode="r").shape
        by_shape.setdefault(shape, []).append(p)

    total_images = 0
    t_pipeline_start = time.time()

    with torch.no_grad():
        for shape, paths in by_shape.items():
            for i in range(0, len(paths), args.batch_size):
                batch_paths = paths[i:i + args.batch_size]

                t0 = time.time()
                batch = load_npy_batch(batch_paths)                 # disk read + preprocessing
                batch = batch.to(device, non_blocking=True)          # CPU -> GPU transfer

                pred = model(batch)                                  # model execution

                if args.clip:
                    pred = pred.clamp(0, 1)
                pred = pred.cpu().numpy()                             # GPU -> CPU transfer

                for path, out_arr in zip(batch_paths, pred):
                    out_arr = out_arr[0] if out_arr.shape[0] == 1 else np.transpose(out_arr, (1, 2, 0))
                    out_path = os.path.join(args.output_dir, Path(path).name)
                    np.save(out_path, out_arr.astype(np.float32))     # save to disk -- INCLUDED in timing
                t1 = time.time()  # matches KLA's runtime definition exactly: read -> ... -> save

                total_images += len(batch_paths)
                print(f"  batch shape={shape} size={len(batch_paths)} -> {t1 - t0:.3f}s "
                      f"({(t1 - t0) / len(batch_paths) * 1000:.1f} ms/image)")

    t_pipeline_total = time.time() - t_pipeline_start
    print("-" * 60)
    print(f"Restored {total_images} images.")
    print(f"End-to-end pipeline runtime: {t_pipeline_total:.2f}s "
          f"({t_pipeline_total / total_images * 1000:.1f} ms/image average)")
    print(f"(Model load time, reported separately, not included above: {t_load_model:.2f}s)")
    print(f"Outputs written to: {args.output_dir}")

    # Write a runtime report for the submission 
    report = {
        "total_images": total_images,
        "batch_size": args.batch_size,
        "end_to_end_runtime_seconds": t_pipeline_total,
        "ms_per_image": t_pipeline_total / total_images * 1000,
        "model_load_time_seconds": t_load_model,
        "timing_method": "wall-clock time.time(), measured per batch from disk-read start "
                          "through model execution, GPU->CPU transfer, and np.save() to disk "
                          "(matches KLA's stated runtime definition); model-load time excluded "
                          "and reported separately.",
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "checkpoint": args.checkpoint,
        "output_clipped": args.clip,
    }
    report_path = os.path.abspath(os.path.join(args.output_dir, "inference_runtime_report.json"))
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Runtime report saved to: {report_path}")


if __name__ == "__main__":
    main()
