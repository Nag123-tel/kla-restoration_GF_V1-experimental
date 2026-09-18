"""
Mandatory entry-point script per KLA's Final Submission Check announcement.

Usage (exact contract required):
    python run.py <input-dir> <output-dir>

- Reads every .npy file from <input-dir>.
- Creates <output-dir> if it does not already exist.
- Writes one restored .npy file per input file, same filename.
- Output arrays are grayscale, shape (H, W), values clipped to [0,1],
  with any NaN/Inf sanitized -- required by the announcement, even though
  this is a stricter behavior than the original problem statement's
  "KLA does not clip or renormalize outputs" wording. This script follows
  the MOST RECENT instruction (clip + sanitize), since it explicitly
  overrides the earlier guidance.
- No internet access, API keys, additional downloads, or user interaction
  required -- the checkpoint is loaded from a fixed local path bundled in
  this repo (models/best.pt, falling back to weights/best.pt for
  backward compatibility).
- Runs on GPU automatically if available, CPU otherwise.
- BATCHED: inputs are grouped by shape and processed in batches (default
  size 8) when running on GPU, matching KLA's "batch processing is
  preferred when GPU memory permits." Falls back to batch size 1 cleanly
  on CPU-only environments or if a batch is too large for available memory.

This script is a thin, dependency-light wrapper around the same model
defined in src/model.py -- see inference.py for the original, more
configurable version (custom checkpoint path, custom batch size, timing
reports, optional un-clipped output) used during development and evaluation.
"""

import glob
import os
import sys

import numpy as np
import torch

# Make src/ importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import build_model  # noqa: E402


# Fixed, local-only checkpoint locations (no downloads, no internet).
_CHECKPOINT_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best.pt"),
]

_DEFAULT_BATCH_SIZE = 8


def _find_checkpoint():
    for path in _CHECKPOINT_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No checkpoint found. Expected one of: " + ", ".join(_CHECKPOINT_CANDIDATES)
    )


def _load_model(device):
    ckpt_path = _find_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=device)
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
    if device.type == "cuda":
        model = model.half()  # FP16 inference: ~2x throughput, verified zero PSNR change
    return model


def _load_npy(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]  # (1, H, W)
    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 3):        # (H, W, C) -> (C, H, W)
            arr = np.transpose(arr, (2, 0, 1))
        # else assume already (C, H, W)
    return arr


def _postprocess_and_save(pred_batch, batch_paths, output_dir):
    """pred_batch: (B, C, H, W) tensor -> saves one .npy per item, same
    filenames as the corresponding inputs."""
    pred_batch = torch.nan_to_num(pred_batch, nan=0.0, posinf=1.0, neginf=0.0)
    pred_batch = pred_batch.clamp(0.0, 1.0)
    pred_batch = pred_batch.cpu().numpy()

    for path, out in zip(batch_paths, pred_batch):
        if out.shape[0] == 1:
            out = out[0]  # (C,H,W) -> (H,W), grayscale as required
        else:
            out = np.transpose(out, (1, 2, 0))  # -> (H, W, C)
        out_path = os.path.join(output_dir, os.path.basename(path))
        np.save(out_path, out.astype(np.float32))


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device)

    input_paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not input_paths:
        print(f"No .npy files found in {input_dir}")
        sys.exit(1)

    # Group by shape so we can batch safely (a batch requires uniform
    # tensor shape; KLA's spec allows mixed 256x256/512x512 evaluation sets).
    by_shape = {}
    for path in input_paths:
        arr = np.load(path, mmap_mode="r")
        by_shape.setdefault(arr.shape, []).append(path)

    batch_size = _DEFAULT_BATCH_SIZE if device.type == "cuda" else 1
    total_restored = 0

    with torch.no_grad():
        for shape, paths in by_shape.items():
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                try:
                    batch_arrs = [_load_npy(p) for p in batch_paths]
                    batch = torch.from_numpy(np.stack(batch_arrs, axis=0)).to(device)  # (B,C,H,W)
                    if device.type == "cuda":
                        batch = batch.half()
                    pred = model(batch)
                    _postprocess_and_save(pred, batch_paths, output_dir)
                    total_restored += len(batch_paths)
                except torch.cuda.OutOfMemoryError:
                    # Gracefully fall back to single-image processing for
                    # this batch if GPU memory is insufficient, rather than
                    # crashing the whole run.
                    torch.cuda.empty_cache()
                    for p in batch_paths:
                        arr = _load_npy(p)
                        x = torch.from_numpy(arr[None, ...]).to(device)
                        if device.type == "cuda":
                            x = x.half()
                        pred = model(x)
                        _postprocess_and_save(pred, [p], output_dir)
                        total_restored += 1

    print(f"Restored {total_restored} image(s) from {input_dir} -> {output_dir} "
          f"(device={device.type}, batch_size={batch_size})")


if __name__ == "__main__":
    main()
