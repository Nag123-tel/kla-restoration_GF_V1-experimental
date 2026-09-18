"""
Run from the kla-restoration_Phase2 repo root in Colab.
Consolidates everything after the baseline training call, so it survives
notebook cleanup/session resets as ONE file instead of many scattered ones.

USAGE: run the numbered sections in order. Each section is independent
enough to skip if already done, but check the printed confirmations.
"""

import os, json, shutil
import numpy as np
import torch
from collections import defaultdict

# =========================================================================
# SECTION 0: Persistent backup setup (run this in EVERY fresh session)
# =========================================================================
def setup_persistence():
    PERSIST_DIR = '/content/drive/MyDrive/Semi_GF/persistent_work'
    os.makedirs(PERSIST_DIR, exist_ok=True)
    print(f'Persistent dir: {PERSIST_DIR}')
    print('Contents:', os.listdir(PERSIST_DIR) if os.path.exists(PERSIST_DIR) else 'EMPTY')
    return PERSIST_DIR

def backup_to_drive(persist_dir, items):
    """items: list of (local_path, name) tuples to copy to Drive."""
    for src, name in items:
        if not os.path.exists(src):
            print(f'SKIP (not found): {src}')
            continue
        dst = os.path.join(persist_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        print(f'Backed up: {src} -> {dst}')


# =========================================================================
# SECTION 1: Reusable full-resolution per-category evaluation function
# (replaces the separate eval_finale_*.py files -- one function, many calls)
# =========================================================================
def evaluate_checkpoint(checkpoint_path, run_val_split_path, final_manifest_path,
                         output_json_path, label):
    from src.metrics import psnr, ssim
    from src.model import build_model

    with open(run_val_split_path) as f:
        run_val = json.load(f)
    with open(final_manifest_path) as f:
        manifest = json.load(f)

    val_stems = sorted(run_val['val_file_stems'])
    noisy_dir = run_val['noisy_dir']
    gt_dir = run_val['gt_dir']
    category_of = manifest['category_of_stem']
    domain_of = manifest['domain_of_stem']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt['args']
    model = build_model(in_ch=args['in_ch'], scale=args['scale'], size=args['model_size'],
                         use_vst=args.get('use_vst', False), vst_k=args.get('vst_k', 4.0)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[{label}] Loaded checkpoint from epoch {ckpt['epoch']}, "
          f"val_psnr(crop-based, DO NOT TRUST)={ckpt.get('val_psnr', 'n/a')}")

    by_category = defaultdict(lambda: {'psnr': [], 'ssim': []})
    by_domain = defaultdict(lambda: {'psnr': [], 'ssim': []})

    with torch.no_grad():
        for stem in val_stems:
            noisy = np.load(os.path.join(noisy_dir, stem + '.npy')).astype(np.float32)
            gt = np.load(os.path.join(gt_dir, stem + '.npy')).astype(np.float32)
            noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
            gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(noisy_t).clamp(0, 1)
            p = float(psnr(pred[0], gt_t[0]))
            s = float(ssim(pred[0], gt_t[0]))
            cat = category_of.get(stem, 'unknown')
            dom = domain_of.get(stem, 'unknown')
            by_category[cat]['psnr'].append(p)
            by_category[cat]['ssim'].append(s)
            by_domain[dom]['psnr'].append(p)
            by_domain[dom]['ssim'].append(s)

    summary = {cat: {'n': len(v['psnr']), 'psnr': float(np.mean(v['psnr'])), 'ssim': float(np.mean(v['ssim']))}
               for cat, v in by_category.items()}
    domain_summary = {dom: {'n': len(v['psnr']), 'psnr': float(np.mean(v['psnr'])), 'ssim': float(np.mean(v['ssim']))}
                      for dom, v in by_domain.items()}
    overall_psnr = float(np.mean([v for cat in by_category.values() for v in cat['psnr']]))
    overall_ssim = float(np.mean([v for cat in by_category.values() for v in cat['ssim']]))

    print(f"[{label}] FULL-RES OVERALL: PSNR={overall_psnr:.3f} dB, SSIM={overall_ssim:.4f} "
          f"(this is the trustworthy number, not the crop-based one above)")

    result = {'label': label, 'per_category': summary, 'per_domain': domain_summary,
              'overall_psnr': overall_psnr, 'overall_ssim': overall_ssim}
    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=2)
    return result


# =========================================================================
# SECTION 2: Idempotent patch for --pretrained_checkpoint support
# (safe to call even if already patched -- checks first)
# =========================================================================
def patch_train_stratified_for_finetuning():
    with open('train_stratified.py') as f:
        content = f.read()

    if 'pretrained_checkpoint' in content:
        print('Already patched -- skipping.')
        return

    old_arg = 'p.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small"])'
    new_arg = old_arg + '''
    p.add_argument("--pretrained_checkpoint", type=str, default=None,
                    help="Path to an existing checkpoint (.pt) to initialize weights from.")'''
    assert old_arg in content, "get_args() structure changed -- patch needs manual adjustment"
    content = content.replace(old_arg, new_arg)

    old_build = '''    model = build_model(in_ch=args.in_ch, scale=args.scale, size=args.model_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: NAFNetSR ({args.model_size}), params={n_params:,}")'''
    new_build = old_build + '''

    if args.pretrained_checkpoint:
        pretrained = torch.load(args.pretrained_checkpoint, map_location=device)
        model.load_state_dict(pretrained['model_state_dict'])
        print(f"Loaded pretrained weights from {args.pretrained_checkpoint} "
              f"(epoch {pretrained.get('epoch', '?')}) -- FINE-TUNING.")
    else:
        print("No --pretrained_checkpoint -- training from scratch.")'''
    assert old_build in content, "model-build structure changed -- patch needs manual adjustment"
    content = content.replace(old_build, new_build)

    with open('train_stratified.py', 'w') as f:
        f.write(content)
    print('Patched train_stratified.py for --pretrained_checkpoint support.')


# =========================================================================
# SECTION 3: Compare any number of evaluated checkpoints side by side
# =========================================================================
def compare_results(*result_dicts):
    labels = [r['label'] for r in result_dicts]
    cats = sorted(result_dicts[0]['per_category'].keys(),
                  key=lambda c: result_dicts[0]['per_category'][c]['psnr'])
    print(f"{'Category':<32}" + "".join(f"{l:>16}" for l in labels))
    for cat in cats:
        print(f"{cat:<32}" + "".join(f"{r['per_category'][cat]['psnr']:>16.3f}" for r in result_dicts))
    print(f"{'OVERALL':<32}" + "".join(f"{r['overall_psnr']:>16.3f}" for r in result_dicts))
    best_idx = max(range(len(result_dicts)), key=lambda i: result_dicts[i]['overall_psnr'])
    print(f"\nBest overall: {labels[best_idx]} ({result_dicts[best_idx]['overall_psnr']:.3f} dB)")
    return labels[best_idx]


if __name__ == '__main__':
    print("Import this module's functions rather than running directly:")
    print("  from finale_pipeline import setup_persistence, backup_to_drive,")
    print("  evaluate_checkpoint, patch_train_stratified_for_finetuning, compare_results")
