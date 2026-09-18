"""
Loss functions for the restoration model.

- CharbonnierLoss: smooth L1-like loss, standard in SR/denoising literature
  (Restormer, SwinIR, MPRNet all use it) -- more robust to outliers than L2,
  sharper reconstructions than L1.
- SSIMLoss: structural similarity term, directly optimizes toward one of
  KLA's reported metrics.
- LPIPSLoss: perceptual similarity term (learned feature-space distance),
  optimizes toward the third of KLA's three reported metrics -- the only
  one of the three the original CombinedLoss didn't touch at all.
- CombinedLoss: weighted sum, the common recipe for restoration training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (1 - SSIM), single-scale, Gaussian window."""

    def __init__(self, window_size=11, sigma=1.5, channels=1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", self._make_window(window_size, sigma, channels))

    @staticmethod
    def _gaussian(window_size, sigma):
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _make_window(self, window_size, sigma, channels):
        g1d = self._gaussian(window_size, sigma).unsqueeze(1)
        g2d = g1d @ g1d.t()
        window = g2d.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred, target):
        c = pred.shape[1]
        if c != self.channels:
            window = self._make_window(self.window_size, 1.5, c).to(pred.device)
        else:
            window = self.window.to(pred.device)

        pad = self.window_size // 2
        mu_p = F.conv2d(pred, window, padding=pad, groups=c)
        mu_t = F.conv2d(target, window, padding=pad, groups=c)

        mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

        sigma_p_sq = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_p_sq
        sigma_t_sq = F.conv2d(target * target, window, padding=pad, groups=c) - mu_t_sq
        sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_pt

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
                   ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))

        return 1.0 - ssim_map.mean()


class LPIPSLoss(nn.Module):
    """Trainable perceptual loss wrapping the `lpips` package (same package
    src/metrics.py's LPIPSMetric uses for reporting).

    Two things distinguish this from LPIPSMetric, both necessary to use it
    as a LOSS rather than a reported number:
    - No @torch.no_grad(): gradients must flow back through to `pred` so
      the restoration model can actually learn from this signal.
    - The LPIPS backbone's own weights are explicitly frozen
      (requires_grad_(False) on every parameter) -- training with this
      loss updates YOUR model only, never the pretrained feature extractor.

    Device handling is lazy (mirrors SSIMLoss's `.to(pred.device)` above):
    the backbone moves to whatever device `pred` is on, the first time it
    sees it, so this doesn't need a separate `.to(device)` call at
    construction the way the model does.
    """

    def __init__(self, net="alex"):
        super().__init__()
        if not _LPIPS_AVAILABLE:
            raise ImportError(
                "pip install lpips  # required for --weight_lpips > 0"
            )
        self.model = lpips_lib.LPIPS(net=net)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._device = torch.device("cpu")

    def forward(self, pred, target):
        if pred.device != self._device:
            self.model = self.model.to(pred.device)
            self._device = pred.device

        pred_c = pred.clamp(0, 1)
        target_c = target.clamp(0, 1)
        if pred_c.shape[1] == 1:
            pred_c = pred_c.repeat(1, 3, 1, 1)
            target_c = target_c.repeat(1, 3, 1, 1)

        # LPIPS expects 3-channel input in [-1, 1].
        pred_n = pred_c * 2 - 1
        target_n = target_c * 2 - 1
        return self.model(pred_n, target_n).mean()


class CombinedLoss(nn.Module):
    """weight_charbonnier * Charbonnier + weight_ssim * (1 - SSIM)
       [+ weight_lpips * LPIPS, only when weight_lpips > 0]

    LPIPS is opt-in and off by default (weight_lpips=0.0), so existing
    training commands and the default loss are completely unchanged --
    this only activates when explicitly asked for."""

    def __init__(self, channels=1, weight_charbonnier=1.0, weight_ssim=0.2,
                 weight_lpips=0.0, lpips_net="alex"):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss(channels=channels)
        self.w_char = weight_charbonnier
        self.w_ssim = weight_ssim
        self.w_lpips = weight_lpips
        self.lpips = LPIPSLoss(net=lpips_net) if weight_lpips > 0 else None

    def forward(self, pred, target):
        pred_c = pred.clamp(0, 1)
        target_c = target.clamp(0, 1)
        l_char = self.charbonnier(pred, target)
        l_ssim = self.ssim(pred_c, target_c)
        total = self.w_char * l_char + self.w_ssim * l_ssim
        parts = {"charbonnier": l_char.item(), "ssim_loss": l_ssim.item()}

        if self.lpips is not None:
            l_lpips = self.lpips(pred_c, target_c)
            total = total + self.w_lpips * l_lpips
            parts["lpips_loss"] = l_lpips.item()

        return total, parts
