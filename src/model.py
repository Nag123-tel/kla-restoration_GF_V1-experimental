"""
NAFNet-based restoration model for the KLA Hackathon 2026 problem:
AI-Based Restoration of Degraded Images for Semiconductor Inspection.

Handles the three specified degradations (speckle noise, additive Gaussian
noise, downsampling) in a single forward pass: input NoisyLR -> output at
GT resolution.

Architecture rationale (see README for full justification + references):
- NAFNet (Chen et al., ECCV 2022, "Simple Baselines for Image Restoration")
  is chosen over Restormer/SwinIR because it matches/beats them on
  real-world denoising benchmarks while being substantially cheaper to run,
  which matters directly for the H100 end-to-end throughput scoring axis.
- NAFNet's SimpleGate + Simplified Channel Attention replace nonlinear
  activations entirely, giving a strong quality/compute trade-off with a
  plain, easy-to-reproduce CNN (no custom CUDA ops, no attention-window
  bookkeeping like Swin-based models).
- A PixelShuffle super-resolution head is appended so the network jointly
  denoises AND upsamples to the GT resolution in one pass (no separate
  SR stage), since NoisyLR is intentionally lower resolution than GT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Core NAFNet building blocks
# ----------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensors, as used in NAFNet."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies them elementwise, replacing
    a nonlinear activation (GELU/ReLU) entirely -- the key NAFNet trick."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Global-average-pool -> 1x1 conv channel gate (no softmax/sigmoid
    nonlinearity needed thanks to the multiplicative gating design)."""

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    """One NAFNet block: LN -> 1x1 conv -> depthwise 3x3 -> SimpleGate ->
    SCA -> 1x1 conv, with a residual connection; then a second LN -> 1x1 ->
    SimpleGate -> 1x1 feed-forward sub-block, also residual. Uses
    LayerScale (learnable per-channel residual scale) for training
    stability, matching the original NAFNet design."""

    def __init__(self, channels, expand_ratio=2, ffn_expand_ratio=2):
        super().__init__()
        hidden = channels * expand_ratio
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(hidden // 2)
        self.conv2 = nn.Conv2d(hidden // 2, channels, 1)
        self.scale1 = nn.Parameter(torch.zeros(1, channels, 1, 1))

        ffn_hidden = channels * ffn_expand_ratio
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_hidden, 1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_hidden // 2, channels, 1)
        self.scale2 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + y * self.scale1

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        x = x + y * self.scale2
        return x


# ----------------------------------------------------------------------
# Full NAFNet encoder-decoder with an SR (PixelShuffle) upsampling head
# ----------------------------------------------------------------------
class NAFNetSR(nn.Module):
    """
    NAFNet U-shaped encoder-decoder + PixelShuffle super-resolution head.

    Input:  (B, in_ch, H, W)               -- degraded NoisyLR image
    Output: (B, in_ch, H*scale, W*scale)   -- restored image at GT resolution

    Args:
        in_ch: number of image channels (1 for grayscale, 3 for RGB)
        width: base channel width
        enc_blk_nums: number of NAFBlocks per encoder stage
        dec_blk_nums: number of NAFBlocks per decoder stage
        middle_blk_num: number of NAFBlocks at the bottleneck
        scale: output/input resolution ratio (e.g. 2 for 128->256).
               Set to 1 if your data has no resolution change.
    """

    def __init__(self, in_ch=1, width=32,
                 enc_blk_nums=(2, 2, 4), dec_blk_nums=(2, 2, 2),
                 middle_blk_num=4, scale=2, use_vst=False, vst_k=4.0):
        super().__init__()
        self.scale = scale
        self.use_vst = use_vst
        self.vst_k = vst_k

        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

        self.ending = nn.Conv2d(width, width, 3, padding=1)

        # SR head: upsample by `scale` via PixelShuffle, then refine + project
        if scale > 1:
            self.sr_head = nn.Sequential(
                nn.Conv2d(width, width * (scale ** 2), 3, padding=1),
                nn.PixelShuffle(scale),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1),
                nn.GELU(),
            )
        else:
            self.sr_head = nn.Identity()

        self.out_conv = nn.Conv2d(width, in_ch, 3, padding=1)
        self.padder_size = 2 ** len(enc_blk_nums)

    def _pad_to_multiple(self, x):
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h)), h, w

    @staticmethod
    def _vst(x, k):
        """Signed log1p variance-stabilizing transform.

        Targets the confirmed multiplicative/speckle noise in this data
        (noise level correlates with local pixel intensity, corr=0.864-0.869,
        measured across two independent SEM datasets): the transform
        compresses input magnitude logarithmically, so the proportionally
        larger noise swings that ride on brighter regions get compressed
        along with them.

        Defined for all real x (including the negative excursions the
        NoisyLR spec allows), and has slope 1 at x=0 for any k, so it is
        ~identity near zero and leaves low-amplitude fine detail (and the
        low end of the intensity range, where noise is already small)
        essentially untouched.
        """
        return torch.sign(x) * torch.log1p(torch.abs(x) * k) / k

    def forward(self, x):
        x_in, h, w = self._pad_to_multiple(x)

        # VST applies ONLY to what the network's internal path sees. `x`
        # (untransformed) is still what feeds the bicubic residual `base`
        # below, so the global residual shortcut's math is unaffected --
        # the network still just predicts a correction on top of `base`,
        # it just extracts features from a variance-stabilized view of
        # the input while doing so.
        net_in = self._vst(x_in, self.vst_k) if self.use_vst else x_in

        feat = self.intro(net_in)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = decoder(feat)

        feat = self.ending(feat)
        feat = feat[:, :, :h, :w]  # undo multiple-of padding, back to input res

        feat = self.sr_head(feat)  # upsample by self.scale (identity if scale=1)

        out = self.out_conv(feat)

        # Global residual: bicubic-upsample the raw input and add the
        # network's (denoised, upsampled) residual on top. This lets the
        # network focus on predicting the noise/high-freq correction
        # rather than reconstructing the whole image from scratch.
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                              align_corners=False) if self.scale > 1 else x
        return base + out


def build_model(in_ch=1, width=32, scale=2, size="tiny", use_vst=False, vst_k=4.0):
    """Convenience factory. size in {'tiny','small','base'} trades quality
    for speed/throughput -- see README for measured runtime on your GPU.

    use_vst / vst_k: enable the signed-log1p variance-stabilizing transform
    on the network's internal input path (see NAFNetSR._vst). Defaults to
    False so existing checkpoints -- trained with the raw input -- keep
    loading and behaving exactly as before; only set True for a run that
    is trained (or fine-tuned) with it enabled from the start, since the
    `intro` conv's weights are only meaningful for the input distribution
    they were trained on."""
    configs = {
        "tiny":  dict(width=24, enc_blk_nums=(1, 1, 2), dec_blk_nums=(1, 1, 1), middle_blk_num=2),
        "small": dict(width=32, enc_blk_nums=(2, 2, 4), dec_blk_nums=(2, 2, 2), middle_blk_num=4),
        "base":  dict(width=48, enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2), middle_blk_num=6),
    }
    cfg = configs[size]
    if size == "base":
        # 4-stage variant needs matching down/up lists; not built by default
        # to keep the reference implementation simple. Use 'small' or 'tiny'
        # unless you extend encoders/decoders to 4 stages yourself.
        raise NotImplementedError("'base' 4-stage config left as an exercise; use 'small' or 'tiny'.")
    return NAFNetSR(in_ch=in_ch, scale=scale, use_vst=use_vst, vst_k=vst_k, **cfg)
