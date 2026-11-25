# SeedVR2 7B video upscale on the shutter engine (3rd stage, video.upscale). Mirrors the ComfyUI e9afbccc2 graph
# (blueprint "Upscale Video (SeedVR2 3B Int8)" with the 7B DiT): resize -> SeedVR2Preprocess -> VAEEncodeTiled ->
# SeedVR2Conditioning -> KSampler (1 step, euler, simple, cfg 1, denoise 1) -> VAEDecodeTiled -> SeedVR2PostProcessing.
# The resize is ours: cover the target (1080x1920 portrait / 1920x1080 landscape) with lanczos, centre crop.
#   s = SeedVR2(); frames = s.upscale(frames_u8, seed, size=None, color="lab")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import models



def target(w, h):
    """Short-form standard: portrait 1080x1920, landscape 1920x1080 (square-ish clips keep their aspect at 1080 short)."""
    if h >= 1.5 * w:
        return 1080, 1920
    if w >= 1.5 * h:
        return 1920, 1080
    s = 1080 / min(w, h)
    return round(w * s / 2) * 2, round(h * s / 2) * 2


def resize_cover(frames, W, H):
    """uint8 [F, h, w, 3] -> uint8 [F, H, W, 3]: PIL lanczos (comfy.utils.lanczos on uint8) to cover W x H, centre crop."""
    h, w = frames.shape[1:3]
    s = max(W / w, H / h)
    rw, rh = max(W, round(w * s)), max(H, round(h * s))
    x0, y0 = (rw - W) // 2, (rh - H) // 2
    if (rw, rh) == (w, h):
        return frames[:, y0:y0 + H, x0:x0 + W]
    return np.stack([np.asarray(Image.fromarray(f).resize((rw, rh), Image.Resampling.LANCZOS))[y0:y0 + H, x0:x0 + W]
                     for f in frames])


def preprocess(px):
    """SeedVR2Preprocess: [F, H, W, 3] 0..1 -> clamp, zero pad bottom / right to /16, repeat the last frame to 4n+1."""
    F_, H, W, _ = px.shape
    px = F.pad(px.clamp(0, 1), (0, 0, 0, (16 - W % 16) % 16, 0, (16 - H % 16) % 16))
    n = F_ if F_ == 1 or (F_ > 4 and (F_ - 1) % 4 == 0) else 5 if F_ <= 4 else F_ + 4 - (F_ - 1) % 4
    return torch.cat([px, px[-1:].expand(n - F_, *px.shape[1:])]) if n > F_ else px


# ---- SeedVR2PostProcessing "lab" (comfy/ldm/seedvr/color_fix.py, per frame, fp32): wavelet detail on the original's
# low frequencies, then CIELAB histogram match of a / b and 20 % of L
DELTA, KAPPA, WX, WZ = 6.0 / 29.0, (29.0 / 3.0) ** 3, 0.95047, 1.08883
RGB2XYZ = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
XYZ2RGB = [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]]


def _blur(x, r):
    r = min(r, max(1, min(x.shape[-2:]) // 8))
    k = torch.tensor([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]], dtype=x.dtype, device=x.device)
    return F.conv2d(F.pad(x, (r, r, r, r), mode="replicate"), k[None, None].repeat(x.shape[1], 1, 1, 1), groups=x.shape[1], dilation=r)


def _wavelet(x, levels=5):
    high = torch.zeros_like(x)
    for i in range(levels):
        low = _blur(x, 2 ** i)
        high.add_(x).sub_(low)
        x = low
    return high, low


def _match(src, ref):
    s, idx = torch.sort(src.flatten())
    return torch.sort(ref.flatten())[0][torch.argsort(idx)].reshape(src.shape)   # same pixel count: no quantile map


def _to_lab(rgb, m):
    lin = torch.where(rgb > 0.04045, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    xyz = (lin.permute(0, 2, 3, 1).reshape(-1, 3) @ m.T).reshape(rgb.shape[0], rgb.shape[2], rgb.shape[3], 3).permute(0, 3, 1, 2)
    xyz[:, 0].div_(WX)
    xyz[:, 2].div_(WZ)
    f = torch.where(xyz > DELTA ** 3, torch.pow(xyz, 1.0 / 3.0), xyz.mul(KAPPA).add_(16.0).div_(116.0))
    return torch.stack([f[:, 1].mul(116.0).sub_(16.0), (f[:, 0] - f[:, 1]).mul_(500.0), (f[:, 1] - f[:, 2]).mul_(200.0)], 1)


def _from_lab(lab, m):
    fy = (lab[:, 0] + 16.0) / 116.0
    fx, fz = lab[:, 1].div(500.0).add_(fy), fy - lab[:, 2] / 200.0
    xyz = [torch.where(t > DELTA, torch.pow(t, 3.0), t.mul(116.0).sub_(16.0).div_(KAPPA)) for t in (fx, fy, fz)]
    xyz = torch.stack([xyz[0].mul_(WX), xyz[1], xyz[2].mul_(WZ)], 1)
    B, _, H, W = xyz.shape
    lin = (xyz.permute(0, 2, 3, 1).reshape(-1, 3) @ m.T).reshape(B, H, W, 3).permute(0, 3, 1, 2)
    rgb = torch.where(lin > 0.0031308, torch.pow(lin.clamp(min=0.0), 1.0 / 2.4).mul_(1.055).sub_(0.055), lin * 12.92)
    return rgb.clamp(0.0, 1.0)


def lab_transfer(content, style, lum=0.8):
    """lab_color_transfer on one frame [1, 3, H, W] in -1..1 (fp32) -> [1, 3, H, W] in -1..1"""
    high, _ = _wavelet(content)
    _, low = _wavelet(style)
    content = high.add_(low).clamp_(-1.0, 1.0).add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    style = style.add(1.0).mul_(0.5).clamp_(0.0, 1.0)
    m = torch.tensor(RGB2XYZ, device=content.device)
    c, s = _to_lab(content, m), _to_lab(style, m)
    L = c[:, 0].mul(lum).add_(_match(c[:, 0], s[:, 0]).mul(1.0 - lum))
    lab = torch.stack([L, _match(c[:, 1], s[:, 1]), _match(c[:, 2], s[:, 2])], 1)
    return _from_lab(lab, torch.tensor(XYZ2RGB, device=content.device)).mul_(2.0).sub_(1.0)


class SeedVR2:
    def __init__(self):
        import sv2_dit
        import sv2_vae
        self.vae = sv2_vae.Vae(models.path(models.SV2_VAE))
        self.dit = sv2_dit.Dit(models.path(models.SV2_DIT))   # load 25-35 s, peak ~17 GiB at 1088x1920 x 125

    def close(self):
        for k in ("dit", "vae"):
            if getattr(self, k, None) is not None:
                getattr(self, k).close()
                setattr(self, k, None)
        torch.cuda.empty_cache()

    def upscale(self, frames, seed, size=None, color="lab", progress=None):
        """uint8 [F, h, w, 3] -> uint8 [F, H, W, 3] at size (W, H) (default target())."""
        p = progress or (lambda f: None)
        W, H = size or target(frames.shape[2], frames.shape[1])
        ref = torch.from_numpy(resize_cover(frames, W, H)).cuda().float().div_(255.0)   # original_resized_images
        p(0.05)
        lat = self.vae.encode(preprocess(ref))                                      # [1, 16, T, h, w]
        p(0.3)
        cond = torch.cat([lat, torch.ones_like(lat[:, :1])], 1)                    # SeedVR2Conditioning
        import sv2_dit
        lat = self.dit.denoise(sv2_dit.comfy_noise(seed, tuple(lat.shape)).cuda(), cond)
        p(0.6)
        out = self.vae.decode(lat)[:ref.shape[0], :H, :W]                          # [F, H, W, 3] 0..1
        p(0.9)
        if color == "lab":
            for i in range(out.shape[0]):
                f = lab_transfer(out[i:i + 1].permute(0, 3, 1, 2).mul(2.0).sub(1.0), ref[i:i + 1].permute(0, 3, 1, 2).mul(2.0).sub(1.0))
                out[i] = f[0].permute(1, 2, 0).add(1.0).div(2.0).clamp(0.0, 1.0)
        p(1.0)
        return (255.0 * out.clamp(0, 1)).to(torch.uint8).cpu().numpy()   # SaveImage-style truncation
