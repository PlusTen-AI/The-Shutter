# SeedVR2 7B DiT on the shutter engine (libshutter.so, engine/seedvr2_dit.cu) + the blueprint's 1-step sampler
# (ComfyUI e9afbccc2: KSampler 1 step, cfg 1, euler, simple, denoise 1, FLOW shift 1 -> sigma 1, timestep 1000).
# torch is used for device memory, ComfyUI-identical noise and the fp32 sampler math.
import ctypes

import torch

import models

LIB = models.LIB
NB, D, NT, NEMB = 36, 3072, 58, 18432

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(LIB)
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        L.sv2_dit_new.restype = P
        L.sv2_dit_new.argtypes = [ctypes.c_char_p]
        L.sv2_dit_forward.argtypes = [P, P, P, I, I, I, F, P]
        L.sv2_dit_taps.argtypes = [P] * 6
        L.sv2_dit_block.argtypes = [P, I, P, P, P, P]
        L.sv2_dit_windows.argtypes = [I, I, I, I, ctypes.POINTER(I), I]
        L.sv2_dit_mem.restype = ctypes.c_size_t
        L.sv2_dit_mem.argtypes = [P, I]
        L.sv2_dit_free.argtypes = [P]
        _lib = L
    return _lib


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr() if t is not None else 0)


def comfy_noise(seed, shape):
    """comfy.sample.prepare_noise: CPU generator, fp32 randn over the whole latent [1, 16, T, H, W]."""
    return torch.randn(shape, dtype=torch.float32, generator=torch.manual_seed(seed), device="cpu")


def windows(T, H, W, shifted):
    """Window list of the latent grid [T, H, W] as [t0, t1, h0, h1, w0, w1] on the token grid (H/2, W/2)."""
    n = lib().sv2_dit_windows(T, H, W, int(shifted), None, 0)
    buf = (ctypes.c_int * (6 * n))()
    lib().sv2_dit_windows(T, H, W, int(shifted), buf, n)
    return [list(buf[6 * i:6 * i + 6]) for i in range(n)]


class Dit:
    def __init__(self, path=models.path(models.SV2_DIT)):
        self.h = lib().sv2_dit_new(path.encode())

    def forward(self, x, cond, timestep):
        """Raw model output v (flow velocity) [1, 16, T, H, W] bf16 for x [1, 16, T, H, W] and the SeedVR2Conditioning
        condition [1, 17, T, H, W] (latent + mask of ones); both are cast to bf16 like ComfyUI's _apply_model."""
        xb = x[0].to(device="cuda", dtype=torch.bfloat16).contiguous()
        cb = cond[0].to(device="cuda", dtype=torch.bfloat16).contiguous()
        assert xb.shape[0] == 16 and cb.shape[0] == 17 and xb.shape[1:] == cb.shape[1:], (x.shape, cond.shape)
        _, T, H, W = xb.shape
        v = torch.empty_like(xb)
        torch.cuda.synchronize()
        lib().sv2_dit_forward(self.h, _ptr(xb), _ptr(cb), T, H, W, float(timestep), _ptr(v))
        return v[None]

    def denoise(self, noise, cond, sigma=1.0):
        """What ComfyUI's 1-step sampler returns as the final latent [1, 16, T, H, W] fp32 (device):
        x = sigma * noise + (1 - sigma) * latent (latent = cond[:, :16]; x = noise at sigma 1), v = DiT(x, sigma * 1000),
        euler to sigma 0: denoised = x - v * sigma, d = (x - denoised) / sigma, x + d * (0 - sigma)."""
        s = torch.tensor(sigma, dtype=torch.float32)
        x = noise.to(device="cuda", dtype=torch.float32)
        if sigma != 1.0:
            x = s * x + (1.0 - s) * cond[:, :16].to(device="cuda", dtype=torch.float32)
        v = self.forward(x, cond, (s * 1000).item()).float()
        den = x - v * s
        d = (x - den) / s
        return x + d * (0.0 - s)

    def taps(self, vid=None, txt=None, emb=None, qk=None, att=None):
        """Debug buffers (bf16, device) filled by the next forward; see sv2_dit_taps in seedvr2_dit.cu for the layouts."""
        lib().sv2_dit_taps(self.h, _ptr(vid), _ptr(txt), _ptr(emb), _ptr(qk), _ptr(att))

    def block(self, i, vid, txt):
        """Debug: block i alone on vid [M, 3072] / txt [58, 3072] bf16 with the grid and timestep of the last forward."""
        vid, txt = vid.contiguous(), txt.contiguous()
        vo, to = torch.empty_like(vid), torch.empty_like(txt)
        torch.cuda.synchronize()
        lib().sv2_dit_block(self.h, i, _ptr(vid), _ptr(txt), _ptr(vo), _ptr(to))
        return vo, to

    def mem(self, peak=False):
        return lib().sv2_dit_mem(self.h, int(peak))

    def close(self):
        if self.h:
            lib().sv2_dit_free(self.h)
            self.h = None
