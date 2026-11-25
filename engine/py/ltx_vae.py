# LTX-2.5 video VAE and latent upsampler x2 on the shutter engine (engine/ltx_vae.cu): ComfyUI CausalDiffusionVAE
# encode / decode (untiled) and LTXVLatentUpsampler, --bf16-vae numerics.
import ctypes

import torch

import models

LIB = models.LIB

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(LIB)
        P, I = ctypes.c_void_p, ctypes.c_int
        L.ltx_vae_new.restype = P
        L.ltx_vae_new.argtypes = [ctypes.c_char_p]
        L.ltx_vae_encode.argtypes = [P, P, I, I, I, P]
        L.ltx_vae_decode.argtypes = [P, P, I, I, I, P, P, P]
        L.ltx_vae_frames.argtypes = [I]
        L.ltx_vae_latent_frames.argtypes = [I]
        L.ltx_vae_free.argtypes = [P]
        L.ltx_up_new.restype = P
        L.ltx_up_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        L.ltx_up_run.argtypes = [P, P, I, I, I, P]
        L.ltx_up_free.argtypes = [P]
        L.ltx_na3d.argtypes = [P, P, P, P] + [I] * 10
        L.ltx_vae_stage5.argtypes = [P, P, I, I, I, P, P, P]
        _lib = L
    return _lib


def _p(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else None


def noise(F, H, W, seed=0):
    """The decoder's x_t for an output of F x H x W: depends on (seed, size) only. torch's CUDA generator in bf16, so
    seed 0 is exactly the noise ComfyUI's untiled decode draws (CausalDiffusionVAE.decode fixes seed 0)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((1, 3, F, H, W), dtype=torch.bfloat16, device="cuda", generator=g)


class Vae:
    def __init__(self, path=None):
        self.h = ctypes.c_void_p(lib().ltx_vae_new((path or models.path(models.LTX_VAE)).encode()))

    def encode(self, px):
        """float32 [F, H, W, 3] (or [1, F, H, W, 3]; an image [1, H, W, 3] is one frame) in 0..1 on cuda, H and W
        multiples of 32 -> normalized latent [1, 128, T, H/32, W/32] float32 (frames past the last 8k + 1 dropped)."""
        x = px.reshape(-1, *px.shape[-3:]).float().contiguous()
        F_, H, W, _ = x.shape
        T = lib().ltx_vae_latent_frames(F_)
        out = torch.empty(128, T, H // 32, W // 32, device="cuda")
        torch.cuda.synchronize()
        lib().ltx_vae_encode(self.h, _p(x), F_, H, W, _p(out))
        return out[None]

    def decode(self, lat, seed=0, x_t=None, raw=False):
        """normalized latent [1, 128, T, h, w] (or [128, T, h, w]) on cuda -> float32 frames [8T-7, 32h, 32w, 3] in 0..1
        (untiled). x_t: the decoder noise [1, 3, F, H, W] bf16 (default noise(F, H, W, seed)). raw: also return x0
        [1, 3, F, H, W] bf16."""
        z = lat.reshape(lat.shape[-4:]).float().contiguous()
        _, T, h, w = z.shape
        F_, H, W = lib().ltx_vae_frames(T), 32 * h, 32 * w
        if x_t is None:
            x_t = noise(F_, H, W, seed)
        x_t = x_t.to(torch.bfloat16).contiguous()
        assert tuple(x_t.shape[-4:]) == (3, F_, H, W), x_t.shape
        frames = torch.empty(F_, H, W, 3, device="cuda")
        r = torch.empty(1, 3, F_, H, W, dtype=torch.bfloat16, device="cuda") if raw else None
        torch.cuda.synchronize()
        lib().ltx_vae_decode(self.h, _p(z), T, h, w, _p(x_t), _p(frames), _p(r))
        return (frames, r) if raw else frames

    def stage5(self, ctx, x_t):
        """test: decoder stage 5 alone on a context [1, F, h5, w5, 256] bf16 -> raw x0 [1, 3, F, 4 h5, 4 w5] bf16."""
        c = ctx.reshape(ctx.shape[-4:]).to(torch.bfloat16).contiguous()
        F_, h5, w5, _ = c.shape
        r = torch.empty(1, 3, F_, 4 * h5, 4 * w5, dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()
        lib().ltx_vae_stage5(self.h, _p(c), F_, h5, w5, _p(x_t.to(torch.bfloat16).contiguous()), None, _p(r))
        return r

    def close(self):
        if self.h:
            lib().ltx_vae_free(self.h)
            self.h = None


class Upsampler:
    def __init__(self, path=None, vae_path=None):
        self.h = ctypes.c_void_p(lib().ltx_up_new((path or models.path(models.LTX_UP)).encode(),
                                                  (vae_path or models.path(models.LTX_VAE)).encode()))

    def __call__(self, lat):
        """normalized latent [1, 128, T, h, w] on cuda -> [1, 128, T, 2h, 2w] float32 (LTXVLatentUpsampler)."""
        z = lat.reshape(lat.shape[-4:]).float().contiguous()
        _, T, h, w = z.shape
        out = torch.empty(128, T, 2 * h, 2 * w, device="cuda")
        torch.cuda.synchronize()
        lib().ltx_up_run(self.h, _p(z), T, h, w, _p(out))
        return out[None]

    def close(self):
        if self.h:
            lib().ltx_up_free(self.h)
            self.h = None


def na3d(q, k, v, kernel, tile=(2, 8, 8)):
    """test: neighborhood attention (scale 1) on [T, H, W, heads, 64] (or with a leading 1) bf16 cuda tensors."""
    q, k, v = (t.reshape(t.shape[-5:]).to(torch.bfloat16).contiguous() for t in (q, k, v))
    T, H, W, nh, hd = q.shape
    assert hd == 64
    o = torch.empty_like(q)
    torch.cuda.synchronize()
    lib().ltx_na3d(_p(q), _p(k), _p(v), _p(o), T, H, W, nh, *kernel, *tile)
    return o
