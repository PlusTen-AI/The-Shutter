# SeedVR2 video VAE on the shutter engine (engine/seedvr2_vae.cu): ComfyUI VAEEncodeTiled / VAEDecodeTiled with the
# blueprint widgets [512, 128, 64, 8] (spatial tiles, causal time slicing, --bf16-vae numerics).
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
        L.sv2_vae_new.restype = P
        L.sv2_vae_new.argtypes = [ctypes.c_char_p]
        L.sv2_vae_encode.argtypes = [P, P, I, I, I, P]
        L.sv2_vae_decode.argtypes = [P, P, I, I, I, P]
        L.sv2_vae_frames.argtypes = [I]
        L.sv2_vae_latent_frames.argtypes = [I]
        L.sv2_vae_free.argtypes = [P]
        _lib = L
    return _lib


class Vae:
    def __init__(self, path=None):
        self.h = ctypes.c_void_p(lib().sv2_vae_new((path or models.path(models.SV2_VAE)).encode()))

    def encode(self, px):
        """float32 [F, H, W, 3] (or [1, F, H, W, 3]) in 0..1 on cuda (F = 4n+1, H, W multiples of 16: after
        SeedVR2Preprocess) -> scaled latent [1, 16, (F+3)//4, H/8, W/8] float32 (what VAEEncodeTiled returns)."""
        x = px.reshape(px.shape[-4:]).float().contiguous()
        F_, H, W, _ = x.shape
        out = torch.empty(16, lib().sv2_vae_latent_frames(F_), H // 8, W // 8, device="cuda")
        torch.cuda.synchronize()
        lib().sv2_vae_encode(self.h, ctypes.c_void_p(x.data_ptr()), F_, H, W, ctypes.c_void_p(out.data_ptr()))
        return out[None]

    def decode(self, lat):
        """scaled latent [1, 16, T, h, w] (or [16, T, h, w]) on cuda -> float32 frames [4T-3, 8h, 8w, 3] in 0..1
        (what VAEDecodeTiled returns)."""
        z = lat.reshape(lat.shape[-4:]).float().contiguous()
        _, T, h, w = z.shape
        out = torch.empty(lib().sv2_vae_frames(T), 8 * h, 8 * w, 3, device="cuda")
        torch.cuda.synchronize()
        lib().sv2_vae_decode(self.h, ctypes.c_void_p(z.data_ptr()), T, h, w, ctypes.c_void_p(out.data_ptr()))
        return out

    def close(self):
        if self.h:
            lib().sv2_vae_free(self.h)
            self.h = None
