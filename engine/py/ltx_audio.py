# LTX-2.5 audio decode on the shutter engine (libshutter.so ltx_audio_*), as ComfyUI e9afbccc2 LTXVAudioVAEDecode:
# audio latent [1, 8, T, 16] (normalized, as the sampler leaves it) -> stereo waveform at 48 kHz, 480 (4T - 3) samples
# (audio VAE decoder -> mel [2, 4T-3, 64] -> vocoder 16 kHz -> BWE 48 kHz).
#   a = LtxAudio()            # ~0.4 GB on the device
#   wav = a.decode(lat)       # cuda f32 [2, 480 (4T - 3)], SAMPLE_RATE
#   a.free()
import ctypes

import torch

import models

LIB = models.LIB
SAMPLE_RATE = 48000
TAPS = ("denorm", "mel", "voc16k", "bwe_mel", "bwe_residual", "bwe_skip")

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(LIB)
        P, I = ctypes.c_void_p, ctypes.c_int
        L.ltx_audio_new.restype = P
        L.ltx_audio_new.argtypes = [ctypes.c_char_p]
        L.ltx_audio_taps.argtypes = [P, P]
        for n in ("ltx_audio_decode", "ltx_audio_vocode", "ltx_audio_bwe"):
            getattr(L, n).restype = I
            getattr(L, n).argtypes = [P, P, I, P]
        L.ltx_audio_free.argtypes = [P]
        _lib = L
    return _lib


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


class LtxAudio:
    def __init__(self, path=None):
        path = path or models.path(models.LTX_AUDIO)
        self.h = ctypes.c_void_p(lib().ltx_audio_new(path.encode()))

    def decode(self, lat):
        """lat [1, 8, T, 16] or [8, T, 16] (any float dtype, cuda) -> cuda f32 [2, 480 (4T - 3)]"""
        z = lat.reshape(8, -1, 16).to(device="cuda", dtype=torch.float32).contiguous()
        T = z.shape[1]
        out = torch.empty(2, 480 * (4 * T - 3), device="cuda")
        lib().ltx_audio_decode(self.h, _p(z), T, _p(out))
        return out

    def vocode(self, mel):
        """mel [2, M, 64] (ComfyUI layout) -> cuda bf16 [2, 160 M] at 16 kHz (vocoder only)"""
        m = mel.reshape(2, -1, 64).to(device="cuda", dtype=torch.bfloat16).contiguous()
        out = torch.empty(2, 160 * m.shape[1], device="cuda", dtype=torch.bfloat16)
        lib().ltx_audio_vocode(self.h, _p(m), m.shape[1], _p(out))
        return out

    def bwe(self, wav):
        """wav [2, L] at 16 kHz -> cuda f32 [2, 3 L] at 48 kHz (bandwidth extension only)"""
        w = wav.reshape(2, -1).to(device="cuda", dtype=torch.bfloat16).contiguous()
        out = torch.empty(2, 3 * w.shape[1], device="cuda")
        lib().ltx_audio_bwe(self.h, _p(w), w.shape[1], _p(out))
        return out

    def taps(self, bufs=None):
        """bufs: dict name -> cuda bf16 tensor (names in TAPS, ComfyUI layouts) filled by later calls; None = off"""
        self._taps = bufs  # keep the tensors alive while the engine holds their pointers
        if not bufs:
            lib().ltx_audio_taps(self.h, None)
            return
        arr = (ctypes.c_void_p * len(TAPS))(*[bufs[n].data_ptr() if n in bufs else None for n in TAPS])
        lib().ltx_audio_taps(self.h, arr)

    def free(self):
        if self.h:
            lib().ltx_audio_free(self.h)
            self.h = None
