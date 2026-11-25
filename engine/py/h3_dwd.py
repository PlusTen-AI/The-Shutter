# pi-H3 DWD head on the shutter engine (libshutter.so h3_dwd_*): the last DiT step's final-block hidden states of the
# target video rows -> pixels, as comfyui-piflow 1.4.1 DWD Decode, in place of the video VAE decode.
# Contract:
#   d = Dwd(path)                        # the pi-H3 adapter file; loads only its dwd_* tensors (~0.65 GB on the device)
#   frames = d.decode(h, temb, grid)     # uint8 cpu [F, 16 lat_h, 16 lat_w, 3], F = 4 T - 3 ceil(T / 5)
#     h:    bf16 cuda [T * lat_h/2 * lat_w/2, 5376] — last step (the last sigma > 0), cond pass, output of the last DiT
#           block (before final_layer), target-video rows only, DiT token order (t, y, x) = dump dwd.in.video_rows
#     temb: fp32 [U, 8] — that step's t_emb table as the DiT computes it (= dump dwd.in.t_emb, h3_dit.cu d->temb);
#           row `row` is used (default 0 = the video stream's t index; ComfyUI takes the video segment's mod row // 3).
#           A single [8] row is also accepted.
#     grid: (T, lat_h, lat_w) of the video latent [1, 24, T, lat_h, lat_w]
#     uint8 = ComfyUI SaveVideo 8-bit: (fp32(px) * 255).clamp(0, 255) truncated (.byte()), px = the fp16 DWD output
#   px = d.decode(h, temb, grid, out="f16")   # cuda fp16 [F, H, W, 3] in [0, 1] = DWD Decode output (before .float())
#   d.close()
import ctypes

import torch

import models

LIB = models.LIB
NTAP = 12   # gate taps: dwd_norm_out, expanded, 8 layers, norm_out, pre-colour (golden dump layouts)

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(LIB)
        P, I = ctypes.c_void_p, ctypes.c_int
        L.h3_dwd_new.restype = P
        L.h3_dwd_new.argtypes = [ctypes.c_char_p]
        L.h3_dwd_frames.argtypes = [I]
        L.h3_dwd_bytes.restype = ctypes.c_longlong
        L.h3_dwd_bytes.argtypes = [P]
        L.h3_dwd_taps.argtypes = [P, P, P]
        L.h3_dwd_decode.argtypes = [P, P, P, I, I, I, P]
        L.h3_dwd_free.argtypes = [P]
        _lib = L
    return _lib


def frames(T):
    return lib().h3_dwd_frames(T)


class Dwd:
    def __init__(self, path=models.path(models.PIH3_ADAPTER % 2), device="cuda"):
        assert device == "cuda"
        self.h = ctypes.c_void_p(lib().h3_dwd_new(path.encode()))

    def bytes(self):
        return lib().h3_dwd_bytes(self.h)

    def taps(self, out=None, inp=None):
        """Gate only: NTAP cuda tensors (or None) receiving stage outputs / replacing stage inputs; taps() clears."""
        arr = lambda ts: None if ts is None else (ctypes.c_void_p * NTAP)(*[t.data_ptr() if t is not None else None for t in ts])  # noqa: E731
        self._keep = (out, inp)
        lib().h3_dwd_taps(self.h, arr(out), arr(inp))

    def decode(self, h, temb, grid, row=0, out="u8"):
        T, lh, lw = (int(v) for v in grid)
        assert h.dtype == torch.bfloat16 and h.is_cuda and tuple(h.shape) == (T * (lh // 2) * (lw // 2), 5376), h.shape
        t = (temb[row] if temb.dim() == 2 else temb).to("cuda", torch.float32).contiguous()
        assert t.numel() == 8
        h = h.contiguous()
        px = torch.empty(frames(T), lh * 16, lw * 16, 3, dtype=torch.float16, device="cuda")
        n = lib().h3_dwd_decode(self.h, ctypes.c_void_p(h.data_ptr()), ctypes.c_void_p(t.data_ptr()), T, lh, lw,
                                ctypes.c_void_p(px.data_ptr()))
        assert n == px.shape[0], (n, px.shape)
        if out == "f16":
            return px
        return torch.cat([(px[i:i + 16].float() * 255).clamp(0, 255).to(torch.uint8).cpu()   # SaveVideo: .byte()
                          for i in range(0, n, 16)])                                            # truncates

    def close(self):
        if self.h:
            lib().h3_dwd_free(self.h)
            self.h = None
