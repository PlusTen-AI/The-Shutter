# LTX-2.5 text encoder on the shutter engine (libshutter.so ltx_te_*): Gemma4-12B + the LTX dual projection.
# Tokens as ComfyUI e9afbccc2 CLIPTextEncode with the LTX-2.5 TE (Gemma4UnifiedSDTokenizer, no chat template): BOS 2 +
# tokenizer.encode(prompt, add_special_tokens=False), from the tokenizer json embedded in the weights file. ComfyUI
# pads on the left to 1024; the engine computes the real tokens only (same rows), so no padding here.
# (sd1_clip's "embedding:<name>" textual-inversion syntax is not handled: the text is tokenized as is.)
#   te = LtxTE()                  # ~16 GB on the device
#   ids, cond = te.encode(text)   # cond: cuda bf16 [n, 6144] = video 4096 | audio 2048 (ComfyUI's conditioning, bf16)
#   te.free()
import ctypes
import json
import struct

import torch
from tokenizers import Tokenizer

import models

LIB = models.LIB
BOS, COND_DIM = 2, 6144

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(LIB)
        P, I = ctypes.c_void_p, ctypes.c_int
        L.ltx_te_new.restype = P
        L.ltx_te_new.argtypes = [ctypes.c_char_p]
        L.ltx_te_encode.restype = I
        L.ltx_te_encode.argtypes = [P, ctypes.POINTER(I), I, P, I]
        L.ltx_te_project.argtypes = [P, P, I, P]
        L.ltx_te_taps.argtypes = [P, P]
        L.ltx_te_free.argtypes = [P]
        _lib = L
    return _lib


def _st_bytes(path, name):
    """One tensor's raw bytes from a safetensors file (header parse, no torch)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        a, b = json.loads(f.read(n))[name]["data_offsets"]
        f.seek(8 + n + a)
        return f.read(b - a)


class LtxTokenizer:
    def __init__(self, path):
        self.t = Tokenizer.from_str(_st_bytes(path, "tokenizer_json").decode("utf-8"))

    def __call__(self, text):
        # sd1_clip escape_important -> unescape_important (weights are off): "\(" / "\)" become "(" / ")"
        text = text.replace("\\)", "\0\1").replace("\\(", "\0\2").replace("\0\1", ")").replace("\0\2", "(")
        return [BOS] + self.t.encode(text, add_special_tokens=False).ids


class LtxTE:
    def __init__(self, path=None):
        path = path or models.path(models.LTX_TE)
        self.tok = LtxTokenizer(path)
        self.h = ctypes.c_void_p(lib().ltx_te_new(path.encode()))

    def encode_ids(self, ids):
        out = torch.empty(len(ids), COND_DIM, dtype=torch.bfloat16, device="cuda")
        r = lib().ltx_te_encode(self.h, (ctypes.c_int * len(ids))(*ids), len(ids), ctypes.c_void_p(out.data_ptr()), len(ids))
        if r != len(ids):
            raise RuntimeError(f"ltx_te_encode returned {r}, expected {len(ids)}")
        return out

    def encode(self, text):
        ids = self.tok(text)
        return ids, self.encode_ids(ids)

    def free(self):
        if self.h:
            lib().ltx_te_free(self.h)
            self.h = None
