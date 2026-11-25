# Qwen-Image 2.1 pipeline on the shutter engine (libshutter.so). Runs in the worker container
# (nvcr pytorch image: torch is used for device memory, ComfyUI-identical noise and image I/O only).
import ctypes

import torch

from qi21_sched import TURBO_SIGMAS, flux_simple, manual
import models

LIB = models.LIB

# latent_format QwenImage21: sampler space <-> VAE space
LAT_MEAN = [0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510, 0.2732, 0.7562,
            -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826,
            3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751,
            -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.0960,
            -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562,
            0.0304]
LAT_STD = [3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675,
           3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719,
           3.7498, 4.2830, 3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833,
           3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
           3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161]

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(LIB)
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        L.qi21_dit_new.restype = P
        L.qi21_dit_new.argtypes = [ctypes.c_char_p]
        L.qi21_dit_prefix.argtypes = [P, P, I, I, ctypes.POINTER(P), ctypes.POINTER(I), ctypes.POINTER(I), I, I]
        L.qi21_dit_step.argtypes = [P, P, F, F]
        L.qi21_dit_step_cfg.argtypes = [P, P, F, F, F]
        L.qi21_dit_swap.argtypes = [P]
        L.qi21_dit_drop_alt.argtypes = [P]
        L.qi21_dit_forward.argtypes = [P, P, F, P]
        L.qi21_dit_free.argtypes = [P]
        _lib = L
    return _lib


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


class Dit:
    def __init__(self, turbo=False):
        self.turbo = turbo
        self.neg = False
        self.h =ctypes.c_void_p(lib().qi21_dit_new(models.path(models.QI_TURBO if turbo else models.QI_DIT).encode()))
        self.mean = torch.tensor(LAT_MEAN, device="cuda").view(64, 1, 1)
        self.std = torch.tensor(LAT_STD, device="cuda").view(64, 1, 1)

    def close(self):
        if self.h:
            lib().qi21_dit_free(self.h)
            self.h = None

    def prefix(self, ctx, refs, slots, H, W, neg=None):
        """ctx [L,4096] bf16 cuda; refs: VAE-space latents [64,h,w] (any float dtype); slots: text index per ref.
        neg: (ctx, slots) of the negative prompt for cfg > 1 — the same references, its own cached prefix (ComfyUI's
        TextEncodeQwenImage21 gives the negative the reference images too)."""
        refs = [((r.float() - self.mean) / self.std).to(torch.bfloat16).contiguous() for r in refs]  # process_in
        n = len(refs)
        rp = (ctypes.c_void_p * max(n, 1))(*[r.data_ptr() for r in refs])
        hw = (ctypes.c_int * max(2 * n, 1))(*[v for r in refs for v in r.shape[1:]])

        def run(c, s):
            sl = (ctypes.c_int * max(n, 1))(*(s or [0]))
            torch.cuda.synchronize()
            lib().qi21_dit_prefix(self.h, _ptr(c), c.shape[0], n, rp, hw, sl if n else None, H, W)
        if neg is not None:
            run(*neg)
            lib().qi21_dit_swap(self.h)     # the negative waits in the second context
        else:
            lib().qi21_dit_drop_alt(self.h)
        run(ctx, slots)
        self.neg = neg is not None
        self._keep = refs

    def sample(self, H, W, seed, steps=None, progress=None, cfg=1.0):
        """Euler flow sampling from ComfyUI-identical noise; returns the VAE-space latent [64,H,W] fp32. cfg != 1 needs
        a negative prefix (two DiT calls a step); cfg 1 skips it, as ComfyUI does."""
        use_cfg = self.neg and abs(float(cfg) - 1.0) > 1e-6
        sig = manual(TURBO_SIGMAS) if self.turbo else flux_simple(steps or 25)
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn((1, 64, H, W), dtype=torch.float32, generator=g)[0].cuda()
        lat_in = (torch.zeros(64, H, W, device="cuda") - self.mean) / self.std  # process_in(empty latent)
        x = (sig[0] * noise + (1.0 - sig[0]) * lat_in).contiguous()
        torch.cuda.synchronize()
        for i in range(len(sig) - 1):
            if use_cfg:
                lib().qi21_dit_step_cfg(self.h, _ptr(x), sig[i], sig[i + 1], float(cfg))
            else:
                lib().qi21_dit_step(self.h, _ptr(x), sig[i], sig[i + 1])
            if progress:
                progress((i + 1) / (len(sig) - 1))
        return x * self.std + self.mean  # process_out


class Vae:
    def __init__(self):
        L = lib()
        P, I = ctypes.c_void_p, ctypes.c_int
        L.qi21_vae_new.restype = P
        L.qi21_vae_new.argtypes = [ctypes.c_char_p]
        L.qi21_vae_decode.argtypes = [P, P, I, I, I, P]
        L.qi21_vae_encode.argtypes = [P, P, I, I, I, P]
        L.qi21_vae_free.argtypes = [P]
        self.h = ctypes.c_void_p(L.qi21_vae_new(models.path(models.QI_VAE).encode()))

    def close(self):
        if self.h:
            lib().qi21_vae_free(self.h)
            self.h = None

    def decode(self, lat):
        """VAE-space latent [64,h,w] -> image [16h,16w,4] float 0..1 (RGBA) on cuda."""
        lat = lat.float().contiguous()
        out = torch.empty(lat.shape[1] * 16, lat.shape[2] * 16, 4, device="cuda")
        torch.cuda.synchronize()
        lib().qi21_vae_decode(self.h, _ptr(lat), 1, lat.shape[1], lat.shape[2], _ptr(out))
        return out

    def encode(self, img):
        """image [H,W,C] float 0..1 -> VAE-space latent [64,H/16,W/16] (center-cropped to multiples of 16)."""
        img = img.float().contiguous().cuda()
        H, W, C = img.shape
        out = torch.empty(64, H // 16, W // 16, device="cuda")
        torch.cuda.synchronize()
        lib().qi21_vae_encode(self.h, _ptr(img), H, W, C, _ptr(out))
        return out


def to_png(img, path):
    """Same conversion as ComfyUI SaveImage (truncating 255*x), RGB only."""
    import numpy as np
    from PIL import Image
    a = (255.0 * img[..., :3].float().cpu().numpy()).clip(0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def t2i_size(opts):
    """Output pixels (W, H), multiples of 16: explicit width/height, else aspect 'a:b' (default 1:1) x megapixels (default 1)."""
    import math
    if opts.get("width") and opts.get("height"):
        w, h = int(opts["width"]), int(opts["height"])
    else:
        a, b = (float(v) for v in str(opts.get("aspect", "1:1")).split(":"))
        total = float(opts.get("mp", 1.0)) * 1024 * 1024
        w = math.sqrt(total * a / b)
        h = w * b / a
    return max(256, round(w / 16) * 16), max(256, round(h / 16) * 16)


def load_ref(path, resolution=1024):
    """LoadImage + TextEncodeQwenImage21 resize: ~resolution^2 pixels at multiples of 32 (0 = own size), PIL lanczos on uint8."""
    import math
    import numpy as np
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w0, h0 = im.size
    if resolution > 0:
        ratio = w0 / h0
        w = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
        h = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
    else:
        w, h = round(w0 / 32) * 32, round(h0 / 32) * 32
    w, h = max(32, w), max(32, h)
    if (w, h) != (w0, h0):
        im = im.resize((w, h), resample=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(im).astype(np.float32) / 255.0)


class Te:
    """Qwen3-VL text encoder + tokenizer. encode(prompt, images) -> (cond [L,4096] bf16 cuda, image slots)."""
    CAP = 8192

    def __init__(self):
        from qi21_tok import QI21Tokenizer
        L = lib()
        P, I = ctypes.c_void_p, ctypes.c_int
        L.qi21_te_new.restype = P
        L.qi21_te_new.argtypes = [ctypes.c_char_p]
        L.qi21_te_encode.restype = I
        L.qi21_te_encode.argtypes = [P, ctypes.POINTER(I), I, I, ctypes.POINTER(P), ctypes.POINTER(I), P, I, ctypes.POINTER(I)]
        L.qi21_te_free.argtypes = [P]
        self.h = ctypes.c_void_p(L.qi21_te_new(models.path(models.QI_TE).encode()))
        self.tok = QI21Tokenizer(models.QWEN_TOKENIZER)
        self.out = torch.empty(self.CAP, 4096, dtype=torch.bfloat16, device="cuda")

    def close(self):
        if self.h:
            lib().qi21_te_free(self.h)
            self.h = None

    def encode(self, prompt, images=()):
        """images: [H,W,3] float 0..1 (already resized to multiples of 32, the same tensors the VAE encodes)."""
        ids = self.tok(prompt, len(images))
        imgs = [im.float().contiguous().cuda() for im in images]
        n = len(imgs)
        ia = (ctypes.c_int * len(ids))(*ids)
        ip = (ctypes.c_void_p * max(n, 1))(*[t.data_ptr() for t in imgs])
        hw = (ctypes.c_int * max(2 * n, 1))(*[v for t in imgs for v in t.shape[:2]])
        sl = (ctypes.c_int * max(n, 1))()
        torch.cuda.synchronize()
        rows = lib().qi21_te_encode(self.h, ia, len(ids), n, ip, hw, _ptr(self.out), self.CAP, sl)
        if rows < 0:
            raise ValueError("prompt too long for the text encoder buffer")
        return self.out[:rows].clone(), list(sl)[:n]
