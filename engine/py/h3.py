# MiniMax H3 Ref2VA on the shutter engine (libshutter.so): packed layout, condition rows, DiT calls and the sampler,
# ported from ComfyUI e9afbccc2 (comfy/ldm/minimax/model.py, model_base.MiniMaxH3, res_multistep + simple, ModelSamplingAV).
# torch is used for device memory, ComfyUI-identical noise and the small fp32 sampler math.
import ctypes
import math

import torch

import models

LIB = models.LIB

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0
SHIFT_V, SHIFT_A = 12.0, 3.0
AUDIO_SCALE = SHIFT_V / SHIFT_A          # ModelSamplingAV.audio_scale: the sampler carries audio * this
SEG_TAG = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}
KIND = {"text": 0, "cond": 1, "ref_img": 1, "cond_audio": 2, "ref_audio": 2, "audio": 3, "video": 4}
f64 = torch.float64

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(LIB)
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        IP, FP = ctypes.POINTER(I), ctypes.POINTER(F)
        L.h3_dit_new.restype = P
        L.h3_dit_new.argtypes = [ctypes.c_char_p]
        L.h3_dit_set.argtypes = [P, ctypes.c_char_p, P, ctypes.c_size_t]
        L.h3_dit_refine.argtypes = [P, P, I, P, P]
        L.h3_dit_prepare.argtypes = [P, P, I, P, I, P, I, IP, FP, I]
        L.h3_dit_forward.argtypes = [P, P, P, I, FP, IP, I, I, P, P]
        L.h3_dit_taps.argtypes = [P] * 5
        L.h3_dit_free.argtypes = [P]
        L.h3_dit_attn_q8.argtypes = [P, I]
        L.h3_dit_video_hidden.argtypes = [P, P, P]
        _lib = L
    return _lib


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr() if t is not None else 0)


# ------------------------------------------------------------------------------------------------- schedule
def time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def simple_sigmas(steps, shift=SHIFT_V):
    """ModelSamplingDiscreteFlow(shift).sigmas + the "simple" scheduler, fp32 like ComfyUI."""
    t = (torch.arange(1, 1001, 1) / 1000) * 1000
    t = t / 1000
    s = shift * t / (1 + (shift - 1) * t)
    ss = len(s) / steps
    return torch.FloatTensor([float(s[-(1 + int(x * ss))]) for x in range(steps)] + [0.0])


def beta_sigmas(steps, shift=SHIFT_V, alpha=0.6, beta=0.6):
    """ModelSamplingDiscreteFlow(shift).sigmas + the "beta" scheduler (comfy/samplers.beta_scheduler)."""
    import numpy as np
    import scipy.stats
    t = torch.arange(1, 1001, 1) / 1000
    s = shift * t / (1 + (shift - 1) * t)
    ts = np.rint(scipy.stats.beta.ppf(1 - np.linspace(0, 1, steps, endpoint=False), alpha, beta) * (len(s) - 1))
    sig, last = [], -1
    for x in ts:
        if x != last:
            sig.append(float(s[int(x)]))
        last = x
    return torch.FloatTensor(sig + [0.0])


def pack(video, audio):
    """sampler-space latents -> the packed row the sampler carries (audio * AUDIO_SCALE, like process_latent_in)."""
    return torch.cat([video.reshape(1, 1, -1), (audio * AUDIO_SCALE).reshape(1, 1, -1)], dim=-1)


def latent_t(frames):
    return ((frames - 5) // 17) * 5 + 2


def audio_t(lat_t):
    return round(sum(FRAME_PER_TOKEN[i % 5] for i in range(lat_t)) * FRAME_RESCALE)


# ------------------------------------------------------------------------------------------------- packed layout
def _axis(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=f64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(_axis(h, 2, area), _axis(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis(w, 2, area)


def _spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_grid(vt, frame, cursor):
    spans = torch.tensor(_spans(vt), dtype=f64)
    tg = float(cursor) + torch.cat([torch.zeros(1, dtype=f64), spans[:-1].cumsum(0)])
    g = torch.empty(vt, frame.shape[0], 3, dtype=f64)
    g[:, :, 0] = tg[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


def _audio_grid(cursor, t, w_low, w_high):
    g = torch.zeros(t * 2, 3, dtype=f64)
    g[:, 0] = (cursor + torch.arange(t, dtype=f64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _ref_span(blk):
    if blk["kind"] == "image":
        return 1.0
    if blk["kind"] == "audio":
        return float(blk["ref_audio_t"])
    return max(float(blk.get("ref_audio_t", 0)), sum(_spans(blk["latent_t"])))


def packed_layout(text_len, lat_t, lat_h, lat_w, aud_t, keyframes=(), refs=()):
    """PackedLayout: [(start, stop, kind)], position_ids [S, 3] float64."""
    frame, w_grid = _frame_grid(lat_h, lat_w)
    segs, pos = [("text", text_len)], []
    g = torch.zeros(text_len, 3, dtype=f64)
    g[:, 0] = torch.arange(text_len, dtype=f64)
    pos.append(g)
    taw = (float(w_grid[0]), float(w_grid[-1]))
    cursor = float(text_len) + sum(_ref_span(b) for b in refs)
    for kf in keyframes:
        ct = cursor + FRAME_RESCALE * kf["resolved_frame_index"]
        if kf.get("latent") is not None:
            vt = kf["latent"].shape[2]
            segs.append(("cond", vt * frame.shape[0]))
            pos.append(_video_grid(vt, frame, ct))
        if kf.get("audio_latent") is not None:
            rt = kf["audio_latent"].shape[-1]
            segs.append(("cond_audio", rt * 2))
            pos.append(_audio_grid(ct, rt, *taw))
    cur = float(text_len)
    for blk in refs:
        if blk["kind"] == "image":
            rf, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
            g = torch.empty(rf.shape[0], 3, dtype=f64)
            g[:, 0] = cur
            g[:, 1:] = rf
            segs.append(("ref_img", rf.shape[0]))
            pos.append(g)
            cur += 1.0
        elif blk["kind"] == "audio":
            rt = blk["ref_audio_t"]
            if rt > 0:
                segs.append(("ref_audio", rt * 2))
                pos.append(_audio_grid(cur, rt, *taw))
            cur += float(rt)
        else:  # video / video_audio: its audio rows first, both from the same cursor
            rt, vt = blk.get("ref_audio_t", 0), blk["latent_t"]
            rf, rw = _frame_grid(blk["latent_h"], blk["latent_w"])
            if rt > 0:
                segs.append(("ref_audio", rt * 2))
                pos.append(_audio_grid(cur, rt, float(rw[0]), float(rw[-1])))
            segs.append(("ref_img", vt * rf.shape[0]))
            pos.append(_video_grid(vt, rf, cur))
            cur += max(float(rt), sum(_spans(vt)))
    if aud_t:
        segs.append(("audio", aud_t * 2))
        pos.append(_audio_grid(cursor, aud_t, *taw))
    segs.append(("video", lat_t * frame.shape[0]))
    pos.append(_video_grid(lat_t, frame, cursor))
    out, off = [], 0
    for kind, n in segs:
        out.append((off, off + n, kind))
        off += n
    return out, torch.cat(pos)


def patchify(z):
    """[1, C, T, H, W] -> [T*h*w, C*4] (patch 1x2x2)."""
    _, c, t, h, w = z.shape
    x = z.reshape(1, c, t, 1, h // 2, 2, w // 2, 2)
    return torch.einsum("nctrhpwq->nthwcrpq", x).reshape(-1, c * 4)


def unpatchify(rows, t, h, w, c=24):
    x = rows.reshape(-1, t, h, w, c, 1, 2, 2)
    return torch.einsum("nthwcrpq->nctrhpwq", x).reshape(-1, c, t, h * 2, w * 2)


def pack_audio(z):
    _, c, ch, t = z.shape
    return z[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def cond_rows(latents, seed, aug, pack, device="cuda"):
    """_cond_video_rows / _cond_audio_rows: every condition restarts the same CPU generator."""
    rows = []
    for z in latents:
        r = pack(z.to(device=device, dtype=torch.float32))
        if aug < 1.0:
            noise = torch.randn(r.shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)
            r = aug * r + (1.0 - aug) * noise.to(r.device)
        rows.append(r)
    return torch.cat(rows).contiguous() if rows else None


# ------------------------------------------------------------------------------------------------- DiT
class Dit:
    def __init__(self, turbo=False, path=None, attn_q8=False, lora=None, adapter=None):
        """turbo: Singularity v1.3 + the ref2v turbo 4-step LoRA (the model we generate with); else the official ref2va
        (golden reference). path / lora / adapter: other DiT, LoRA or pi-H3 adapter files. LoRA and adapter are merged
        into the loaded weights the way ComfyUI computes them at run time (merge.py, ~5-15 s)."""
        self.turbo = turbo
        self.vsa = None   # h3_vsa.Vsa while the option vsa is attached
        if path is None:
            path = models.path(models.H3_SINGULARITY if turbo else models.H3_REF2VA)
            lora = lora or (models.path(models.H3_TURBO_LORA) if turbo else None)
        self.h = ctypes.c_void_p(lib().h3_dit_new(path.encode()))
        if lora or adapter:
            import merge
            for name, t in (merge.lora_patch(path, lora) if lora else merge.adapter_patch(path, adapter)):
                t = t.contiguous()
                torch.cuda.synchronize()
                lib().h3_dit_set(self.h, name.encode(), _ptr(t), t.numel() * t.element_size())
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.set_attn_q8(attn_q8)

    def set_attn_q8(self, on):
        """lossy switch: SageAttention2-style int8 QK / e4m3 PV attention (what ComfyUI runs with --use-sage-attention)."""
        lib().h3_dit_attn_q8(self.h, int(bool(on)))

    def close(self):
        if self.h:
            lib().h3_dit_free(self.h)
            self.h = None

    def refine(self, cross_attn):
        """condition_proj + token refiner: Qwen layer-50 states [L, 5120] -> context [L, 5376] bf16."""
        x = cross_attn.to(device="cuda", dtype=torch.bfloat16).contiguous()
        out = torch.empty(x.shape[0], 5376, dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()
        lib().h3_dit_refine(self.h, _ptr(x), x.shape[0], _ptr(out), None)
        return out

    def prepare(self, ctx, tags, lat_shape, aud_t, keyframes=(), refs=(), seed=0,
                vis_aug=VISUAL_COND_TIMESTEP, aud_aug=AUDIO_COND_TIMESTEP):
        """ctx [L, 5376] bf16 (refined); tags [L] modality tag per text token; lat_shape (T, H, W) of the target video latent;
        keyframes / refs as ComfyUI's minimax payload (latents normalized [1, 24, t, h, w], fp32)."""
        T, H, W = lat_shape
        H2, W2 = (H + 1) // 2 * 2, (W + 1) // 2 * 2
        self.segs, posid = packed_layout(ctx.shape[0], T, H2, W2, aud_t, keyframes, refs)
        self.shape, self.aud_t, self.vis_aug, self.aud_aug = (T, H, W), aud_t, vis_aug, aud_aug
        vl = [k["latent"] for k in keyframes if k.get("latent") is not None] + [r["latent"] for r in refs if r.get("latent") is not None]
        al = [k["audio_latent"] for k in keyframes if k.get("audio_latent") is not None] + \
             [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
        self.vis = cond_rows(vl, int(seed), vis_aug, patchify)
        self.aud = cond_rows(al, int(seed) + 1, aud_aug, pack_audio)
        kinds = [KIND[k] for a, b, k in self.segs for _ in range(b - a)]
        self.tags = [int(v) for v in torch.as_tensor(tags).view(-1).tolist()]
        S = len(kinds)
        pos = posid.to(torch.float32).contiguous()
        self.posid = posid
        ka = (ctypes.c_int * S)(*kinds)
        ctx = ctx.to(torch.bfloat16).contiguous()
        torch.cuda.synchronize()
        lib().h3_dit_prepare(self.h, _ptr(ctx), ctx.shape[0], _ptr(self.vis), 0 if self.vis is None else self.vis.shape[0],
                             _ptr(self.aud), 0 if self.aud is None else self.aud.shape[0], ka,
                             ctypes.cast(ctypes.c_void_p(pos.data_ptr()), ctypes.POINTER(ctypes.c_float)), S)
        self.S = S

    def _rows(self, t_v, t_a):
        seg_t = {"text": t_v, "video": t_v, "audio": t_a, "cond": max(t_v, self.vis_aug), "ref_img": max(t_v, self.vis_aug),
                 "cond_audio": max(t_a, self.aud_aug), "ref_audio": max(t_a, self.aud_aug)}
        uniq = sorted({t_v, t_a} | {seg_t[k] for _, _, k in self.segs})
        tr = {t: i for i, t in enumerate(uniq)}
        mrow = []
        for a, b, k in self.segs:
            base = tr[seg_t[k]] * 3
            mrow += [base + v for v in self.tags[:b - a]] if k == "text" else [base + SEG_TAG[k]] * (b - a)
        return uniq, mrow, tr[t_v], tr[t_a]

    def inner(self, video_x, audio_x, sigma_v, t_v, t_a):
        """_forward: stream latents (bf16) -> raw network outputs [-v] (bf16), like ComfyUI."""
        T, H, W = self.shape
        uniq, mrow, iv, ia = self._rows(t_v, t_a)
        vr = patchify(video_x.float()).contiguous()
        ar = pack_audio(audio_x.float()).contiguous()
        vo = torch.empty(vr.shape[0], 96, device="cuda")
        ao = torch.empty(ar.shape[0], 32, device="cuda")
        tv = (ctypes.c_float * len(uniq))(*uniq)
        mr = (ctypes.c_int * len(mrow))(*mrow)
        torch.cuda.synchronize()
        lib().h3_dit_forward(self.h, _ptr(vr), _ptr(ar), len(uniq), tv, mr, iv, ia, _ptr(vo), _ptr(ao))
        if self.vsa is not None:
            self.vsa.check()
        self.last_rows, self.last_u, self.last_iv = (vo, ao), len(uniq), iv
        v = unpatchify(vo, T, (H + 1) // 2, (W + 1) // 2)[:, :, :T, :H, :W]
        return -v.to(torch.bfloat16), -unpack_audio(ao).to(torch.bfloat16)

    def video_hidden(self):
        """After a forward: the last block's target-video rows (bf16 [T*h*w, 5376], before final_layer, patch order) and
        that call's time embedding rows (fp32 [U, 8], sorted unique t; the video row is self.last_iv) = the DWD inputs."""
        T, H, W = self.shape
        h = torch.empty(T * ((H + 1) // 2) * ((W + 1) // 2), 5376, dtype=torch.bfloat16, device="cuda")
        temb = torch.empty(self.last_u, 8, device="cuda")
        torch.cuda.synchronize()
        lib().h3_dit_video_hidden(self.h, _ptr(h), _ptr(temb))
        return h, temb

    def forward(self, xv, xa, sigma):
        """MiniMaxH3Model.forward: xv / xa bf16 as the sampler carries them (audio scaled), sigma fp32 tensor [1]."""
        timestep = (sigma * 1000).float()
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        sigma_a = time_shift_sigma(sigma_v, SHIFT_V, SHIFT_A)
        carry = (sigma_a / sigma_v).to(xa.dtype)
        xa = xa * carry
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, SHIFT_V, SHIFT_A))
        ov, oa = self.inner(xv, xa, sigma_v, t_v, t_a)
        oa = (1.0 - AUDIO_SCALE) * xa + (1.0 + (AUDIO_SCALE - 1.0) * sigma_a).to(oa.dtype) * oa
        return ov, oa

    def denoise(self, x, sigma):
        """BaseModel._apply_model on the packed fp32 latent x [1, 1, N]: returns denoised (fp32)."""
        T, H, W = self.shape
        nvid = 24 * T * H * W
        xc = x.to(torch.bfloat16)
        ov, oa = self.forward(xc[..., :nvid].reshape(1, 24, T, H, W), xc[..., nvid:].reshape(1, 32, 2, self.aud_t), sigma)
        out = torch.cat([ov.reshape(1, 1, -1), oa.reshape(1, 1, -1)], dim=-1)
        return x - out.float() * sigma.view(())

    def sample(self, seed, steps=None, sigmas=None, noise=None, callback=None, progress=None, init=None,
               sampler="res_multistep", denoised=False):
        """BasicGuider + res_multistep (eta 0) or euler from ComfyUI-identical noise; returns (video [1,24,T,H,W], audio [1,32,2,Ta]).
        init: packed clean latent (pack(video, audio)) to re-noise at sigmas[0] instead of the empty latent (second pass).
        denoised: return the last step's prediction instead of x (SamplerCustomAdvanced's denoised_output, the one a split
        first pass hands to the upscaler)."""
        T, H, W = self.shape
        sig = (sigmas if sigmas is not None else simple_sigmas(steps or (4 if self.turbo else 20))).float().cuda()
        if noise is None:
            gen = torch.manual_seed(seed)
            noise = [torch.randn([1, 24, T, H, W], dtype=torch.float32, generator=gen, device="cpu"),
                     torch.randn([1, 32, 2, self.aud_t], dtype=torch.float32, generator=gen, device="cpu")]
        n = torch.cat([t.reshape(1, 1, -1) for t in noise], dim=-1).cuda()
        z = torch.zeros_like(n) if init is None else init.to(device=n.device, dtype=n.dtype)
        x = sig[0] * (1.0 * n) + (1.0 - sig[0]) * z   # CONST.noise_scaling (empty latent, or init for a second pass)
        nvid = 24 * T * H * W
        s_in = x.new_ones([1])
        old_denoised = old_sigma_down = None
        for i in range(len(sig) - 1):
            den = self.denoise(x, sig[i] * s_in)
            sigma_down = sig[i + 1]
            if callback:
                callback(i, x, den)
            if sampler == "euler" or sigma_down == 0 or old_denoised is None:
                d = (x - den) / sig[i]
                x = x + d * (sigma_down - sig[i])
            else:
                t, t_old, t_next, t_prev = (-s.log() for s in (sig[i], old_sigma_down, sigma_down, sig[i - 1]))
                h = t_next - t
                c2 = (t_prev - t_old) / h
                phi1 = torch.expm1(-h) / (-h)
                phi2 = (phi1 - 1.0) / (-h)
                b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
                b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
                x = (-h).exp() * x + h * (b1 * den + b2 * old_denoised)
            old_denoised, old_sigma_down = den, sigma_down
            if progress:
                progress((i + 1) / (len(sig) - 1))
        x = den.clone() if denoised else x / (1.0 - sig[-1])   # denoised_output, or inverse_noise_scaling
        x[..., nvid:] *= 1.0 / AUDIO_SCALE           # process_latent_out: undo the audio carry scale
        return x[..., :nvid].reshape(1, 24, T, H, W), x[..., nvid:].reshape(1, 32, 2, self.aud_t)
