# pi-H3 on the shutter engine: MiniMax H3 FL2VA + the LakonLab pi-Flow DWD adapter (2 or 4 steps) as the author's graph
# (comfyui-piflow workflows/pi-H3-DWD.json; docs/pih3.md §3), mirroring ComfyUI e9afbccc2 + comfyui-piflow 1.4.1 as
# golden/pih3_dump.py drove it:
#   LoadImage -> ImageScale lanczos center W x H -> MiniMaxH3ImageToVideo (first frame: stretch to W x H, keyframe 0;
#   last frame: cover crop, keyframe frame_count - 1; TE "<Picture 1>" [, "<Picture 2>"] in that order; T2V: no picture)
#   -> MiniMaxH3SigmaShift 12 / 3 -> pi-Flow Scheduler + euler + BasicGuider (cfg 1, dense attention)
#   -> DWD Decode (the last step's hidden rows -> pixels, no VAE decode) + VAEDecodeAudio.
#   p = PiH3(steps=2)
#   frames, wav = p.generate(first_u8_or_None, prompt, W, H, frames, seed, last=last_u8_or_None)
#                                                                    # uint8 [F, H, W, 3], f32 [2, n] at 32 kHz
# Weights: FL2VA + the pi-H3 <n>-step adapter merged at load as ComfyUI computes with it (merge.adapter_patch); the
# adapter's dwd_* tensors are the DWD head (h3_dwd).
import ctypes
import time

import numpy as np
import torch

import h3
import h3_pipe
import models
from h3_pipe import AUDIO_LATENT_FPS, _center, _sync
from h3_tok import FPS, H3Tokenizer, _lanczos, token_tags

MAX_RAW_T, FSS = {2: 0.93, 4: 0.96}, 0.6   # pi-Flow Scheduler settings of the author's graph (min_raw_t 0)


def pi_sigmas(steps, max_raw_t=None, min_raw_t=0.0, fss=FSS, shift=h3.SHIFT_V):
    """comfyui-piflow get_piflow_schedule: fp32 linspace of raw t, the model's percent_to_sigma (python floats:
    ModelSamplingDiscreteFlow time_snr_shift), fp32 sigmas + 0. DiT t_v = 1 - sigma."""
    m = MAX_RAW_T[steps] if max_raw_t is None else max_raw_t
    raw = torch.linspace(m, (m - min_raw_t) * fss / (steps - 1 + fss) + min_raw_t, steps)

    def p2s(p):
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0
        t = 1.0 - p
        return shift * t / (1 + (shift - 1) * t)
    s = torch.tensor([p2s(1.0 - t) for t in raw.tolist()])
    return torch.cat([s, s.new_zeros(1)])


class PiH3(h3_pipe.H3):
    """Text encoder, video VAE (first-frame encode only), audio VAE, and the DiT + DWD head of the chosen step count
    (switching reloads FL2VA and merges the other adapter)."""

    def __init__(self, steps=2):
        L = h3_pipe.lib()
        t0 = time.perf_counter()
        self.tok = H3Tokenizer(models.QWEN_TOKENIZER)
        self.te = ctypes.c_void_p(L.h3_te_new(models.path(models.H3_TE).encode()))
        self.vae = ctypes.c_void_p(L.h3_vae_new(models.path(models.H3_VAE).encode()))
        self.aud = ctypes.c_void_p(L.h3_audio_new(models.path(models.H3_AUDIO).encode()))
        self.face = self.ups = self.dit = self.dwd = self.steps = None
        self.turbo = False
        self.load(steps)
        self.load_s = time.perf_counter() - t0

    def load(self, steps):
        if steps == self.steps:
            return
        if steps not in MAX_RAW_T:
            raise ValueError(f"steps {steps}: one of {sorted(MAX_RAW_T)}")
        self._free_model()
        path = models.path(models.PIH3_ADAPTER % steps)
        self.dit = h3.Dit(path=models.path(models.H3_FL2VA), adapter=path)
        try:
            from h3_dwd import Dwd
        except ImportError:   # DWD head not built yet: run() still gives its inputs
            Dwd = None
        self.dwd = Dwd(path) if Dwd else None
        self.steps = steps

    def _free_model(self):
        if self.dit is not None:
            self.dit.close()
        if self.dwd is not None and hasattr(self.dwd, "close"):
            self.dwd.close()
        self.dit = self.dwd = self.steps = None
        torch.cuda.empty_cache()

    def close(self):
        self._free_model()
        super().close()

    # ---- MiniMaxH3ImageToVideo
    def condition(self, image, prompt, W, H, tm=None, last=None, frames=None):
        """-> (ids, vis, tags, cross_attn [rows, 5120] fp32, keyframes). image (first frame, keyframe 0) / last (last
        frame, keyframe frames - 1): uint8 [h, w, 3] or None; neither = T2V, last only = L2VA."""
        tm = {} if tm is None else tm
        t0 = _sync()
        # the graph's ImageScale (lanczos, center crop), then the node's own resize to W x H (first: stretch, last: cover
        # crop — same size either way: a u8 trip)
        at = [(g, i) for g, i in ((image, 0), (last, None if frames is None else frames - 1)) if g is not None]
        if any(i is None for _, i in at):
            raise ValueError("a last frame needs frames")
        vis = [_lanczos(_center(np.asarray(g), W, H), W, H) for g, _ in at]
        ids = self.tok(prompt, [{"type": "image"}] * len(vis))
        t1 = _sync()
        cross = self.encode_text(ids, vis)
        t2 = _sync()
        kf = [{"resolved_frame_index": i, "latent": self.encode(v)} for v, (_, i) in zip(vis, at)]
        tm["image_prep"], tm["te"], tm["vae_encode"] = t1 - t0, t2 - t1, _sync() - t2
        return ids, vis, token_tags(ids, vis), cross, kf

    def run(self, image, prompt, W, H, frames, seed, steps=None, q8=True, progress=None, trace=None, last=None, vsa=False):
        """Everything up to the DWD decode: returns (h bf16 [T*lat_h/2*lat_w/2, 5376] = the last step's final-block
        video rows, temb fp32 [U, 8] = that step's time embedding rows (video row 0), grid (T, lat_h, lat_w),
        audio latent [1, 32, 2, Ta]). Times in self.times. trace: optional dict, filled with intermediates.
        vsa: block-sparse attention as ComfyUI's "Model Sparse Attention" vsa (h3_vsa.py, lossy option, session #12)."""
        if W % 32 or H % 32:
            raise ValueError(f"size {W}x{H}: width and height must be multiples of 32")
        if frames % 17 != 5:
            raise ValueError(f"frames {frames} is not 17k + 5")
        self.load(steps or self.steps)
        self.dit.set_attn_q8(q8)
        p = progress or (lambda f: None)
        tm = self.times = {"load": self.load_s}
        tr = trace if trace is not None else {}
        grid, aud_t = (h3.latent_t(frames), H // 16, W // 16), round(frames / FPS * AUDIO_LATENT_FPS)
        self.info = dict(model="pih3_%dstep" % self.steps, steps=self.steps, i2v=image is not None, last=last is not None, width=W, height=H,
                         frames=frames, seed=seed, q8=bool(q8), vsa=bool(vsa), max_raw_t=MAX_RAW_T[self.steps], fss=FSS)
        t0 = _sync()
        ids, vis, tags, cross, kf = self.condition(image, prompt, W, H, tm, last, frames)
        t1 = _sync()
        p(0.05)
        ctx = self.dit.refine(cross)
        self.dit.prepare(ctx, torch.tensor(tags), grid, aud_t, kf, (), seed=seed)
        t2 = _sync()
        tm["dit_prepare"] = t2 - t1
        p(0.1)
        steps_t, last = [], [t2]

        def cb(i, x, den):
            now = _sync()
            steps_t.append(now - last[0])
            last[0] = now
            if trace is not None:
                tr.setdefault("steps", []).append({"x": x.clone(), "den": den.clone(),
                                                   "final": [r.clone() for r in self.dit.last_rows]})
                if "on_step" in tr:
                    tr["on_step"](i)

        sig = pi_sigmas(self.steps)
        if vsa:
            import h3_vsa
            self.dit.vsa = h3_vsa.Vsa(self.dit)
        try:
            v, a = self.dit.sample(seed, sigmas=sig, sampler="euler", callback=cb, progress=lambda f: p(0.1 + 0.75 * f))
        finally:
            if self.dit.vsa is not None:
                self.dit.vsa.detach()
                self.dit.vsa = None
        h, temb = self.dit.video_hidden()
        t3 = _sync()
        tm["dit_steps"], tm["dit_total"], tm["S"] = steps_t, t3 - t2, self.dit.S
        tm["total_to_dwd"] = t3 - t0
        if trace is not None:
            tr.update(ids=ids, vis=vis, tags=tags, cross_attn=cross, context=ctx, keyframes=kf, sigmas=sig,
                      cond_rows=self.dit.vis, posid=self.dit.posid, segs=self.dit.segs, video_latent=v, audio_latent=a,
                      hidden=h, temb=temb, grid=grid)
        del v, ctx, cross
        return h, temb, grid, a

    def generate(self, image, prompt, W, H, frames, seed, steps=None, q8=True, progress=None, trace=None, last=None,
                 vsa=False):
        """image uint8 [h, w, 3] (first frame, I2V) or None (T2V); last: the same for the last frame (FL2VA / L2VA —
        the prompt's first line then aligns both pictures, MiniMax's base guide §2.1); W, H multiples of 32; frames 17k + 5 at 24 fps;
        steps 2 | 4 (None = the loaded model). Returns (uint8 frames [frames, H, W, 3] cpu, f32 waveform [2, n] at
        32 kHz cpu); times in self.times, settings in self.info. trace, vsa: as run()."""
        p = progress or (lambda f: None)
        if self.dwd is None:
            raise RuntimeError("h3_dwd (the DWD head) is not available")
        h, temb, grid, a = self.run(image, prompt, W, H, frames, seed, steps, q8, lambda f: p(0.9 * f), trace, last, vsa)
        tm = self.times
        t0 = _sync()
        out = self.dwd.decode(h, temb, grid, row=self.dit.last_iv).numpy()   # SaveVideo 8-bit (truncating)
        del h
        t1 = _sync()
        wav = self.decode_audio(a).cpu().numpy()
        t2 = _sync()
        tm["dwd"], tm["audio_decode"] = t1 - t0, t2 - t1
        tm["total"] = tm["total_to_dwd"] + t2 - t0
        if out.shape[0] != frames:
            raise RuntimeError(f"DWD returned {out.shape[0]} frames, expected {frames}")
        p(1.0)
        return out, wav
