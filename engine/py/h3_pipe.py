# MiniMax H3 Ref2VA pipeline on the shutter engine: one video.swap segment.
# Mirrors ComfyUI e9afbccc2 as golden/h3_dump.py drove it:
#   MiniMaxH3ReferenceToVideo(ref_image, ref_video, ref_image_size="match") -> MiniMaxH3AddGuide(first, 0)
#   -> MiniMaxH3AddGuide(last, -1) -> BasicGuider + res_multistep/simple -> VAE decode (comfy_extras/nodes_minimax_h3.py).
#   h = H3(turbo=True)
#   frames = h.generate(ref_video, ref_image, first, last, gen_frames, prompt, seed)   # uint8 [gen_frames, H, W, 3]
#   face_check(ENG, frames, master)                                                 # #18
import ctypes
import hashlib
import math
import time

import numpy as np
import torch

import h3
import models
from h3_tok import FPS, H3Tokenizer, _lanczos, prepare_refs, token_tags

CANVAS, BASE_SHORT, MAX_PIXELS = 32, 768, 768 * 1344
AUDIO_LATENT_FPS = 40

_api = None


def lib():
    """h3.lib() plus the text encoder / VAE entry points."""
    global _api
    L = h3.lib()
    if _api is None:
        P, I = ctypes.c_void_p, ctypes.c_int
        for n in ("h3_te_new", "h3_vae_new", "h3_audio_new"):
            getattr(L, n).restype = P
            getattr(L, n).argtypes = [ctypes.c_char_p]
        L.h3_te_encode.restype = I
        L.h3_te_encode.argtypes = [P, ctypes.POINTER(I), I, I, ctypes.POINTER(P), ctypes.POINTER(I), P, I]
        L.h3_te_encode_prefix.restype = I
        L.h3_te_encode_prefix.argtypes = [P, ctypes.POINTER(I), I, I, ctypes.POINTER(P), ctypes.POINTER(I), P, I,
                                          ctypes.c_ulonglong, I]
        L.h3_vae_encode.argtypes = [P, P, I, I, I, P]
        L.h3_vae_decode.argtypes = [P, P, I, I, I, P]
        L.h3_vae_frames.argtypes = [I]
        L.h3_vae_latent_frames.argtypes = [I]
        L.h3_audio_decode.argtypes = [P, P, I, P]
        for n in ("h3_te_free", "h3_vae_free", "h3_audio_free"):
            getattr(L, n).argtypes = [P]
        _api = L
    return L


def _f32(u8):
    return np.asarray(u8).astype(np.float32) / 255.0   # LoadImage


def _ref_video_size(vw, vh):
    """MiniMaxH3ReferenceToVideo: adapt_canvas (768 short edge, <= 768x1344, /32), never upscaled."""
    r = vw / vh
    nw, nh = (BASE_SHORT * r, BASE_SHORT) if r >= 1.0 else (BASE_SHORT, BASE_SHORT / r)
    if nw * nh > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nw * nh))
        nw, nh = nw * s, nh * s
    cw, ch = max(CANVAS, round(nw / CANVAS) * CANVAS), max(CANVAS, round(nh / CANVAS) * CANVAS)
    if vw * vh < cw * ch:
        cw, ch = max(CANVAS, round(vw / CANVAS) * CANVAS), max(CANVAS, round(vh / CANVAS) * CANVAS)
    return cw, ch


def _center(img, w, h):
    """common_upscale(lanczos, crop="center") of one u8 image -> float32 [1, h, w, 3] (MiniMaxH3AddGuide)."""
    oh, ow = img.shape[:2]
    oa, na = ow / oh, w / h
    x = y = 0
    if oa > na:
        x = round((ow - ow * (na / oa)) / 2)
    elif oa < na:
        y = round((oh - oh * (oa / na)) / 2)
    return _lanczos(_f32(img[y:oh - y, x:ow - x])[None], w, h)


def _sync():
    torch.cuda.synchronize()
    return time.perf_counter()


class H3:
    def __init__(self, turbo=True):
        import face
        L = lib()
        t0 = time.perf_counter()
        self.tok = H3Tokenizer(models.QWEN_TOKENIZER)
        self.te = ctypes.c_void_p(L.h3_te_new(models.path(models.H3_TE).encode()))
        self.dit = h3.Dit(turbo=turbo)
        self.vae = ctypes.c_void_p(L.h3_vae_new(models.path(models.H3_VAE).encode()))
        self.aud = ctypes.c_void_p(L.h3_audio_new(models.path(models.H3_AUDIO).encode()))
        self.face = face.FaceCheck()
        self.turbo = turbo
        self.load_s = time.perf_counter() - t0

    def close(self):
        L = lib()
        if self.dit is not None:
            self.dit.close()
            self.dit = None
        for k, f in (("te", L.h3_te_free), ("vae", L.h3_vae_free), ("aud", L.h3_audio_free)):
            if getattr(self, k):
                f(getattr(self, k))
                setattr(self, k, None)
        self.face = self.ups = None
        torch.cuda.empty_cache()

    # ---- pieces (device fp32 in / out)
    def encode_text(self, ids, vis):
        """Text encoder rows. The prefix up to the last picture's <|vision_end|> (the reference pictures) is cached in the
        engine: the next prompt with the same pictures and labels computes only its own text rows (lossless, causal)."""
        vt = [torch.from_numpy(np.ascontiguousarray(v)).cuda() for v in vis]
        rows = len(token_tags(ids, vis))
        key, P = 0, 0
        if -1 in ids and all(v.shape[0] == 1 for v in vis):   # pictures only (video refs differ per segment)
            j = len(ids) - 1 - ids[::-1].index(-1) + 2          # tokens through the last block's <|vision_end|>
            if j < len(ids):
                P = len(token_tags(ids[:j], vis))
                hsh = hashlib.blake2b(np.asarray(ids[:j], np.int64).tobytes(), digest_size=8)
                for v in vis:
                    hsh.update(np.ascontiguousarray(v).tobytes())
                key = int.from_bytes(hsh.digest(), "little") or 1
        out = torch.empty(rows, 5120, dtype=torch.float32, device="cuda")
        arr = (ctypes.c_void_p * max(len(vt), 1))(*[v.data_ptr() for v in vt])
        thw = (ctypes.c_int * max(3 * len(vt), 1))(*[d for v in vt for d in v.shape[:3]])
        torch.cuda.synchronize()
        r = lib().h3_te_encode_prefix(self.te, (ctypes.c_int * len(ids))(*ids), len(ids), len(vt), arr, thw,
                                      ctypes.c_void_p(out.data_ptr()), rows, key, P)
        if r != rows:
            raise RuntimeError(f"h3_te_encode returned {r} rows, expected {rows}")
        return out

    def encode(self, px):
        """float32 [F, H, W, 3] in 0..1 -> normalized latent [1, 24, T', H/16, W/16] (VAE.encode)."""
        x = torch.from_numpy(np.ascontiguousarray(px)).cuda()
        F_, H, W, _ = x.shape
        out = torch.empty(24, lib().h3_vae_latent_frames(F_), H // 16, W // 16, device="cuda")
        torch.cuda.synchronize()
        lib().h3_vae_encode(self.vae, ctypes.c_void_p(x.data_ptr()), F_, H, W, ctypes.c_void_p(out.data_ptr()))
        torch.cuda.synchronize()
        return out[None]

    def decode(self, lat):
        """[1, 24, T, h, w] -> float32 frames [F, 16h, 16w, 3] in 0..1 (device)."""
        z = lat[0].float().contiguous()
        _, T, lh, lw = z.shape
        out = torch.empty(lib().h3_vae_frames(T), lh * 16, lw * 16, 3, device="cuda")
        torch.cuda.synchronize()
        lib().h3_vae_decode(self.vae, ctypes.c_void_p(z.data_ptr()), T, lh, lw, ctypes.c_void_p(out.data_ptr()))
        torch.cuda.synchronize()
        return out

    def decode_audio(self, lat):
        """[1, 32, 2, Ta] -> waveform [2, Ta * 800] at 32 kHz (device)."""
        z = lat[0].float().contiguous()
        out = torch.empty(2, z.shape[-1] * 800, device="cuda")
        torch.cuda.synchronize()
        lib().h3_audio_decode(self.aud, ctypes.c_void_p(z.data_ptr()), z.shape[-1], ctypes.c_void_p(out.data_ptr()))
        torch.cuda.synchronize()
        return out

    # ---- one segment
    def generate(self, ref_video, ref_image, first, last, gen_frames, prompt, seed, progress=None, steps=None,
                 trace=None, audio=False, last_idx=-1, canvas=None, up=None, sampler="res_multistep", sigmas=None,
                 ref_size="match", stage="full", init1=None, keys=()):
        """ref_video uint8 [N, H, W, 3] at the target canvas (24 fps), or None for pure reference generation with
        canvas=(W, H) (with a ref_video: the first-pass canvas, the reference keeps its own size); ref_image uint8 [h, w, 3] or a list of them (<Picture 1>, <Picture 2> … in order); first / last uint8 guide
        frames (frame 0, frame last_idx: -1 = the last generated frame, or the kept length - 1 of a trimmed last
        segment; None = no guide on that side); keys: more guides [(frame index, uint8 image)] anywhere on the output
        time axis (ComfyUI "Add Guide for MiniMax H3", the official multiframe template: a multi-shot's shot first
        frames); gen_frames = 17k+5. Returns uint8 [gen_frames, H, W, 3].
        sampler "res_multistep" | "euler"; sigmas: explicit first-pass schedule (else `steps` of "simple");
        ref_size "match" | "max" (ComfyUI ref_image_size: "max" keeps refs up to a 2048 short edge).
        up=(W2, H2, sigmas2): two passes as the Singularity author's graph — the first pass (sigmas, usually the head of a
        split schedule) hands its last prediction to the 3D latent upscaler, the second re-noises it at W2 x H2 with the
        same seed and runs sigmas2 (the tail); audio passes through and is re-noised too. Returns frames at W2 x H2.
        stage (with up): "full" both passes; "first" only the first pass — returns its prediction decoded at the canvas
        (a preview) and keeps the latents in self.last_latent = (video, audio); "second" only the second pass from
        init1 = that (video, audio) — same arguments as the full call, so the result equals the full run.
        trace: optional dict, filled with intermediates and stage times. audio=True also decodes the generated soundtrack
        into self.last_audio ([2, n] float32 at 32 kHz, cpu)."""
        H, W = canvas[::-1] if canvas is not None else ref_video.shape[1:3]   # canvas wins (two-pass swap: smaller first pass)
        imgs = list(ref_image) if isinstance(ref_image, (list, tuple)) else [ref_image]
        vids = [] if ref_video is None else [ref_video]
        if H % 16 or W % 16:
            raise ValueError(f"canvas {W}x{H} is not a multiple of 16")
        if gen_frames % 17 != 5:
            raise ValueError(f"gen_frames {gen_frames} is not 17k+5")
        if stage != "full" and (up is None or (stage == "second") != (init1 is not None)):
            raise ValueError(f"stage {stage!r} needs up=, and init1= exactly for 'second'")
        p = progress or (lambda f: None)
        tr = trace if trace is not None else {}
        tm = tr.setdefault("t", {})
        # temporal_shape(): latent T and audio length from the frame count
        T = h3.latent_t(gen_frames)
        aud_t = round(gen_frames / FPS * AUDIO_LATENT_FPS)
        t0 = _sync()

        # MiniMaxH3ReferenceToVideo: Qwen presentation (ref image at "match" size, ref video at 2 fps) + text encoder
        refs_q, vis = prepare_refs(W, H, gen_frames, images=imgs, videos=vids, ref_image_size=ref_size)
        ids = self.tok(prompt, refs_q)
        tags = token_tags(ids, vis)
        t1 = _sync()
        cross = self.encode_text(ids, vis)
        t2 = _sync()
        tm["te_prep"], tm["te"] = t1 - t0, t2 - t1
        p(0.05)

        # VAE encodes in ComfyUI order: ref image (the Qwen copy), ref video (all frames, cut to 17k+5), guides
        img_px = vis[:len(imgs)]
        if vids:
            cw, ch = _ref_video_size(ref_video.shape[2], ref_video.shape[1])
            n = min(len(ref_video), gen_frames)
            while n % 17 != 5:
                n -= 1
            vid_px = _lanczos(_f32(ref_video[:n]), cw, ch)
        g_px = [None if g is None else _center(g, W, H) for g in (first, last)]
        k_px = [(int(i), _center(g, W, H)) for i, g in keys]
        t3 = _sync()
        z_imgs = [self.encode(x) for x in img_px]
        z_img = z_imgs[0]
        t4 = _sync()
        z_vid = self.encode(vid_px) if vids else None
        t5 = _sync()
        z_g = [None if g is None else self.encode(g) for g in g_px]
        z_k = [(i, self.encode(g)) for i, g in k_px]
        t6 = _sync()
        tm["vae_prep"], tm["vae_ref_image"], tm["vae_ref_video"], tm["vae_guides"] = t3 - t2, t4 - t3, t5 - t4, t6 - t5
        if vids:
            del vid_px
        p(0.1)

        # payload: keyframes (AddGuide 0, -1 -> frame_count - 1), then refs; condition_proj + token refiner; layout
        frame_count = sum(h3.FRAME_PER_TOKEN[k % 5] for k in range(T))
        li = last_idx if last_idx >= 0 else frame_count + last_idx
        if not 0 <= li < frame_count:
            raise ValueError(f"last_idx {last_idx} is outside the video's {frame_count} frames")
        if any(not 0 <= i < frame_count for i, _ in z_k):
            raise ValueError(f"a key frame index is outside the video's {frame_count} frames")
        keyframes = [{"resolved_frame_index": i, "latent": z} for i, z in [(0, z_g[0]), (li, z_g[1])] + z_k if z is not None]
        refs = [{"kind": "image", "latent_h": x.shape[1] // 16, "latent_w": x.shape[2] // 16, "latent": z}
                for x, z in zip(img_px, z_imgs)]
        if vids:
            refs.append({"kind": "video", "latent_t": z_vid.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                         "ref_audio_t": 0, "latent": z_vid, "audio_latent": None})
        ctx = self.dit.refine(cross)
        if stage != "second":
            self.dit.prepare(ctx, torch.tensor(tags), (T, H // 16, W // 16), aud_t, keyframes, refs, seed=seed)
        t7 = _sync()
        tm["dit_prepare"] = t7 - t6

        # sampler
        steps_t = []
        last_t = [t7]

        def cb(i, x, den):
            now = _sync()
            steps_t.append(now - last_t[0])
            last_t[0] = now

        f1 = 0.8 if up is None or stage == "first" else 0.25
        if stage == "second":
            v, a = (t.to(device="cuda", dtype=torch.float32) for t in init1)
        else:
            sig1 = None if sigmas is None else torch.as_tensor(sigmas, dtype=torch.float32)
            v, a = self.dit.sample(seed, steps=steps, sigmas=sig1, sampler=sampler, denoised=up is not None, callback=cb,
                                   progress=lambda f: p(0.1 + f1 * f))
        if stage == "first":
            self.last_latent = (v, a)
        elif up is not None:   # second pass: latent upscale, re-noise at the target size, same text / refs / seed
            W2, H2, sig2 = up
            if getattr(self, "ups", None) is None:
                import h3_up
                self.ups = h3_up.Upscaler(models.path(models.H3_UP))
            ta = _sync()
            v2 = self.ups(v, (v.shape[2], H2 // 16, W2 // 16))
            tm["upscale"] = _sync() - ta
            g2 = [None if g is None else self.encode(_center(g, W2, H2)) for g in (first, last)]
            k2 = [(int(i), self.encode(_center(g, W2, H2))) for i, g in keys]
            kf2 = [{"resolved_frame_index": i, "latent": z} for i, z in [(0, g2[0]), (li, g2[1])] + k2 if z is not None]
            self.dit.prepare(ctx, torch.tensor(tags), (T, H2 // 16, W2 // 16), aud_t, kf2, refs, seed=seed)
            v, a = self.dit.sample(seed, sigmas=torch.tensor(sig2, dtype=torch.float32), init=h3.pack(v2, a), sampler=sampler,
                                   callback=cb, progress=lambda f: p(0.35 + 0.55 * f))
            del v2
        t8 = _sync()
        tm["dit_steps"], tm["dit_total"] = steps_t, t8 - t7
        tm["S"] = self.dit.S

        # decode
        px = self.decode(v)
        t9 = _sync()
        tm["vae_decode"] = t9 - t8
        if audio:
            self.last_audio = self.decode_audio(a).cpu()
            tm["audio_decode"] = _sync() - t9
        frames = (255.0 * px.clamp(0, 1)).to(torch.uint8).cpu().numpy()   # SaveImage: clip(255 x) -> uint8 (truncating)
        tm["total"] = time.perf_counter() - t0
        if trace is not None:
            tr.update(ids=ids, tags=tags, vis=vis, cross_attn=cross, context=ctx, pixels=[img_px[0], g_px[0], g_px[1]],
                      latents={"ref_image": z_img, "ref_video": z_vid, "guide_first": z_g[0], "guide_last": z_g[1]},
                      cond_rows=self.dit.vis, posid=self.dit.posid, segs=self.dit.segs, video_latent=v, audio_latent=a,
                      frames_f32=px)
        else:
            del px, cross, ctx
        p(1.0)
        return frames


def face_check(ENG, frames, master, n=8):
    """#18: InsightFace similarity of ~n evenly spaced frames to the character master: per frame the best of all its
    faces (others in the shot are not the character). Frames without a face are reported with sim None and left out
    of the median."""
    fc = getattr(ENG.get("h3"), "face", None) or ENG.get("face")
    if fc is None:
        import face
        fc = ENG.setdefault("face", face.FaceCheck())
    ref = fc.embed(np.ascontiguousarray(master))
    idx = sorted({int(round(i)) for i in np.linspace(0, len(frames) - 1, min(n, len(frames)))})
    per = []
    for i in idx:
        f = np.ascontiguousarray(frames[i])
        best = [] if ref is None else [fc.similarity(ref, fc.embed(f, face)) for face in fc.detect(f)]
        per.append({"frame": i, "sim": round(max(best), 4) if best else None})
    sims = [q["sim"] for q in per if q["sim"] is not None]
    return {"median": round(float(np.median(sims)), 4) if sims else None, "faces": len(sims), "master_face": ref is not None,
            "per_frame": per}
