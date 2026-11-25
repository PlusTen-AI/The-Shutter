# LTX-2.5 pipeline on the shutter engine (video.i2v): first frame + prompt -> video with sound.
# The "LTX-2.5 I2V" template (tools/h3_5s/comfy_ltx25_i2v.py, golden/ltx25_dump.py), reproduced node by node (ComfyUI
# e9afbccc2 + ComfyUI-LTXVideo nodes):
#    LoadImage -> ResizeImageMaskNode (longer side 1536, lanczos) -> LTXVPreprocess (crf 18) -> CLIPTextEncode (cfg 1:
#    the negative prompt is never evaluated) -> stage 1 at W/2 x H/2: EmptyLTXVLatentVideo + LTXVImgToVideoInplace 0.7
#    + empty audio, euler_ancestral over SIGMAS_S1, noise seed -> LTXVLatentUpsampler x2 -> LTXVImgToVideoInplace 1.0 +
#    the stage-1 audio, SIGMAS_S2, noise seed2 -> video VAE decode (untiled; decoder noise from seed 0 and the output
#    size) + audio VAE decode (48 kHz).
#   p = Ltx()
#   frames, wav = p.generate(image_u8, prompt, W, H, frames, seed, seed2)   # uint8 [F, H, W, 3], f32 [2, n] at 48 kHz
import time
from io import BytesIO

import av
import numpy as np
import torch

import ltx_audio
import ltx_dit as L
import ltx_te
import ltx_vae
from h3_tok import _lanczos

FPS = 24
LONGER, CRF = 1536, 18                  # template: ResizeImageMaskNode longer side, LTXVPreprocess img_compression
STRENGTH = (0.7, 1.0)                   # LTXVImgToVideoInplace per stage (template)
SAMPLE_RATE = ltx_audio.SAMPLE_RATE


def _sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def _resize_longer(x, n):
    """ResizeImageMaskNode "scale longer dimension" (lanczos): float32 [H, W, 3] -> float32 [h, w, 3]."""
    h, w = x.shape[:2]
    if h > w:
        w, h = round((w / h) * n), n
    elif w > h:
        h, w = round((h / w) * n), n
    else:
        h = w = n
    return _lanczos(x[None], w, h)[0]


def _crop_box(oh, ow, w, h):
    """comfy.utils.common_upscale crop "center": the (y0, x0) offsets that cut [oh, ow] to the w / h aspect."""
    oa, na = ow / oh, w / h
    cx = cy = 0
    if oa > na:
        cx = round((ow - ow * (na / oa)) / 2)
    elif oa < na:
        cy = round((oh - oh * (oa / na)) / 2)
    return cy, cx


def _preprocess(x, crf):
    """LTXVPreprocess: even crop, one-frame libx264 (veryfast, yuv420p) round trip, as comfy_extras/nodes_lt.preprocess."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    arr = (t[:(t.shape[0] // 2) * 2, :(t.shape[1] // 2) * 2] * 255.0).byte().numpy()
    buf = BytesIO()
    c = av.open(buf, "w", format="mp4")
    s = c.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
    s.height, s.width = arr.shape[:2]
    c.mux(s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")))
    c.mux(s.encode())
    c.close()
    c = av.open(BytesIO(buf.getvalue()))
    out = next(c.decode(next(s for s in c.streams if s.type == "video"))).to_ndarray(format="rgb24")
    c.close()
    return torch.tensor(out, dtype=torch.float32) / 255.0


def _bilinear(x, w, h, crop=False):
    """comfy.utils.common_upscale(bilinear, crop "center" or "disabled") of images (cpu float32 [B, H, W, 3])."""
    s = x.movedim(-1, 1)
    if crop:
        cy, cx = _crop_box(s.shape[-2], s.shape[-1], w, h)
        s = s.narrow(-2, cy, s.shape[-2] - cy * 2).narrow(-1, cx, s.shape[-1] - cx * 2)
    if s.shape[-2:] != (h, w):
        s = torch.nn.functional.interpolate(s, size=(h, w), mode="bilinear")
    return s.movedim(1, -1)


def _center_bilinear(x, w, h):
    """comfy.utils.common_upscale(bilinear, crop "center") of one image (cpu float32 [H, W, 3]) -> [1, h, w, 3]."""
    return _bilinear(x[None], w, h, crop=True)


class Ltx:
    def __init__(self, attn_q8=False):
        t0 = time.perf_counter()
        self.te = ltx_te.LtxTE()
        self.dit = L.Dit(attn_q8=attn_q8)
        self.vae = ltx_vae.Vae()
        self.ups = ltx_vae.Upsampler()
        self.aud = ltx_audio.LtxAudio()
        self.load_s = time.perf_counter() - t0
        self.times = {}
        self.info = {}

    def close(self):
        for k, f in (("dit", "close"), ("vae", "close"), ("ups", "close"), ("te", "free"), ("aud", "free")):
            if getattr(self, k, None) is not None:
                getattr(getattr(self, k), f)()
                setattr(self, k, None)
        torch.cuda.empty_cache()

    def _guide(self, pre, samples):
        """LTXVImgToVideoInplace: the image's latent (bilinear center resize to the latent's pixel size) on frame 0."""
        _, _, _, lh, lw = samples.shape
        z = self.vae.encode(_center_bilinear(pre, 32 * lw, 32 * lh).cuda())
        samples = samples.clone()
        samples[:, :, :z.shape[2]] = z
        return samples

    def generate(self, image, prompt, W, H, frames, seed, seed2=42, progress=None, fps=float(FPS)):
        """image uint8 [h, w, 3] (the first frame); W, H multiples of 64 (stage 1 runs at W/2 x H/2); frames 8k + 1.
        Returns (uint8 frames [frames, H, W, 3] cpu, f32 waveform [2, n] at 48 kHz cpu); times in self.times,
        the settings used in self.info."""
        p = progress or (lambda f: None)
        tm = self.times = {"load": self.load_s}
        t0 = _sync()
        if W % 64 or H % 64:
            raise ValueError(f"size {W}x{H}: width and height must be multiples of 64")
        if frames % 8 != 1:
            raise ValueError(f"frames {frames} is not 8k + 1")
        s1, s2 = STRENGTH
        self.info = dict(width=W, height=H, frames=frames, fps=fps, seed=seed, seed2=seed2, strength=[s1, s2],
                         sigmas2=L.SIGMAS_S2)

        # LoadImage -> resize (longer 1536, lanczos) -> LTXVPreprocess; CLIPTextEncode + the text connectors
        pre = _preprocess(_resize_longer(np.asarray(image).astype(np.float32) / 255.0, LONGER), CRF)
        t1 = _sync()
        _, cond = self.te.encode(prompt.strip())
        ctx = self.dit.connect(cond)
        t2 = _sync()
        tm["image_prep"], tm["te"] = t1 - t0, t2 - t1
        p(0.03)

        T, ta = (frames - 1) // 8 + 1, round(frames / FPS * 25)   # video latent frames; audio latents (25 per second)
        lh, lw = (H // 2) // 32, (W // 2) // 32                   # stage-1 latent grid

        # stage 1 at W/2 x H/2: the image on frame 0 (LTXVImgToVideoInplace), empty audio
        v = torch.zeros(1, 128, T, lh, lw, device="cuda")
        a = torch.zeros(1, 8, ta, 16, device="cuda")
        mv, ma = L.i2v_mask(v.shape, a.shape, 0.0)
        mv[:, :, 0] = 1.0 - s1
        v = self._guide(pre, v)
        tm["guide_encode"] = _sync() - t2
        self.dit.prepare(ctx, (T, lh, lw), ta, float(fps))
        sig = L.parse_sigmas(L.SIGMAS_S1)
        nz = list(L.init_noise(seed, v.shape, a.shape))
        n = len(sig) - 1
        out = self.dit.sample((v, a), (mv, ma), sig, nz, seed=seed,
                              callback=lambda i, x, den: p(0.03 + 0.22 * (i + 1) / n))
        v1, a1 = out[0], out[1]
        del v, out
        t3 = _sync()
        tm["stage1"] = t3 - t2

        # latent upsampler x2, then stage 2 at W x H: the image on frame 0 at full strength, the stage-1 audio re-noised
        up = self.ups(v1)
        t4 = _sync()
        tm["upsample"] = t4 - t3
        mv, ma = L.i2v_mask(up.shape, a1.shape, 0.0)
        mv[:, :, 0] = 1.0 - s2
        v = self._guide(pre, up)
        del up
        self.dit.prepare(ctx, tuple(v.shape[2:]), ta, float(fps))
        sig = L.parse_sigmas(L.SIGMAS_S2)
        n = len(sig) - 1
        out = self.dit.sample((v, a1), (mv, ma), sig, list(L.init_noise(seed2, v.shape, a1.shape)), seed=seed2,
                              callback=lambda i, x, den: p(0.25 + 0.45 * (i + 1) / n))
        v2, a2 = out[0], out[1]
        del v, a1, out
        t5 = _sync()
        tm["stage2"] = t5 - t4

        # decode: video (untiled, decoder noise seed 0) and audio
        px = self.vae.decode(v2, seed=0)
        out = (255.0 * px.clamp(0, 1)).to(torch.uint8).cpu().numpy()   # SaveImage: clip(255 x) -> uint8 (truncating)
        del px
        t6 = _sync()
        wav = self.aud.decode(a2).cpu().numpy()
        t7 = _sync()
        tm["decode_video"], tm["decode_audio"], tm["total"] = t6 - t5, t7 - t6, t7 - t0
        tm["latent"] = [list(v2.shape), list(a2.shape)]
        p(0.95)
        return out, wav
