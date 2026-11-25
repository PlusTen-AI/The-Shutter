# Video I/O and segment planning for video.swap (#8, design §4). PyAV (shutter-worker image) + numpy.
#   frames, audio, info = load(path)            24 fps RGB uint8 frames [N,H,W,3], audio as (fltp pcm [ch,n], rate, layout)
#                                               (30 fps pulldown film: its unique 24 fps frames)
#   shots(frames)                               [(a, b)] split at hard cuts (colour histogram jump)
#   plan_segments(frames)                       within one shot: cut points at low frame change; lengths 17k+5 (H3)
#   save(path, frames, audio)                   H.264 + original audio
# Segments are generated per shot: a cut needs no continuity, so seams sit on cuts where possible. Inside a long shot,
# neighbouring segments share their boundary frame (seam plan ①). Shots without a usable face keep the source frames.
from fractions import Fraction

import av
import numpy as np

FPS = 24
LENGTHS = (107, 124)          # 17k+5 frames: 4.46 s, 5.17 s at 24 fps
SHORT = (22, 39, 56, 73, 90)  # allowed only for the last segment when the clip doesn't divide
MIN_TAIL = 24                 # a last segment shorter than 1 s is merged into the one before (swap 11: a 6-frame tail froze)


def load(path, fps=FPS, max_side=None, telecine=True):
    """Decode to constant `fps` (nearest source frame per output timestamp) + the audio stream as int16/float pcm.
    A 30 fps file carrying 24 fps film (one repeated frame in every 5, pulldown) gives its unique frames instead:
    resampling it would repeat one frame and skip another in every 4 (FL02 29 s swap, session #7)."""
    c = av.open(path)
    vs = c.streams.video[0]
    rate = float(vs.average_rate or fps)
    frames, times = [], []
    for f in c.decode(vs):
        if f.time is None:
            continue
        img = f.to_ndarray(format="rgb24")
        if max_side and max(img.shape[:2]) > max_side:
            s = max_side / max(img.shape[:2])
            img = f.reformat(width=round(img.shape[1] * s / 2) * 2, height=round(img.shape[0] * s / 2) * 2,
                             format="rgb24").to_ndarray()
        frames.append(img)
        times.append(float(f.time))
    c.close()
    if not frames:
        raise ValueError("no video frames")
    if telecine and fps == FPS and abs(rate - 30) < 0.1:
        p = pulldown(frames)
        if p is not None:
            keep = [i for i in range(len(frames)) if i % 5 != p]
            out = np.stack([frames[i] for i in keep])
            return out, load_audio(path), {"src_frames": len(frames), "frames": len(out), "fps": fps,
                                           "size": out.shape[2:0:-1], "pulldown": p}
    t0, dur = times[0], times[-1] - times[0] + 1.0 / rate
    n = max(1, int(round(dur * fps)))
    idx = np.searchsorted(np.array(times) - t0, (np.arange(n) + 0.5) / fps) - 1
    out = np.stack([frames[max(0, min(len(frames) - 1, i))] for i in idx])
    return out, load_audio(path), {"src_frames": len(frames), "frames": n, "fps": fps, "size": out.shape[2:0:-1]}


def load_audio(path):
    """The first audio stream as (planar float pcm [ch, n], rate, layout name), or None."""
    audio = None
    c = av.open(path)
    if c.streams.audio:
        a = c.streams.audio[0]
        rs = av.AudioResampler(format="fltp", layout=a.layout.name, rate=a.rate)   # planar float, original rate/layout
        chunks = [r.to_ndarray() for fr in c.decode(a) for r in rs.resample(fr)]
        chunks += [r.to_ndarray() for r in rs.resample(None)]
        if chunks:
            audio = (np.concatenate(chunks, axis=1), a.rate, a.layout.name)
    c.close()
    return audio


def pulldown(frames, side=64):
    """Phase (0..4) of the repeated frame if `frames` (30 fps) carry 24 fps film, else None: one phase of the
    frame-to-frame change is near zero while the others move. One phase for the whole clip (single-shot sources)."""
    if len(frames) < 25:
        return None
    step = max(1, max(frames[0].shape[:2]) // side)
    g = [f[::step, ::step].astype(np.float32).mean(-1) for f in frames]
    d = np.array([np.abs(g[i] - g[i - 1]).mean() for i in range(1, len(g))])   # d[i-1]: frame i vs i-1
    m = np.array([np.median(d[(np.arange(1, len(g)) % 5) == q]) for q in range(5)])
    p = int(m.argmin())
    rest = np.median(np.delete(m, p))
    return p if rest > 0.5 and m[p] < 0.2 * rest else None


def change(frames, side=64):
    """Per-boundary change score d[i] = mean |f[i] - f[i-1]| on a small grey copy (d[0] = 0)."""
    step = max(1, max(frames.shape[1:3]) // side)
    g = frames[:, ::step, ::step].astype(np.float32).mean(-1)
    d = np.zeros(len(frames), np.float32)
    d[1:] = np.abs(g[1:] - g[:-1]).mean((1, 2))
    return d


CUT = 0.5   # colour histogram jump (0..3) above this is a hard cut (measured: cuts 0.82~2.8, fast dance moves <= 0.39)


def shots(frames, thr=CUT):
    """Split at hard cuts -> [(a, b)] inclusive, covering every frame."""
    s = frames[:, ::8, ::8]
    h = np.stack([np.concatenate([np.bincount(f[..., c].ravel() >> 3, minlength=32) for c in range(3)])
                  for f in s]).astype(np.float32)
    h /= h.sum(1, keepdims=True) / 3
    jump = np.abs(h[1:] - h[:-1]).sum(1) / 2          # jump[i-1]: between frame i-1 and i
    starts = [0] + [i for i in range(1, len(frames)) if jump[i - 1] > thr]
    return [(a, b - 1) for a, b in zip(starts, starts[1:] + [len(frames)])]


def plan_segments(frames, lengths=LENGTHS, short=SHORT):
    """Cut the clip into segments [(start, end)] (end inclusive, shared with the next start) minimising the sum of
    frame-change scores at the cuts. Every segment has 17k+5 frames; the last may be one of `short` or be trimmed."""
    n = len(frames)
    d = change(frames)
    INF = float("inf")
    best = [INF] * n          # best[i]: cost of a plan whose last segment ends at frame i
    prev = [-1] * n
    best[0] = 0.0
    for i in range(n):
        if best[i] == INF:
            continue
        for L in lengths:
            j = i + L - 1
            if j < n and best[i] + d[j] < best[j]:
                best[j], prev[j] = best[i] + d[j], i
    # finish: from the reachable end closest to the tail, one more segment (allowed lengths, or a long one trimmed).
    # A tail shorter than MIN_TAIL frames is merged into the segment before it (one trimmed long segment; H3 pins the
    # end guide early in a very short tail, so those frames freeze); if that would be too long, the closest end that
    # leaves a tail of at least MIN_TAIL is used instead
    ends = [i for i in range(n) if best[i] < INF]
    last = max(ends)
    if last < n - 1 and n - last < MIN_TAIL:
        if prev[last] >= 0 and n - prev[last] <= lengths[-1]:
            last = prev[last]
        else:
            last = max([i for i in ends if MIN_TAIL <= n - i <= lengths[-1]] or [last])
    segs = []
    if last < n - 1:
        rest = n - last
        L = next((s for s in short + lengths if s >= rest), lengths[-1])
        segs.append((last, n - 1, L))   # generate L frames, keep n - last
    i = last
    while i > 0:
        p = prev[i]
        segs.append((p, i, i - p + 1))
        i = p
    return sorted(segs)


def stitch(base, parts):
    """base: source frames at the canvas size [N,H,W,3]; parts: [(start, frames)] generated segments, written in place
    (a shared boundary frame takes the later segment's copy). Frames no segment covers keep the source."""
    out = base.copy()
    for start, fr in parts:
        out[start:start + len(fr)] = fr[:len(out) - start]
    return out


def save(path, frames, audio=None, fps=FPS, crf=18):
    c = av.open(path, "w")
    vs = c.add_stream("libx264", rate=fps)
    vs.width, vs.height, vs.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
    vs.options = {"crf": str(crf), "preset": "medium"}
    ast = None
    if audio is not None:
        pcm, rate, layout = audio
        ast = c.add_stream("aac", rate=rate)
        ast.layout = layout
    for f in frames:
        for p in vs.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            c.mux(p)
    for p in vs.encode():
        c.mux(p)
    if ast is not None:
        pcm = pcm[:, : int(len(frames) / fps * rate)].astype(np.float32)
        for k in range(0, pcm.shape[1], 1024):
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm[:, k:k + 1024]), format="fltp", layout=layout)
            fr.sample_rate, fr.pts, fr.time_base = rate, k, Fraction(1, rate)
            for p in ast.encode(fr):
                c.mux(p)
        for p in ast.encode():
            c.mux(p)
    c.close()


def resize(frames, w, h):
    """Center-crop to the w:h aspect, then resize (PIL lanczos) — frames [N,H,W,3] uint8."""
    from PIL import Image
    H, W = frames.shape[1:3]
    if W * h > H * w:
        cw, ch = round(H * w / h), H
    else:
        cw, ch = W, round(W * h / w)
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    return np.stack([np.array(Image.fromarray(f[y0:y0 + ch, x0:x0 + cw]).resize((w, h), Image.Resampling.LANCZOS))
                     for f in frames])
