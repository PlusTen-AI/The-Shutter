# Drama editing (design §12 A). Two steps, neither holds a whole video in memory:
#   render(...)  video.edit — one scene's cuts: the used span of each generation (기능 1), sound laid from any generation
#                (기능 2), loudness matched per generation (기능 3), screen text (기능 18) → part files split at the
#                episode breaks inside the scene. The one encode between the generated clips and the film.
#   join(...)    video.join — scene parts in order → film.mp4 by stream copy (기능 4·5); with breaks a white flash at each
#                break and one file per episode (기능 6). Parts of another size are re-encoded to the first one's (기능 7).
# Every encode uses the same libx264 settings, so parts made by different jobs (render, upscale, flash) concatenate
# without re-encoding the video. Audio is decoded and encoded once more at the join: AAC parts copied as they are would
# each carry their 1024-sample priming into the film (about 21 ms drift per part).
from fractions import Fraction

import av
import numpy as np

FPS = 24
RATE = 48000
SPF = RATE // FPS            # audio samples per video frame
CRF, PRESET = '12', 'medium'  # as video.save(crf=12): the upscaler's output joins the same way
ABR = 192000
FADE = 384                   # 8 ms fade where the sound source changes
SNAP = 8                     # frames: a cut edge this close to a shot change of a multi-shot generation moves onto it
LIMIT = 10 ** (-1 / 20)      # -1 dBFS peak


# ---- reading ----
def frames(path, a, n, size=None):
    """n frames at 24 fps from `a` seconds on (per output time the last source frame at or before it, as video.load),
    as rgb uint8 [H, W, 3], center-cropped and resized to size=(w, h) when it differs."""
    c = av.open(path)
    vs = c.streams.video[0]
    vs.thread_type = 'AUTO'
    want = [a + (k + 0.5) / FPS for k in range(n)]
    k, last, t0 = 0, None, None
    try:
        for f in c.decode(vs):
            if f.time is None:
                continue
            t0 = float(f.time) if t0 is None else t0
            t = float(f.time) - t0
            while k < n and t > want[k] and last is not None:
                yield fit(last, size)
                k += 1
            if k >= n:
                break
            last = f.to_ndarray(format='rgb24')
        while k < n and last is not None:   # the source is shorter: hold its last frame
            yield fit(last, size)
            k += 1
    finally:
        c.close()
    if last is None:
        raise ValueError('no video frames: %s' % path)


def fit(img, size):
    if not size or (img.shape[1], img.shape[0]) == tuple(size):
        return img
    from PIL import Image
    w, h = size
    H, W = img.shape[:2]
    cw, ch = (round(H * w / h), H) if W * h > H * w else (W, round(W * h / w))
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    return np.asarray(Image.fromarray(img[y0:y0 + ch, x0:x0 + cw]).resize((w, h), Image.Resampling.LANCZOS))


def audio48(path):
    """The first audio stream as 48 kHz stereo float32 [2, n] (linear resampling as video.concat), or None."""
    c = av.open(path)
    try:
        if not c.streams.audio:
            return None
        rs = av.AudioResampler(format='fltp', layout='stereo', rate=RATE)
        chunks = [r.to_ndarray() for fr in c.decode(c.streams.audio[0]) for r in rs.resample(fr)]
        chunks += [r.to_ndarray() for r in rs.resample(None)]
    finally:
        c.close()
    return np.concatenate(chunks, axis=1).astype(np.float32) if chunks else None


def size_of(path):
    c = av.open(path)
    try:
        s = c.streams.video[0]
        return s.codec_context.width, s.codec_context.height
    finally:
        c.close()


# ---- loudness (ITU-R BS.1770: K-weighting, 400 ms blocks, 75 % overlap, gates -70 LUFS and -10 LU) ----
def _kweight(n):
    """|H(f)|^2 of the two-stage K-weighting filter (48 kHz coefficients) at the rfft bins of an n-sample block."""
    z = np.exp(-1j * np.pi * np.arange(n // 2 + 1) / (n // 2))
    def biquad(b, a):
        return np.abs((b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)) ** 2
    return (biquad((1.53512485958697, -2.69169618940638, 1.19839281085285), (1.0, -1.69065929318241, 0.73248077421585)) *
            biquad((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621)))


def loudness(pcm):
    """Integrated loudness (LUFS) of 48 kHz pcm [ch, n]; -inf for silence or under 400 ms. The filter is applied per
    block in the frequency domain (Parseval) — within a fraction of a dB of the time-domain filter."""
    n, step = int(0.4 * RATE), int(0.1 * RATE)
    if pcm is None or pcm.shape[1] < n:
        return float('-inf')
    w = _kweight(n)
    w[1:-1] *= 2
    starts = np.arange(0, pcm.shape[1] - n + 1, step)
    z = np.zeros(len(starts))
    for ch in pcm[:2]:
        for k0 in range(0, len(starts), 256):   # blocks in batches (memory)
            idx = starts[k0:k0 + 256, None] + np.arange(n)
            spec = np.abs(np.fft.rfft(ch[idx], axis=1)) ** 2
            z[k0:k0 + 256] += (spec * w).sum(1) / (n * n)
    lj = -0.691 + 10 * np.log10(np.maximum(z, 1e-12))
    keep = lj > -70
    if not keep.any():
        return float('-inf')
    rel = -0.691 + 10 * np.log10(z[keep].mean()) - 10
    keep &= lj > rel
    return float(-0.691 + 10 * np.log10(z[keep].mean()))


def gain_db(lufs, speech, target):
    """Gain for one generation's sound: speech toward the target (-15 … +12 dB); room tone and effects are only brought
    down when louder than 8 LU under the target (a quiet room is not pumped up to dialogue level)."""
    if lufs == float('-inf'):
        return 0.0
    if speech:
        return float(np.clip(target - lufs, -15, 12))
    return float(np.clip(min(0.0, target - 8 - lufs), -15, 0))


def limit(pcm):
    """Peak limiter at -1 dBFS: per 5 ms block the gain that keeps it under, held over the neighbouring blocks and
    interpolated per sample, then a hard ceiling."""
    blk = 240
    n = pcm.shape[1]
    nb = -(-n // blk)
    pad = np.zeros((pcm.shape[0], nb * blk), np.float32)
    pad[:, :n] = pcm
    peak = np.abs(pad).reshape(pcm.shape[0], nb, blk).max((0, 2))
    g = np.minimum(1.0, LIMIT / np.maximum(peak, 1e-9))
    if g.min() < 1:
        h = np.pad(g, 4, mode='edge')
        g = np.min(np.stack([h[i:i + nb] for i in range(9)]), 0)   # hold ±4 blocks (20 ms)
        pcm = pcm * np.interp(np.arange(n), np.arange(nb) * blk + blk / 2, g).astype(np.float32)
    return np.clip(pcm, -LIMIT, LIMIT)


# ---- writing ----
class Writer:
    """One mp4 with the shared settings: frames encoded as they come, the audio (one part: a few minutes at most) at
    the end. Both streams are declared before the first packet (the header is written then)."""

    def __init__(self, path, w, h):
        self.c = av.open(path, 'w')
        self.vs = self.c.add_stream('libx264', rate=FPS)
        self.vs.width, self.vs.height, self.vs.pix_fmt = w, h, 'yuv420p'
        self.vs.options = {'crf': CRF, 'preset': PRESET}
        self.ast = add_audio_stream(self.c)
        self.n = 0

    def frame(self, rgb):
        for p in self.vs.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format='rgb24')):
            self.c.mux(p)
        self.n += 1

    def close(self, pcm):
        """pcm [2, n] (padded or cut to the frame count)"""
        for p in self.vs.encode():
            self.c.mux(p)
        encode_audio(self.c, self.ast, pad_to(pcm, self.n * SPF))
        for p in self.ast.encode():
            self.c.mux(p)
        self.c.close()


def add_audio_stream(c):
    a = c.add_stream('aac', rate=RATE)
    a.layout = 'stereo'
    a.bit_rate = ABR
    return a


def encode_audio(c, ast, pcm, start=0):
    """Encode pcm [2, n] on ast with pts from `start` (samples) → the next start. The caller flushes at the end."""
    for k in range(0, pcm.shape[1], 1024):
        fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm[:, k:k + 1024]), format='fltp', layout='stereo')
        fr.sample_rate, fr.pts, fr.time_base = RATE, start + k, Fraction(1, RATE)
        for p in ast.encode(fr):
            c.mux(p)
    return start + pcm.shape[1]


def write_clip(path, rgb_frames, w, h, pcm):
    """A small clip (flash, re-encoded part) with the shared settings → frame count."""
    wr = Writer(path, w, h)
    for f in rgb_frames:
        wr.frame(f)
    wr.close(pcm)
    return wr.n


def pad_to(pcm, m):
    out = np.zeros((2, m), np.float32)
    if pcm is not None:
        k = min(m, pcm.shape[1])
        out[:, :k] = pcm[:2, :k]
    return out


# ---- multi-shot generations (기능 8) ----
DW = 8                       # frames: half width of the dissolve test


def shot_changes(path, want=0):
    """Frame indices where a new shot starts in a generated clip, on small frames (1/8):
    - hard cut: the grey difference to the previous frame stands far above its neighbourhood, or the colour histogram
      jumps (fast motion and dust around a cut hide the first: LTX 3-shot test, 48 against 15~17 around it);
    - dissolve (LTX crossfades some shot changes over ~15 frames): frame k is close to the mean of frames k-DW and k+DW
      while those two differ a lot (a camera move does not average that way); the shot starts where the grey
      difference falls back after it.
    Each is the largest within its reach. want: the shots the cuts on this clip number up to — a flash of light or a
    sudden big move can pass for a cut (5 of 74 one-shot clips of the final test), so only the want - 1 strongest are
    kept. → (indices, frame count)"""
    c = av.open(path)
    vs = c.streams.video[0]
    vs.thread_type = 'AUTO'
    g, h = [], []
    try:
        for f in c.decode(vs):
            a = f.to_ndarray(format='rgb24')[::8, ::8].astype(np.float32)
            g.append(a.mean(2))
            x = np.concatenate([np.histogram(a[..., ch], bins=16, range=(0, 256))[0] for ch in range(3)]).astype(np.float32)
            h.append(x / max(x.sum(), 1))
    finally:
        c.close()
    n = len(g)
    if n < 2:
        return [], n
    g, h = np.array(g), np.array(h)
    d = np.r_[0.0, np.abs(np.diff(g, axis=0)).mean((1, 2))]
    hd = np.r_[0.0, np.abs(np.diff(h, axis=0)).sum(1)]
    out = []
    for i in range(1, n):
        lo, hi = max(1, i - 12), i + 13
        near = np.concatenate([d[lo:max(1, i - 1)], d[i + 2:hi]])
        hnear = np.concatenate([hd[lo:max(1, i - 1)], hd[i + 2:hi]])
        pix = d[i] >= 10 and d[i] > 4 * (float(np.median(near)) if len(near) else 0.0) and d[i] == d[max(1, i - 3):i + 4].max()
        col = hd[i] >= 0.3 and hd[i] > 5 * (float(np.median(hnear)) if len(hnear) else 0.0) and hd[i] == hd[max(1, i - 3):i + 4].max()
        if pix or col:
            out.append((i, max(d[i] / 10, hd[i] / 0.3)))
    ds = np.zeros(n)
    for k in range(DW, n - DW):
        A, B = g[k - DW], g[k + DW]
        ab = float(np.abs(A - B).mean())
        if ab > 0:
            ds[k] = ab * max(0.0, 1 - 2 * float(np.abs(g[k] - (A + B) / 2).mean()) / ab)
    for k in range(DW, n - DW):
        if ds[k] < 6 or ds[k] < ds[k - DW:k + DW + 1].max() or any(abs(k - j) <= 2 * DW for j, _ in out):
            continue
        top = d[k - DW:k + DW + 1].max()
        e = next((j for j in range(k + 1, min(n, k + 2 * DW + 1)) if d[j] < 0.5 * top), min(n - 1, k + DW))
        out.append((e, ds[k] / 6))
    if want > 0:
        out = sorted(out, key=lambda x: -x[1])[:want - 1]
    return sorted(j for j, _ in out), n


def snap(cuts, sources):
    """Cuts whose picture comes from a multi-shot generation. With `shot` (1-based) the cut's start a counts from that
    shot's start as found in the clip, and it ends at the latest where the shot ends. Otherwise each edge within SNAP
    frames of a shot change moves onto it (the model cuts a few frames off the prompt's times). A cut playing its own
    sound moves that with it. → (cuts, report: shot changes per source, moved edges, inner edges with no shot change
    near, shots not found — those cuts take the last shot found)."""
    ch, moved, lone, gone, out = {}, [], [], [], []
    want = {}   # source → the highest shot number its cuts take (0: cuts by seconds only, every change found counts)
    for c in cuts:
        want[c['v']] = max(want.get(c['v'], 0), int(c.get('shot') or 0))
    for ci, c in enumerate(cuts):
        src = sources[c['v']]
        if not src.get('multi'):
            out.append(c)
            continue
        if c['v'] not in ch:
            ch[c['v']] = shot_changes(src['path'], want[c['v']])
        sc, nf = ch[c['v']]
        own = c['s'] == c['v'] and abs(c['sa'] - c['a']) < 1e-6
        if c.get('shot'):
            starts, k = [0] + sc, int(c['shot'])
            if k > len(starts):
                gone.append({'cut': ci, 'shot': k, 'found': len(starts)})
                k = len(starts)
            lo, hi = starts[k - 1], (starts + [nf])[k]
            a0 = min(hi - 1, lo + int(round(c['a'] * FPS)))
            c = dict(c, a=a0 / FPS, n=max(1, min(c['n'], hi - a0)))
            if own:
                c['sa'] = c['a']
            out.append(c)
            continue
        i0 = int(round(c['a'] * FPS))
        edges = []
        for e in (i0, i0 + c['n']):
            j = min(sc, key=lambda x: abs(x - e)) if sc else None
            if j is not None and abs(j - e) <= SNAP:
                edges.append(j)
            else:
                edges.append(e)
                if 0 < e < nf - 1:   # an edge inside the generation with no cut near: the model may have merged the shots
                    lone.append({'cut': ci, 'sec': round(e / FPS, 3)})
        if edges[1] - edges[0] < 1 or edges == [i0, i0 + c['n']]:
            out.append(c)
            continue
        c = dict(c, a=edges[0] / FPS, n=edges[1] - edges[0])
        if own:
            c['sa'] = c['a']
        moved.append({'cut': ci, 'from': [round(i0 / FPS, 3), round((i0 + cuts[ci]['n']) / FPS, 3)],
                      'to': [round(edges[0] / FPS, 3), round(edges[1] / FPS, 3)]})
        out.append(c)
    return out, {'changes': {k: [round(x / FPS, 3) for x in v[0]] for k, v in ch.items()}, 'moved': moved, 'lone': lone,
                 'gone': gone}


# ---- video.edit ----
def fades(c):
    """A cut's fade from / to black (seconds {in, out}) → frame counts, each at most the cut's length."""
    f = c.get('fade') or {}
    return tuple(min(c['n'], max(0, int(round(float(f.get(x) or 0) * FPS)))) for x in ('in', 'out'))


def render(cuts, sources, out_dir, size=None, target=-16.0, overlay=None, progress=lambda p: None):
    """cuts: [{v, a, n, s, sa, part, text[, fade]}] — v: source index of the picture, a: its start (s), n: frames; s:
    source index of the sound (None = silence), sa: its start (s); part: index of the output part; fade: {in, out}
    seconds from / to black, picture and sound. sources: [{path, speech, multi}] — multi: a multi-shot generation,
    whose shot changes the cut edges snap to.
    overlay(rgb, cut, k) draws the cut's screen text on frame k (기능 18). → summary dict (parts, per-source loudness)."""
    size = tuple(size) if size else size_of(sources[cuts[0]['v']]['path'])
    cuts, shots = snap(cuts, sources)
    pcm = {}
    def snd(i):
        if i not in pcm:
            pcm[i] = audio48(sources[i]['path'])
        return pcm[i]
    # loudness of each sound source over the span the cuts use
    spans = {}
    for c in cuts:
        if c['s'] is not None:
            a0, a1 = int(round(c['sa'] * RATE)), int(round(c['sa'] * RATE)) + c['n'] * SPF
            lo, hi = spans.get(c['s'], (a0, a1))
            spans[c['s']] = (min(lo, a0), max(hi, a1))
    gains, info = {}, []
    for i, (lo, hi) in sorted(spans.items()):
        p = snd(i)
        lu = loudness(p[:, lo:hi]) if p is not None else float('-inf')
        gains[i] = 10 ** (gain_db(lu, sources[i].get('speech', True), target) / 20)
        info.append({'source': i, 'lufs': None if lu == float('-inf') else round(lu, 1),
                     'gain_db': round(20 * np.log10(gains[i]), 1)})
    parts, total, done = [], sum(c['n'] for c in cuts), 0
    for pi in sorted({c['part'] for c in cuts}):
        pc = [c for c in cuts if c['part'] == pi]
        name = 'part_%02d.mp4' % pi
        w = Writer('%s/%s' % (out_dir, name), *size)
        track, prev = [], None
        for c in pc:
            fi, fo = fades(c)
            for k, img in enumerate(frames(sources[c['v']]['path'], c['a'], c['n'], size)):
                if overlay and c.get('text'):
                    img = overlay(img, c, k)
                g = min(k / fi if k < fi else 1.0, (c['n'] - 1 - k) / fo if k >= c['n'] - fo else 1.0)
                w.frame(img if g >= 1 else (img.astype(np.float32) * g).astype(np.uint8))
                done += 1
                if done % 24 == 0:
                    progress(0.95 * done / total)
            m = c['n'] * SPF
            if c['s'] is None or snd(c['s']) is None:
                piece, key = np.zeros((2, m), np.float32), None
            else:
                a0 = int(round(c['sa'] * RATE))
                piece, key = pad_to(snd(c['s'])[:, a0:a0 + m], m) * gains[c['s']], (c['s'], a0 + m)
                if not (prev and prev == (c['s'], a0)):   # not continuing the same sound: short fades at the seam
                    piece[:, :FADE] *= np.linspace(0, 1, FADE, dtype=np.float32)
                    if track and prev is not None:
                        track[-1][:, -FADE:] *= np.linspace(1, 0, FADE, dtype=np.float32)
            if fi:
                piece[:, :fi * SPF] *= np.linspace(0, 1, fi * SPF, dtype=np.float32)
            if fo:
                piece[:, -fo * SPF:] *= np.linspace(1, 0, fo * SPF, dtype=np.float32)
                key = None   # the next cut starts from silence, not continuing this sound
            track.append(piece)
            prev = key
        w.close(limit(np.concatenate(track, axis=1)) if track else None)
        parts.append({'name': name, 'frames': w.n, 'sec': round(w.n / FPS, 3), 'w': size[0], 'h': size[1]})
    progress(1.0)
    r = {'parts': parts, 'loudness': info, 'target': target}
    if shots['changes']:
        r['shots'] = shots
    return r


# ---- video.join ----
class Remux:
    """film / episode file: video packets copied from the parts (pts shifted), audio re-encoded once, piece by piece."""

    def __init__(self, path, template):
        self.c = av.open(path, 'w')
        self.vs = self.c.add_stream_from_template(template)
        self.ast = add_audio_stream(self.c)
        self.frames, self.apts = 0, 0

    def add(self, path):
        src = av.open(path)
        try:
            ivs = src.streams.video[0]
            n = 0
            for pkt in src.demux(ivs):
                if pkt.dts is None:
                    continue
                shift = int(round(Fraction(self.frames, FPS) / pkt.time_base))
                pkt.pts += shift
                pkt.dts += shift
                pkt.stream = self.vs
                self.c.mux(pkt)
                n += 1
        finally:
            src.close()
        self.apts = encode_audio(self.c, self.ast, pad_to(audio48(path), n * SPF), self.apts)
        self.frames += n
        return n

    def close(self):
        for p in self.ast.encode():
            self.c.mux(p)
        self.c.close()


def same_stream(a, b):
    """True when part b can be appended to a by stream copy: same size and the same H.264 parameter sets."""
    return (a.codec_context.width, a.codec_context.height) == (b.codec_context.width, b.codec_context.height) and \
        a.codec_context.extradata == b.codec_context.extradata


def join(paths, out_dir, tmp_dir, breaks=(), flash=0.0, progress=lambda p: None):
    """paths: parts in order. breaks: indices of the parts an episode ends with (the last part needs none). flash: white
    flash seconds at each break (0 = none). → film.mp4 (+ ep_NN.mp4 with breaks), summary dict."""
    first = av.open(paths[0])
    tmpl = first.streams.video[0]
    w, h = tmpl.codec_context.width, tmpl.codec_context.height
    fixed, mixed = [], []
    for k, p in enumerate(paths):   # 기능 7: a part of another size (upscaled or not) is re-encoded to the first one's
        c = av.open(p)
        ok = same_stream(tmpl, c.streams.video[0])
        size = (c.streams.video[0].codec_context.width, c.streams.video[0].codec_context.height)
        c.close()
        if ok:
            fixed.append(p)
            continue
        q = '%s/fit_%02d.mp4' % (tmp_dir, k)
        src = av.open(p)
        cnt = sum(1 for pk in src.demux(src.streams.video[0]) if pk.dts is not None)
        src.close()
        write_clip(q, frames(p, 0.0, cnt, (w, h)), w, h, audio48(p))
        fixed.append(q)
        mixed.append({'part': k, 'size': list(size)})
    fl = None
    if flash and breaks:
        fl = '%s/flash.mp4' % tmp_dir
        write_clip(fl, (np.full((h, w, 3), 255, np.uint8) for _ in range(max(1, round(flash * FPS)))), w, h, None)
    eps, cur = [], []
    for k, p in enumerate(fixed):
        cur.append(p)
        if k in breaks and k < len(fixed) - 1:
            eps.append(cur)
            cur = []
    eps.append(cur)
    films = [('film.mp4', [p for i, e in enumerate(eps) for p in e + ([fl] if fl and i < len(eps) - 1 else [])])]
    if breaks:
        films += [('ep_%02d.mp4' % (i + 1), e + ([fl] if fl and i < len(eps) - 1 else [])) for i, e in enumerate(eps)]
    total, done, out = sum(len(f[1]) for f in films), 0, []
    for name, ps in films:
        r = Remux('%s/%s' % (out_dir, name), tmpl)
        for p in ps:
            r.add(p)
            done += 1
            progress(done / total)
        r.close()
        out.append({'name': name, 'frames': r.frames, 'sec': round(r.frames / FPS, 3)})
    first.close()
    return {'files': out, 'size': [w, h], 'reencoded': mixed, 'episodes': len(eps) if breaks else 1}
