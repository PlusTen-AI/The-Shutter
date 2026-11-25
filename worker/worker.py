#!/usr/bin/env python3
"""shutter worker (one per C·D node, container): takes units from the server, runs them and uploads the results.

Each node keeps only one model loaded. Told the currently loaded model, the server hands out that model's units first (§3).
D's worker calls the server at its CX7 address, so input and result files travel over CX7 too.
"""
import ctypes, gc, hashlib, json, os, shutil, signal, socket, sys, tempfile, threading, time, traceback
import urllib.error, urllib.request

SERVER = os.environ.get('SHUTTER_SERVER', 'http://127.0.0.1:8300')
NAME = os.environ.get('SHUTTER_NODE', socket.gethostname())
ASSETS = os.environ.get('SHUTTER_WORK', '/work') + '/cache/assets'   # on C the server's own folder: nothing to fetch
CHUNK = 1 << 20
BEAT = 30   # seconds. Well below the server's STALE (120)
MIN_FACE = 64   # boundary frame face height (px); below this, no face check
MIN_LUMA = 60   # mean brightness of the face box; below this is a silhouette (measured: silhouettes 34~39, normal 99~117)
MIN_SHOT = 3    # shots shorter than this (transition frames between shots) stay as in the source
POSE_MAX = 45   # boundary edit vs source: mean arm or leg angle difference (deg) above this = pose lost. FL02 29 s
                # swap (session #7, pose.py): kept boundaries 7~29 deg, arms-down redraws 64~80 deg -> midpoint
POSE_RETRIES = 2
FRAME_MIN, FRAME_MAX = 0.75, 1.33   # boundary edit vs source: largest face height ratio. Zoomed-out redraws of a medium
                                    # shot and a 9:16 close-up were about 0.5 (swap 16·17, session #11, by eye)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'engine', 'py'))   # engine Python


class Abort(Exception):
    """The server no longer treats this unit as ours (taken back after a timeout etc.)."""


class Stop(Exception):
    """Stop signal (docker stop = SIGTERM): the running unit is let go at its next progress report and goes back to the
    queue (not counted as a try, §5)."""


STOP = threading.Event()   # the signal handler only sets a flag. GPU work is never cut off midway


def call(method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(SERVER + path, data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        if e.code == 409:
            raise Abort(path)
        raise


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


class Ctx:
    """What the job type functions use: input file paths, output folder, progress reports."""

    def __init__(self, unit, out):
        self.unit, self.out = unit, out
        self.p, self.lost = 0.0, False

    def input(self, name):
        sha = self.unit['inputs'][name]
        path = '%s/%s' % (ASSETS, sha)
        if not os.path.isfile(path):   # fetch only when it is not in the node cache
            tmp = path + '.%s.part' % NAME
            with urllib.request.urlopen(SERVER + '/assets/' + sha, timeout=60) as r, open(tmp, 'wb') as f:
                shutil.copyfileobj(r, f, CHUNK)
            if sha_file(tmp) != sha:
                os.remove(tmp)
                raise IOError('asset sha mismatch ' + sha)
            os.replace(tmp, path)
        return path

    def progress(self, p):
        self.p = p
        if STOP.is_set():                 # progress reports come between engine calls (at step ends)
            raise Stop(self.unit['id'])
        if self.lost:
            raise Abort(self.unit['id'])
        call('POST', '/units/%s/progress' % self.unit['id'], {'worker': NAME, 'p': round(p, 3)})

    def heartbeat(self, stop):
        """Report alive every BEAT seconds while the unit runs. One H3 step can outlast the server's STALE (120 s)."""
        while not stop.wait(BEAT):
            try:
                call('POST', '/units/%s/progress' % self.unit['id'], {'worker': NAME, 'p': round(self.p, 3)})
            except Abort:
                self.lost = True          # stops at the next progress report
                return
            except Exception:
                pass                      # server briefly unreachable: try again on the next beat


# ---- job types: unit → write files to the output folder ----
def run_fake(u, ctx):
    """Fake job without a GPU: sleeps a few seconds and writes a small text file derived from the inputs."""
    o, i = u['opts'], u['spec']['i']
    sec, steps = float(o.get('sec', 3)), 6
    for s in range(steps):
        time.sleep(sec / steps)
        ctx.progress((s + 1) / steps)
    if i == 0 and u['total'] <= int(o.get('fail', 0)):   # for failure/retry tests
        raise RuntimeError('fake failure (attempt %d)' % u['total'])
    lines = ['unit %d/%d  node %s  model %s  attempt %d' % (i + 1, u['spec']['n'], NAME, u['model'], u['total'])]
    for name in sorted(u['inputs']):
        p = ctx.input(name)
        lines.append('%s  sha256 %s  %d bytes' % (name, sha_file(p), os.path.getsize(p)))
    with open('%s/out_%02d.txt' % (ctx.out, i), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_probe(u, ctx):
    """video.probe (cpu unit): what the UI shows before a swap — info.json (seconds, fps, size, audio) and the lit faces of
    the first frame (within the first 3 s) that has any, left to right, as face_<k>.png crops (the 사람 table)."""
    import av
    import numpy as np
    from PIL import Image
    import face
    path = ctx.input('video')
    with av.open(path) as c:
        s = c.streams.video[0]
        dur = float(c.duration / 1e6) if c.duration else float(s.duration * s.time_base) if s.duration else None
        fps = float(s.average_rate) if s.average_rate else None
        info = {'sec': dur, 'fps': fps, 'width': s.codec_context.width, 'height': s.codec_context.height,
                'audio': bool(c.streams.audio), 'faces': [], 'at': None}
        fc = ENG.setdefault('face', face.FaceCheck())
        for fr in c.decode(video=0):
            t = float(fr.time or 0)
            if t > 3:
                break
            if info['at'] is not None and t < info['at'] + 0.5:
                continue
            img = fr.to_ndarray(format='rgb24')
            faces = sorted(fc.visible(img, MIN_LUMA), key=lambda f: f[0][0])
            info['at'] = t
            if faces:
                for k, f in enumerate(faces):
                    x0, y0, x1, y1 = [float(v) for v in f[0][:4]]
                    cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.8
                    box = [max(0, int(cx - r)), max(0, int(cy - r)), min(img.shape[1], int(cx + r)), min(img.shape[0], int(cy + r))]
                    Image.fromarray(img[box[1]:box[3], box[0]:box[2]]).resize((160, 160)).save('%s/face_%d.png' % (ctx.out, k))
                    info['faces'].append({'k': k, 'box': box, 'size': int(y1 - y0)})
                break
    with open(ctx.out + '/info.json', 'w') as f:
        json.dump(info, f)


# ---- engine (Qwen-Image 2.1). Base and Turbo share the encoder and VAE; switching swaps only the DiT ----
ENG = {}
LEGS_MIN = 0.45   # full-body panel leg ratio floor (pose.leg_ratio; good panels 0.50-0.56, squeezed MS01_ref_v1 0.37)


def run_image(u, ctx):
    """One image for image.t2i / image.edit: encoder → prefix → sampling → VAE → PNG + settings json."""
    import qi21
    import torch
    o, spec = u['opts'], u['spec']
    images = []
    if u['inputs']:
        names = sorted(u['inputs'], key=lambda n: int(n.rsplit('_', 1)[-1]) if n.rsplit('_', 1)[-1].isdigit() else 0)
        images = [qi21.load_ref(ctx.input(n), int(o.get('ref_resolution', 1024))) for n in names]
    if images and not (o.get('width') and o.get('height')) and not o.get('mp'):
        W, H = images[0].shape[1], images[0].shape[0]     # size of the first reference (ComfyUI edit template default)
    else:
        W, H = qi21.t2i_size(o)
    cond, slots = ENG['te'].encode(o['prompt'], images)
    # cfg > 1: the negative prompt (empty = ' ', like ComfyUI) with the same references — two DiT calls a step
    cfg = float(o.get('cfg') or 1)
    neg = ENG['te'].encode(o.get('neg') or '', images) if abs(cfg - 1) > 1e-6 else None
    refs = [ENG['vae'].encode(im) for im in images]
    dit = ENG['dit']
    dit.prefix(cond, refs, slots, H // 16, W // 16, neg)
    # face check (#18): check = name of the input to compare with. Below threshold → redraw with a new seed (#16, no
    # picking among several candidates)
    face_ref, sims = None, []
    if o.get('check'):
        import face
        ENG.setdefault('face', face.FaceCheck())
        import numpy as np
        from PIL import Image
        face_ref = ENG['face'].embed(np.array(Image.open(ctx.input(o['check'])).convert('RGB')))
    # body check (pose.py DWPose): 'framing' = a waist-up master must not show the ankles; 'legs' = a full-body sheet
    # panel's leg ratio must be >= LEGS_MIN (drama test 02: good panels 0.50-0.56, the squeezed MS01_ref_v1 0.37)
    body_check, bodies = o.get('body_check') or (u['type'] in ('char.ref', 'char.from_photo') and 'framing'), []
    if body_check:
        import pose
        ENG.setdefault('pose', pose.Pose())
    face_retries = int(o.get('retries', 3)) if face_ref is not None else 0
    # a body miss redraws up to body_retries times (default 2), even on a no-retry panel. Both are off when the UI's
    # "미달 시 자동 다시 뽑기" is off — the check then only reports (boss, session #10)
    tries = 1 + max(face_retries, int(o.get('body_retries', 2)) if body_check else 0)
    min_sim = float(o.get('min_sim', 0.5))       # provisional threshold — to be set by running our characters
    for t in range(tries):
        seed = spec['seed'] + t * 1000003
        lat = dit.sample(H // 16, W // 16, seed, int(o.get('steps', 25)),
                         progress=lambda p: ctx.progress((t + 0.95 * p) / tries), cfg=cfg)
        img = ENG['vae'].decode(lat)
        sim, body_ok = None, True
        if face_ref is not None or body_check:
            arr = (255.0 * img[..., :3].clamp(0, 1)).to(torch.uint8).cpu().numpy()
        if face_ref is not None:   # the best-matching face (the largest one was the wrong person in a two-shot, swap 17)
            ds = ENG['face'].detect(arr)
            sim = round(max(ENG['face'].similarity(face_ref, ENG['face'].embed(arr, d)) for d in ds), 4) if ds else None
            sims.append(sim)
        if body_check:
            kp = ENG['pose'].body([arr])[0]
            b = {'seed': seed, 'ankles': pose.ankles_seen(kp), 'legs': pose.leg_ratio(kp)}
            body_ok = (not b['ankles']) if body_check == 'framing' else (b['legs'] is None or b['legs'] >= LEGS_MIN)
            bodies.append(dict(b, ok=body_ok))
        face_ok = face_ref is None or (sim is not None and sim >= min_sim)
        if body_ok and (face_ok or t >= face_retries):
            break
    base = '%s/%02d_%d' % (ctx.out, spec['i'], seed)
    people = _plate_people(u, ctx, img) if u['type'] == 'scene.plate' else None
    qi21.to_png(img, base + '.png')
    with open(base + '.json', 'w') as f:
        json.dump({'type': u['type'], 'model': u['model'], 'seed': seed, 'width': W, 'height': H,
                   'steps': None if dit.turbo else int(o.get('steps', 25)), 'prompt': o['prompt'],
                   'cfg': cfg if neg else None, 'neg': (o.get('neg') or '') if neg else None,
                   'face_sim': sims or None, 'min_sim': min_sim if face_ref is not None else None,
                   'passed': None if face_ref is None else bool(sims and sims[-1] is not None and sims[-1] >= min_sim),
                   'body_check': body_check or None, 'body': bodies or None, 'people': people,
                   'inputs': u['inputs'], 'node': NAME}, f, ensure_ascii=False, indent=1)


def _plate_people(u, ctx, img):
    """scene.plate: where each person stands — for every person picture (image_2, image_3, …) the box of the face most
    like it (one face per person, best pairs first) as fractions of the image, or None. A cut box drawn on the plate turns
    on only the people whose face is inside it (boss, session #14)."""
    import numpy as np
    import torch
    from PIL import Image
    import face
    fc = ENG.setdefault('face', face.FaceCheck())
    arr = (255.0 * img[..., :3].clamp(0, 1)).to(torch.uint8).cpu().numpy()
    names = sorted((n for n in u['inputs'] if n.startswith('image_') and n != 'image_1'), key=lambda n: int(n.split('_')[1]))
    refs = [fc.embed(np.array(Image.open(ctx.input(n)).convert('RGB'))) for n in names]
    faces = fc.detect(arr)
    embs = [fc.embed(arr, f) for f in faces]
    pairs = sorted(((float(fc.similarity(r, e)), i, k) for i, r in enumerate(refs) if r is not None
                    for k, e in enumerate(embs)), reverse=True)
    h, w = arr.shape[:2]
    out, used = [None] * len(refs), set()
    for _, i, k in pairs:
        if out[i] is None and k not in used:
            x0, y0, x1, y1 = (float(v) for v in faces[k][0][:4])
            out[i] = [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]
            used.add(k)
    return out


# ---- character sheet (char.sheet): 8 derive panels from the master → one H3 reference sheet ----
SHEET_HEADS = (('sh_head_front', 0, 0), ('sh_head_34L', 640, 0), ('sh_head_34R', 0, 640), ('sh_head_side', 640, 640))
SHEET_FULLS = ('sh_full_front', 'sh_full_34', 'sh_full_side', 'sh_full_back')


def run_sheet(u, ctx):
    """char.sheet. panel: one derive panel (image_1 = master, face check, retries per panel) → sh_<kind>.png/json.
    compose: the 8 panels → sheet.png, left 2x2 heads (640x640 each, square from the upper part), right 4 full bodies
    (540x1280 each, centre strip, head to shoes) = 3440x1280."""
    sp = u['spec']
    if sp['step'] == 'panel':
        legs = sp['kind'] in ('sh_full_front', 'sh_full_34', 'sh_full_side')   # the back panel has no nose to measure from
        run_image(dict(u, opts=dict(u['opts'], prompt=sp['prompt'], retries=sp['retries'], body_check=legs and 'legs'),
                       spec={'i': 0, 'seed': sp['seed']}), ctx)
        for n in os.listdir(ctx.out):   # 00_<seed>.png/json → sh_<kind>.png/json (the unit's own output folder)
            os.replace(os.path.join(ctx.out, n), os.path.join(ctx.out, sp['kind'] + os.path.splitext(n)[1]))
        return
    from PIL import Image
    sheet = Image.new('RGB', (1280 + 540 * len(SHEET_FULLS), 1280), 'white')
    for n, x, y in SHEET_HEADS:
        im = Image.open(fetch_result(u, n + '.png', ctx.out + '/_p.png')).convert('RGB')
        s = im.width
        y0 = int((im.height - s) * 0.08)
        sheet.paste(im.crop((0, y0, s, y0 + s)).resize((640, 640), Image.LANCZOS), (x, y))
    for i, n in enumerate(SHEET_FULLS):
        im = Image.open(fetch_result(u, n + '.png', ctx.out + '/_p.png')).convert('RGB')
        w = int(im.height * 540 / 1280)
        x0 = (im.width - w) // 2
        sheet.paste(im.crop((x0, 0, x0 + w, im.height)).resize((540, 1280), Image.LANCZOS), (1280 + 540 * i, 0))
    os.remove(ctx.out + '/_p.png')
    sheet.save(ctx.out + '/sheet.png')


# ---- person swap (video.swap, design §4): plan → boundary edit (qi21) → segments (h3) → stitch ----
def fetch_result(u, name, dest):
    with urllib.request.urlopen('%s/results/%s/%s' % (SERVER, u['job'], name), timeout=600) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f, CHUNK)
    return dest


def _bg_dist(a, b):
    """Mean colour distance between the outer thirds (left and right, upper 60 %) of image a (HxWx3 uint8) and image b
    (PIL) at 24x48 blocks — the person stands mostly in the middle. Used to tell whether a boundary edit really took the
    new background."""
    from PIL import Image
    import numpy as np
    x = np.asarray(Image.fromarray(a).resize((24, 48), Image.BOX), np.float32)
    y = np.asarray(b.resize((24, 48), Image.BOX), np.float32)
    m = np.zeros((48, 24), bool)
    m[:29, :8] = m[:29, 16:] = True
    return float(np.abs(x - y)[m].mean())


UP_LEN, UP_OVERLAP = 121, 8   # upscale chunks (swap, video.upscale): frames per SeedVR2 unit (5 s) and frames shared


def up_chunks(n, even=False):
    """[(start, end)] over n frames: UP_LEN long, consecutive chunks share UP_OVERLAP frames (crossfaded in 'final':
    each chunk draws its own restoration noise, a hard cut would show). A tail shorter than 2 overlaps joins the previous.
    even (video.upscale): chunks of one length 4m + 1 (SeedVR2 pads to that), at most UP_LEN + 2 overlaps (the longest
    merged tail above), in an even number so C·D take the same share — a 5~10 s cut is two halves made at once.
    Under 2 s: one chunk (the second node's model load would eat the gain)."""
    if even:
        top = UP_LEN + 2 * UP_OVERLAP
        k = max(1, -(-(n - UP_OVERLAP) // (top - UP_OVERLAP)))
        k += k % 2 if n >= 48 else 0
        if k == 1:
            return [(0, n - 1)]
        L = -(-(n + (k - 1) * UP_OVERLAP) // k)
        L = min(top, L + (1 - L) % 4)
        return [(a, a + L - 1) for a in (round(j * (n - L) / (k - 1)) for j in range(k))]
    out, a = [], 0
    while True:
        b = min(n - 1, a + UP_LEN - 1)
        out.append((a, b))
        if b == n - 1:
            break
        a = b - UP_OVERLAP + 1
    if len(out) > 1 and out[-1][1] - out[-1][0] + 1 < 2 * UP_OVERLAP:
        out[-2:] = [(out[-2][0], n - 1)]
    return out


def up_units(u, n, after, even=False):
    """Append the SeedVR2 chunk units over n frames (C·D share them, each after the units `after`) and the cpu unit
    'final' that joins them. even (video.upscale): even chunks (up_chunks), and a video of one chunk is a single unit
    without chunks or join."""
    ch = up_chunks(n, even)
    path = '/jobs/%s/units' % u['job']
    if even and len(ch) == 1:
        call('POST', path, {'units': [{'model': 'sv2', 'spec': {'step': 'up', 'after': after}}]})
        return
    first = call('POST', path, {'units': [{'model': 'sv2', 'spec': {'step': 'up', 'k': k, 'start': a, 'end': b,
                                                                     'after': after}} for k, (a, b) in enumerate(ch)]})['first']
    call('POST', path, {'units': [{'model': 'cpu', 'spec': {'step': 'final', 'chunks': ch,
                                                            'after': list(range(first, first + len(ch)))}}]})


def up_join(u, ctx, ch):
    """The chunk results up_KK.mp4 → one frame array: linear crossfade over the frames consecutive chunks share (each
    chunk draws its own restoration noise, a hard cut would show)."""
    import numpy as np
    import video
    out = None
    for k, (a, b) in enumerate(ch):
        fr, _, _ = video.load(fetch_result(u, 'up_%02d.mp4' % k, ctx.out + '/_u.mp4'))
        if out is None:
            out = np.empty((ch[-1][1] + 1,) + fr.shape[1:], np.uint8)
        n = ch[k - 1][1] - a + 1 if k else 0   # frames shared with the previous chunk
        if n:
            w = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))[:, None, None, None]
            out[a:a + n] = (out[a:a + n] * (1 - w) + fr[:n] * w).round().astype(np.uint8)
        out[a + n:b + 1] = fr[n:b - a + 1]
    os.remove(ctx.out + '/_u.mp4')
    return out


def up_drop(u, ch):
    """The chunks are intermediates (about 50 MB each): dropped once the joined result exists."""
    for k in range(len(ch)):
        call('DELETE', '/results/%s/up_%02d.mp4' % (u['job'], k))


def _face_sims(fc, frames, e0):
    """Similarity to e0 of the best face in every 4th frame → {median, min, no_face} (as the face pass test)."""
    import numpy as np
    v, none = [], 0
    for i in range(0, len(frames), 4):
        ds = fc.detect(frames[i])
        if ds:
            v.append(max(fc.similarity(e0, fc.embed(frames[i], d)) for d in ds))
        else:
            none += 1
    return {'median': round(float(np.median(v)), 4) if v else None, 'min': round(min(v), 4) if v else None, 'no_face': none}


def _face_pass(u, ctx, src, dst, crf):
    """Face pass (video.swap after the join, video.i2v after LTX): every frame's face → the character's identity
    (hyperswap_1a_256 + XSeg occlusion). Identity = the master (inputs master, or image_1 when there is none) + the
    sheet's 4 head panels + image_1 (swap: the sheet's full-body front panel) when given; a second character
    (master_b [+ sheet_b, image_b]) → every face goes to the nearer one. Reads the job result src (None: the job's
    input video, video.face), writes dst (with its audio) and face.json (similarity to the master before / after)."""
    import numpy as np
    from PIL import Image
    import face
    import faceswap
    import video
    inp = u['inputs']
    if src is None:
        frames, audio, _ = video.load(ctx.input('video'))
    else:
        frames, audio, _ = video.load(fetch_result(u, src, ctx.out + '/_f.mp4'))
        os.remove(ctx.out + '/_f.mp4')
    fc = ENG.setdefault('face', face.FaceCheck())
    if 'faceswap' not in ENG:
        ENG['faceswap'] = faceswap.FaceSwap(fc)
    rgb = lambda n: np.array(Image.open(ctx.input(n)).convert('RGB'))

    def heads(name):
        sh = Image.open(ctx.input(name)).convert('RGB')
        return [np.array(sh.crop((x, y, x + 640, y + 640))) for _, x, y in SHEET_HEADS]

    imgs = [rgb('master' if 'master' in inp else 'image_1')]   # imgs[0] = the identity anchor
    if 'sheet' in inp:
        imgs += heads('sheet') + ([rgb('image_1')] if 'image_1' in inp and 'master' in inp else [])
    ident, isims = ENG['faceswap'].identity(imgs)
    two = 'master_b' in inp
    if two:
        imgs_b = [rgb('master_b')] + (heads('sheet_b') if 'sheet_b' in inp else []) + ([rgb('image_b')] if 'image_b' in inp else [])
        ident_b, isims_b = ENG['faceswap'].identity(imgs_b)
        out, st = ENG['faceswap'].video_multi(frames, [ident, ident_b], progress=lambda p: ctx.progress(0.95 * p))
        eb = fc.embed(imgs_b[0])
    else:
        out, st = ENG['faceswap'].video(frames, ident, progress=lambda p: ctx.progress(0.95 * p))
    e0 = fc.embed(imgs[0])
    stats = {'identity_sims': isims, 'missed': len(st['missed']), 'missed_frames': st['missed'],
             'seconds': st['seconds'], 'before': _face_sims(fc, frames, e0), 'after': _face_sims(fc, out, e0), 'node': NAME}
    if two:
        stats.update(identity_sims_b=isims_b, before_b=_face_sims(fc, frames, eb), after_b=_face_sims(fc, out, eb),
                     **{k: st[k] for k in ('tracks', 'conflicts', 'track_char')})
    video.save(ctx.out + '/' + dst, out, audio, crf=crf)
    with open(ctx.out + '/face.json', 'w') as fp:
        json.dump(stats, fp, indent=1)


def _src(ctx, o):
    """The source video at 24 fps, trimmed to opts start / end (seconds) → (frames, audio, info)."""
    import video
    frames, audio, info = video.load(ctx.input('video'))
    a = int(round(float(o.get('start') or 0) * video.FPS))
    b = int(round(float(o['end']) * video.FPS)) if str(o.get('end', '')).strip() else len(frames)
    if a or b < len(frames):
        frames = frames[a:b]
        audio = _cut_audio(audio, a, b)
        info = dict(info, trim=[a, b], frames=len(frames))
    if o.get('crop') == '9:16' and frames.shape[2] * 16 > frames.shape[1] * 9:   # UI "세로 9:16로 자르기": centre strip
        w = (frames.shape[1] * 9 // 16) // 2 * 2
        x0 = (frames.shape[2] - w) // 2
        frames = frames[:, :, x0:x0 + w]
        info = dict(info, crop=[x0, w])
    return frames, audio, info


def _cut_audio(audio, a, b):
    """Audio of frames a..b-1 (24 fps). The source audio starts with the first video frame (load keeps both from 0)."""
    if audio is None:
        return None
    pcm, rate, lay = audio
    return pcm[:, int(a / 24 * rate):int(b / 24 * rate)], rate, lay


def run_swap(u, ctx):
    import numpy as np
    from PIL import Image
    import video
    o, sp = u['opts'], u['spec']
    step = sp['step']
    two = 'sheet_b' in u['inputs']   # two-person swap (#11)
    if step == 'plan':             # split into shots (cuts); only shots with a visible face are made, in 4~5 s segments
        import face
        frames, audio, info = _src(ctx, o)
        # creator recipe: two passes (2/3, then latent x1.5), both with even latent sides — canvas in multiples of 96.
        # A source smaller than the training canvas (a 9:16 strip cut from a landscape video: 608x1080 → 576x1056 before)
        # is drawn grown to it, 768x1344 (boss, session #14); larger sources are scaled down as before
        fw, fh = frames.shape[2], frames.shape[1]
        s = min(768 / min(fw, fh), (768 * 1344 / (fw * fh)) ** 0.5)
        a = max(96, round(min(fw, fh) * s / 96) * 96)            # short side, then the long side from the aspect
        b = max(96, round(a * max(fw, fh) / min(fw, fh) / 96) * 96)
        while a * b > 768 * 1344:
            b -= 96
        W, H = (a, b) if fw <= fh else (b, a)
        fc = ENG.setdefault('face', face.FaceCheck())
        plan, kept = [], []
        for a, b in video.shots(frames):
            probe = np.linspace(a, b, min(5, b - a + 1)).round().astype(int)
            if b - a + 1 < MIN_SHOT or not any(fc.visible(frames[i], MIN_LUMA) for i in probe):
                kept.append([a, b])        # faceless shots (silhouette, graphics, hands) and very short shots: source as is
                continue
            plan += [(a + s0, a + e0, L) for s0, e0, L in video.plan_segments(frames[a:b + 1])]
        segs_all = plan
        if o.get('segments') is not None:   # test option: only these segment indices (plan made on the whole video)
            plan = [plan[k] for k in o['segments']]
        bounds = sorted({p[0] for p in plan} | {p[1] for p in plan})
        bnd = {f: video.resize(frames[f:f + 1], W, H)[0] for f in bounds}
        for f in bounds:
            Image.fromarray(bnd[f]).save('%s/bnd_%05d.png' % (ctx.out, f))
        # output range: the whole (trimmed) video, or only the chosen segments (test option) — stitch/up/final use it
        rng = [0, len(frames) - 1] if o.get('segments') is None else [plan[0][0], plan[-1][1]]
        with open(ctx.out + '/plan.json', 'w') as fp:
            json.dump({'canvas': [W, H], 'segments': plan, 'all_segments': segs_all, 'kept': kept, 'bounds': bounds,
                       'range': rng, 'info': info}, fp)
        me = u['idx']
        if two:
            # two people: no boundary edits (Qwen redrew close-ups as zoomed-out full-body shots and mixed the two up,
            # tests 13/14) — segments run in order, each starting from the previous segment's last frame (its only
            # guide; the first segment has none). One node at a time, but the seams are continuous by construction
            first = prev = None
            for k, (a, b, L) in enumerate(plan):
                prev = call('POST', '/jobs/%s/units' % u['job'], {'units': [{'model': 'h3', 'spec': {
                    'step': 'segment', 'k': k, 'start': a, 'end': b, 'gen': L, 'canvas': [W, H], 'chain': True,
                    'after': [me if prev is None else prev]}}]})['first']
                first = prev if first is None else first
        else:
            units = [{'model': 'qi21', 'spec': {'step': 'edit', 'frame': f, 'after': [me]}} for f in bounds]
            first = call('POST', '/jobs/%s/units' % u['job'], {'units': units})['first']
            if o.get('edit_only'):              # test option: plan + boundary edits only
                return
            eidx = {f: first + k for k, f in enumerate(bounds)}
            segs = [{'model': 'h3', 'spec': {'step': 'segment', 'k': k, 'start': a, 'end': b, 'gen': L, 'canvas': [W, H],
                                             'after': [eidx[a], eidx[b]]}} for k, (a, b, L) in enumerate(plan)]
            first = call('POST', '/jobs/%s/units' % u['job'], {'units': segs})['first']
        post = call('POST', '/jobs/%s/units' % u['job'],
                    {'units': [{'model': 'cpu', 'spec': {'step': 'stitch', 'after': list(range(first, first + len(plan)))}}]})['first']
        if o.get('face'):        # face pass on the joined video (one unit)
            post = call('POST', '/jobs/%s/units' % u['job'],
                        {'units': [{'model': 'face', 'spec': {'step': 'face', 'after': [post]}}]})['first']
        if o.get('upscale'):     # SeedVR2 in time chunks (C·D share them), then one cpu unit blends and adds the audio
            up_units(u, rng[1] - rng[0] + 1, [post])
    elif step == 'edit':           # boundary frame → character (seam ①), face check against the master
        import face
        src = fetch_result(u, 'bnd_%05d.png' % sp['frame'], ctx.out + '/_src.png')
        im = np.array(Image.open(src).convert('RGB'))
        fc = ENG.setdefault('face', face.FaceCheck())
        faces = fc.visible(im, MIN_LUMA)   # silhouettes (dark faces) excluded
        if not faces:   # faceless boundary (hands, back, silhouette): an edit would invent a person → drop this side's guide
            with open('%s/bndc_%05d.json' % (ctx.out, sp['frame']), 'w') as fp:
                json.dump({'guide': None, 'reason': 'no face in source frame', 'node': NAME}, fp)
            os.remove(src)
            return
        box = faces[0][0]
        small = box[3] - box[1] < MIN_FACE   # small faces score low in the check and redraws don't help → one draw, no check
        W, H = [int(v) for v in json.load(open(fetch_result(u, 'plan.json', ctx.out + '/_plan.json')))['canvas']]
        import pose
        P = ENG.setdefault('pose', pose.Pose())
        src_kp = P.body([im])[0]
        # image 2 and wording by the source's shot size: the full-body panel made medium shots and close-ups zoom out to
        # full body (swap 16·17, session #11) — feet seen → full-body panel + SWAP_EDIT, otherwise the master (waist up)
        # + SWAP_EDIT_PART. (A head panel for close-ups lost the source's expression and shows no clothes, session #12)
        shot = pose.shot_size(src_kp, im.shape[0])
        ref, prompt = ctx.input('image_1'), o['edit_prompt']
        if shot != 'full':
            prompt = o.get('edit_prompt_part') or prompt
            ref = ctx.input('master') if 'master' in u['inputs'] else ref
        # checks on every draw, any miss → redraw with the next seed; still missing after POSE_RETRIES → no guide on this
        # boundary (H3 follows the reference video alone there): pose (an arms-down redraw makes H3 freeze before the
        # boundary), framing (largest face height edit / source — a zoomed-out redraw), face (below min_sim after
        # run_image's own redraws — used to be kept, swap 16·17)
        e_ref = fc.embed(np.array(Image.open(ref).convert('RGB')))
        centre = lambda d: ((d[0][0] + d[0][2]) / 2, (d[0][1] + d[0][3]) / 2)
        tries, ok = [], False
        for a in range(1 + POSE_RETRIES):
            u2 = dict(u, inputs={'image_1': None, 'image_2': None},
                      opts=dict(o, prompt=prompt, check='' if small else 'image_2',
                                min_sim=o.get('min_sim', 0.5), width=W, height=H),
                      spec={'i': 0, 'seed': int(sp.get('seed', o['seed'])) + a})   # the job seed for every boundary (boss), + retry
            ctx2 = Ctx(u2, ctx.out)
            ctx2.input = lambda n: src if n == 'image_1' else ref
            run_image(u2, ctx2)
            outs = sorted(n for n in os.listdir(ctx.out) if n.startswith('00_'))
            png = os.path.join(ctx.out, next(n for n in outs if n.endswith('.png')))
            with open(png[:-4] + '.json') as fp:
                person = json.load(fp)
            if o.get('bg_prompt'):   # new place (#10): second edit puts the background image behind her (same image on
                u3 = dict(u, inputs={'image_1': None, 'image_2': u['inputs']['background']},   # every boundary)
                          opts=dict(o, prompt=o['bg_prompt'], check='', width=W, height=H),
                          spec={'i': 1, 'seed': int(sp.get('seed', o['seed'])) + a})
                ctx3 = Ctx(u3, ctx.out)
                ctx3.input = lambda n: png if n == 'image_1' else ctx.input('background')
                run_image(u3, ctx3)
                for n in outs:
                    os.remove(os.path.join(ctx.out, n))
                outs = sorted(n for n in os.listdir(ctx.out) if n.startswith('01_'))
                png = os.path.join(ctx.out, next(n for n in outs if n.endswith('.png')))
                with open(png[:-4] + '.json') as fp:   # person edit record (face check) goes into the final json
                    j = json.load(fp)
                j['person_edit'] = {k: person.get(k) for k in ('prompt', 'face_sim', 'min_sim', 'passed')}
                with open(png[:-4] + '.json', 'w') as fp:
                    json.dump(j, fp, ensure_ascii=False, indent=1)
            edit = np.array(Image.open(png).convert('RGB'))
            arm, leg = pose.limb_diff(src_kp, P.body([edit])[0])
            # framing on the replaced person: the edit's face nearest the character, against the source face at that
            # place (the largest face was the other person in a two-shot and hid a zoomed-out redraw, swap 17 retest)
            ef, fr = fc.detect(edit), None
            if ef:
                e = ef[0] if e_ref is None else max(ef, key=lambda d: fc.similarity(e_ref, fc.embed(edit, d)))
                (ex, ey), sf = centre(e), faces
                s0 = min(sf, key=lambda d: (centre(d)[0] - ex) ** 2 + (centre(d)[1] - ey) ** 2)[0]
                fr = round(float((e[0][3] - e[0][1]) / (s0[3] - s0[1])), 3)
            t = {'seed': int(sp.get('seed', o['seed'])) + a, 'arm': arm, 'leg': leg, 'framing': fr,
                 'face': person.get('passed')}
            miss = [k for k, bad in (('pose', max(arm or 0, leg or 0) > POSE_MAX),
                                     ('framing', not small and fr is not None and not FRAME_MIN <= fr <= FRAME_MAX),
                                     ('face', person.get('passed') is False)) if bad]
            if o.get('bg_prompt'):   # the background edit sometimes keeps the old place: it must be nearer the new one
                d_bg, d_src = _bg_dist(edit, Image.open(ctx.input('background')).convert('RGB')), _bg_dist(edit, Image.fromarray(im))
                t.update(bg_dist=round(d_bg, 1), src_dist=round(d_src, 1))
                miss += ['background'] if d_bg >= d_src else []
            t['miss'] = miss
            tries.append(t)
            ok = not miss
            if ok:
                break
            for n in outs:   # a missed draw stays as bndx_<frame>_<seed>.png (to see what the checks turned down)
                if n.endswith('.png'):
                    os.replace(os.path.join(ctx.out, n), '%s/bndx_%05d_%d.png' % (ctx.out, sp['frame'], t['seed']))
                else:
                    os.remove(os.path.join(ctx.out, n))
        res = {'shot': shot, 'max_deg': POSE_MAX, 'framing': [FRAME_MIN, FRAME_MAX], 'tries': tries}
        if ok:
            for n in outs:    # 00_<seed> (01_<seed> with a background) .png/json → bndc_<frame>.*
                os.replace(os.path.join(ctx.out, n), os.path.join(ctx.out, 'bndc_%05d%s' % (sp['frame'], os.path.splitext(n)[1])))
            jp = '%s/bndc_%05d.json' % (ctx.out, sp['frame'])
            with open(jp) as fp:
                j = json.load(fp)
            j['pose'] = res
            with open(jp, 'w') as fp:
                json.dump(j, fp, ensure_ascii=False, indent=1)
        else:
            with open('%s/bndc_%05d.json' % (ctx.out, sp['frame']), 'w') as fp:
                json.dump({'guide': None, 'reason': 'edit keeps missing: ' + ', '.join(tries[-1]['miss']), 'pose': res,
                           'node': NAME}, fp)
        for n in ('_src.png', '_plan.json'):
            os.remove(os.path.join(ctx.out, n))
    elif step == 'segment':        # H3 Ref2VA: source segment + master + start/end guides
        import h3_pipe
        frames, _, _ = _src(ctx, o)
        W, H = sp['canvas']
        clip = frames[sp['start']:sp['end'] + 1]
        if len(clip) < sp['gen']:   # truncated segment (shot end): repeat the last reference frame up to gen frames
            clip = np.concatenate([clip, np.repeat(clip[-1:], sp['gen'] - len(clip), 0)])
        ref = video.resize(clip, W, H)
        guides = []
        if sp.get('chain'):   # chained segments: the previous segment's last frame is the first guide, no end guide
            if sp['k'] > 0:
                prev, _, _ = video.load(fetch_result(u, 'seg_%02d.mp4' % (sp['k'] - 1), ctx.out + '/_p.mp4'))
                os.remove(ctx.out + '/_p.mp4')
                guides = [prev[-1], None]
            else:
                guides = [None, None]
        for k, f in enumerate(() if sp.get('chain') else (sp['start'], sp['end'])):
            try:
                guides.append(np.array(Image.open(fetch_result(u, 'bndc_%05d.png' % f, ctx.out + '/_g%d.png' % k)).convert('RGB')))
                os.remove(ctx.out + '/_g%d.png' % k)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                guides.append(None)   # unedited boundary: no guide
        master = np.array(Image.open(ctx.input('image_1')).convert('RGB'))   # master, or the sheet's full-body front panel
        pic = np.array(Image.open(ctx.input('sheet')).convert('RGB')) if 'sheet' in u['inputs'] else master
        if two:   # <Picture 1> = character A's sheet, <Picture 2> = B's (SWAP2_H3)
            pic = [pic, np.array(Image.open(ctx.input('sheet_b')).convert('RGB'))]
        ENG['h3'].dit.set_attn_q8(bool(o.get('q8')))
        # Singularity author's recipe as video.r2v: euler·beta 12 = 8 at 2/3 size + 4 after x1.5
        import h3
        sig = h3.beta_sigmas(12)
        out = ENG['h3'].generate(ref_video=ref, ref_image=pic, first=guides[0], last=guides[1], gen_frames=sp['gen'],
                                 prompt=o['h3_prompt'], seed=int(sp.get('seed', o['seed'])),   # redo: new seed
                                 progress=lambda p: ctx.progress(0.95 * p),
                                 last_idx=sp['end'] - sp['start'],   # truncated last segment: end guide at source end
                                 ref_size=o.get('ref_size', 'match'), sampler='euler', sigmas=sig[:9],
                                 canvas=(W * 2 // 3, H * 2 // 3), up=(W, H, sig[8:].tolist()))
        out = out[:sp['end'] - sp['start'] + 1]
        sims = h3_pipe.face_check(ENG, out, master)
        if two:   # per character, against its master
            rgbm = lambda n: np.array(Image.open(ctx.input(n)).convert('RGB'))
            sims = {'a': h3_pipe.face_check(ENG, out, rgbm('master')), 'b': h3_pipe.face_check(ENG, out, rgbm('master_b'))}
        video.save('%s/seg_%02d.mp4' % (ctx.out, sp['k']), out, crf=12)
        with open('%s/seg_%02d.json' % (ctx.out, sp['k']), 'w') as fp:
            json.dump({'start': sp['start'], 'end': sp['end'], 'gen': sp['gen'], 'face_sim': sims, 'node': NAME}, fp)
    elif step == 'stitch':         # paste segments in place over the source (canvas size), add source audio (drop generated)
        plan = json.load(open(fetch_result(u, 'plan.json', ctx.out + '/_plan.json')))
        parts = []
        for k, (a, b, L) in enumerate(plan['segments']):
            fr, _, _ = video.load(fetch_result(u, 'seg_%02d.mp4' % k, ctx.out + '/_s.mp4'))
            parts.append((a, fr))
        for n in ('_s.mp4', '_plan.json'):
            if os.path.exists(ctx.out + '/' + n):
                os.remove(ctx.out + '/' + n)
        frames, audio, _ = _src(ctx, o)
        a, b = plan.get('range') or [0, len(frames) - 1]   # output range (test option segments: only those)
        last = not (o.get('face') or o.get('upscale'))   # the join is the result, or an intermediate for the next steps
        video.save(ctx.out + ('/swap.mp4' if last else '/join.mp4'),
                   video.stitch(video.resize(frames[a:b + 1], *plan['canvas']), [(s0 - a, fr) for s0, fr in parts]),
                   _cut_audio(audio, a, b + 1), crf=18 if last else 12)
    elif step == 'face':           # face pass on the join (hyperswap_1a_256)
        _face_pass(u, ctx, 'join.mp4', 'face.mp4' if o.get('upscale') else 'swap.mp4', 12 if o.get('upscale') else 18)
    elif step == 'up':             # SeedVR2 7B on one time chunk of the joined (face-passed) video, 1080x1920
        frames, _, _ = video.load(fetch_result(u, 'face.mp4' if o.get('face') else 'join.mp4', ctx.out + '/_v.mp4'))
        os.remove(ctx.out + '/_v.mp4')
        out = ENG['sv2'].upscale(frames[sp['start']:sp['end'] + 1], int(o['seed']), None, 'lab',
                                 progress=lambda p: ctx.progress(0.95 * p))
        video.save('%s/up_%02d.mp4' % (ctx.out, sp['k']), out, crf=10)
    elif step == 'final':          # chunks → one video (crossfaded), source audio, CRF 18
        out = up_join(u, ctx, sp['chunks'])
        plan = json.load(open(fetch_result(u, 'plan.json', ctx.out + '/_plan.json')))
        os.remove(ctx.out + '/_plan.json')
        a, b = plan.get('range') or [0, len(out) - 1]
        t0 = int(round(float(o.get('start') or 0) * video.FPS))   # trim offset (audio only, no frame decode)
        video.save(ctx.out + '/swap.mp4', out, _cut_audio(video.load_audio(ctx.input('video')), t0 + a, t0 + b + 1))
        up_drop(u, sp['chunks'])


def run_r2v(u, ctx):
    """video.r2v: H3 makes one shot (with sound) from the reference images (image_1..N) and the prompt. If first_<shot>
    exists, that scene image is the shot's first-frame guide; last_<shot> the same at the shot's last frame. key_<shot>_<j>
    at frame opts.keys[shot][j] (a multi-shot's shot first frames, the official multiframe template): each is a guide at
    that frame and also a reference picture after the sheets (<Picture N+1> …). The concat unit joins the shots in order.
    Stages (opts.stage): preview = pass 1 only → shot_KK_pre.mp4 (preview at pass-1 size) + shot_KK_pre.lat (pass-1
    latent and the settings pass 2 needs), refine = pass 2 only from the picked shot's .lat (lat_<k>) → shot_KK.mp4 (same
    as a single run), full = all in one go."""
    import numpy as np
    import video
    from PIL import Image
    o, sp = u['opts'], u['spec']
    stage = o.get('stage', 'full')
    tail = '_pre' if stage == 'preview' else ''
    if sp.get('step') == 'concat':
        frames, pcm, ar = [], [], None
        for k in range(sp['n']):
            fr, au, _ = video.load(fetch_result(u, 'shot_%02d%s.mp4' % (k, tail), ctx.out + '/_s.mp4'))
            frames.append(fr)
            if au is not None:
                pcm.append(au[0][:, :round(len(fr) / video.FPS * au[1])])   # shot length only (aac tail padding dropped)
                ar = au[1:]
        os.remove(ctx.out + '/_s.mp4')
        video.save('%s/r2v%s.mp4' % (ctx.out, tail), np.concatenate(frames),
                   (np.concatenate(pcm, axis=1),) + ar if ar else None)
        return
    k = sp['shot']
    names = sorted((n for n in u['inputs'] if n.startswith('image_')), key=lambda n: int(n.split('_')[1]))
    imgs = [np.array(Image.open(ctx.input(n)).convert('RGB')) for n in names]
    fk, lk = 'first_%d' % k, 'last_%d' % k
    first, last = (np.array(Image.open(ctx.input(n)).convert('RGB')) if n in u['inputs'] else None for n in (fk, lk))
    knames = sorted((n for n in u['inputs'] if n.startswith('key_%d_' % k)), key=lambda n: int(n.rsplit('_', 1)[1]))
    kimgs = [np.array(Image.open(ctx.input(n)).convert('RGB')) for n in knames]
    import h3
    h = ENG['h3']
    if stage == 'refine':   # all settings from the .lat. Checks the references / start frame are the same files as in pass 1
        from safetensors import safe_open
        with safe_open(ctx.input('lat_%d' % k), 'pt') as f:
            m = json.loads(f.metadata()['shutter'])
            init1 = (f.get_tensor('video'), f.get_tensor('audio'))
        if ([u['inputs'][n] for n in names] != m['refs'] or u['inputs'].get(fk) != m['first']
                or u['inputs'].get(lk) != m.get('last') or u['model'] != m['dit']
                or [u['inputs'][n] for n in knames] != [s for _, s in m.get('keys', [])]):
            raise ValueError('shot %d: reference / first- or last-frame / key images or model differ from the preview' % k)
        h.dit.set_attn_q8(m['q8'])
        out = h.generate(ref_video=None, ref_image=imgs + kimgs, first=first, last=last, gen_frames=m['frames'], prompt=m['prompt'],
                         keys=[(f, g) for (f, _), g in zip(m.get('keys', []), kimgs)],
                         seed=m['seed'], progress=lambda p: ctx.progress(0.95 * p), canvas=tuple(m['canvas']), audio=True,
                         up=tuple(m['up']), sampler=m['sampler'], ref_size=m['ref_size'], stage='second', init1=init1)
        video.save('%s/shot_%02d.mp4' % (ctx.out, k), out, (h.last_audio.numpy(), 32000, 'stereo'), crf=12)
        return
    W, H = int(o.get('width', 768)), int(o.get('height', 1344))
    # default = Singularity creator recipe: euler·beta 12 steps split at split (8); pass 1 at 1/up (1.5) of width x height,
    # grown by the 3D latent upscaler, pass 2 with the same seed. Reference images max (short side up to 2048).
    # up 1 = single pass (no stages). sched simple + sampler res_multistep = the turbo LoRA's own one-pass setting
    # (ComfyUI H3 template, 4 steps) — the calm-shot comparison of drama test 01 (session #7).
    sched = h3.simple_sigmas if o.get('sched') == 'simple' else h3.beta_sigmas
    sig, n, x = sched(int(o.get('steps', 12))), int(o.get('split', 8)), float(o.get('up', 1.5))
    up = None
    if x > 1:
        up, sig = (W, H, sig[n:].tolist()), sig[:n + 1]
        W, H = max(32, round(W / x / 32) * 32), max(32, round(H / x / 32) * 32)
    q8 =bool(o.get('q8', True))   # Sage-style attention (lossy, ComfyUI default family) — default on, false = lossless
    h.dit.set_attn_q8(q8)
    m = dict(frames=int(o.get('frames', 124)), prompt=o['shots'][k], seed=int(o['seed']) + k, canvas=[W, H], up=up,
             sampler=o.get('sampler', 'euler'), ref_size=o.get('ref_size', 'max'), q8=q8)
    kf = [int(f) for f in ((o.get('keys') or [])[k] if k < len(o.get('keys') or []) else [])]
    out = h.generate(ref_video=None, ref_image=imgs + kimgs, first=first, last=last, gen_frames=m['frames'], prompt=m['prompt'],
                     keys=list(zip(kf, kimgs)),
                     seed=m['seed'], progress=lambda p: ctx.progress(0.95 * p), canvas=(W, H), audio=True, up=up,
                     sampler=m['sampler'], sigmas=sig, ref_size=m['ref_size'], stage='first' if stage == 'preview' else 'full')
    # preview = viewing only (community default CRF 18, as ComfyUI SaveVideo); full output may feed the upscaler: CRF 12
    video.save('%s/shot_%02d%s.mp4' % (ctx.out, k, tail), out, (h.last_audio.numpy(), 32000, 'stereo'),
               crf=18 if stage == 'preview' else 12)
    if stage == 'preview':
        from safetensors.torch import save_file
        m.update(refs=[u['inputs'][n] for n in names], first=u['inputs'].get(fk), last=u['inputs'].get(lk), dit=u['model'],
                 keys=[[f, u['inputs'][n]] for f, n in zip(kf, knames)])
        save_file({'video': h.last_latent[0].cpu().contiguous(), 'audio': h.last_latent[1].cpu().contiguous()},
                  '%s/shot_%02d_pre.lat' % (ctx.out, k), metadata={'shutter': json.dumps(m, ensure_ascii=False)})


def run_concat(u, ctx):
    """video.concat: video_0..N joined in order → seq.mp4. Every clip at the first one's size (center crop + resize where
    it differs, e.g. an upscaled cut), sound as 48 kHz stereo (LTX makes 48 kHz, H3 / pi-H3 32 kHz: resampled linearly;
    a clip without sound adds silence)."""
    import numpy as np
    import video
    o, n = u['opts'], u['spec']['n']
    frames, pcm, size = [], [], None
    for k in range(n):
        fr, au, _ = video.load(ctx.input('video_%d' % k))
        size = size or ((int(o['width']), int(o['height'])) if o.get('width') and o.get('height') else (fr.shape[2], fr.shape[1]))
        if (fr.shape[2], fr.shape[1]) != size:
            fr = video.resize(fr, *size)
        m = int(round(len(fr) / video.FPS * 48000))
        a = np.zeros((2, m), np.float32)
        if au is not None:
            p, rate = au[0].astype(np.float32), au[1]
            p = np.repeat(p, 2, 0) if p.shape[0] == 1 else p[:2]
            x = np.arange(m) * (rate / 48000)
            a = np.stack([np.interp(x, np.arange(p.shape[1]), c, right=0) for c in p]).astype(np.float32)
        frames.append(fr)
        pcm.append(a)
        ctx.progress(0.9 * (k + 1) / n)
    video.save(ctx.out + '/seq.mp4', np.concatenate(frames), (np.concatenate(pcm, axis=1), 48000, 'stereo'), crf=12)


def run_edit(u, ctx):
    """video.edit: a scene's cuts → part_NN.mp4 + edit.json (engine/py/edit.py render; screen text by overlay.py)."""
    import edit
    o = u['opts']
    n = sum(k.startswith('video_') for k in u['inputs'])
    speech = list(o.get('speech') or []) + [True] * n
    multi = list(o.get('multi') or []) + [False] * n
    src = [{'path': ctx.input('video_%d' % k), 'speech': bool(speech[k]), 'multi': bool(multi[k])} for k in range(n)]
    cuts = [dict(c, v=int(c['v']), s=None if c.get('s') is None else int(c['s']), a=float(c['a']), sa=float(c.get('sa') or 0),
                 n=int(c['n']), part=int(c.get('part') or 0)) for c in o['cuts']]
    size = (int(o['width']), int(o['height'])) if o.get('width') and o.get('height') else None
    draw = None
    if any(c.get('text') for c in cuts):
        import overlay
        draw = overlay.Overlay(o.get('font')).draw
    r = edit.render(cuts, src, ctx.out, size=size, target=float(o.get('target') or -16), overlay=draw, progress=ctx.progress)
    with open(ctx.out + '/edit.json', 'w') as f:
        json.dump(r, f, ensure_ascii=False)


def run_join(u, ctx):
    """video.join: scene parts → film.mp4 (+ ep_NN.mp4 with breaks) + join.json, by stream copy (engine/py/edit.py)."""
    import edit
    o = u['opts']
    n = sum(k.startswith('part_') for k in u['inputs'])
    paths = [ctx.input('part_%d' % k) for k in range(n)]
    tmp = tempfile.mkdtemp(prefix='join-')
    try:
        r = edit.join(paths, ctx.out, tmp, breaks=[int(b) for b in o.get('breaks') or []], flash=float(o.get('flash') or 0),
                      progress=ctx.progress)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(ctx.out + '/join.json', 'w') as f:
        json.dump(r, f, ensure_ascii=False)


def run_voice(u, ctx):
    """audio.voice: the clip's dialogue split from the rest, each spoken turn converted to its speaker's reference voice
    (same length), mixed back → voice.mp4 (video stream copied) + voice.json (turns, speaker similarity before / after)."""
    import edit, voice
    o = u['opts']
    if 'vc' not in ENG:   # small (separation + Seed-VC + CAM++): stays next to whatever is loaded, as the face pass
        ENG['vc'] = voice.Voice()
    src = ctx.input('video')
    pcm = edit.audio48(src)
    if pcm is None:
        raise ValueError('the clip has no sound')
    turns = [int(t) for t in o['turns']]
    refs = {t: voice.load_wav(ctx.input('voice_%d' % t)) for t in set(turns) if t >= 0}   # speaker → (wav, sr)
    out, rep = ENG['vc'].clip(pcm, edit.RATE, turns, refs, progress=lambda p: ctx.progress(0.9 * p))
    voice.remux(src, out, edit.RATE, ctx.out + '/voice.mp4')
    with open(ctx.out + '/voice.json', 'w') as f:
        json.dump(rep, f, ensure_ascii=False)


def run_tts(u, ctx):
    """audio.tts: one reference voice candidate (Qwen3-TTS VoiceDesign) → voice_<k>.wav."""
    import voice
    o, k = u['opts'], int(u['spec']['k'])
    if 'tts' not in ENG:
        ENG['tts'] = voice.VoiceDesign()
    wav, sr = ENG['tts'].generate(o['text'], o['instruct'], seed=int(o['seed']) + k)
    voice.save_wav('%s/voice_%d.wav' % (ctx.out, k), wav, sr)


def run_upscale(u, ctx):
    """video.upscale (pass 3): upscale the video with one SeedVR2 7B step (default 1080x1920) → up.mp4, the source audio
    attached unchanged. As the swap's upscale (boss, session #13): the plan unit splits the video into 5 s chunks that
    C·D share (up_units) and a cpu unit joins them; a video of one chunk is a single unit (spec without k)."""
    import video
    o, sp = u['opts'], u['spec']
    step = sp.get('step', 'up')   # units queued before the chunks existed have no step: the whole video
    if step == 'plan':
        frames, _, _ = video.load(ctx.input('video'))   # the frame count as every chunk unit decodes it
        up_units(u, len(frames), [u['idx']], even=True)
        return
    if step == 'final':
        video.save(ctx.out + '/up.mp4', up_join(u, ctx, sp['chunks']), video.load_audio(ctx.input('video')), crf=12)
        up_drop(u, sp['chunks'])
        return
    frames, audio, _ = video.load(ctx.input('video'))
    if 'k' in sp:
        frames = frames[sp['start']:sp['end'] + 1]
    size = (int(o['width']), int(o['height'])) if o.get('width') and o.get('height') else None
    out = ENG['sv2'].upscale(frames, int(o['seed']), size, o.get('color', 'lab'), progress=lambda p: ctx.progress(0.95 * p))
    if 'k' in sp:
        video.save('%s/up_%02d.mp4' % (ctx.out, sp['k']), out, crf=10)
    else:
        video.save(ctx.out + '/up.mp4', out, audio, crf=12)


def run_i2v(u, ctx):
    """video.i2v: LTX-2.5 makes one shot with sound from a first frame (inputs image) and a prompt — the LTX-2.5 I2V
    template (stage 1 at W/2 x H/2, latent x2 upsampler, stage 2 at W x H; engine/py/ltx_pipe.py) → i2v.mp4 + i2v.json
    (settings, times). With opts face the LTX shot is ltx.mp4 and a face unit after it writes i2v.mp4 + face.json
    (_face_pass)."""
    import numpy as np
    import video
    from PIL import Image, ImageOps
    o = u['opts']
    if u['spec'].get('step') == 'face':
        _face_pass(u, ctx, 'ltx.mp4', 'i2v.mp4', 12)
        return
    img = np.array(ImageOps.exif_transpose(Image.open(ctx.input('image'))).convert('RGB'))   # LoadImage
    p = ENG['ltx']
    q8 = bool(o.get('q8', False))   # lossless by default (speed is already past ComfyUI without it)
    p.dit.set_attn_q8(q8)
    W, H, n = int(o['width']), int(o['height']), int(o['frames'])
    frames, wav = p.generate(img, o['prompt'], W, H, n, int(o['seed']), int(o['seed2']),
                             progress=lambda f: ctx.progress(0.95 * f))
    t0 = time.perf_counter()
    video.save(ctx.out + ('/ltx.mp4' if o.get('face') else '/i2v.mp4'), frames, (wav, 48000, 'stereo'), crf=12)
    p.times['save'] = time.perf_counter() - t0
    with open(ctx.out + '/i2v.json', 'w') as fp:
        json.dump(dict(p.info, q8=q8, node=NAME,
                       times={k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.times.items()}),
                  fp, indent=1)

def run_pih3(u, ctx):
    """video.pih3: pi-H3 (FL2VA + pi-H3 DWD adapter, engine/py/pih3_pipe.py) — first frame (inputs image) and/or last
    frame (inputs last), or text only → pih3.mp4 (32 kHz audio) + pih3.json (settings, times). With opts face the shot is pih3_raw.mp4 and a face unit
    after it writes pih3.mp4 + face.json (_face_pass)."""
    import numpy as np
    import video
    from PIL import Image, ImageOps
    o = u['opts']
    if u['spec'].get('step') == 'face':
        _face_pass(u, ctx, 'pih3_raw.mp4', 'pih3.mp4', 12)
        return
    img, last = (np.array(ImageOps.exif_transpose(Image.open(ctx.input(k))).convert('RGB'))
                 if k in u['inputs'] else None for k in ('image', 'last'))
    p = ENG['pih3']
    frames, wav = p.generate(img, o['prompt'], int(o['width']), int(o['height']), int(o['frames']), int(o['seed']),
                             steps=int(o['steps']), q8=bool(o.get('q8', True)), progress=lambda f: ctx.progress(0.95 * f),
                             last=last, vsa=o.get('vsa') not in (None, '', False, 0, '0', 'false', 'off'))
    t0 = time.perf_counter()
    video.save(ctx.out + ('/pih3_raw.mp4' if o.get('face') else '/pih3.mp4'), frames, (wav, 32000, 'stereo'), crf=12)
    p.times['save'] = time.perf_counter() - t0
    with open(ctx.out + '/pih3.json', 'w') as fp:
        json.dump(dict(p.info, node=NAME, times={k: (round(v, 2) if isinstance(v, float) else
                                                     [round(x, 2) for x in v] if isinstance(v, list) else v)
                                                 for k, v in p.times.items()}), fp, indent=1)


def run_face(u, ctx):
    """video.face: the face pass alone on an uploaded video (inputs video) → face.mp4 + face.json (_face_pass)."""
    _face_pass(u, ctx, None, 'face.mp4', 12)


RUN = {
    'test.fake': run_fake,
    'video.probe': run_probe,
    'image.t2i': run_image,
    'image.edit': run_image,
    'char.ref': run_image, 'char.from_photo': run_image, 'char.derive': run_image,
    'place.anchor': run_image, 'place.derive': run_image, 'prop.ref': run_image, 'prop.derive': run_image,
    'scene.compose': run_image, 'scene.plate': run_image,
    'scene.zoom': run_image,
    'char.sheet': run_sheet,
    'video.swap': run_swap,
    'video.r2v': run_r2v,
    'video.upscale': run_upscale,
    'video.i2v': run_i2v,
    'video.pih3': run_pih3,
    'video.face': run_face,
    'video.concat': run_concat,
    'video.edit': run_edit,
    'video.join': run_join,
    'audio.voice': run_voice,
    'audio.tts': run_tts,
}


def gpu_idle():
    """Wait until all work queued on the GPU has finished (torch default stream + engine streams = the whole device)."""
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return torch


def drop(*keys):
    """Unload models: after the GPU work is done, free the engine buffers, streams and handles (close) and return the
    torch cache as well."""
    keys = [k for k in keys if k in ENG]
    if not keys:
        return
    gpu_idle()
    for k in keys:   # not bound to a name (a leftover reference would keep empty_cache from freeing its torch tensors)
        getattr(ENG.pop(k), 'close', lambda: None)()
    gc.collect()
    torch = gpu_idle()
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        print('unload %s: %.1f GiB free' % ('+'.join(keys), torch.cuda.mem_get_info()[0] / 2**30), flush=True)


def load(model):
    """Keep one model loaded on the node. Within the qi21 family only the DiT changes. To switch, unload (drop) first,
    then load."""
    if model in ('face', 'vc', 'tts'):      # face pass, voice conversion, TTS: small, load themselves on use next to
        return                              # whatever is loaded
    if model == 'sv2':                     # upscale: unload every other model, then SeedVR2 7B (DiT + VAE)
        drop('dit', 'te', 'vae', 'h3', 'face', 'faceswap', 'ltx', 'pih3')
        if 'sv2' not in ENG:
            import seedvr2
            ENG['sv2'] = seedvr2.SeedVR2()
        return
    drop('sv2')
    if model == 'ltx25':                    # LTX-2.5 (TE + DiT + video/audio VAE + upsampler, ~40 GB)
        drop('dit', 'te', 'vae', 'h3', 'face', 'faceswap', 'pih3')
        if 'ltx' not in ENG:
            import ltx_pipe
            ENG['ltx'] = ltx_pipe.Ltx()
        return
    drop('ltx')
    if model.startswith('pih3_'):           # pi-H3 (drama H3 path): one PiH3; switching 2 <-> 4 steps reloads its DiT
        drop('dit', 'te', 'vae', 'h3', 'face', 'faceswap')
        steps = int(model.split('_')[1])
        if 'pih3' not in ENG:
            import pih3_pipe
            ENG['pih3'] = pih3_pipe.PiH3(steps)
        else:
            ENG['pih3'].load(steps)
        return
    drop('pih3')
    if model == 'h3':                       # video: unload the image models, load H3 (#2: never two models at once)
        import h3_pipe
        drop('dit', 'te', 'vae', 'face', 'faceswap')   # H3 holds its own face checker (the worker rebuilds its own on use)
        if 'h3' not in ENG:
            ENG['h3'] = h3_pipe.H3()        # Singularity v1.3 + turbo 4-step LoRA merged at load
        return
    if not model.startswith('qi21'):
        return
    drop('h3')
    import qi21
    if 'te' not in ENG:
        ENG['te'], ENG['vae'] = qi21.Te(), qi21.Vae()
    drop('dit')
    ENG['dit'] = qi21.Dit(turbo=model.endswith('turbo'))


def upload(u, path):
    name = os.path.basename(path)
    tmp = u['spec'].get('step') == 'up' and name.startswith('up_')   # upscale chunks: the final step deletes them
    with open(path, 'rb') as f:
        req = urllib.request.Request('%s/results/%s/%s%s' % (SERVER, u['job'], name, '?tmp=1' if tmp else ''),
                                     data=f, method='PUT',
                                     headers={'Content-Length': str(os.path.getsize(path))})
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()


SHARED = os.environ.get('SHUTTER_SHARED_CACHE') == '1'   # C: the server's cache folder, the server cleans it
CACHE = {'at': 0, 'bytes': None, 'gen': None}


def cache_round():
    """Between units, at most every 10 min: report the cache size, and on a node with its own cache (D) apply the node
    cache rules (§8) — unused for cache_days, then the oldest until under cache_gb, everything on "노드 캐시 비우기".
    D's cache is a copy of C's, so whatever goes is fetched again when needed."""
    now = time.time()
    if now - CACHE['at'] < 600:
        return
    CACHE['at'] = now
    files = []
    for n in os.listdir(ASSETS) if os.path.isdir(ASSETS) else []:
        try:
            st = os.stat('%s/%s' % (ASSETS, n))
        except OSError:
            continue
        files.append((st.st_mtime, st.st_size, n))
    total = sum(f[1] for f in files)
    if not SHARED:
        try:
            c = call('GET', '/config')
        except Exception:
            c = None
        if c:
            clear = CACHE['gen'] is not None and c['clear_gen'] != CACHE['gen']
            CACHE['gen'] = c['clear_gen']
            for mtime, size, n in sorted(files):
                if clear or mtime < now - c['cache_days'] * 86400 or total > c['cache_gb'] * 2 ** 30:
                    os.remove('%s/%s' % (ASSETS, n))
                    total -= size
    CACHE['bytes'] = total


def blocking_sync():
    """Sleep instead of spinning a CPU core while waiting on the GPU. Same results, only some tens of µs later per wait.
    Sets the primary context flags before torch or the engine first uses CUDA."""
    try:
        cu = ctypes.CDLL('libcuda.so.1')
        dev = ctypes.c_int()
        r = cu.cuInit(0) or cu.cuDeviceGet(ctypes.byref(dev), 0) or cu.cuDevicePrimaryCtxSetFlags_v2(dev, 4)  # BLOCKING_SYNC
        print('cuda blocking sync: %s' % ('on' if r == 0 else 'error %d' % r), flush=True)
    except OSError as e:
        print('cuda blocking sync: %s' % e, flush=True)


def bye(unit=None):
    """Tell the server we are stopping: the held unit goes back to the queue without counting a try, and this worker is
    removed from the 'idle with a model loaded' workers."""
    try:
        call('POST', '/bye', {'worker': NAME, 'unit': unit})
    except Exception as e:
        print('bye: %s' % e, flush=True)   # if the server can't be told, the unit comes back via the timeout (STALE)


def main():
    for sig in (signal.SIGTERM, signal.SIGINT):   # container PID 1: without a handler SIGTERM is ignored and SIGKILL follows
        signal.signal(sig, lambda *a: STOP.set())
    loaded = None
    blocking_sync()
    print('shutter worker %s → %s' % (NAME, SERVER), flush=True)
    while not STOP.is_set():
        try:
            cache_round()
        except Exception as e:
            print('cache: %s' % e, flush=True)
        try:
            u = call('POST', '/next', {'worker': NAME, 'loaded': loaded, 'wait': 20, 'cache': CACHE['bytes']},
                     timeout=40)['unit']
        except Exception as e:
            print('next: %s' % e, flush=True)
            STOP.wait(3)
            continue
        if not u:
            continue
        if STOP.is_set():
            bye(u['id'])
            break
        print('run %s %s model=%s attempt=%d' % (u['id'], u['type'], u['model'], u['total']), flush=True)
        out = tempfile.mkdtemp(prefix='unit-')
        ctx, stop = Ctx(u, out), threading.Event()
        threading.Thread(target=ctx.heartbeat, args=(stop,), daemon=True).start()   # during the model load too
        try:
            if u['model'] not in ('cpu', loaded):     # cpu units don't change the loaded model
                load(u['model'])
                loaded = u['model']
            RUN[u['type']](u, ctx)
            stop.set()
            for name in sorted(os.listdir(out)):
                upload(u, os.path.join(out, name))
            call('POST', '/units/%s/done' % u['id'], {'worker': NAME})
        except Stop:
            print('stop: %s back to the queue' % u['id'], flush=True)
            bye(u['id'])
        except Abort:
            print('abort %s (server took it back)' % u['id'], flush=True)
        except Exception as e:
            err = '%s: %s' % (type(e).__name__, e)
            print('fail %s %s\n%s' % (u['id'], err, traceback.format_exc()), flush=True)
            try:
                call('POST', '/units/%s/fail' % u['id'], {'worker': NAME, 'error': err})
            except Exception:
                pass   # if the server can't be told, the unit comes back via the timeout
        finally:
            stop.set()
            shutil.rmtree(out, ignore_errors=True)
    # stop: wait for the GPU work to finish, unload every model (engine buffers, streams, handles, torch cache), exit cleanly
    bye()
    drop(*list(ENG))
    print('worker stopped', flush=True)


if __name__ == '__main__':
    main()
