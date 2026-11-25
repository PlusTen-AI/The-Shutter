#!/usr/bin/env python3
"""shutter server (C, container): job queue · dispatch · status API. Standard library only.

The UI (Mac) and, later, the Hermes skill call the same API (design §2).
  POST /jobs {type, inputs:{name: sha256}, options:{}, project}  → split the job into units and queue them
  GET  /jobs[?limit=N&open=1&project=P&unrecorded=1&ids=a,b]   GET /jobs/<id>   POST /jobs/<id>/retry|cancel|first|close|recorded
  POST /jobs/<id>/units {units:[{model, spec}]}   (append units; spec.after = unit indices that must finish first)
  POST /jobs/<id>/redo {idx}   (one unit again with a new seed, and the units after it)   POST /prompt {type, options}
  GET  /files?unsynced=1   POST /files/ack {job, name, sha}   (the Mac takes each result as soon as its unit ends, §8)
  GET|POST /config {cache_days, cache_gb}   POST /cache/clear   (node cache rules, §8)
  GET  /queue  (ComfyUI /queue shape, §5)   GET /types   GET /workers
  HEAD/PUT/GET /assets/<sha256>   (content-addressed. The same file is not sent twice)
  GET  /results/<job>/<name>
  for workers: POST /next  POST /units/<id>/progress|done|fail  POST /bye   PUT|DELETE /results/<job>/<name>[?tmp=1]
"""
import hashlib, json, os, re, secrets, shutil, signal, sqlite3, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import templates  # noqa: E402

WORK = os.environ.get('SHUTTER_WORK', '/work')
PORT = int(os.environ.get('SHUTTER_PORT', '8300'))
MAX_TRIES = int(os.environ.get('SHUTTER_MAX_TRIES', '3'))   # automatic retry limit per unit (#16)
STALE = int(os.environ.get('SHUTTER_STALE', '120'))         # no progress report for this long (s) counts as one failure
ASSETS, RESULTS, DB = WORK + '/cache/assets', WORK + '/results', WORK + '/state/queue.db'
CONFIG = WORK + '/state/config.json'
CHUNK = 1 << 20
RESUME = 3 * 86400   # a failed multi-unit job waits this long for "이어서 하기", then it is closed (boss, session #10)
CONFIG_DEFAULT = {'cache_days': 3, 'cache_gb': 20, 'clear_gen': 0}   # node cache: unused files older than this, size cap


# ---- job types: split one job into mutually independent units → [(model, spec)] (§3) ----
def split_fake(inputs, opts):
    n = max(1, min(int(opts.get('units', 2)), 64))
    return [(opts.get('model', 'fake'), {'i': i, 'n': n}) for i in range(n)]

def split_image(inputs, opts, kind=None):
    """Image job: one image = one unit. The seed is chosen and recorded here (a retry gives the same image).
    Character/place jobs fill the prompt and default settings from the templates.JOBS templates and go to the same run."""
    if kind:
        fill, defaults = templates.JOBS[kind]
        merged = dict(defaults)
        merged.update(opts)
        if kind == 'char.derive' and str(opts.get('kind', '')).startswith('back'):
            merged.pop('check', None)          # no face check for back views
        if merged.get('check') and 'min_sim' not in opts:
            merged['min_sim'] = templates.min_sim(kind, merged)
        merged['prompt'] = fill(merged)
        opts.clear()
        opts.update(merged)
    if not str(opts.get('prompt', '')).strip():
        raise ValueError('prompt required')
    n = max(1, min(int(opts.get('count', 1)), 16))
    seed = int(opts['seed']) if str(opts.get('seed', '')).strip() else secrets.randbelow(2 ** 48)
    opts['seed'] = seed                        # recorded so the UI shows it and a re-run can reuse it
    model = 'qi21-turbo' if opts.get('turbo') in (True, 'true', '1', 1) else 'qi21'
    return [(model, {'i': i, 'n': n, 'seed': seed + i}) for i in range(n)]


TYPES = {
    'test.fake': split_fake,
    'image.t2i': split_image,    # text → image (master, place anchor)
    'image.edit': split_image,   # image + instruction → image (inputs: image_1..image_N)
    **{k: (lambda i, o, k=k: split_image(i, o, k)) for k in templates.JOBS},   # char.ref, char.from_photo, char.derive, place.*
    'char.sheet': lambda i, o: split_sheet(i, o),
    'video.swap': lambda i, o: split_swap(i, o),
    'video.r2v': lambda i, o: split_r2v(i, o),
    'video.upscale': lambda i, o: split_upscale(i, o),
    'video.i2v': lambda i, o: split_i2v(i, o),
    'video.pih3': lambda i, o: split_pih3(i, o),
    'video.face': lambda i, o: split_face(i, o),
    'video.probe': lambda i, o: split_probe(i, o),
    'video.concat': lambda i, o: split_concat(i, o),
    'video.edit': lambda i, o: split_edit(i, o),
    'video.join': lambda i, o: split_join(i, o),
    'audio.voice': lambda i, o: split_voice(i, o),
    'audio.tts': lambda i, o: split_tts(i, o),
}


def numbered(inputs, prefix):
    """How many inputs prefix_0 .. prefix_<n-1> there are (all of them, in order)."""
    n = sum(k.startswith(prefix + '_') for k in inputs)
    if not n or any('%s_%d' % (prefix, k) not in inputs for k in range(n)):
        raise ValueError('%s_0 .. %s_<n-1> required' % (prefix, prefix))
    return n


def split_edit(inputs, opts):
    """video.edit (design §12 A): one scene's cuts in one encode → part_NN.mp4 (a new part after each episode break in
    the scene) + edit.json. inputs video_0..N = the generations the cuts use. opts cuts [{v, a, n, s, sa, part, text}]
    (v / s: input index of picture / sound, s null = silence; a / sa: start seconds; n: frames at 24 fps), speech
    [bool per input] (dialogue is brought to the target, room tone only down), multi [bool per input: a multi-shot
    generation — cut edges near its shot changes move onto them], target (LUFS, default -16), width·height (default the
    first cut's source size); a cut's fade {in, out} (seconds from / to black). One cpu unit."""
    n = numbered(inputs, 'video')
    cuts = opts.get('cuts') or []
    if not cuts:
        raise ValueError('cuts required')
    for c in cuts:
        if not (0 <= int(c['v']) < n) or (c.get('s') is not None and not 0 <= int(c['s']) < n) or int(c['n']) < 1:
            raise ValueError('cut source index or frame count out of range')
    return [('cpu', {})]


def split_join(inputs, opts):
    """video.join (design §12 A): scene parts (inputs part_0..N, in order) → film.mp4 by stream copy; opts breaks
    [indices of the parts an episode ends with] + flash (white flash seconds at each break, 0 = none) → also one
    ep_NN.mp4 per episode. Parts of another size are re-encoded to the first one's (reported in join.json)."""
    numbered(inputs, 'part')
    return [('cpu', {})]


def split_voice(inputs, opts):
    """audio.voice (기능 17, docs/drama.md §13.6): a generated clip's dialogue in the characters' own voices — inputs
    video (the clip) + voice_<i> (speaker i's reference voice); opts turns [speaker index per spoken turn, in order, -1 =
    left as generated]. The dialogue is split from the rest, each turn's speech converted (same length, so the lips
    still match), mixed back. One vc unit → voice.mp4 (the clip's video stream copied, the new sound) + voice.json."""
    turns = [int(t) for t in opts.get('turns') or []]
    if 'video' not in inputs or not turns or any(t >= 0 and 'voice_%d' % t not in inputs for t in turns) or max(turns) < 0:
        raise ValueError('video, turns and voice_<i> for every speaker index used')
    return [('vc', {})]


def split_tts(inputs, opts):
    """audio.tts: reference voice candidates for a character (Qwen3-TTS VoiceDesign) — opts instruct (the voice in
    words, the card's voice sentence), text (a Korean line to say), count (1-4), seed → one tts unit per candidate →
    voice_<k>.wav."""
    if not str(opts.get('instruct', '')).strip() or not str(opts.get('text', '')).strip():
        raise ValueError('instruct and text required')
    n = max(1, min(int(opts.get('count', 2)), 4))
    opts['seed'] = int(opts['seed']) if str(opts.get('seed', '')).strip() else secrets.randbelow(2 ** 31)
    return [('tts', {'k': k}) for k in range(n)]


def split_concat(inputs, opts):
    """video.concat: finished clips (inputs video_0..N, in order) → one video, one cpu unit — the 시퀀스 "한 편으로
    내보내기". opts width·height (default the first clip's size)."""
    n = sum(k.startswith('video_') for k in inputs)
    if not n or any('video_%d' % k not in inputs for k in range(n)):
        raise ValueError('video_0 .. video_<n-1> required')
    return [('cpu', {'n': n})]


def split_face(inputs, opts):
    """video.face: the face pass alone on a finished video (inputs video) — 영상 생성 › 마무리 works from an uploaded
    file (boss, session #13). Identity as video.i2v's face unit: inputs master [+ sheet] (+ master_b [+ sheet_b]) → one
    face unit → face.mp4 (the source audio) + face.json."""
    if 'video' not in inputs or 'master' not in inputs:
        raise ValueError('video and master required')
    return [('face', {})]


def split_probe(inputs, opts):
    """video.probe: one cpu unit that reads a video's length / size and the faces at its start (the swap screen's 사람
    table) → info.json + face_<k>.png."""
    if 'video' not in inputs:
        raise ValueError('video required')
    return [('cpu', {})]


def split_sheet(inputs, opts):
    """char.sheet: 8 panel units (qi21, image_1 = the master, face check against it) and a compose unit (cpu) after
    them → sh_<panel>.png x8 + sheet.png (3440x1280, the H3 reference), plus one unit per expression → ex_<id>.png. opts:
    gender, clothing (card), seed (chosen and recorded), turbo, expr [ids], strength 0/1/2."""
    if 'image_1' not in inputs:
        raise ValueError('image_1 (master) required')
    seed = int(opts['seed']) if str(opts.get('seed', '')).strip() else secrets.randbelow(2 ** 48)
    opts.update(seed=seed, check='image_1', min_sim=opts.get('min_sim', templates.MIN_SIM_FRONT))
    model = 'qi21-turbo' if opts.get('turbo') in (True, 'true', '1', 1) else 'qi21'
    auto = opts.get('auto_retry', True) not in (False, 0, '0', 'false', 'off')   # off: checks only report, no redraws
    if not auto:
        opts['body_retries'] = 0
    panels = [(k, p, r if auto else 0) for k, p, r in templates.sheet_panels(opts)]
    units = [(model, {'step': 'panel', 'kind': k, 'prompt': p, 'retries': r, 'seed': seed + i})
             for i, (k, p, r) in enumerate(panels)]
    units.append(('cpu', {'step': 'compose', 'after': list(range(len(panels)))}))
    # expressions (opts expr [ids], strength): separate images in the same job, not part of the sheet (boss, session #10)
    return units + [(model, {'step': 'panel', 'kind': k, 'prompt': p, 'retries': r if auto else 0, 'seed': seed + 100 + i})
                    for i, (k, p, r) in enumerate(templates.expr_prompts(opts))]


def split_upscale(inputs, opts):
    """video.upscale (pass 3): upscale one finished video (inputs video) with SeedVR2 7B → up.mp4. Starts with a plan
    unit (cpu) that reads the length and appends 5 s SeedVR2 chunks (C·D share them) and a join unit, as the swap's
    upscale (boss, session #13); a video of one chunk gets a single unit. opts: width·height (default portrait
    1080x1920, landscape 1920x1080 — cover and center crop), color lab (default)|none, seed (chosen and recorded, the
    same for every chunk)."""
    if 'video' not in inputs:
        raise ValueError('video required')
    if opts.get('color', 'lab') not in ('lab', 'none'):
        raise ValueError('color is lab or none')
    opts['seed'] = int(opts['seed']) if str(opts.get('seed', '')).strip() else secrets.randbelow(2 ** 48)
    return [('cpu', {'step': 'plan'})]


def split_i2v(inputs, opts):
    """video.i2v: LTX-2.5 first frame (inputs image) + prompt → one shot with sound (one ltx25 unit). opts: prompt,
    width·height (final, multiples of 64, default 1088x1920 = Lightricks node pack size; stage 1 runs at half size), sec
    (default 5 → frames 8k+1 at 24 fps, recorded), seed (chosen and recorded), seed2 (stage-2 noise, default 42 as the
    template), q8 (lossy attention, default off). Face pass (boss, session #8 — the drama LTX path): inputs master
    [+ sheet] (+ master_b [+ sheet_b] for a second character) → a face unit after the shot puts the character's face
    back (as video.swap); opts face (default on when master is given). Default size 768x1344 = 1 MP (landscape
    1344x768), the drama size."""
    if 'image' not in inputs or not str(opts.get('prompt', '')).strip():
        raise ValueError('image and prompt required')
    w, h = int(opts.get('width') or 768), int(opts.get('height') or 1344)
    if w % 64 or h % 64:
        raise ValueError('width and height must be multiples of 64')
    sec = float(opts.get('sec') or 5)
    seed = int(opts['seed']) if str(opts.get('seed', '')).strip() else secrets.randbelow(2 ** 48)
    seed2 = int(opts['seed2']) if str(opts.get('seed2', '')).strip() else 42   # stage-2 noise: the template's fixed 42
    v = opts.get('face', '')
    face = 'master' in inputs if v in ('', None) else v not in (False, 0, '0', 'false', 'off')
    if face and 'master' not in inputs:
        raise ValueError('face pass needs inputs master (character master)')
    opts.update(width=w, height=h, sec=sec, frames=8 * max(1, round(sec * 24 / 8)) + 1, seed=seed, seed2=seed2, face=face)
    return [('ltx25', {})] + ([('face', {'step': 'face', 'after': [0]})] if face else [])

def split_pih3(inputs, opts):
    """video.pih3: the drama line's H3 path (boss, session #8) — pi-H3 as the author's graph: MiniMax H3 FL2VA + the pi-H3
    DWD adapter (engine/py/pih3_pipe.py), first frame (inputs image, I2V) and/or last frame (inputs last, FL2VA / L2VA)
    or none (T2V), prompt in the FL2VA format (with a last frame its first line aligns the pictures, base guide §2.1).
    opts: prompt, steps 2 (default) | 4 (important shots), width·height (multiples of 32, default 768x1344; landscape
    1344x768), frames (17k + 5 at 24 fps) or sec (default 5 → nearest 17k + 5), seed (chosen and recorded), q8 (Sage-style
    attention, default on — off is slower than ComfyUI), vsa (block-sparse attention as ComfyUI's vsa, lossy, default
    off — boss, session #12: selectable; drama 40·41). Face pass as video.i2v: inputs master [+ sheet] (+ master_b
    [+ sheet_b]) → a face unit after the shot; opts face (default on when master is given)."""
    if not str(opts.get('prompt', '')).strip():
        raise ValueError('prompt required')
    steps = int(opts.get('steps') or 2)
    if steps not in (2, 4):
        raise ValueError('steps is 2 or 4')
    w, h = int(opts.get('width') or 768), int(opts.get('height') or 1344)
    if w % 32 or h % 32:
        raise ValueError('width and height must be multiples of 32')
    if str(opts.get('frames', '')).strip():
        n = int(opts['frames'])
        if n % 17 != 5:
            raise ValueError('frames must be 17k + 5 (124 = 5.2 s, 175 = 7.3 s, 243 = 10.1 s)')
    else:
        n = 17 * max(1, round((float(opts.get('sec') or 5) * 24 - 5) / 17)) + 5
    seed = int(opts['seed']) if str(opts.get('seed', '')).strip() else secrets.randbelow(2 ** 48)
    v = opts.get('face', '')
    face = 'master' in inputs if v in ('', None) else v not in (False, 0, '0', 'false', 'off')
    if face and 'master' not in inputs:
        raise ValueError('face pass needs inputs master (character master)')
    q = opts.get('q8', True)
    opts.update(steps=steps, width=w, height=h, frames=n, sec=round(n / 24, 2), seed=seed, face=face,
                q8=q not in (False, 0, '0', 'false', 'off'),
                vsa=opts.get('vsa') not in (None, '', False, 0, '0', 'false', 'off'))
    return [('pih3_%d' % steps, {})] + ([('face', {'step': 'face', 'after': [0]})] if face else [])


def split_r2v(inputs, opts):
    """video.r2v: reference images (character sheets etc., image_1..N = <Picture 1..N>) + an H3 prompt per shot → one h3
    unit per shot (C·D share them); with two or more shots a cpu unit joins them in order. inputs first_<k> / last_<k>:
    start / end frame of shot k (scene image, guide at frame 0 / the last frame), optional; key_<k>_<j> + opts keys
    [[frame index …] per shot]: more guides on shot k's time axis, also reference pictures after image_N (a multi-shot's
    shot first frames, the official multiframe template). opts: shots [prompt…], width·height (final, default 768x1344 = H3 training canvas),
    frames, seed; defaults are the creator recipe — steps 12 (euler·beta), split 8, up 1.5 (1 = single pass), ref_size
    max, sampler euler; q8 = Sage-style attention (lossy, on by default — boss 2026-11-20, no visible difference),
    false = lossless.
    stage: preview (default) = pass 1 only (preview + pass-1 latent .lat), refine = pass 2 only from the picked shots' .lat
    (inputs lat_0..; settings come from the .lat, references and start frames are the same files as in pass 1), full = all
    in one go. With up 1 there are no stages, so full."""
    stage = opts.get('stage') or 'preview'
    if stage != 'refine' and float(opts.get('up', 1.5)) <= 1:
        stage = 'full'
    if stage not in ('preview', 'refine', 'full') or 'image_1' not in inputs:
        raise ValueError('image_1 required; stage is preview, refine or full')
    if opts.get('ref_size', 'max') not in ('max', 'match'):   # max: full-res sheet (faithful), match: canvas fit (~2x faster, lossy)
        raise ValueError('ref_size is max or match')
    opts['stage'] = stage
    if stage == 'refine':
        n = sum(k.startswith('lat_') for k in inputs)
        if not n or any('lat_%d' % k not in inputs for k in range(n)):
            raise ValueError('refine needs lat_0 .. lat_<n-1>')
    else:
        if not opts.get('shots'):
            raise ValueError('shots required')
        opts['seed'] = int(opts['seed']) if str(opts.get('seed', '')).strip() else secrets.randbelow(2 ** 48)
        n = len(opts['shots'])
        keys = opts.get('keys') or []
        frames = int(opts.get('frames', 124))
        if len(keys) > n or any(not 0 <= int(f) < frames for kk in keys for f in kk) or \
                any(('key_%d_%d' % (k, j) in inputs) != (j < len(kk)) for k, kk in enumerate(keys) for j in range(len(kk) + 1)):
            raise ValueError('keys: frame indices per shot inside the video, key_<shot>_<j> for each')
    units = [('h3', {'shot': k}) for k in range(n)]
    return units + ([('cpu', {'step': 'concat', 'after': list(range(n)), 'n': n})] if n > 1 else [])


def split_swap(inputs, opts):
    """video.swap: starts with a single plan unit. The plan unit (worker) reads the video and appends boundary-edit (qi21),
    segment (h3) and stitch units with spec.after (§4). inputs: video, image_1 = character master [, background = image
    of the new place (#10), image 3 of every boundary edit]. opts place = text of the new place (alone or with the image).
    opts start / end (seconds) trim the source. Segments use the creator recipe 8+4 (as video.r2v; one-pass turbo 4
    steps broke in fast motion — boss, session #7). Two people (#11): + image_b / master_b / sheet_b (second
    character), opts person_a / person_b (how the two look in the source; A = the front one at the start), gender_b /
    desc_b, taller ('a' | 'b'), clean_logo; the segments run in order, each from the previous one's last frame
    (two-person boundary edits don't work on close-ups)."""
    if 'video' not in inputs or 'image_1' not in inputs:
        raise ValueError('video and image_1 (character master) required')
    seed = int(opts['seed']) if str(opts.get('seed', '')).strip() else secrets.randbelow(2 ** 48)
    opts['seed'] = seed                       # every segment uses the same seed (§4)
    # after the join (session #7): face pass (hyperswap, default on when a character sheet is given) and SeedVR2 upscale
    # to 1080x1920 (#12 optional, default off — boss, session #10). Switches accept true/false, '' = default
    for k, d in (('face', 'sheet' in inputs), ('upscale', False)):
        v = opts.get(k, '')
        opts[k] = d if v in ('', None) else v not in (False, 0, '0', 'false', 'off')
    if 'sheet_b' in inputs:   # two people (#11): image_b / master_b / sheet_b = the second character, person_a/_b name
        if 'sheet' not in inputs or not opts.get('person_a') or not opts.get('person_b'):   # who is who in the source
            raise ValueError('two-person swap needs both sheets and person_a / person_b')
        if 'background' in inputs or opts.get('place'):
            raise ValueError('two-person swap: background change not supported yet')
        opts['h3_prompt'] = templates.swap2_prompt(opts)
        return [('cpu', {'step': 'plan'})]
    opts['edit_prompt'], opts['edit_prompt_part'], opts['h3_prompt'], bg = templates.swap_prompts(opts, 'background' in inputs)
    if bg:
        opts['bg_prompt'] = bg
    return [('cpu', {'step': 'plan'})]

# ---- queue (sqlite, C disk) ----
for d in (ASSETS, RESULTS, os.path.dirname(DB)):
    os.makedirs(d, exist_ok=True)
db = sqlite3.connect(DB, check_same_thread=False, isolation_level=None)
db.row_factory = sqlite3.Row
db.executescript('''
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, type TEXT, inputs TEXT, opts TEXT,
  state TEXT, created REAL, synced INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY, job TEXT, idx INTEGER, model TEXT, spec TEXT,
  state TEXT, tries INTEGER DEFAULT 0, total INTEGER DEFAULT 0, worker TEXT, progress REAL DEFAULT 0,
  started REAL, beat REAL, error TEXT);
CREATE TABLE IF NOT EXISTS files(job TEXT, name TEXT, size INTEGER, sha TEXT, PRIMARY KEY(job, name));
''')


def add_cols(table, cols):
    have = {r[1] for r in db.execute('PRAGMA table_info(%s)' % table)}
    new = [c for c in cols if c[0] not in have]
    for name, decl in new:
        db.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table, name, decl))
    return [c[0] for c in new]


# session #10: project per job, "먼저 하기" priority, closed jobs, end time; per-file sync. Files from before per-file
# sync count as taken unless their job is done and not fetched yet (the old sync took whole done jobs)
add_cols('jobs', [('project', 'TEXT'), ('prio', 'INTEGER DEFAULT 0'), ('closed', 'INTEGER DEFAULT 0'), ('ended', 'REAL')])
if add_cols('files', [('synced', 'INTEGER DEFAULT 0'), ('tmp', 'INTEGER DEFAULT 0')]):
    db.execute("UPDATE files SET synced=1 WHERE job NOT IN (SELECT id FROM jobs WHERE state='done' AND synced=0)")
# the Mac keeps each ended job's full record as job.json in its job folder (boss, session #10). Jobs from before that
# count as recorded (their folders were cleaned up)
if add_cols('jobs', [('recorded', 'INTEGER DEFAULT 0')]):
    db.execute('UPDATE jobs SET recorded=1')
cv = threading.Condition()   # all DB access under this lock. notify when new units appear
workers = {}                 # name → {seen, loaded, unit, cache}


def config():
    try:
        with open(CONFIG) as f:
            return dict(CONFIG_DEFAULT, **json.load(f))
    except (OSError, ValueError):
        return dict(CONFIG_DEFAULT)


def drop_results(job):
    """Delete the job's result files still on the node. Files the Mac already took stay listed (they are on the Mac)."""
    shutil.rmtree('%s/%s' % (RESULTS, job), ignore_errors=True)
    db.execute('DELETE FROM files WHERE job=? AND synced=0', (job,))


def finish_sync(job):
    """A done job whose results are all on the Mac: mark it synced and drop what is left of its result folder
    (intermediates)."""
    r = db.execute('SELECT state FROM jobs WHERE id=?', (job,)).fetchone()
    if r and r['state'] == 'done' and not db.execute('SELECT 1 FROM files WHERE job=? AND synced=0 AND tmp=0',
                                                     (job,)).fetchone():
        db.execute('UPDATE jobs SET synced=1 WHERE id=?', (job,))
        shutil.rmtree('%s/%s' % (RESULTS, job), ignore_errors=True)


def refresh(job):
    s = [r[0] for r in db.execute('SELECT state FROM units WHERE job=?', (job,))]
    st = ('pending' if all(x == 'pending' for x in s) else
          'running' if 'running' in s or 'pending' in s else
          'failed' if 'failed' in s else 'done')
    old = db.execute('SELECT state FROM jobs WHERE id=?', (job,)).fetchone()[0]
    if st == old:
        return
    db.execute('UPDATE jobs SET state=?, ended=?, recorded=0 WHERE id=?', (st, time.time() if st in ('done', 'failed') else None, job))
    if st == 'failed' and len(s) == 1:   # a single-unit job that failed or was cancelled has nothing to resume: its files go
        drop_results(job)                # at once (boss, session #10). The record stays (settings, seed → re-run)
    if st == 'done':
        finish_sync(job)


def fail_unit(uid, error):
    u = db.execute('SELECT job, idx, tries FROM units WHERE id=?', (uid,)).fetchone()
    st = 'failed' if u['tries'] >= MAX_TRIES else 'pending'
    db.execute('UPDATE units SET state=?, error=?, worker=NULL WHERE id=?', (st, error, uid))
    if st == 'failed':
        fail_dependents(u['job'], {u['idx']})
    refresh(u['job'])
    cv.notify_all()


def fail_dependents(job, dead):
    """Units that depend (spec.after) on a failed unit fail too (so they never sit 'pending' forever). A retry requeues
    them along with it."""
    changed = True
    while changed:
        changed = False
        for r in db.execute("SELECT idx, spec FROM units WHERE job=? AND state='pending'", (job,)).fetchall():
            if set(json.loads(r['spec']).get('after') or []) & dead:
                db.execute("UPDATE units SET state='failed', error='dependency failed' WHERE job=? AND idx=?", (job, r['idx']))
                dead.add(r['idx'])
                changed = True


def ready(r):
    after = json.loads(r['spec']).get('after') or []
    if not after:
        return True
    st = dict(db.execute('SELECT idx, state FROM units WHERE job=?', (r['job'],)).fetchall())
    return all(st.get(i) == 'done' for i in after)


def pick(worker, loaded):
    # among units whose predecessors (spec.after) are done: units of the already loaded model (and cpu units, which don't
    # change the model) first, another model only when there are none (§3, #2). Within that, arrival order
    rows = [r for r in db.execute("SELECT u.id, u.job, u.model, u.spec FROM units u JOIN jobs j ON j.id=u.job "
                                  "WHERE u.state='pending' ORDER BY j.prio DESC, j.rowid, u.idx").fetchall() if ready(r)]
    now = time.time()
    # if another worker with that model loaded is idle, leave the unit to it (no new model load, and with the same
    # references its TE cache hits too)
    idle = {v.get('loaded') for k, v in workers.items() if k != worker and not v.get('unit') and now - v.get('seen', 0) < 30}
    r = next((x for x in rows if x['model'] in (loaded, 'cpu')), None) or \
        next((x for x in rows if x['model'] not in idle), None)
    if not r:
        return None
    now = time.time()
    db.execute("UPDATE units SET state='running', worker=?, tries=tries+1, total=total+1, progress=0, "
               "started=?, beat=?, error=NULL WHERE id=?", (worker, now, now, r['id']))
    refresh(r['job'])
    u = dict(db.execute('SELECT * FROM units WHERE id=?', (r['id'],)).fetchone())
    j = db.execute('SELECT type, inputs, opts FROM jobs WHERE id=?', (r['job'],)).fetchone()
    u.update(type=j['type'], inputs=json.loads(j['inputs']), opts=json.loads(j['opts']),
             spec=json.loads(u['spec']))
    return u


def give_back(uid, job):
    """Put a held unit back in the queue without counting a try (disconnected worker, stopping worker)."""
    db.execute("UPDATE units SET state='pending', worker=NULL, tries=tries-1, total=total-1 WHERE id=?", (uid,))
    refresh(job)
    cv.notify_all()


def reap():
    for r in db.execute("SELECT id FROM units WHERE state='running' AND beat<?",
                        (time.time() - STALE,)).fetchall():
        fail_unit(r['id'], 'stale: no progress for %ds' % STALE)


def job_view(r, units=False):
    j = dict(r)
    j['inputs'], j['opts'] = json.loads(j['inputs']), json.loads(j['opts'])
    us = [dict(u) for u in db.execute('SELECT * FROM units WHERE job=? ORDER BY idx', (j['id'],))]
    j['count'] = {s: sum(u['state'] == s for u in us) for s in ('pending', 'running', 'done', 'failed')}
    j['progress'] = sum(1.0 if u['state'] == 'done' else u['progress'] or 0 for u in us) / max(len(us), 1)
    j['errors'] = [u['error'] for u in us if u['state'] == 'failed']
    j['files'] = [dict(f) for f in db.execute('SELECT name, size, sha, synced, tmp FROM files WHERE job=? ORDER BY name',
                                                (j['id'],))]
    if units:
        for u in us:
            u['spec'] = json.loads(u['spec'])
        j['units'] = us
    return j


def safe(name):
    return bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,200}', name))


def cleanup(days=None, gb=None):
    """Node cache rules (§8, boss session #10) for C's cache (inputs from the Mac + results the Mac already took):
    failed multi-unit jobs past RESUME are closed; result folders of closed / fully synced jobs go; cache files unused
    for `days` go, then the oldest until the cache is under `gb`. Files of queued, running and resumable jobs are kept.
    A missing input is resent by the Mac. Call under cv."""
    c, now = config(), time.time()
    days, gb = c['cache_days'] if days is None else days, c['cache_gb'] if gb is None else gb
    for r in db.execute("SELECT id FROM jobs WHERE state='failed' AND closed=0 AND ended<?", (now - RESUME,)).fetchall():
        db.execute('UPDATE jobs SET closed=1 WHERE id=?', (r['id'],))
        drop_results(r['id'])
    for jid in os.listdir(RESULTS):
        r = db.execute('SELECT closed, synced FROM jobs WHERE id=?', (jid,)).fetchone()
        if not r or r['closed'] or r['synced']:
            shutil.rmtree('%s/%s' % (RESULTS, jid), ignore_errors=True)
    keep = set()
    for r in db.execute("SELECT id, inputs FROM jobs WHERE closed=0 AND (state IN ('pending','running') "
                        "OR (state='failed' AND ended>=?))", (now - RESUME,)).fetchall():
        keep.update(v for v in json.loads(r['inputs']).values() if isinstance(v, str))
        keep.update(f[0] for f in db.execute('SELECT sha FROM files WHERE job=?', (r['id'],)))
    files = []
    for n in os.listdir(ASSETS):
        p = '%s/%s' % (ASSETS, n)
        try:
            st = os.stat(p)
        except OSError:
            continue
        if n.endswith('.part'):          # an upload cut off over a day ago
            if st.st_mtime < now - 86400:
                os.remove(p)
            continue
        files.append((st.st_mtime, st.st_size, p, n))
    files.sort()
    total, gone = sum(f[1] for f in files), 0
    for mtime, size, p, n in files:
        if n in keep or (mtime >= now - days * 86400 and total <= gb * 2 ** 30):
            continue
        os.remove(p)
        total -= size
        gone += size
    return {'freed': gone, 'left': total}


def cleanup_loop():
    while True:
        time.sleep(600)
        try:
            with cv:
                cleanup()
        except Exception as e:
            print('cleanup: %s: %s' % (type(e).__name__, e), flush=True)


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def send(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(b)

    def body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'{}') if n else {}

    def recv_file(self, dest):
        """Receive the body into a temp file of this request (two uploads of the same new asset at once must not share
        one) → (temp path, size, sha256); the caller renames it once the whole body arrived."""
        n, h, tmp = int(self.headers['Content-Length']), hashlib.sha256(), '%s.%d.part' % (dest, threading.get_ident())
        with open(tmp, 'wb') as f:
            left = n
            while left:
                b = self.rfile.read(min(CHUNK, left))
                if not b:
                    raise IOError('short body')
                h.update(b); f.write(b); left -= len(b)
        return tmp, n, h.hexdigest()

    def send_file(self, path):
        if not os.path.isfile(path):
            return self.send({'error': 'not found'}, 404)
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(os.path.getsize(path)))
        self.end_headers()
        if self.command == 'GET':
            with open(path, 'rb') as f:
                shutil.copyfileobj(f, self.wfile, CHUNK)

    def route(self):
        u = urlparse(self.path)
        p, q, m = u.path.rstrip('/'), parse_qs(u.query), self.command
        try:
            for pat, meth, fn in ROUTES:
                g = re.fullmatch(pat, p)
                if g and m == meth:
                    return fn(self, q, *g.groups())
            self.send({'error': 'no route'}, 404)
        except Exception as e:
            self.send({'error': '%s: %s' % (type(e).__name__, e)}, 500)

    do_GET = do_POST = do_PUT = do_HEAD = do_DELETE = route

    # ---- job API ----
    def jobs_post(self, q):
        b = self.body()
        t, inputs, opts = b.get('type'), b.get('inputs') or {}, b.get('options') or {}
        if t not in TYPES:
            return self.send({'error': 'unknown type %r' % t}, 400)
        missing = [k for k, s in inputs.items() if not os.path.isfile('%s/%s' % (ASSETS, s))]
        if missing:
            return self.send({'error': 'assets not uploaded', 'missing': missing}, 400)
        try:
            units = TYPES[t](inputs, opts)
        except (ValueError, TypeError) as e:
            return self.send({'error': str(e)}, 400)
        jid = time.strftime('%Y%m%d-%H%M%S-') + secrets.token_hex(2)
        for s in inputs.values():   # used again: the cache keeps it (§8)
            os.utime('%s/%s' % (ASSETS, s))
        with cv:
            db.execute('INSERT INTO jobs(id,type,inputs,opts,state,created,project) VALUES(?,?,?,?,?,?,?)',
                       (jid, t, json.dumps(inputs), json.dumps(opts), 'pending', time.time(), b.get('project')))
            for i, (model, spec) in enumerate(units):
                db.execute("INSERT INTO units(id,job,idx,model,spec,state) VALUES(?,?,?,?,?,'pending')",
                           ('%s.%d' % (jid, i), jid, i, model, json.dumps(spec)))
            cv.notify_all()
        self.send({'id': jid, 'units': len(units)})

    def jobs_list(self, q):
        lim = int(q.get('limit', ['50'])[0])
        where, args = [], []
        if q.get('open'):        # not closed (the 작업 screen)
            where.append('closed=0')
        if q.get('ids'):         # these jobs (a production's: more than the open list holds)
            ids = [x for x in q['ids'][0].split(',') if x][:1000]
            where.append('id IN (%s)' % ','.join('?' * len(ids)))
            args += ids
        if q.get('project'):
            where.append('project=?')
            args.append(q['project'][0])
        if q.get('unrecorded'):  # Mac: ended jobs whose record (job.json) it has not written yet — with their units
            where.append("recorded=0 AND state IN ('done','failed')")
        with cv:
            rs = db.execute('SELECT * FROM jobs %s ORDER BY rowid DESC LIMIT ?' % ('WHERE ' + ' AND '.join(where) if where else ''),
                            args + [lim]).fetchall()
            self.send([job_view(r, bool(q.get('unrecorded'))) for r in rs])

    def job_recorded(self, q, jid):
        with cv:
            db.execute('UPDATE jobs SET recorded=1 WHERE id=?', (jid,))
        self.send({'ok': True})

    def job_get(self, q, jid):
        with cv:
            r = db.execute('SELECT * FROM jobs WHERE id=?', (jid,)).fetchone()
            self.send(job_view(r, True) if r else {'error': 'no job'}, 200 if r else 404)

    def job_retry(self, q, jid):
        with cv:
            r = db.execute('SELECT inputs FROM jobs WHERE id=?', (jid,)).fetchone()
            missing = [k for k, s in json.loads(r['inputs']).items() if not os.path.isfile('%s/%s' % (ASSETS, s))] if r else []
            if missing:   # cleared from the node cache meanwhile: the Mac resends them, then retries
                return self.send({'error': 'assets not on the node', 'missing': missing}, 409)
            n = db.execute("UPDATE units SET state='pending', tries=0, error=NULL, progress=0 "
                           "WHERE job=? AND state='failed'", (jid,)).rowcount
            refresh(jid)
            cv.notify_all()
        self.send({'requeued': n})

    def job_redo(self, q, jid):
        """"이 칸만 다시": one unit (body idx) runs again with a new seed, and every unit after it (spec.after, e.g. the
        sheet compose) runs again too. Its result file is replaced and goes to the Mac again."""
        idx = int(self.body().get('idx', -1))
        with cv:
            u = db.execute('SELECT id, spec, state FROM units WHERE job=? AND idx=?', (jid, idx)).fetchone()
            if not u or u['state'] == 'running':
                return self.send({'error': 'no such unit or running'}, 409)
            spec = json.loads(u['spec'])
            if 'seed' not in spec:   # swap boundary edits and segments use the job seed until they are redone
                seed = json.loads(db.execute('SELECT opts FROM jobs WHERE id=?', (jid,)).fetchone()['opts']).get('seed')
                if seed is None:
                    return self.send({'error': 'unit has no seed'}, 400)
                spec['seed'] = int(seed)
            spec['redo'] = spec.get('redo', 0) + 1
            spec['seed'] = spec['seed'] + 7919   # a new seed, still recorded in the unit and the result json
            db.execute("UPDATE units SET spec=?, state='pending', tries=0, error=NULL, progress=0 WHERE id=?",
                       (json.dumps(spec), u['id']))
            dead, changed = {idx}, True
            while changed:
                changed = False
                for r in db.execute('SELECT idx, spec FROM units WHERE job=? AND idx NOT IN (%s)' % ','.join('?' * len(dead)),
                                    [jid] + sorted(dead)).fetchall():
                    if set(json.loads(r['spec']).get('after') or []) & dead:
                        db.execute("UPDATE units SET state='pending', tries=0, error=NULL, progress=0 WHERE job=? AND idx=?",
                                   (jid, r['idx']))
                        dead.add(r['idx'])
                        changed = True
            db.execute('UPDATE jobs SET synced=0, closed=0 WHERE id=?', (jid,))
            refresh(jid)
            cv.notify_all()
        self.send({'redo': sorted(dead), 'seed': spec['seed']})

    def prompt_preview(self, q):
        """The English prompt a job would get from its template (the UI's "영어 프롬프트"), without queuing anything."""
        b = self.body()
        t, opts = b.get('type'), dict(b.get('options') or {})
        if t == 'char.sheet':
            return self.send({'prompts': [[k, p] for k, p, r in templates.sheet_panels(opts) + templates.expr_prompts(opts)]})
        if t not in templates.JOBS:
            return self.send({'error': 'no template for %r' % t}, 400)
        fill, defaults = templates.JOBS[t]
        merged = dict(defaults)
        merged.update(opts)
        try:
            return self.send({'prompt': fill(merged)})
        except KeyError as e:
            return self.send({'error': 'missing %s' % e}, 400)

    def job_cancel(self, q, jid):
        # remaining units become failed ('cancelled'). A running worker gets 409 on its next progress report and stops.
        # A retry brings them back
        with cv:
            n = db.execute("UPDATE units SET state='failed', error='cancelled', worker=NULL "
                           "WHERE job=? AND state IN ('pending','running')", (jid,)).rowcount
            refresh(jid)
            cv.notify_all()
        self.send({'cancelled': n})

    def job_units_post(self, q, jid):
        # a worker appends units to a job (e.g. the video.swap plan unit creates boundary-edit, segment and stitch units).
        # body {units: [{model, spec}]}, spec.after = list of unit indices (idx) within this job
        b = self.body()
        with cv:
            if not db.execute('SELECT 1 FROM jobs WHERE id=?', (jid,)).fetchone():
                return self.send({'error': 'no job'}, 404)
            base = db.execute('SELECT COALESCE(MAX(idx), -1) + 1 FROM units WHERE job=?', (jid,)).fetchone()[0]
            for k, u in enumerate(b.get('units') or []):
                spec = dict(u.get('spec') or {})
                db.execute("INSERT INTO units(id,job,idx,model,spec,state) VALUES(?,?,?,?,?,'pending')",
                           ('%s.%d' % (jid, base + k), jid, base + k, u['model'], json.dumps(spec)))
            refresh(jid)
            cv.notify_all()
        self.send({'first': base, 'n': len(b.get('units') or [])})

    def job_first(self, q, jid):
        # "먼저 하기": this job's waiting units go before every other job's (pick orders by prio)
        with cv:
            top = db.execute('SELECT COALESCE(MAX(prio), 0) FROM jobs').fetchone()[0]
            n = db.execute('UPDATE jobs SET prio=? WHERE id=?', (top + 1, jid)).rowcount
            cv.notify_all()
        self.send({'ok': bool(n)}, 200 if n else 404)

    def job_close(self, q, jid):
        # "닫기": the job leaves the 작업 screen and its files on the node go. The record stays (settings, seeds)
        with cv:
            r = db.execute('SELECT state FROM jobs WHERE id=?', (jid,)).fetchone()
            if not r:
                return self.send({'error': 'no job'}, 404)
            if r['state'] in ('pending', 'running'):
                return self.send({'error': 'stop it first'}, 409)
            db.execute('UPDATE jobs SET closed=1, recorded=0 WHERE id=?', (jid,))   # the Mac rewrites job.json (closed)
            drop_results(jid)
        self.send({'ok': True})

    def files_list(self, q):
        # Mac sync: every result not taken yet, as soon as its unit has uploaded it (intermediates excluded)
        with cv:
            rs = db.execute("SELECT f.job, f.name, f.size, f.sha, j.project, j.type FROM files f JOIN jobs j ON j.id=f.job "
                            "WHERE f.synced=0 AND f.tmp=0 AND j.closed=0 ORDER BY j.rowid, f.name").fetchall()
        self.send([dict(r) for r in rs])

    def files_ack(self, q):
        # the Mac has this file → it moves from the result folder into the node cache (by sha: later units of the job
        # still read it through /results, and it can serve as an input without a resend)
        b = self.body()
        with cv:
            r = db.execute('SELECT sha FROM files WHERE job=? AND name=?', (b.get('job'), b.get('name'))).fetchone()
            if not r or r['sha'] != b.get('sha') or not safe(b['name']):
                return self.send({'error': 'no such file'}, 409)
            src, dest = '%s/%s/%s' % (RESULTS, b['job'], b['name']), '%s/%s' % (ASSETS, r['sha'])
            if os.path.isfile(src):
                if os.path.isfile(dest):
                    os.remove(src)
                else:
                    os.replace(src, dest)
            db.execute('UPDATE files SET synced=1 WHERE job=? AND name=?', (b['job'], b['name']))
            finish_sync(b['job'])
        self.send({'ok': True})

    def config_get(self, q):
        self.send(config())

    def config_post(self, q):
        b = self.body()
        c = config()
        for k in ('cache_days', 'cache_gb'):
            if k in b:
                c[k] = max(0, float(b[k]))
        with open(CONFIG + '.part', 'w') as f:
            json.dump(c, f)
        os.replace(CONFIG + '.part', CONFIG)
        self.send(c)

    def cache_clear(self, q):
        # "노드 캐시 비우기": C now (protected files stay), the other workers on their next round (clear_gen)
        c = config()
        c['clear_gen'] += 1
        with open(CONFIG + '.part', 'w') as f:
            json.dump(c, f)
        os.replace(CONFIG + '.part', CONFIG)
        with cv:
            self.send(cleanup(0, 0))

    def queue(self, q):
        # ComfyUI /queue shape: [number, id, content]. The engine router looks at the lengths of the two lists (§5)
        with cv:
            rs = db.execute("SELECT u.id, u.job, u.model, u.state, u.worker, u.progress, j.type FROM units u "
                            "JOIN jobs j ON j.id=u.job WHERE u.state IN ('running','pending') "
                            "ORDER BY j.rowid, u.idx").fetchall()
        out = {'queue_running': [], 'queue_pending': []}
        for n, r in enumerate(rs):
            out['queue_' + r['state']].append(
                [n, r['id'], {k: r[k] for k in ('job', 'type', 'model', 'worker', 'progress')}])
        self.send(out)

    def types(self, q):
        self.send(sorted(TYPES))

    def workers_get(self, q):
        now = time.time()
        self.send({k: dict(v, ago=round(now - v['seen'], 1)) for k, v in workers.items()})

    # ---- asset · result files ----
    def asset_head(self, q, sha):
        self.send_file('%s/%s' % (ASSETS, sha))

    def asset_put(self, q, sha):
        dest = '%s/%s' % (ASSETS, sha)
        if os.path.isfile(dest):
            return self.send({'ok': True, 'had': True})
        tmp, n, h = self.recv_file(dest)
        if h != sha:
            os.remove(tmp)
            return self.send({'error': 'sha mismatch'}, 400)
        os.replace(tmp, dest)
        self.send({'ok': True, 'size': n})

    def result_put(self, q, jid, name):
        # ?tmp=1: an intermediate a later unit of the job deletes (upscale chunks) — never sent to the Mac
        if not safe(name):
            return self.send({'error': 'bad name'}, 400)
        d = '%s/%s' % (RESULTS, jid)
        os.makedirs(d, exist_ok=True)
        tmp, n, h = self.recv_file('%s/%s' % (d, name))
        os.replace(tmp, '%s/%s' % (d, name))
        with cv:
            db.execute('INSERT OR REPLACE INTO files(job,name,size,sha,synced,tmp) VALUES(?,?,?,?,0,?)',
                       (jid, name, n, h, int(q.get('tmp') == ['1'])))
        self.send({'size': n, 'sha': h})

    def result_del(self, q, jid, name):   # a worker drops an intermediate it made (e.g. upscale chunks)
        if not safe(name):
            return self.send({'error': 'bad name'}, 400)
        f = '%s/%s/%s' % (RESULTS, jid, name)
        if os.path.isfile(f):
            os.remove(f)
        with cv:
            db.execute('DELETE FROM files WHERE job=? AND name=?', (jid, name))
        self.send({'ok': True})

    def result_get(self, q, jid, name):
        if not safe(name):
            return self.send({}, 400)
        path = '%s/%s/%s' % (RESULTS, jid, name)
        if not os.path.isfile(path):   # already taken by the Mac: moved into the cache under its sha
            with cv:
                r = db.execute('SELECT sha FROM files WHERE job=? AND name=?', (jid, name)).fetchone()
            path = '%s/%s' % (ASSETS, r['sha']) if r else path
        self.send_file(path)

    # ---- worker ----
    def next_unit(self, q):
        b = self.body()
        w, loaded = b['worker'], b.get('loaded')
        end = time.time() + float(b.get('wait', 20))   # long poll: returns as soon as work appears
        with cv:
            # this worker asked again = the unit it held was cut off → back to the queue without counting a try
            for r in db.execute("SELECT id, job FROM units WHERE state='running' AND worker=?", (w,)).fetchall():
                give_back(r['id'], r['job'])
            while True:
                reap()
                workers[w] = {'seen': time.time(), 'loaded': loaded, 'unit': None, 'cache': b.get('cache')}
                u = pick(w, loaded)
                if u or time.time() >= end:
                    break
                cv.wait(min(5, end - time.time()))
            if u:
                workers[w]['unit'] = u['id']
        self.send({'unit': u})

    def unit_progress(self, q, uid):
        b = self.body()
        with cv:
            mine = self.owned(uid, b)
            if mine:
                db.execute('UPDATE units SET progress=?, beat=? WHERE id=?', (b.get('p', 0), time.time(), uid))
                workers.setdefault(b['worker'], {})['seen'] = time.time()
        self.send({'ok': bool(mine)}, 200 if mine else 409)   # 409 → the worker drops this unit

    def owned(self, uid, b):
        r = db.execute('SELECT job, state, worker FROM units WHERE id=?', (uid,)).fetchone()
        return r and r['state'] == 'running' and r['worker'] == b.get('worker') and r['job']

    def unit_done(self, q, uid):
        b = self.body()
        with cv:
            job = self.owned(uid, b)
            if job:
                db.execute("UPDATE units SET state='done', progress=1, error=NULL WHERE id=?", (uid,))
                refresh(job)
        self.send({'ok': bool(job)}, 200 if job else 409)

    def bye(self, q):
        """The worker is stopping (docker stop): its held unit goes back to the queue and the worker leaves the list — if
        it stayed listed as an idle worker with a model loaded, pick would not give that model's units to other workers."""
        b = self.body()
        with cv:
            job = b.get('unit') and self.owned(b['unit'], b)
            if job:
                give_back(b['unit'], job)
            workers.pop(b.get('worker'), None)
            cv.notify_all()
        self.send({'ok': True})

    def unit_fail(self, q, uid):
        b = self.body()
        with cv:
            job = self.owned(uid, b)
            if job:
                fail_unit(uid, str(b.get('error', 'error'))[:2000])
        self.send({'ok': bool(job)}, 200 if job else 409)


J, S, N = r'/jobs/([\w.-]+)', r'([0-9a-f]{64})', r'([^/]+)'
ROUTES = [
    (r'/jobs', 'POST', H.jobs_post), (r'/jobs', 'GET', H.jobs_list),
    (J, 'GET', H.job_get), (J + '/retry', 'POST', H.job_retry), (J + '/cancel', 'POST', H.job_cancel),
    (J + '/first', 'POST', H.job_first), (J + '/close', 'POST', H.job_close), (J + '/units', 'POST', H.job_units_post),
    (J + '/redo', 'POST', H.job_redo), (J + '/recorded', 'POST', H.job_recorded), (r'/prompt', 'POST', H.prompt_preview),
    (r'/files', 'GET', H.files_list), (r'/files/ack', 'POST', H.files_ack),
    (r'/config', 'GET', H.config_get), (r'/config', 'POST', H.config_post), (r'/cache/clear', 'POST', H.cache_clear),
    (r'/queue', 'GET', H.queue), (r'/types', 'GET', H.types), (r'/workers', 'GET', H.workers_get),
    (r'/assets/' + S, 'HEAD', H.asset_head), (r'/assets/' + S, 'GET', H.asset_head),
    (r'/assets/' + S, 'PUT', H.asset_put),
    (r'/results/([\w.-]+)/' + N, 'PUT', H.result_put), (r'/results/([\w.-]+)/' + N, 'GET', H.result_get),
    (r'/results/([\w.-]+)/' + N, 'DELETE', H.result_del),
    (r'/next', 'POST', H.next_unit), (r'/bye', 'POST', H.bye),
    (r'/units/([\w.-]+)/progress', 'POST', H.unit_progress),
    (r'/units/([\w.-]+)/done', 'POST', H.unit_done), (r'/units/([\w.-]+)/fail', 'POST', H.unit_fail),
]

if __name__ == '__main__':
    s = ThreadingHTTPServer(('0.0.0.0', PORT), H)
    s.daemon_threads = True
    print('shutter server :%d  work=%s' % (PORT, WORK), flush=True)
    # docker stop: container PID 1 without a handler ignores SIGTERM and waits out the stop timeout. The DB commits per
    # statement, so exiting right away is fine
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    threading.Thread(target=cleanup_loop, daemon=True).start()
    s.serve_forever()
