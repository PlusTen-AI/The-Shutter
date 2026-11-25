#!/usr/bin/env python3
"""shutter-mac: browser UI server + result sync on the machine you work from. macOS: standard library only (stock
python3 3.9). Linux / WSL: python3 + curl, and Pillow + PyAV in .venv for crops and thumbnails (start.sh uses it).

  python3 mac/mac.py   → http://localhost:8310
The UI forwards the job API as /api/srv/... to the server (C) unchanged. The Hermes skill can call the server API directly.
Source assets and results live in the storage location (settings):  <storage>/assets/<sha256><ext> (shared),
<storage>/projects/<project>/jobs/<job>/<file> (each result as soon as its unit ends, §8), saved project items in
<storage>/projects/<project>/{scenes,videos}/. Jobs from before projects: <storage>/jobs/<job>/.
Without the storage location (external drive unplugged) results wait on the node and are saved once it is back (#21);
new jobs cannot be submitted meanwhile.

Asset codes (production ch. 2 file names):  <storage>/characters/FL01/{card.json, FL01_ref_v1.png, FL01_angle_34L_v1.png …}
                                            <storage>/locations/L01/{card.json, L01_anchor_v1.png, L01_<view>_v1.png …}
The master (ref) / anchor is the one with the highest version number. Derives always start from the master. Adopting
copies to the next version name (never overwrites).
"""
import hashlib, json, mimetypes, os, re, shutil, subprocess, threading, time
import script   # 화 파일 (docs/script.md)
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.environ.get('SHUTTER_MAC_SETTINGS', os.path.join(HERE, 'settings.json'))
PORT = int(os.environ.get('SHUTTER_MAC_PORT', '8310'))
DEFAULTS = {'server': 'http://127.0.0.1:8300', 'storage': '', 'project': '', 'lang': 'ko', 'notify': True,
            'defaults': {}, 'record_days': 30}   # settings.json (template settings.example.json). defaults = per-model job defaults (UI)
CHUNK = 1 << 20
JOBS = 'jobs'   # job results arrive in <storage>/jobs/<job>/. Adopting copies into the code folder, deletes the job folder
status = {'server_ok': False, 'waiting': 0, 'error': None, 'last_sync': None, 'workers': {}}
UI = os.path.join(HERE, 'ui')
PROJECT_RE = r'[^./\\:\x00-\x1f][^/\\:\x00-\x1f]{0,59}'   # folder name: no slash, no leading dot
LIB_KINDS = {'scenes': '.png', 'videos': '.mp4'}   # saved project items


def settings():
    try:
        with open(SETTINGS) as f:
            return dict(DEFAULTS, **json.load(f))
    except (OSError, ValueError):
        return dict(DEFAULTS)


def storage_ok(s):
    # never create the storage folder. With the external drive unplugged, the same path must not appear on the Mac disk
    return bool(s['storage']) and os.path.isdir(s['storage'])


class Unreachable(Exception):
    pass


CURL = shutil.which('curl') or '/usr/bin/curl'


def srv(method, path, body=None, upload=None, out=None, timeout=30):
    """Server call → (HTTP code, body). python lacks the macOS 'Local Network' permission, so LAN connections go through
    curl (macOS, Linux and WSL ship it).  upload: send this file as the body.  out: receive the body into this file."""
    cmd = [CURL, '-sS', '-m', str(timeout), '-X', method, '-w', '\n%{http_code}', '-o', out or '-']
    if method == 'HEAD':
        cmd[cmd.index('-X'):cmd.index('-X') + 2] = ['-I']
    if upload:
        cmd += ['-T', upload]
    elif body is not None:
        cmd += ['-H', 'Content-Type: application/json', '--data-binary', '@-']
    try:
        r = subprocess.run(cmd + [settings()['server'] + path], input=body, capture_output=True)
    except FileNotFoundError:
        raise Unreachable('curl 없음 — curl 을 설치해야 서버에 연결됨')
    text, _, code = r.stdout.rpartition(b'\n')
    if r.returncode or code == b'000':
        raise Unreachable(r.stderr.decode(errors='replace').strip() or 'curl exit %d' % r.returncode)
    return int(code), text


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


# ---- pictures on the machine the UI runs on (crops, thumbnails of unsaved candidates): Pillow / PyAV when installed
# (Linux, WSL: pip install pillow av into .venv — start.sh runs it), else the macOS tools sips / qlmanage. With neither,
# a crop fails with a message and a candidate stays whole instead of becoming a thumbnail ----
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import av
except ImportError:
    av = None
SIPS, QLMANAGE = shutil.which('sips'), shutil.which('qlmanage')


def img_crop(src, box, dst):
    """box = [x0, y0, x1, y1] as fractions of the picture → png dst (at least 16 px a side)."""
    if Image:
        with Image.open(src) as im:
            w, h = im.size
            x, y = round(box[0] * w), round(box[1] * h)
            im.crop((x, y, min(w, x + max(16, round((box[2] - box[0]) * w))), min(h, y + max(16, round((box[3] - box[1]) * h))))).save(dst, 'PNG')
        return
    if not SIPS:
        raise Bad('그림 자르기에 Pillow 필요 (.venv 에 pip install pillow av)')
    w, h = sips_size(src)
    x, y = round(box[0] * w), round(box[1] * h)
    cw, ch = min(max(16, round((box[2] - box[0]) * w)), w - x), min(max(16, round((box[3] - box[1]) * h)), h - y)
    # sips crops wrongly at the edges (400x200 grid): offset "0 0" takes the centre, and at x 0 a box reaching the
    # bottom comes back uncropped. A 2 px border first (padding is centred) keeps every offset off 0 and off the edges.
    pad = dst + '.pad.png'
    subprocess.run([SIPS, '-s', 'format', 'png', '-p', str(h + 4), str(w + 4), src, '--out', pad], capture_output=True)
    r = subprocess.run([SIPS, '-s', 'format', 'png', '-c', str(ch), str(cw), '--cropOffset', str(y + 2), str(x + 2), pad,
                        '--out', dst], capture_output=True)
    if os.path.isfile(pad):
        os.remove(pad)
    if r.returncode or not os.path.isfile(dst) or sips_size(dst) != (cw, ch):
        if os.path.isfile(dst):
            os.remove(dst)
        raise Bad('자르기 실패 (sips) — .venv 에 pip install pillow av 권장: %s' % r.stderr.decode(errors='replace')[:200])


def sips_size(path):
    out = subprocess.run([SIPS, '-g', 'pixelWidth', '-g', 'pixelHeight', path], capture_output=True, text=True).stdout
    return tuple(int(re.search(r'%s: (\d+)' % k, out).group(1)) for k in ('pixelWidth', 'pixelHeight'))


def img_thumb(src, dst, size=256):
    """A small jpg of a picture → True when it was made."""
    if Image:
        with Image.open(src) as im:
            im = im.convert('RGB')
            im.thumbnail((size, size))
            im.save(dst, 'JPEG', quality=85)
    elif SIPS:
        subprocess.run([SIPS, '-Z', str(size), '-s', 'format', 'jpeg', src, '--out', dst], capture_output=True)
    return os.path.isfile(dst)


def video_thumb(src, dst, size=256):
    """A small jpg of a video (its middle frame; Quick Look's frame on macOS without PyAV) → True when it was made."""
    if av and Image:
        try:
            with av.open(src) as c:
                if c.duration:
                    c.seek(c.duration // 2)
                im = next(c.decode(video=0)).to_image()
            im.thumbnail((size, size))
            im.save(dst, 'JPEG', quality=85)
        except (av.error.FFmpegError, StopIteration, OSError):
            pass
    elif QLMANAGE and SIPS:
        d = os.path.dirname(dst)
        subprocess.run([QLMANAGE, '-t', '-s', str(size), '-o', d, src], capture_output=True)
        png = os.path.join(d, os.path.basename(src) + '.png')
        if os.path.isfile(png):
            subprocess.run([SIPS, '-s', 'format', 'jpeg', png, '--out', dst], capture_output=True)
            os.remove(png)
    return os.path.isfile(dst)


def send_to_node(path):
    """Send a Mac file to the node asset cache (skipped if the node already has it) → (sha, sent)"""
    sha = sha_file(path)
    sent = srv('HEAD', '/assets/' + sha)[0] != 200
    if sent:
        code, b = srv('PUT', '/assets/' + sha, upload=path, timeout=3600)
        if code != 200:
            raise IOError(b.decode(errors='replace'))
    return sha, sent


# ---- projects (boss, session #10): characters/, locations/, assets/ shared at the storage root; per project
# projects/<name>/ holds job results, scenes and videos. The current project is in the settings ----
def projects(s):
    d = os.path.join(s['storage'], 'projects')
    return sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)) and not n.startswith('.')) \
        if storage_ok(s) and os.path.isdir(d) else []


def new_project(s, name):
    name = str(name or '').strip()
    if not re.fullmatch(PROJECT_RE, name):
        raise Bad('프로젝트 이름이 맞지 않음 (/ : 와 앞 . 은 안 됨)')
    if not storage_ok(s):
        raise Bad('저장 위치 없음')
    os.makedirs(os.path.join(s['storage'], 'projects', name), exist_ok=True)
    return name


def job_dir(s, job, project=None):
    """Where a job's results go on the Mac: <storage>/projects/<project>/jobs/<job>, or <storage>/jobs/<job> for jobs
    without a project (before session #10)."""
    if project:
        return os.path.join(s['storage'], 'projects', project, 'jobs', job)
    return os.path.join(s['storage'], JOBS, job)


def find_job_dir(s, job):
    """A job's result folder on the Mac, whichever project it belongs to (None if not there)."""
    if not re.fullmatch(r'[\w.-]+', job or '') or not storage_ok(s):
        return None
    for d in [job_dir(s, job)] + [job_dir(s, job, p) for p in projects(s)]:
        if os.path.isdir(d):
            return d
    return None


def submit(s, job):
    """Submit a job to the server under the current project. No storage location → no new jobs (results would have
    nowhere to go)."""
    if not storage_ok(s):
        raise Bad('저장 위치 없음 · 새 작업은 연결된 뒤에')
    job = dict(job)
    if s.get('project'):
        job['project'] = s['project']
    return srv('POST', '/jobs', json.dumps(job).encode())


def retry(s, b):
    """"이어서 하기": requeue the failed units. Inputs cleared from the node cache meanwhile are resent first."""
    job = str(b.get('job', ''))
    if not re.fullmatch(r'[\w.-]+', job):
        raise Bad('작업 이름이 맞지 않음')
    code, body = srv('POST', '/jobs/%s/retry' % job, b'{}')
    if code == 409:
        missing = json.loads(body).get('missing') or []
        j = json.loads(srv('GET', '/jobs/' + job)[1])
        for name in missing:
            on_node(s, j['inputs'][name])
        code, body = srv('POST', '/jobs/%s/retry' % job, b'{}')
    return code, body


# ---- 보관함: shared characters / locations (code folders) and the current project's saved scenes / videos ----
def project_items(s, project):
    out = []
    root = os.path.join(s['storage'], 'projects', project)
    for kind, ext in LIB_KINDS.items():
        d = os.path.join(root, kind)
        for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if n.endswith(ext):
                meta = read_json(os.path.join(d, n[:-len(ext)] + '.json'), {})
                out.append({'kind': kind, 'name': n, 'path': 'projects/%s/%s/%s' % (project, kind, n),
                            'meta': meta, 'mtime': os.path.getmtime(os.path.join(d, n))})
    return out


def code_users(s, code):
    """Projects whose saved items were made with this character / location code (shown before deleting it)."""
    return [p for p in projects(s) if any(code in str(i['meta'].get('asset_code', '')).split()
                                         for i in project_items(s, p))]


def library(s, scope):
    if not storage_ok(s):
        return {'error': '저장 위치 없음'}
    if scope == 'shared':
        out = {cat: list_codes(s, cat) for cat in CODE_RE}
        for cat in CODE_RE:
            for c in out[cat]:
                c['users'] = code_users(s, c['code'])
        return out
    return {'items': project_items(s, s['project'])} if s.get('project') else {'items': []}


def save_item(s, b):
    """"보관함에 저장": copy one result of a job into the current project (scenes / videos) under a free name, with its
    settings json (result json + job id)."""
    job, name = str(b.get('job', '')), str(b.get('name', ''))
    if not s.get('project'):
        raise Bad('프로젝트 없음')
    kind = 'videos' if name.endswith('.mp4') else 'scenes' if name.endswith('.png') else None
    src_dir = find_job_dir(s, job)
    if not kind or not src_dir or not re.fullmatch(r'[\w.-]+', name) or not os.path.isfile(os.path.join(src_dir, name)):
        raise Bad('결과가 맥에 없음: %s/%s' % (job, name))
    d = os.path.join(s['storage'], 'projects', s['project'], kind)
    os.makedirs(d, exist_ok=True)
    stem = re.sub(r'[^\w.-]+', '_', str(b.get('title') or '').strip()) or '%s_%s' % (job, name[:-4])
    ext, k = LIB_KINDS[kind], 1
    dest = os.path.join(d, stem + ext)
    while os.path.exists(dest):
        k += 1
        dest = os.path.join(d, '%s_%d%s' % (stem, k, ext))
    shutil.copyfile(os.path.join(src_dir, name), dest)
    meta = read_json(os.path.join(src_dir, name[:-4] + '.json'), {})
    code, body = srv('GET', '/jobs/' + job)
    j = json.loads(body) if code == 200 else {}
    meta.update(job=job, result=name, type=j.get('type'), asset_code=(j.get('opts') or {}).get('asset_code', ''))
    write_json(dest[:-len(ext)] + '.json', meta)
    return os.path.basename(dest)


def delete_item(s, b):
    """Delete one 보관함 item: a saved project scene / video, or a whole character / location code folder."""
    if not storage_ok(s):
        raise Bad('저장 위치 없음')
    if b.get('code'):
        code = str(b['code'])
        d = code_dir(s, code)
        if base_of(code) == code and cat_of(code) == 'characters':   # a character goes with its outfits
            for n in os.listdir(os.path.dirname(d)):
                if n.startswith(code + '-') and re.fullmatch(CODE_RE['characters'], n):
                    shutil.rmtree(os.path.join(os.path.dirname(d), n))
        shutil.rmtree(d)
        return
    kind, name = str(b.get('kind', '')), str(b.get('name', ''))
    if kind not in LIB_KINDS or not re.fullmatch(r'[\w.-]+', name) or not name.endswith(LIB_KINDS[kind]):
        raise Bad('항목 이름이 맞지 않음')
    path = os.path.join(s['storage'], 'projects', s['project'], kind, name)
    if not os.path.isfile(path):
        raise Bad('항목 없음')
    os.remove(path)
    if os.path.isfile(path[:-4] + '.json'):
        os.remove(path[:-4] + '.json')


# ---- asset codes (character FL01, place L01) ----
# a character code, or one of its outfits: <code>-<style> (wardrobe, boss session #13 — style in English, else a number)
CODE_RE = {'characters': r'[FM][LSE]\d{2,3}(?:-[a-z0-9]{1,20})?', 'locations': r'L\d{2,3}', 'props': r'P\d{2,3}'}   # props: 기능 10
MASTER = {'characters': 'ref', 'locations': 'anchor', 'props': 'ref'}
CARD = {'characters': ('who', 'face', 'hair', 'body', 'hands', 'clothing', 'name', 'age', 'extra', 'voice'),   # the card's text
        'locations': ('kind', 'space', 'scale', 'body', 'name'), 'props': ('name', 'what', 'desc')}
# card keys passed as options, per job. who is the master's first line ('a beautiful Korean woman who …'), so only the
# master gets it (the derive template says 'Keep the {who}', which needs a noun → woman/man from gender)
CARD_OPTS = {'char.ref': ('who', 'face', 'hair', 'body', 'hands', 'clothing', 'gender', 'child'), 'char.from_photo': ('clothing', 'gender', 'child'),
             'char.derive': ('gender', 'child'), 'place.anchor': CARD['locations'], 'place.derive': (),
             'char.sheet': ('gender', 'child', 'clothing'), 'video.swap': ('gender', 'child', 'clothing'),
             'prop.ref': ('what', 'desc'), 'prop.derive': ('what',)}
JOB_CAT = {'char.ref': 'characters', 'char.from_photo': 'characters', 'char.derive': 'characters',
           'place.anchor': 'locations', 'place.derive': 'locations', 'char.sheet': 'characters', 'video.swap': 'characters',
           'prop.ref': 'props', 'prop.derive': 'props'}
FIXED_KIND = {'char.ref': 'ref', 'char.from_photo': 'ref', 'place.anchor': 'anchor', 'char.sheet': 'sheet', 'video.swap': 'swap',
              'prop.ref': 'ref'}
KIND_RE = r'[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*'


class Bad(Exception):
    pass


def cat_of(code):
    for cat, rx in CODE_RE.items():
        if re.fullmatch(rx, code or ''):
            return cat
    raise Bad('코드 형식이 맞지 않음: %r (캐릭터 FL01·MS02, 배경 L01, 사물 P01)' % code)


def code_dir(s, code, must=True):
    if not storage_ok(s):
        raise Bad('저장 위치 없음')
    d = os.path.join(s['storage'], cat_of(code), code)
    if must and not os.path.isdir(d):
        raise Bad('코드 없음: %s' % code)
    return d


def images(d, code):
    """Images in a code folder → [(kind, version, file name)]"""
    out = []
    for n in os.listdir(d):
        m = re.fullmatch(re.escape(code) + r'_(%s)_v(\d+)\.png' % KIND_RE, n)
        if m:
            out.append((m.group(1), int(m.group(2)), n))
    return sorted(out)


def latest(d, code, kind):
    v = [x for x in images(d, code) if x[0] == kind]
    return os.path.join(d, max(v)[2]) if v else None


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    with open(path + '.part', 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(path + '.part', path)


def list_codes(s, cat):
    root = os.path.join(s['storage'], cat)
    out = []
    for code in sorted(os.listdir(root)) if storage_ok(s) and os.path.isdir(root) else []:
        d = os.path.join(root, code)
        if not (re.fullmatch(CODE_RE[cat], code) and os.path.isdir(d)):
            continue
        im = images(d, code)
        m = latest(d, code, MASTER[cat])
        out.append({'code': code, 'card': read_json(os.path.join(d, 'card.json'), {}),
                    'images': [{'kind': k, 'v': v, 'name': n} for k, v, n in im],
                    'master': os.path.basename(m) if m else None})
    return out


def base_of(code):
    """FL01-suit → FL01 (an outfit's character); other codes as they are."""
    return code.split('-')[0]


def read_card(s, code):
    """A code's card. An outfit (FL01-suit) has only its clothing (+ of, style): the character's card fills the rest."""
    card = read_json(os.path.join(code_dir(s, code), 'card.json'), {})
    if base_of(code) != code:
        card = dict(read_json(os.path.join(code_dir(s, base_of(code)), 'card.json'), {}), **card)
    return card


def save_card(s, b):
    code = b.get('code', '').strip()
    cat = cat_of(code)
    if base_of(code) != code:   # an outfit: its clothing only, the rest stays the character's
        code_dir(s, base_of(code))
        d = code_dir(s, code, must=False)
        os.makedirs(d, exist_ok=True)
        card = {'code': code, 'of': base_of(code), 'style': code.split('-', 1)[1],
                'clothing': str((b.get('card') or {}).get('clothing', '')).strip()}
        write_json(os.path.join(d, 'card.json'), card)
        return card
    d = code_dir(s, code, must=False)
    if not os.path.isdir(os.path.dirname(d)):
        os.mkdir(os.path.dirname(d))
    if not os.path.isdir(d):
        os.mkdir(d)
    card = {k: str(v).strip() for k, v in (b.get('card') or {}).items() if k in CARD[cat] + ('notes',)}
    card['code'] = code
    old = read_json(os.path.join(d, 'card.json'), {})
    if old.get('views'):   # a location's view names (set when its views are registered)
        card['views'] = old['views']
    if old.get('voice') and 'voice' not in card:   # the voice sentence is set in its own block (캐릭터 › 목소리)
        card['voice'] = old['voice']
    if cat == 'characters':
        card['gender'] = 'female' if code[0] == 'F' else 'male'   # gender is the code's first letter
        m = re.match(r'\d+', card.get('age', ''))
        card['child'] = bool(m and int(m.group()) < 13)   # 기능 13: girl / boy in every template (e.g. a 7-year-old)
        if not card.get('who'):   # the master's first line from the UI card (age, one-line type)
            card['who'] = 'a Korean %s who looks %s' % (('girl' if card['child'] else 'woman') if code[0] == 'F' else ('boy' if card['child'] else 'man'),
                                                     '%s years old' % card['age'] if card.get('age') else 'in the twenties')
            if card.get('extra'):
                card['who'] += ', ' + card['extra']
    write_json(os.path.join(d, 'card.json'), card)
    return card


def next_outfit(s, code, style):
    """An outfit code for character code: <code>-<style> when given (English lowercase / digits, must be free), else the next
    number from 2 (the character's own folder is outfit 1)."""
    if cat_of(code) != 'characters' or base_of(code) != code:
        raise Bad('캐릭터 코드가 아님: %s' % code)
    root = os.path.dirname(code_dir(s, code))
    style = str(style or '').strip().lower()
    if style:
        if not re.fullmatch(r'[a-z0-9]{1,20}', style):
            raise Bad('스타일은 영문 소문자·숫자 20자까지')
        if os.path.exists(os.path.join(root, '%s-%s' % (code, style))):
            raise Bad('이미 있는 스타일: %s-%s' % (code, style))
        return '%s-%s' % (code, style)
    used = [int(n.split('-')[1]) for n in os.listdir(root) if re.fullmatch(re.escape(code) + r'-\d+', n)]
    return '%s-%d' % (code, max(used + [1]) + 1)


def next_code(s, gender, rank):
    """Next free character code: {F/M}{L lead / S supporting / E extra}{number} (production ch. 2), e.g. FL03. gender 'L' =
    the next location code (L01, L02 …)."""
    if gender in ('L', 'P'):   # the next location / object code
        root = os.path.join(s['storage'], 'locations' if gender == 'L' else 'props')
        used = [int(n[1:]) for n in (os.listdir(root) if storage_ok(s) and os.path.isdir(root) else []) if re.fullmatch(gender + r'\d{2,3}', n)]
        return '%s%02d' % (gender, max(used or [0]) + 1)
    if gender not in 'FM' or rank not in 'LSE' or len(gender) != 1 or len(rank) != 1:
        raise Bad('성별 F/M, 등급 L/S/E')
    root = os.path.join(s['storage'], 'characters')
    used = [int(n[2:]) for n in (os.listdir(root) if storage_ok(s) and os.path.isdir(root) else [])
            if re.fullmatch(gender + rank + r'\d{2,3}', n)]
    return '%s%s%02d' % (gender, rank, max(used or [0]) + 1)


# ---- character workflow session (the UI's state for one character until 등록 / 닫기): characters/<code>/session.json
def session_path(s, code):
    return os.path.join(code_dir(s, code), 'session.json')


def sessions(s, cat='characters'):
    root = os.path.join(s['storage'], cat if cat in CODE_RE else 'characters')
    return sorted(n for n in (os.listdir(root) if storage_ok(s) and os.path.isdir(root) else [])
                  if os.path.isfile(os.path.join(root, n, 'session.json')))


def thumb_and_drop(path):
    """Keep a small jpg of an unsaved candidate (the job record) and delete the full image (boss, session #10). Without
    a way to make the thumbnail the image stays whole."""
    if img_thumb(path, os.path.join(os.path.dirname(path), '_thumb_' + os.path.basename(path)[:-4] + '.jpg')):
        os.remove(path)


def finish_character(s, b, register):
    """End a character or location session. register: the ● base master → <code>_ref_vN (a location's anchor →
    <code>_anchor_vN), the sheet job → sheet + panels + expressions, a location's views → <code>_<view kind>_vN, ☆ saved
    candidates → <code>_cand_vN (on 닫기 too). Then every candidate image left in the session's job folders becomes a
    thumbnail (settings and seeds stay in the json, so any can be made again), the jobs are closed, and the session file
    goes. Without register (닫기) a code folder left with only its card is removed."""
    code = str(b.get('code', ''))
    d = code_dir(s, code)
    ses = read_json(session_path(s, code), {})
    saved = []
    base = ses.get('base') or {}
    master = MASTER[cat_of(code)]
    if register:
        if not base.get('job') and not latest(d, code, master):
            raise Bad('● 선택 없음')
        if base.get('job'):
            saved.append(save_version(d, code, master, os.path.join(find_job_dir(s, base['job']) or '', base['name']),
                                      base['job'], base['name']))
        names = {}
        for v in ses.get('views') or []:   # location views: one image each, + the view's short name (the script's `L01 소파`)
            jd = v.get('job') and find_job_dir(s, v['job'])
            n = next((x for x in sorted(os.listdir(jd)) if x.endswith('.png')), None) if jd else None
            if n and re.fullmatch(KIND_RE, str(v.get('kind', ''))):
                saved.append(save_version(d, code, v['kind'], os.path.join(jd, n), v['job'], n))
                if str(v.get('name') or '').strip():
                    names[v['kind']] = str(v['name']).strip()
        if names:
            card = read_json(os.path.join(d, 'card.json'), {})
            card['views'] = dict(card.get('views') or {}, **names)
            write_json(os.path.join(d, 'card.json'), card)
        sj = ses.get('sheetJob')
        jd = sj and find_job_dir(s, sj)
        if jd:
            for n in sorted(os.listdir(jd)):
                if n == 'sheet.png' or (n.endswith('.png') and n[:3] in ('sh_', 'ex_')):
                    saved.append(save_version(d, code, 'sheet' if n == 'sheet.png' else n[:-4], os.path.join(jd, n), sj, n))
    for c in ses.get('saved') or []:   # ☆ candidates stay on 닫기 too (design §11 후보: only unsaved ones go)
        src = os.path.join(find_job_dir(s, c.get('job', '')) or '', c.get('name', ''))
        if not (register and c == base) and os.path.isfile(src):
            saved.append(save_version(d, code, 'cand', src, c['job'], c['name']))
    jobs = list(ses.get('masterJobs') or []) + ([ses['sheetJob']] if ses.get('sheetJob') else []) + list(ses.get('oldSheetJobs') or []) + \
        [v['job'] for v in ses.get('views') or [] if v.get('job')] + list(ses.get('oldViewJobs') or [])
    for job in jobs:
        jd = find_job_dir(s, job)
        for n in sorted(os.listdir(jd)) if jd else []:
            if n.endswith('.png'):
                thumb_and_drop(os.path.join(jd, n))
        try:
            srv('POST', '/jobs/%s/close' % job, b'{}')
        except Unreachable:
            pass   # the server closes it by itself later (done jobs only leave the list)
    if os.path.isfile(session_path(s, code)):
        os.remove(session_path(s, code))
    if not register and not images(d, code):
        shutil.rmtree(d)   # nothing was registered: the card-only folder made by 시작 goes
    return saved


def code_job(s, b):
    """Submit a job by code: the card text as options, the master/anchor as image_1, sent to the server → (code, body)"""
    t, code = b.get('type'), b.get('code', '').strip()
    if t not in JOB_CAT or cat_of(code) != JOB_CAT[t]:
        raise Bad('%s 작업에 %s 코드는 맞지 않음' % (t, code))
    cat, d = JOB_CAT[t], code_dir(s, code)
    card = read_card(s, code)
    opts = {k: card[k] for k in CARD_OPTS[t] if str(card.get(k, '')).strip()}
    opts.update({k: v for k, v in (b.get('options') or {}).items() if v not in ('', None)})
    kind = FIXED_KIND.get(t) or b.get('file_kind', '')
    if not re.fullmatch(KIND_RE, kind):
        raise Bad('파일 종류 이름 필요 (영문·숫자·_)')
    opts.update(asset_code=code, asset_kind=kind)   # used on adopt. Templates ignore unknown keys
    inputs = dict(b.get('inputs') or {})
    need = {'char.ref': 'who', 'place.anchor': 'space', 'place.derive': 'view', 'prop.ref': 'desc', 'prop.derive': 'view'}.get(t)
    if need and not opts.get(need) and not opts.get('prompt'):
        raise Bad('%s 비어 있음' % need)
    if t == 'char.from_photo' and not inputs.get('image_1'):
        raise Bad('사진 없음')
    if t == 'video.swap':
        if not inputs.get('video'):
            raise Bad('영상 없음')
        bc = str(b.get('background_code') or '').strip()
        if bc:   # a shared location's anchor as the new background (UI 배경 › 이미지)
            if cat_of(bc) != 'locations':
                raise Bad('%s 는 배경 코드가 아님' % bc)
            anchor = latest(code_dir(s, bc), bc, MASTER['locations'])
            if not anchor:
                raise Bad('%s 의 대표 이미지 없음' % bc)
            inputs['background'] = send_to_node(anchor)[0]
        if not opts.get('desc'):   # person description for the H3 prompt: the card's face and hair sentences
            opts['desc'] = ' '.join(str(card[k]).strip().rstrip('.') + '.' for k in ('face', 'hair') if str(card.get(k, '')).strip())
    if t == 'char.derive' and opts.get('kind') == 'outfit':   # wardrobe: the new outfit on the character's own master
        if b.get('outfit_image'):
            inputs['image_2'] = send_to_node(storage_file(s, b['outfit_image']))[0]
            opts['outfit_pic'] = True
        if not opts.get('clothing') and not opts.get('outfit_pic') and not opts.get('prompt'):
            raise Bad('새 옷 설명이나 옷 사진 필요')
        d, code = code_dir(s, base_of(code)), base_of(code)
    if t in ('char.derive', 'place.derive', 'prop.derive', 'char.sheet', 'video.swap'):
        m = latest(d, code, MASTER[cat])
        base = b.get('master') or {}
        if base.get('job'):   # the character workflow's ● base candidate, not registered yet
            m = os.path.join(find_job_dir(s, str(base['job'])) or '', str(base.get('name', '')))
            if not re.fullmatch(r'[\w.-]+\.png', str(base.get('name', ''))) or not os.path.isfile(m):
                raise Bad('선택한 대표 이미지가 맥에 없음')
        if not m:
            raise Bad('%s 의 대표 이미지 없음 · 먼저 등록 필요' % code)
        inputs['image_1'] = send_to_node(m)[0]
        full, sheet = latest(d, code, 'sh_full_front'), latest(d, code, 'sheet')
        if t == 'video.swap' and full and sheet:   # a sheet pins the whole outfit (length, shoes): boundary edits and the
            inputs['master'] = inputs['image_1']       # face check use its full-body front panel, H3 gets the sheet,
            inputs['image_1'] = send_to_node(full)[0]   # the face pass takes its identity from the master + head panels
            inputs['sheet'] = send_to_node(sheet)[0]
    code_b = str(b.get('code_b') or opts.pop('code_b', '') or '').strip()   # the UI sends it among the options
    if t == 'video.swap' and code_b:   # two people (#11): the second character's master, sheet and full-body front panel
        if cat_of(code_b) != 'characters':
            raise Bad('%s 는 캐릭터 코드가 아님' % code_b)
        db = code_dir(s, code_b)
        mb, fb, sb = (latest(db, code_b, k) for k in (MASTER['characters'], 'sh_full_front', 'sheet'))
        if not (mb and fb and sb and 'sheet' in inputs):
            raise Bad('두 사람 교체는 두 캐릭터 모두 캐릭터 시트 필요')
        inputs.update(master_b=send_to_node(mb)[0], image_b=send_to_node(fb)[0], sheet_b=send_to_node(sb)[0])
        cb = read_card(s, code_b)
        opts.setdefault('gender_b', cb.get('gender', 'male' if code_b[0] == 'M' else 'female'))
        opts.setdefault('desc_b', ' '.join(str(cb[k]).strip().rstrip('.') + '.' for k in ('face', 'hair') if str(cb.get(k, '')).strip()))
    return submit(s, {'type': t, 'inputs': inputs, 'options': opts})


def r2v_job(s, b):
    """Video (video.r2v) pass-1 preview: the latest sheets of the character codes (<code>_sheet_vN.png) as image_1..N
    (<Picture 1..N>), the shot prompts, seed, reference size (max faithful / match fast), uploaded start / end frames per
    shot (first / last: lists of sha, index = shot), key frames per shot (keys: [[{sha, frame}]] — a multi-shot's shot
    first frames: guides at those frames and pictures after the sheets) and optional width / height / frames."""
    codes = str(b.get('codes', '')).split()
    shots = [x.strip() for x in b.get('shots') or [] if str(x).strip()]
    if not codes or not shots:
        raise Bad('캐릭터 코드와 컷 프롬프트 필요')
    inputs = {}
    for i, code in enumerate(codes):
        if cat_of(code) != 'characters':
            raise Bad('캐릭터 코드가 아님: %s' % code)
        sheet = latest(code_dir(s, code), code, 'sheet')
        if not sheet:
            raise Bad('%s 의 캐릭터 시트 없음' % code)
        inputs['image_%d' % (i + 1)] = send_to_node(sheet)[0]
    for key in ('first', 'last'):   # uploaded start / end frames per shot (list index = shot, '' = none)
        for k, sha in enumerate(b.get(key) or []):
            if k >= len(shots):
                raise Bad('%s 프레임이 컷 수보다 많음' % {'first': '첫', 'last': '끝'}[key])
            if sha:
                inputs['%s_%d' % (key, k)] = sha
    keys = [list(kk or []) for kk in b.get('keys') or []][:len(shots)]
    for k, kk in enumerate(keys):
        for j, x in enumerate(kk):
            inputs['key_%d_%d' % (k, j)] = x['sha']
    rs = b.get('ref_size') or 'max'
    opts = {'shots': shots, 'stage': 'preview', 'ref_size': rs, 'asset_code': ' '.join(codes),   # shown in the job list
            'asset_kind': str(b.get('label') or '') or '참조 ' + {'max': '원래 크기', 'match': '축소'}.get(rs, rs)}
    for k in ('width', 'height', 'frames'):   # optional (API), defaults in the server
        if str(b.get(k, '')).strip():
            opts[k] = int(b[k])
    if any(keys):
        opts['keys'] = [[int(x['frame']) for x in kk] for kk in keys]
    if str(b.get('seed', '')).strip():
        opts['seed'] = b['seed']
    return submit(s, {'type': 'video.r2v', 'inputs': inputs, 'options': opts})


def i2v_job(s, b):
    """Video (video.i2v = LTX-2.5, or b type video.pih3 = pi-H3, the drama H3 path, first frame optional): the uploaded
    first frame (inputs image) + options as typed, and up to two character codes → their latest master (+ sheet) as
    master / sheet (master_b / sheet_b): the face pass after the shot (drama LTX path, boss session #8). Without codes
    the shot is left as the model made it."""
    inputs, opts = dict(b.get('inputs') or {}), dict(b.get('options') or {})
    codes = str(b.get('codes', '')).split()
    if len(codes) > 2:
        raise Bad('캐릭터 코드는 둘까지')
    for code, sfx in zip(codes, ('', '_b')):
        if cat_of(code) != 'characters':
            raise Bad('캐릭터 코드가 아님: %s' % code)
        d = code_dir(s, code)
        m, sheet = latest(d, code, MASTER['characters']), latest(d, code, 'sheet')
        if not m:
            raise Bad('%s 의 대표 이미지 없음' % code)
        inputs['master' + sfx] = send_to_node(m)[0]
        if sheet and 'sheet' + sfx not in inputs:   # a sheet sent by the caller wins
            inputs['sheet' + sfx] = send_to_node(sheet)[0]
    if codes:
        opts.setdefault('asset_code', ' '.join(codes))   # shown in the job list
    t = b.get('type') or 'video.i2v'
    if t not in ('video.i2v', 'video.pih3', 'video.face'):
        raise Bad('작업 종류가 맞지 않음: %s' % t)
    return submit(s, {'type': t, 'inputs': inputs, 'options': opts})


def storage_file(s, rel):
    """A file under the storage location by its relative path (upload, library scene, location anchor) → absolute path."""
    root = os.path.realpath(s['storage'])
    p = os.path.realpath(os.path.join(root, str(rel)))
    if not storage_ok(s) or not p.startswith(root + os.sep) or not os.path.isfile(p):
        raise Bad('파일이 맥에 없음: %s' % rel)
    return p


VIDEO_TYPES = {'ltx25': 'video.i2v', 'pih3': 'video.pih3', 'h3': 'video.r2v'}


# ---- 목소리 (기능 17, docs/drama.md §13.6): a character's reference voice (Qwen3-TTS VoiceDesign candidates → ●) and the
# voice step of a cut (audio.voice: the generated dialogue in the speakers' own voices, same length)
VOICE_EXT = 'wav|mp3|m4a|flac|ogg|aac'   # a reference voice: a TTS candidate (wav) or the boss's own recording


def voice_files(s, code):
    """A character's registered reference voices, <code>_voice_vN.<ext>, oldest first (an outfit speaks with its
    character's voice)."""
    d = code_dir(s, base_of(code), must=False)
    if not os.path.isdir(d):
        return []
    v = [(int(m.group(1)), n) for n in os.listdir(d) for m in [re.fullmatch(re.escape(base_of(code)) + r'_voice_v(\d+)\.(%s)' % VOICE_EXT, n)] if m]
    return [n for _, n in sorted(v)]


def tts_job(s, b):
    """캐릭터 › 목소리: reference voice candidates (audio.tts) — the voice in words (the card's voice sentence), a
    Korean line, count, seed."""
    code = str(b.get('code', '')).strip()
    if cat_of(code) != 'characters':
        raise Bad('캐릭터 코드가 아님: %s' % code)
    instruct, text = str(b.get('instruct') or '').strip(), str(b.get('text') or '').strip()
    if not instruct or not text:
        raise Bad('목소리 문장과 대사 필요')
    opts = {'instruct': instruct, 'text': text, 'count': max(1, min(int(b.get('count') or 2), 4)),
            'asset_code': base_of(code), 'asset_kind': str(b.get('label') or '목소리')[:80]}
    if str(b.get('seed', '')).strip():
        opts['seed'] = int(b['seed'])
    return submit(s, {'type': 'audio.tts', 'inputs': {}, 'options': opts})


def save_voice(s, b):
    """● on a voice candidate (job, name: voice_<k>.wav of an audio.tts job) or an uploaded recording (asset:
    assets/<sha>.<ext>, file: its name) → <code>_voice_vN.<ext> (never overwrites) + its json."""
    code = str(b.get('code', '')).strip()
    if cat_of(code) != 'characters':
        raise Bad('캐릭터 코드가 아님: %s' % code)
    if b.get('asset'):
        rel = str(b['asset'])
        if not re.fullmatch(r'assets/[0-9a-f]{64}\.(%s)' % VOICE_EXT, rel.lower()):
            raise Bad('목소리 파일은 %s' % VOICE_EXT.replace('|', '·'))
        src, ext, meta = storage_file(s, rel), rel.rsplit('.', 1)[1].lower(), {'asset': rel, 'file': str(b.get('file') or '')[:200]}
    else:
        job, name = (str(b.get(k, '')).strip() for k in ('job', 'name'))
        if not re.fullmatch(r'[\w.-]+', job) or not re.fullmatch(r'voice_\d+\.wav', name):
            raise Bad('코드·작업·파일 이름이 맞지 않음')
        src, ext, meta = os.path.join(find_job_dir(s, job) or '', name), 'wav', {'job': job, 'result': name, 'instruct': b.get('instruct'), 'text': b.get('text')}
    if not os.path.isfile(src):
        raise Bad('파일이 아직 맥에 없음: %s' % os.path.basename(src))
    d = code_dir(s, base_of(code))
    v = max([int(re.search(r'_v(\d+)\.\w+$', n).group(1)) for n in voice_files(s, code)] or [0])
    while True:
        v += 1
        dest = os.path.join(d, '%s_voice_v%d.%s' % (base_of(code), v, ext))
        if os.path.exists(dest[:-len(ext)] + 'json'):
            continue
        try:
            out = open(dest, 'xb')
        except FileExistsError:
            continue
        with out, open(src, 'rb') as f:
            shutil.copyfileobj(f, out, CHUNK)
        break
    write_json(dest[:-len(ext)] + 'json', meta)
    return os.path.basename(dest)


def voice_job(s, b):
    """영상 생성 › 목소리: a cut's final video (job, name on the Mac) + its say lines → audio.voice with each speaking
    character's newest reference voice. turns = the speakers in order (consecutive lines of one speaker are one turn);
    a speaker without a registered voice (a robot, an extra) keeps the generated voice (-1). Nobody has one → skip."""
    job, name = str(b.get('job', '')), str(b.get('name', ''))
    src = os.path.join(find_job_dir(s, job) or '', name)
    if not (re.fullmatch(r'[\w.-]+\.mp4', name) and os.path.isfile(src)):
        raise Bad('컷 결과가 맥에 없음: %s/%s' % (job, name))
    codes = []
    for x in b.get('say') or []:
        c = str((x or {}).get('code', '')).strip()
        if c and (not codes or codes[-1] != c):
            codes.append(c)
    spk, inputs, turns = {}, {}, []
    for c in codes:
        vs = voice_files(s, c) if re.fullmatch(CODE_RE['characters'], c) else []
        if not vs:
            turns.append(-1)
            continue
        if base_of(c) not in spk:
            spk[base_of(c)] = len(spk)
            inputs['voice_%d' % spk[base_of(c)]] = send_to_node(os.path.join(code_dir(s, base_of(c)), vs[-1]))[0]
        turns.append(spk[base_of(c)])
    if not spk:
        return 200, {'skip': 'novoice'}
    inputs['video'] = send_to_node(src)[0]
    opts = {'turns': turns, 'speakers': list(spk), 'asset_code': ' '.join(spk), 'asset_kind': str(b.get('label') or '')[:80]}
    return submit(s, {'type': 'audio.voice', 'inputs': inputs, 'options': opts})


def face_job(s, b):
    """영상 생성 › 마무리: the face pass alone (video.face) on a video under the storage location (an upload) — codes: the
    characters whose faces go on (one or two). label: the job list's title."""
    codes = [str(c) for c in b.get('codes') or [] if str(c).strip()][:2]
    if not codes:
        raise Bad('얼굴 보정은 캐릭터 필요')
    inputs = {'video': send_to_node(storage_file(s, b.get('video')))[0]}
    opts = {'asset_kind': str(b.get('label') or '')[:80]}
    return i2v_job(s, {'type': 'video.face', 'inputs': inputs, 'options': opts, 'codes': ' '.join(codes)})


def video_job(s, b):
    """영상 생성 (one candidate = one job): model ltx25 → video.i2v, pih3 → video.pih3, h3 → video.r2v pass-1 preview.
    first / last: frames as paths under the storage location. codes: the people in the cut (up to two) — H3 takes their
    sheets as references, LTX-2.5 / pi-H3 their faces (face pass) when face is on. orient v | h (768x1344 | 1344x768),
    sec, seed, steps · vsa (pi-H3), ref_size (H3). keys (H3 only): a multi-shot's shot first frames [{rel, at (s)}] —
    then the first frame goes in as the key at 0 too, so the pictures after the sheets are shot 1, 2 … in order.
    label: the job list's title."""
    m = b.get('model')
    if m not in VIDEO_TYPES:
        raise Bad('모델이 맞지 않음: %s' % m)
    codes = [str(c) for c in b.get('codes') or [] if str(c).strip()][:2]
    w, h = (1344, 768) if b.get('orient') == 'h' else (768, 1344)
    sec = float(b.get('sec') or 5)
    fr = {k: send_to_node(storage_file(s, b[k]))[0] for k in ('first', 'last') if b.get(k)}
    label = str(b.get('label') or '')[:80]
    prompt = str(b.get('prompt') or '').strip()
    keys = [{'sha': send_to_node(storage_file(s, x['rel']))[0], 'frame': round(float(x['at']) * 24)} for x in b.get('keys') or []]
    if keys and m != 'h3':
        raise Bad('샷 첫 프레임(멀티샷)은 H3 만')
    if m == 'h3':
        if keys and fr.get('first'):   # shot 1's frame: a picture as well, numbered before the later shots'
            keys = [{'sha': fr.pop('first'), 'frame': 0}] + keys
        return r2v_job(s, {'codes': ' '.join(codes), 'shots': [prompt], 'seed': b.get('seed'), 'ref_size': b.get('ref_size') or 'max',
                           'first': [fr.get('first', '')], 'last': [fr.get('last', '')], 'keys': [keys], 'width': w, 'height': h,
                           'frames': 17 * max(1, round((sec * 24 - 5) / 17)) + 5, 'label': label})
    if m == 'ltx25' and ('first' not in fr or 'last' in fr):
        raise Bad('LTX-2.5 는 첫 프레임만 받음')
    inputs = {k2: fr[k1] for k1, k2 in (('first', 'image'), ('last', 'last')) if k1 in fr}
    face = bool(b.get('face')) and bool(codes)
    opts = {'prompt': prompt, 'width': w, 'height': h, 'sec': sec, 'seed': b.get('seed'), 'face': face,
            'asset_code': ' '.join(codes), 'asset_kind': label}
    if m == 'pih3':
        opts['steps'] = int(b.get('steps') or 2)
        opts['vsa'] = bool(b.get('vsa'))   # block-sparse attention (boss, session #12: selectable)
    opts = {k: v for k, v in opts.items() if v is not None}
    return i2v_job(s, {'type': VIDEO_TYPES[m], 'inputs': inputs, 'options': opts, 'codes': ' '.join(codes) if face else ''})


IMAGE_TYPES = ('image.t2i', 'image.edit', 'scene.compose', 'scene.plate', 'scene.zoom')


def image_job(s, b):
    """이미지 생성 · 편집 · 장면: images = files under the storage location (uploads, library items, job results) →
    image_1..N in order; options as typed (prompt, aspect · mp or width · height, count, seed, steps; the scene
    templates' fields)."""
    t = b.get('type')
    if t not in IMAGE_TYPES:
        raise Bad('작업 종류가 맞지 않음: %s' % t)
    inputs = {'image_%d' % (k + 1): send_to_node(storage_file(s, rel))[0] for k, rel in enumerate(b.get('images') or [])}
    opts = {k: v for k, v in (b.get('options') or {}).items() if v not in ('', None)}
    return submit(s, {'type': t, 'inputs': inputs, 'options': opts})


def crop_image(s, b):
    """A part of a storage image (box = [x0, y0, x1, y1] as fractions) → a new png in <storage>/assets → its path. The
    scene workflow cuts shots out of the seated wide plate this way (scene.zoom)."""
    src = storage_file(s, b.get('rel'))
    box = [min(1.0, max(0.0, float(v))) for v in b.get('box') or []]
    d = os.path.join(s['storage'], 'assets')
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, '.crop-%d.png' % threading.get_ident())
    img_crop(src, box, tmp)
    sha = sha_file(tmp)
    os.replace(tmp, os.path.join(d, sha + '.png'))
    return 'assets/%s.png' % sha


def concat_job(s, b):
    """시퀀스 "한 편으로 내보내기": the cuts' results on the Mac (items [{job, name}], in order) → video.concat."""
    items = b.get('items') or []
    if not items:
        raise Bad('이을 컷이 없음')
    inputs = {}
    for k, it in enumerate(items):
        job, name = str(it.get('job', '')), str(it.get('name', ''))
        src = os.path.join(find_job_dir(s, job) or '', name)
        if not (re.fullmatch(r'[\w.-]+\.mp4', name) and os.path.isfile(src)):
            raise Bad('컷 결과가 맥에 없음: %s/%s' % (job, name))
        inputs['video_%d' % k] = send_to_node(src)[0]
    return submit(s, {'type': 'video.concat', 'inputs': inputs, 'options': {'asset_kind': str(b.get('label') or '')[:80]}})


def mac_video(s, it):
    """A video the Mac has: a job result {job, name} or a file under the storage location {path} → absolute path."""
    if it.get('path'):
        return storage_file(s, it['path'])
    job, name = str(it.get('job', '')), str(it.get('name', ''))
    src = os.path.join(find_job_dir(s, job) or '', name)
    if not (re.fullmatch(r'[\w.-]+', job) and re.fullmatch(r'[\w.-]+\.mp4', name) and os.path.isfile(src)):
        raise Bad('영상이 맥에 없음: %s/%s' % (job, name))
    return src


def edit_job(s, b):
    """시퀀스 내보내기 (design §12 A): sources [{job, name} | {path}] = the generations the cuts use (each once), speech
    [bool per source: dialogue → loudness target, else room tone only down], cuts [{v, a, n, s, sa, part, text}] (v / s:
    source index of picture / sound, s null = silence, n frames at 24 fps, part: output part — a new one after each
    episode break), multi [bool per source: a multi-shot generation — cut edges move onto its shot changes] →
    video.edit. width·height / target optional."""
    srcs, cuts = b.get('sources') or [], b.get('cuts') or []
    if not srcs or not cuts:
        raise Bad('이을 컷이 없음')
    inputs = {'video_%d' % k: send_to_node(mac_video(s, it))[0] for k, it in enumerate(srcs)}
    opts = {'cuts': cuts, 'speech': [bool(x) for x in b.get('speech') or []], 'multi': [bool(x) for x in b.get('multi') or []],
            'asset_kind': str(b.get('label') or '')[:80],
            'asset_code': str(b.get('codes') or '')[:80]}
    for k in ('width', 'height', 'target'):
        if b.get(k) not in (None, ''):
            opts[k] = b[k]
    return submit(s, {'type': 'video.edit', 'inputs': inputs, 'options': opts})


def join_job(s, b):
    """한 편 합치기 (design §12 A): parts [{job, name} | {path}] in order → video.join. breaks: indices of the parts an
    episode ends with (화 끊기 on), flash: white flash seconds at each break."""
    parts = b.get('parts') or []
    if not parts:
        raise Bad('합칠 조각이 없음')
    inputs = {'part_%d' % k: send_to_node(mac_video(s, it))[0] for k, it in enumerate(parts)}
    opts = {'breaks': sorted({int(x) for x in b.get('breaks') or [] if 0 <= int(x) < len(parts) - 1}),
            'flash': float(b.get('flash') or 0), 'asset_kind': str(b.get('label') or '')[:80]}
    return submit(s, {'type': 'video.join', 'inputs': inputs, 'options': opts})


# ---- 대본 → 제작 (design §12 기능 16, docs/script.md): the project's script/ folder → one sequence per scene (its
# cuts as cut docs, its first frames in the sequence doc) and the first-frame jobs with every reference resolved here.
# What runs next (frame ● → cut first frame → video) is the UI's production step (mac/ui/prod.js). ----
SHOT_H = {'ms': 6.2, 'mcu': 4.3, 'cu': 3.0}   # cut box height in face heights: head top … waist / chest / shoulders


def script_load(s):
    """The project's script (docs/script.md) parsed, + made: {scene no: sequence id} of the sequences made from it."""
    if not storage_ok(s) or not s.get('project'):
        raise Bad('저장 위치·프로젝트 없음')
    d = os.path.join(s['storage'], 'projects', s['project'], 'script')
    if not os.path.isdir(d):
        return {'files': [], 'scenes': [], 'errors': [], 'warnings': [], 'made': {}, 'dir': d}
    out = script.load([d])
    out['made'] = {str(q['script']['scene']): q['id'] for q in docs(s, 'sequences') if q and q.get('script')}
    out['dir'] = d
    return out


def rel_of(s, path):
    return os.path.relpath(path, s['storage'])


def ref_path(s, r, frames=None):
    """A script reference (script.ref) → its picture's path under the storage location (the latest version)."""
    k = r['kind']
    if k == 'char':
        d = code_dir(s, r['code'])
        p = latest(d, r['code'], r['pic'])
        if not p:
            raise Bad('%s 그림 없음: %s' % (r['code'], {'ref': '대표', 'sh_head_front': '얼굴 칸'}.get(r['pic'], r['pic'])))
        return rel_of(s, p)
    if k in ('place', 'prop'):
        d = code_dir(s, r['code'])
        if not r['view']:
            p = latest(d, r['code'], MASTER[cat_of(r['code'])])
        else:   # a view by its name (배경 화면의 시점 이름) or its kind
            names = read_card(s, r['code']).get('views') or {}
            kind = next((kv for kv, nm in names.items() if nm == r['view']), r['view'])
            p = latest(d, r['code'], kind) if re.fullmatch(KIND_RE, kind) else None
        if not p:
            raise Bad('%s 그림 없음%s' % (r['code'], ': ' + r['view'] if r['view'] else ''))
        return rel_of(s, p)
    if k == 'lib':
        p = os.path.join(s['storage'], 'projects', s['project'], 'scenes', r['name'] + '.png')
        if not os.path.isfile(p):
            raise Bad('보관함에 없음: %s' % r['name'])
        return rel_of(s, p)
    if k == 'frame':
        rel = (frames or {}).get(r['name']) or (prod_frames(s, [r['name']]).get(r['name']) or {}).get('rel')
        if not rel:
            raise Bad('첫 프레임 %s 가 아직 없음' % r['name'])
        return rel
    return rel_of(s, storage_file(s, r['name']))


def person_of(s, code):
    """gender / child of a character code (its card): the scene templates' woman · man · girl · boy."""
    c = read_card(s, code)
    return {'gender': c.get('gender') or ('female' if code[0] == 'F' else 'male'), 'child': bool(c.get('child'))}


def noun_of(p):
    return ('girl' if p['child'] else 'woman') if p['gender'] == 'female' else ('boy' if p['child'] else 'man')


def box_for(f, plate, people, size):
    """A cut box (fractions of the plate) for a box frame: around the faces of the people it turns on (their order in the
    plate's person lines ↔ the face boxes the worker left on the plate result), tall by the shot size, in the cut's
    aspect. full = the plate's full height."""
    if isinstance(f.get('box'), list):
        return f['box']
    codes = [q['pic']['code'] for q in plate['people']]
    faces = [people[codes.index(r['code'])] for r in f['on'] if r['code'] in codes and codes.index(r['code']) < len(people)]
    faces = [b for b in faces if b]
    if not faces:
        raise Bad('판 결과에서 %s 얼굴을 못 찾음 · 상자를 직접' % ', '.join(r['code'] for r in f['on']))
    W, H = size
    asp = (9 / 16) if f['orient'] == 'v' else (16 / 9)   # w / h of the cut
    x0, y0 = min(b[0] for b in faces) * W, min(b[1] for b in faces) * H
    x1, y1 = max(b[2] for b in faces) * W, max(b[3] for b in faces) * H
    fh = max((b[3] - b[1]) * H for b in faces)
    shot = 'full' if f.get('box') == 'full' else f.get('shot') or 'ms'
    ch = H if shot == 'full' else min(H, fh * SHOT_H.get(shot, 6.2))
    cw = max(ch * asp, (x1 - x0) * 1.5)
    if cw > W:
        cw = W
    ch = min(H, cw / asp)
    cw = ch * asp
    cx = (x0 + x1) / 2
    top = 0 if shot == 'full' else y0 - 0.45 * fh - 0.06 * ch   # hair above the face box, a little air above it
    bx0 = min(max(0, cx - cw / 2), W - cw)
    by0 = min(max(0, top), H - ch)
    return [round(bx0 / W, 4), round(by0 / H, 4), round((bx0 + cw) / W, 4), round((by0 + ch) / H, 4)]


def prod_frames(s, ids):
    """First frames of other scenes' sequences (a plate or a picture reused across scenes) → {id: {rel, people, spec, key}}
    of the ones with a ● picture."""
    want, out = set(ids), {}
    for q in docs(s, 'sequences'):
        for f in (q or {}).get('frames') or []:
            if f.get('id') in want and f.get('rel'):
                out[f['id']] = {k: f.get(k) for k in ('rel', 'people', 'spec', 'key')}
    return out


def frame_job(s, b):
    """One first frame of the script (b: spec = a parsed frame, frames {id: rel} of the ● frames so far, plate {spec, rel,
    people} for a box, seed, count, label, cfg options) → the image job, as the image screens would make it."""
    f, frames = b.get('spec') or {}, b.get('frames') or {}
    t = f.get('type')
    v = f.get('orient', 'v') == 'v'
    size = (1088, 1920) if v else (1920, 1088)
    o = {'count': int(b.get('count') or 1), 'seed': int(b.get('seed') or 1), 'asset_kind': str(b.get('label') or f.get('id', ''))[:80]}
    o.update({k: b[k] for k in ('cfg', 'neg', 'retries') if b.get(k) not in (None, '')})
    if t == 'compose':
        p = person_of(s, f['who']['code'])
        o.update(width=size[0], height=size[1], gender=p['gender'], child=p['child'], who=noun_of(p), shot=f['shot'],
                 frame=f.get('frame') or '', action=f['action'], behind=f['behind'], asset_code=f['who']['code'])
        o.update({k: f[k] for k in ('clothing', 'light', 'extra') if f.get(k)})
        if f.get('blur') if f.get('blur') is not None else f['shot'] == 'close':   # CU·MCU blurred by default (boss, session #18)
            o['blur'] = True
        if not o['frame']:
            pos = 'her' if p['gender'] == 'female' else 'his'
            o['frame'] = {'wide': 'full length, from the top of {p} head down to {p} shoes', 'medium': 'a medium shot from the top of {p} head to {p} waist',
                          'close': 'a close-up from the top of {p} head to {p} chest'}[f['shot']].format(p=pos)
        return image_job(s, {'type': 'scene.compose', 'images': [ref_path(s, f['who']), ref_path(s, f['place'])], 'options': o})
    if t == 'plate':
        ppl = []
        for q in f['people']:
            p = person_of(s, q['pic']['code'])
            cl = read_card(s, q['pic']['code']).get('clothing', '') if q['clothing'] == 'card' else q['clothing']
            ppl.append({'gender': p['gender'], 'child': p['child'], 'who': noun_of(p), 'frame': q['frame'], 'action': q['action'], 'clothing': cl})
        o.update(width=1920 if not v else 1088, height=1088 if not v else 1920, plate=f['plate'], people=ppl,
                 things=[{'what': read_card(s, x['pic']['code']).get('what') or 'object', 'action': x['action']} for x in f.get('things') or []],
                 asset_code=' '.join(q['pic']['code'] for q in f['people'] + (f.get('things') or [])))
        if f.get('extra'):
            o['extra'] = f['extra']
        imgs = [ref_path(s, f['place'])] + [ref_path(s, dict(q['pic'], pic='ref')) for q in f['people']] + \
            [ref_path(s, x['pic']) for x in f.get('things') or []]
        return image_job(s, {'type': 'scene.plate', 'images': imgs, 'options': o})
    if t == 'box':
        pl = b.get('plate') or {}
        if not pl.get('rel'):
            raise Bad('판 %s ● 없음' % f.get('plate'))
        box = box_for(f, pl['spec'], pl.get('people') or [], (1920, 1088) if pl['spec'].get('orient', 'h') == 'h' else (1088, 1920))
        crop = crop_image(s, {'rel': pl['rel'], 'box': box})
        on, codes, faces = f['on'], [q['pic']['code'] for q in pl['spec']['people']], pl.get('people') or []

        def fx(r):   # left to right on the plate: the face box centre (the person line order when no face was found)
            i = codes.index(r['code'])
            return (faces[i][0] + faces[i][2]) / 2 if i < len(faces) and faces[i] else i / max(1, len(codes))
        order = sorted(on, key=fx)
        ppl = []
        for r in on:
            p = person_of(s, r['code'])
            n = noun_of(p)
            if len(on) == 1:
                where = 'the ' + n
            else:
                i = order.index(r)
                where = 'the %s %s' % (n, 'on the left' if i == 0 else 'on the right' if i == len(on) - 1 else 'in the middle')
            cl = next((q['clothing'] for q in pl['spec']['people'] if q['pic']['code'] == r['code']), '')
            if cl == 'card':
                cl = read_card(s, r['code']).get('clothing', '')
            ppl.append({'gender': p['gender'], 'child': p['child'], 'who': n, 'where': where, 'action': f['acts'].get(r['code'], ''), 'clothing': cl})
        things = f.get('things') or []
        if f.get('blur') if f.get('blur') is not None else f.get('shot') in ('cu', 'mcu'):   # as compose: CU·MCU blurred
            o['blur'] = True
        o.update(width=size[0], height=size[1], what=pl['spec']['plate'], frame=f['frame'], people=ppl,
                 things=[{'what': read_card(s, r['code']).get('what') or 'object'} for r in things],
                 asset_code=' '.join(r['code'] for r in on + things))
        code, body = image_job(s, {'type': 'scene.zoom', 'images': [crop] + [ref_path(s, r) for r in on + things], 'options': o})
        if code == 200:
            body = json.dumps(dict(json.loads(body), crop=crop, box=box)).encode()
        return code, body
    if t == 't2i':
        o.update(prompt=f['prompt'], aspect=f.get('aspect') or '9:16', mp=f.get('mp') or 1)
        return image_job(s, {'type': 'image.t2i', 'images': [], 'options': o})
    if t == 'edit':
        o.update(prompt=f['prompt'], aspect=f.get('aspect') or '9:16', mp=f.get('mp') or 1)
        return image_job(s, {'type': 'image.edit', 'images': [ref_path(s, r, frames) for r in f['images']], 'options': o})
    raise Bad('이 첫 프레임은 작업이 없음: %s' % t)


def file_frame(s, b):
    """A `file` first frame: its picture's path → {rel}."""
    f = b.get('spec') or {}
    return {'rel': ref_path(s, f['file'], b.get('frames') or {})}


PROD_OVER = ['model', 'sec', 'steps', 'codes', 'cands', 'face', 'refineOn', 'upOn', 'orient']   # set per cut by the script


def script_make(s, b):
    """대본 › 시퀀스 만들기: for each picked scene a sequence doc (script {scene, file, title}, frames, prod) + its cut
    docs, or the existing one brought up to date (cut and frame state kept, script fields replaced; cuts no longer in the
    script are dropped when they made nothing, else kept and marked gone). mode: all | step."""
    sc = script_load(s)
    if sc['errors']:
        raise Bad('대본 오류 %d개 · 먼저 고침 (%s:%d %s)' % (len(sc['errors']), sc['errors'][0]['file'], sc['errors'][0]['line'], sc['errors'][0]['msg']))
    want = {int(x) for x in b.get('scenes') or []}
    mode = 'step' if b.get('mode') == 'step' else 'all'
    seqs = {q['script']['scene']: q for q in docs(s, 'sequences') if q and q.get('script')}
    allcuts = {c['id']: c for c in docs(s, 'cuts') if c and c.get('id')}
    made = []
    for scn in sc['scenes']:
        if scn['no'] not in want:
            continue
        sq = seqs.get(scn['no']) or {'name': 'S%02d %s' % (scn['no'], scn['title']), 'common': {'mode': mode, 'orient': sc['orient'], 'cands': 1},
                                     'cuts': [], 'export': None, 'prod': {'started': False}}
        sq['script'] = {'scene': scn['no'], 'file': scn['file'], 'title': scn['title'], 'notes': scn.get('notes', '')}
        old = {f['id']: f for f in sq.get('frames') or []}
        sq['frames'] = [dict(old.get(f['id'], {'jobs': [], 'base': None, 'rel': None}), id=f['id'], spec=f) for f in scn['frames']]
        sid = sq.get('id') or save_doc(s, {'kind': 'sequences', 'doc': sq})
        sq['id'] = sid
        mine = {allcuts[c]['script']['id']: allcuts[c] for c in sq['cuts'] if c in allcuts and allcuts[c].get('script')}
        ids = {}
        for c in scn['cuts'] + scn['gens']:   # doc ids first (sound / video borrow by id)
            ids[c['id']] = (mine.get(c['id']) or {}).get('id') or save_doc(s, {'kind': 'cuts', 'doc': {'seq': sid}})
        order = []
        for c in scn['cuts'] + scn['gens']:
            d = dict(mine.get(c['id']) or {'cand': [], 'saved': [], 'base': None, 'dropped': [], 'seed': 1, 'mode': mode})
            d.update(id=ids[c['id']], seq=sid, title=c['id'], over=PROD_OVER, orient=sc['orient'], cands=1, upOn=False, brk=c['break'],
                     say=c['say'], texts=c['text'], fade=c.get('fade'), genOnly=c['kind'] == 'gen',
                     script={'scene': scn['no'], 'id': c['id'], 'frame': c.get('frame') or None, 'last': c.get('last') or None,
                             'keys': c.get('keys') or [], 'shot': c['shot'], 'note': c['note']})
            if c.get('video'):
                d.update(src=ids[c['video']['src']], use=[c['video']['a'], c['video']['b']], shot=c['video'].get('shot'),
                         prompt=d.get('prompt') or '')
            else:
                d.update(src=None, shot=None, prompt=c['prompt'], model=c['model'], steps=c['steps'] or 2, codes=c['who'], sec=c['gen'],
                         face=c['model'] != 'h3', refineOn=c['model'] == 'h3',
                         use=None if c['kind'] == 'gen' or (c['use'][0] == 0 and abs(c['use'][1] - c['gen']) < 1e-6) else c['use'])
                if not c.get('frame'):
                    d['first'] = None
                if not c.get('keys'):
                    d['keys'] = []
            snd = c['sound']
            d['sound'] = None if snd['mode'] == 'self' else 'none' if snd['mode'] == 'none' else {'cut': ids[snd['src']], 'at': snd['a']}
            save_doc(s, {'kind': 'cuts', 'id': d['id'], 'doc': d})
            order.append(d['id'])
        for cid in sq['cuts']:   # cuts the script no longer has
            c = allcuts.get(cid)
            if c and cid not in order and (c.get('cand') or c.get('base')):
                c['script'] = dict(c.get('script') or {}, gone=True)
                save_doc(s, {'kind': 'cuts', 'id': cid, 'doc': c})
                order.append(cid)
            elif c and cid not in order:
                os.remove(doc_path(s, 'cuts', cid))
        sq['cuts'] = order
        save_doc(s, {'kind': 'sequences', 'id': sid, 'doc': sq})
        made.append({'scene': scn['no'], 'id': sid})
    return made


# ---- project docs: a cut (영상 생성 › 한 컷) and a sequence, the UI's state until 보관함에 저장 / 닫기 ----
DOC_KINDS = ('cuts', 'sequences', 'drafts', 'films')   # drafts: the image screens (생성 · 편집 · 장면); films: 한 편


def doc_path(s, kind, did):
    if kind not in DOC_KINDS or not re.fullmatch(r'[\w-]{1,60}', str(did or '')):
        raise Bad('문서 이름이 맞지 않음')
    if not storage_ok(s) or not s.get('project'):
        raise Bad('저장 위치·프로젝트 없음')
    return os.path.join(s['storage'], 'projects', s['project'], kind, did + '.json')


def docs(s, kind, ids=None):
    d = os.path.dirname(doc_path(s, kind, 'x'))
    if ids is not None:
        return [read_json(doc_path(s, kind, i)) for i in ids if os.path.isfile(doc_path(s, kind, i))]
    return [read_json(os.path.join(d, n)) for n in sorted(os.listdir(d)) if n.endswith('.json')] if os.path.isdir(d) else []


def save_doc(s, b):
    kind, did = b.get('kind'), b.get('id') or time.strftime('%Y%m%d-%H%M%S-') + os.urandom(2).hex()
    p = doc_path(s, kind, did)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    write_json(p, dict(b.get('doc') or {}, id=did))
    return did


def thumb_video(path):
    """Keep a small jpg of an unsaved video candidate and delete the video (design §11 후보). Without a way to make the
    thumbnail the video stays."""
    d, n = os.path.split(path)
    if video_thumb(path, os.path.join(d, '_thumb_' + n[:-4] + '.jpg')):
        os.remove(path)


def finish(s, b):
    """보관함에 저장 / 닫기 for a cut or a sequence: items [{job, name, title}] are copied into the project library,
    then every video left in the jobs' folders becomes a small thumbnail (the job record keeps settings and seeds) and
    pass-1 latents go, the jobs are closed and the docs ([{kind, id}]) deleted."""
    saved = [save_item(s, it) for it in b.get('items') or []]
    for job in b.get('jobs') or []:
        jd = find_job_dir(s, str(job))
        for n in sorted(os.listdir(jd)) if jd else []:
            if n.endswith('.mp4'):
                thumb_video(os.path.join(jd, n))
            elif n.endswith('.png'):
                thumb_and_drop(os.path.join(jd, n))
            elif n.endswith('.lat'):
                os.remove(os.path.join(jd, n))
        try:
            srv('POST', '/jobs/%s/close' % job, b'{}')
        except Unreachable:
            pass
    for d in b.get('docs') or []:
        p = doc_path(s, d.get('kind'), d.get('id'))
        if os.path.isfile(p):
            os.remove(p)
    return saved


def on_node(s, sha):
    """Make sure the sha file is in the node asset cache. If not, resend the Mac original (<storage>/assets/<sha>.*)."""
    if srv('HEAD', '/assets/' + sha)[0] == 200:
        return
    src = find_sha(s, sha)
    if not src:
        raise Bad('노드에도 맥에도 원본 없음: %s' % sha[:12])
    send_to_node(src)


def refine_job(s, b):
    """video.r2v pass 2: send the pass-1 latents (.lat) of the shots picked in the preview (picks [{job, shot}], pick order
    = join order) to the node as lat_<i> and submit the pass-2 job. All picked shots must share the same reference images
    (one job's references = image_1..N)."""
    picks = b.get('picks') or []
    if not picks:
        raise Bad('선택한 컷 없음')
    if not storage_ok(s):
        raise Bad('저장 위치 없음')
    jobs, inputs, refs = {}, {}, None
    for i, p in enumerate(picks):
        job, k = str(p.get('job', '')), int(p.get('shot', -1))
        if not re.fullmatch(r'[\w.-]+', job):
            raise Bad('작업 이름이 맞지 않음')
        if job not in jobs:
            code, body = srv('GET', '/jobs/' + job)
            jobs[job] = json.loads(body) if code == 200 else {}
        j = jobs[job]
        if j.get('type') != 'video.r2v' or j['opts'].get('stage') != 'preview':
            raise Bad('%s 는 미리보기 작업이 아님' % job)
        r = {n: v for n, v in j['inputs'].items() if n.startswith('image_')}
        if refs is not None and r != refs:
            raise Bad('참조 이미지가 다른 컷은 같이 고화질 생성 불가')
        refs = r
        lat = os.path.join(find_job_dir(s, job) or '', 'shot_%02d_pre.lat' % k)
        if not os.path.isfile(lat):
            raise Bad('미리보기 잠재 파일이 맥에 없음: %s 컷 %d' % (job, k))
        inputs['lat_%d' % i] = send_to_node(lat)[0]
        for g in ('first', 'last'):   # the shot's start / end frame guides, as in pass 1
            if j['inputs'].get('%s_%d' % (g, k)):
                inputs['%s_%d' % (g, i)] = j['inputs']['%s_%d' % (g, k)]
        for n, sha in j['inputs'].items():   # key frames (frame indices travel in the .lat)
            if n.startswith('key_%d_' % k):
                inputs['key_%d_%s' % (i, n.rsplit('_', 1)[1])] = sha
    inputs.update(refs)
    for n, sha in inputs.items():
        if not n.startswith('lat_'):
            on_node(s, sha)
    opts = {'stage': 'refine', 'from': [[p['job'], int(p['shot'])] for p in picks],   # pass 2 reuses each shot's pass-1 seed
            'seeds': [int(jobs[p['job']]['opts']['seed']) + int(p['shot']) for p in picks],
            'asset_code': jobs[picks[0]['job']]['opts'].get('asset_code', ''), 'asset_kind': str(b.get('label') or '')[:80]}
    return submit(s, {'type': 'video.r2v', 'inputs': inputs, 'options': opts})


def upscale_job(s, b):
    """Pass-3 upscale: send one video already on the Mac — a job result (job, name) or a saved 보관함 item (path under
    the storage location) — to the node and submit a video.upscale job."""
    if not storage_ok(s):
        raise Bad('저장 위치 없음')
    if b.get('path'):
        root = os.path.realpath(s['storage'])
        src = os.path.realpath(os.path.join(root, str(b['path'])))
        if not src.startswith(root + os.sep) or not src.endswith('.mp4') or not os.path.isfile(src):
            raise Bad('영상이 맥에 없음: %s' % b['path'])
        opts = {'from': [str(b['path'])]}
    else:
        job, name = str(b.get('job', '')), str(b.get('name', ''))
        if not (re.fullmatch(r'[\w.-]+', job) and re.fullmatch(r'[\w.-]+\.mp4', name)):
            raise Bad('작업·파일 이름이 맞지 않음')
        src = os.path.join(find_job_dir(s, job) or '', name)
        if not os.path.isfile(src):
            raise Bad('영상이 맥에 없음: %s/%s' % (job, name))
        opts = {'from': [job, name]}
    opts.update({k: v for k, v in (b.get('options') or {}).items() if v not in ('', None)})
    return submit(s, {'type': 'video.upscale', 'inputs': {'video': send_to_node(src)[0]}, 'options': opts})


def save_version(d, code, kind, src, job, name):
    """Copy src into the code folder as <code>_<kind>_v<next>.png (never overwrites) + its settings json → file name."""
    v = max([x[1] for x in images(d, code) if x[0] == kind] or [0])
    with open(src, 'rb') as f:
        while True:
            v += 1
            dest = os.path.join(d, '%s_%s_v%d.png' % (code, kind, v))
            try:
                out = open(dest, 'xb')   # if it exists, the next number (concurrent adopts don't overwrite either)
            except FileExistsError:
                continue
            try:
                with out:
                    shutil.copyfileobj(f, out, CHUNK)
            except BaseException:
                os.remove(dest)          # never leave a half-written version file
                raise
            break
    meta = read_json(src[:-4] + '.json', {})
    meta.update(job=job, result=name)
    write_json(dest[:-4] + '.json', meta)
    return os.path.basename(dest)


def adopt(s, b):
    """Copy one result image into the code folder under the next version name. Never overwrites. The settings json goes
    next to it. A char.sheet job (sheet.png + sh_*.png) is adopted as one set: the sheet under the given kind, each panel
    under its own name (sh_head_front, ...)."""
    job, name, code, kind = (str(b.get(k, '')).strip() for k in ('job', 'name', 'code', 'kind'))
    if not (re.fullmatch(r'[\w.-]+', job) and re.fullmatch(r'[\w.-]+\.png', name)):
        raise Bad('작업·파일 이름이 맞지 않음')
    if not re.fullmatch(KIND_RE, kind):
        raise Bad('파일 종류 이름 필요 (영문·숫자·_)')
    d = code_dir(s, code)
    jd = find_job_dir(s, job) or ''
    if not os.path.isfile(os.path.join(jd, name)):
        raise Bad('결과가 아직 맥에 없음: %s/%s' % (job, name))
    pick = {name: kind}
    if os.path.isfile(os.path.join(jd, 'sheet.png')):
        pick = {n: (kind if n == 'sheet.png' else n[:-4]) for n in os.listdir(jd)
                if n == 'sheet.png' or (n.startswith('sh_') and n.endswith('.png'))}
    saved = [save_version(d, code, k, os.path.join(jd, n), job, n) for n, k in sorted(pick.items())]
    shutil.rmtree(jd)   # the adopted result is in the code folder. The job folder has no further use
    return ' '.join(saved)


def result_meta(s, job):
    """Summary of the settings json of a job's results on the Mac → {png name: {seed, face_sim, min_sim, passed, body,
    people (scene.plate: each person's face box)}}"""
    if not re.fullmatch(r'[\w.-]+', job) or not storage_ok(s):
        return {}
    d = find_job_dir(s, job)
    if not d:
        return {'_gone': True}   # job adopted and cleaned up, or not arrived yet
    out = {}
    for n in sorted(os.listdir(d)):
        if n.endswith('.json') and os.path.isfile(os.path.join(d, n[:-5] + '.png')):
            m = read_json(os.path.join(d, n), {})
            out[n[:-5] + '.png'] = {k: m.get(k) for k in ('seed', 'face_sim', 'min_sim', 'passed', 'body', 'body_check', 'people')}
    return out


def sync_once():
    """Take every result the node has uploaded and the Mac doesn't have yet — per file, as each unit ends (§8) — into
    the job's project folder, then tell the node (it moves the file into its cache)."""
    s = settings()
    status['workers'] = json.loads(srv('GET', '/workers')[1])
    code, b = srv('GET', '/files?unsynced=1')
    files = json.loads(b) if code == 200 else []
    status.update(server_ok=True, waiting=len(files))
    if not storage_ok(s):
        return
    for f in files:
        d = job_dir(s, f['job'], f.get('project'))
        os.makedirs(d, exist_ok=True)   # under the storage location, which exists (checked above)
        dest = os.path.join(d, f['name'])
        if not (os.path.isfile(dest) and os.path.getsize(dest) == f['size'] and sha_file(dest) == f['sha']):
            tmp = dest + '.part'
            code, _ = srv('GET', '/results/%s/%s' % (f['job'], f['name']), out=tmp, timeout=3600)
            if code != 200 or sha_file(tmp) != f['sha']:   # rename only when fully received and matching
                os.remove(tmp)
                raise IOError('download failed %s/%s (%d)' % (f['job'], f['name'], code))
            os.replace(tmp, dest)
        srv('POST', '/files/ack', json.dumps({'job': f['job'], 'name': f['name'], 'sha': f['sha']}).encode())
        status['waiting'] -= 1
        print('saved %s/%s' % (f['job'], f['name']), flush=True)
    code, b = srv('GET', '/jobs?unrecorded=1&limit=100')
    for j in json.loads(b) if code == 200 else []:   # each ended job's full record next to its results (job.json)
        d = job_dir(s, j['id'], j.get('project'))
        os.makedirs(d, exist_ok=True)
        write_json(os.path.join(d, 'job.json'), j)
        srv('POST', '/jobs/%s/recorded' % j['id'], b'{}')
    status['last_sync'] = time.time()


def records(s, q='', project=None, limit=300, every=False):
    """The 기록 tab: the job.json files under the storage location (all projects, or one) from the last record_days
    (settings, default 30 — boss, session #10: the list need not go back far; files are never deleted for it), newest
    first, filtered by a text match on the id, type, codes, seeds and options."""
    if not storage_ok(s):
        return []
    since = 0 if every else time.time() - float(s.get('record_days') or 30) * 86400
    roots = [os.path.join(s['storage'], 'projects', p, 'jobs') for p in ([project] if project else projects(s))]
    if not project:
        roots.append(os.path.join(s['storage'], JOBS))
    out, q = [], q.strip().lower()
    for r in roots:
        for jid in os.listdir(r) if os.path.isdir(r) else []:
            j = read_json(os.path.join(r, jid, 'job.json'))
            if not j or j.get('created', 0) < since or (q and q not in json.dumps([j.get('id'), j.get('type'), j.get('opts'), [u.get('spec') for u in j.get('units') or []]],
                                                      ensure_ascii=False).lower()):
                continue
            o = j.get('opts') or {}
            seeds = sorted({u['spec']['seed'] for u in j.get('units') or [] if isinstance(u.get('spec'), dict) and 'seed' in u['spec']})
            out.append({'id': j['id'], 'type': j['type'], 'project': j.get('project'), 'state': j['state'], 'created': j['created'],
                        'ended': j.get('ended'), 'seed': o.get('seed'), 'seeds': seeds[:20], 'asset_code': o.get('asset_code', ''),
                        'asset_kind': o.get('asset_kind', ''), 'errors': j.get('errors') or [], 'opts': o, 'inputs': j.get('inputs'),
                        'files': [f for f in j.get('files') or [] if not f.get('tmp')],
                        'here': [n for n in os.listdir(os.path.join(r, jid))]})
    out.sort(key=lambda x: -x['created'])
    return out[:limit]


def find_sha(s, sha):
    """A file the Mac holds under this sha: an uploaded original, or a job result listed in a job.json."""
    d = os.path.join(s['storage'], 'assets')
    for n in os.listdir(d) if os.path.isdir(d) else []:
        if n.split('.')[0] == sha:
            return os.path.join(d, n)
    for r in records(s, limit=100000, every=True):
        for f in r['files']:
            if f.get('sha') == sha:
                path = os.path.join(find_job_dir(s, r['id']) or '', f['name'])
                if os.path.isfile(path) and sha_file(path) == sha:
                    return path
    return None


def rerun(s, b):
    """"같은 설정으로 다시 걸기": the recorded job's type, inputs and options (seed included — same result), under the
    current project. Inputs no longer on the node are resent from the Mac."""
    d = find_job_dir(s, str(b.get('job', '')))
    j = read_json(os.path.join(d, 'job.json')) if d else None
    if not j:
        raise Bad('기록 없음')
    for name, sha in (j.get('inputs') or {}).items():
        if srv('HEAD', '/assets/' + sha)[0] != 200:
            src = find_sha(s, sha)
            if not src:
                raise Bad('입력이 노드에도 맥에도 없음: %s' % name)
            send_to_node(src)
    return submit(s, {'type': j['type'], 'inputs': j.get('inputs') or {}, 'options': j.get('opts') or {}})


def sync_loop():
    while True:
        try:
            sync_once()
            status['error'] = None
        except Exception as e:
            status['error'] = '%s: %s' % (type(e).__name__, e)
            if isinstance(e, Unreachable):
                status['server_ok'] = False   # server unreachable (Mac asleep, network)
        time.sleep(3)


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def reply(self, code, body, ctype='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        return self.rfile.read(int(self.headers.get('Content-Length') or 0))

    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        if p == '/' or p.startswith('/ui/'):   # the UI (mac/ui/)
            path = os.path.realpath(os.path.join(UI, 'index.html' if p == '/' else p[len('/ui/'):]))
            if not path.startswith(UI + os.sep) or not os.path.isfile(path):
                return self.reply(404, {'error': 'not found'})
            with open(path, 'rb') as f:
                return self.reply(200, f.read(), (mimetypes.guess_type(path)[0] or 'text/plain') + '; charset=utf-8')
        if p == '/api/status':
            s = settings()
            return self.reply(200, dict(status, settings=s, storage_ok=storage_ok(s), projects=projects(s)))
        if p == '/api/asset':   # where an uploaded original is kept (by sha) → its path under the storage location
            s, sha = settings(), (parse_qs(urlparse(self.path).query).get('sha') or [''])[0]
            d = os.path.join(s['storage'], 'assets')
            hit = next((n for n in (os.listdir(d) if storage_ok(s) and os.path.isdir(d) else [])
                        if re.fullmatch(r'[0-9a-f]{64}', sha) and n.split('.')[0] == sha), None)
            return self.reply(200, {'rel': 'assets/' + hit if hit else None})
        if p == '/api/records':
            q = parse_qs(urlparse(self.path).query)
            return self.reply(200, records(settings(), (q.get('q') or [''])[0], (q.get('project') or [None])[0]))
        if p == '/api/library':
            q = parse_qs(urlparse(self.path).query)
            return self.reply(200, library(settings(), (q.get('scope') or ['shared'])[0]))
        if p.startswith('/files/'):   # view results saved in the storage location
            s, rel = settings(), p[len('/files/'):]
            path = os.path.realpath(os.path.join(s['storage'], rel))
            if not storage_ok(s) or not path.startswith(os.path.realpath(s['storage']) + os.sep) \
                    or not os.path.isfile(path):
                return self.reply(404, {'error': 'not found'})
            size = os.path.getsize(path)   # byte ranges: seeking in a long film (Safari plays video only with them)
            m = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range') or '')
            a, z = (0, size - 1) if not m or not (m.group(1) or m.group(2)) else \
                (int(m.group(1)), min(size - 1, int(m.group(2)) if m.group(2) else size - 1)) if m.group(1) else \
                (max(0, size - int(m.group(2))), size - 1)
            if a > z:
                self.send_response(416)
                self.send_header('Content-Range', 'bytes */%d' % size)
                self.end_headers()
                return
            self.send_response(206 if m else 200)
            self.send_header('Content-Type', mimetypes.guess_type(path)[0] or 'application/octet-stream')
            self.send_header('Accept-Ranges', 'bytes')
            if m:
                self.send_header('Content-Range', 'bytes %d-%d/%d' % (a, z, size))
            self.send_header('Content-Length', str(z - a + 1))
            self.end_headers()
            with open(path, 'rb') as f:
                f.seek(a)
                left = z - a + 1
                while left:
                    b = f.read(min(CHUNK, left))
                    if not b:
                        break
                    self.wfile.write(b)
                    left -= len(b)
            return
        if p in ('/api/doc', '/api/docs'):
            s, q = settings(), parse_qs(urlparse(self.path).query)
            try:
                kind = (q.get('kind') or [''])[0]
                if p == '/api/docs':
                    ids = [x for x in (q.get('ids') or [''])[0].split(',') if x] if q.get('ids') else None
                    return self.reply(200, docs(s, kind, ids) if storage_ok(s) and s.get('project') else [])
                return self.reply(200, read_json(doc_path(s, kind, (q.get('id') or [''])[0])) or {'error': 'no doc'})
            except Bad as e:
                return self.reply(400, {'error': str(e)})
        if p in ('/api/nextcode', '/api/session', '/api/sessions'):
            s, q = settings(), parse_qs(urlparse(self.path).query)
            try:
                if p == '/api/nextcode':
                    if q.get('of'):   # an outfit of a character (wardrobe)
                        return self.reply(200, {'code': next_outfit(s, q['of'][0], (q.get('style') or [''])[0])})
                    return self.reply(200, {'code': next_code(s, (q.get('g') or [''])[0], (q.get('r') or [''])[0])})
                if p == '/api/sessions':
                    return self.reply(200, sessions(s, (q.get('cat') or ['characters'])[0]))
                code = (q.get('code') or [''])[0]
                d = code_dir(s, code)
                return self.reply(200, {'session': read_json(session_path(s, code)),
                                        'card': read_card(s, code),
                                        'images': [{'kind': k, 'v': v, 'name': n} for k, v, n in images(d, code)],
                                        'voices': voice_files(s, code) if cat_of(code) == 'characters' else []})
            except Bad as e:
                return self.reply(400, {'error': str(e)})
        if p == '/api/script/frames':   # ● first frames of other scenes (cross-scene reuse)
            ids = [x for x in (parse_qs(urlparse(self.path).query).get('ids') or [''])[0].split(',') if x]
            s = settings()
            return self.reply(200, prod_frames(s, ids) if storage_ok(s) and s.get('project') else {})
        if p == '/api/script':
            try:
                return self.reply(200, script_load(settings()))
            except Bad as e:
                return self.reply(400, {'error': str(e)})
        if p in ('/api/codes', '/api/meta'):
            q = parse_qs(urlparse(self.path).query)
            arg = (q.get('cat') or q.get('job') or [''])[0]
            if p == '/api/codes':
                return self.reply(200, list_codes(settings(), arg) if arg in CODE_RE else [])
            return self.reply(200, result_meta(settings(), arg))
        if p.startswith('/api/srv/'):
            return self.proxy('GET')
        self.reply(404, {'error': 'no route'})

    def do_POST(self):
        p = urlparse(self.path).path
        if p == '/api/settings':
            s = settings()
            for k, v in json.loads(self.read_body()).items():
                if k in DEFAULTS and type(v) is type(DEFAULTS[k]):
                    s[k] = v.strip() if isinstance(v, str) else v
            if s['project'] and s['project'] not in projects(s):
                s['project'] = ''
            write_json(SETTINGS, s)
            return self.reply(200, dict(s, storage_ok=storage_ok(s), projects=projects(s)))
        if p == '/api/assets':
            return self.asset()
        if p in ('/api/codes', '/api/codejob', '/api/adopt', '/api/refine', '/api/upscale', '/api/r2v', '/api/i2v',
                 '/api/projects', '/api/retry', '/api/save', '/api/delete', '/api/srv/jobs', '/api/session',
                 '/api/register', '/api/discard', '/api/rerun', '/api/video', '/api/concat', '/api/doc', '/api/finish',
                 '/api/image', '/api/crop', '/api/face', '/api/edit', '/api/join', '/api/texts', '/api/script/make',
                 '/api/script/frame', '/api/script/file', '/api/tts', '/api/voice', '/api/voice/save'):
            s = settings()
            try:
                b = json.loads(self.read_body() or b'{}')
                if p == '/api/srv/jobs':   # a job from the UI as is: under the current project
                    return self.reply(*submit(s, b))
                if p == '/api/session':
                    write_json(session_path(s, str(b.get('code', ''))), b.get('session') or {})
                    return self.reply(200, {'ok': True})
                if p in ('/api/register', '/api/discard'):
                    return self.reply(200, {'saved': finish_character(s, b, p == '/api/register')})
                if p == '/api/projects':
                    s['project'] = new_project(s, b.get('name'))
                    write_json(SETTINGS, s)
                    return self.reply(200, {'project': s['project'], 'projects': projects(s)})
                if p == '/api/retry':
                    return self.reply(*retry(s, b))
                if p == '/api/video':
                    return self.reply(*video_job(s, b))
                if p == '/api/face':
                    return self.reply(*face_job(s, b))
                if p == '/api/tts':
                    return self.reply(*tts_job(s, b))
                if p == '/api/voice':
                    return self.reply(*voice_job(s, b))
                if p == '/api/voice/save':
                    return self.reply(200, {'saved': save_voice(s, b), 'voices': voice_files(s, b.get('code', ''))})
                if p == '/api/concat':
                    return self.reply(*concat_job(s, b))
                if p == '/api/texts':   # screen text typed in the UI (one item a line, script syntax without 'text:')
                    out, errs = [], []
                    for k, line in enumerate(str(x).strip() for x in b.get('lines') or []):
                        if line:
                            r = script.text_item(line)
                            (errs.append('%d: %s' % (k + 1, r)) if isinstance(r, str) else out.append(r))
                    return self.reply(400 if errs else 200, {'error': ' · '.join(errs)} if errs else {'texts': out})
                if p == '/api/edit':
                    return self.reply(*edit_job(s, b))
                if p == '/api/script/make':
                    return self.reply(200, {'made': script_make(s, b)})
                if p == '/api/script/frame':
                    return self.reply(*frame_job(s, b))
                if p == '/api/script/file':
                    return self.reply(200, file_frame(s, b))
                if p == '/api/join':
                    return self.reply(*join_job(s, b))
                if p == '/api/image':
                    return self.reply(*image_job(s, b))
                if p == '/api/crop':
                    return self.reply(200, {'rel': crop_image(s, b)})
                if p == '/api/doc':
                    return self.reply(200, {'id': save_doc(s, b)})
                if p == '/api/finish':
                    return self.reply(200, {'saved': finish(s, b)})
                if p == '/api/rerun':
                    return self.reply(*rerun(s, b))
                if p == '/api/save':
                    return self.reply(200, {'saved': save_item(s, b)})
                if p == '/api/delete':
                    delete_item(s, b)
                    return self.reply(200, {'ok': True})
                if p == '/api/codes':
                    return self.reply(200, save_card(s, b))
                if p == '/api/adopt':
                    return self.reply(200, {'saved': adopt(s, b)})
                if p == '/api/refine':
                    return self.reply(*refine_job(s, b))
                if p == '/api/upscale':
                    return self.reply(*upscale_job(s, b))
                if p == '/api/r2v':
                    return self.reply(*r2v_job(s, b))
                if p == '/api/i2v':
                    return self.reply(*i2v_job(s, b))
                return self.reply(*code_job(s, b))
            except Bad as e:
                return self.reply(400, {'error': str(e)})
            except Exception as e:
                return self.reply(502, {'error': '%s: %s' % (type(e).__name__, e)})
        if p.startswith('/api/srv/'):
            return self.proxy('POST')
        self.reply(404, {'error': 'no route'})

    def proxy(self, method):
        path = self.path[len('/api/srv'):]
        try:
            self.reply(*srv(method, path, self.read_body() if method == 'POST' else None))
        except Exception as e:
            self.reply(502, {'error': 'server unreachable: %s' % e})

    def asset(self):
        """Keep the uploaded file as the original in the storage location and send it only if the node lacks it. → {sha}"""
        s = settings()
        if not storage_ok(s):
            self.read_body()
            return self.reply(409, {'error': '저장 위치 없음'})
        d = os.path.join(s['storage'], 'assets')
        if not os.path.isdir(d):
            os.mkdir(d)
        ext = os.path.splitext(unquote(self.headers.get('X-Name', '')))[1].lower()[:10]
        tmp, h, left = os.path.join(d, '.upload-%d.part' % threading.get_ident()), hashlib.sha256(), \
            int(self.headers['Content-Length'])
        with open(tmp, 'wb') as f:
            while left:
                b = self.rfile.read(min(CHUNK, left))
                h.update(b)
                f.write(b)
                left -= len(b)
        sha = h.hexdigest()
        dest = os.path.join(d, sha + ext)
        os.replace(tmp, dest)
        try:
            sent = send_to_node(dest)[1]
        except Exception as e:
            return self.reply(502, {'error': 'upload to server failed: %s' % e})
        self.reply(200, {'sha': sha, 'sent': sent, 'saved': dest})


if __name__ == '__main__':
    threading.Thread(target=sync_loop, daemon=True).start()
    srv_ = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    srv_.daemon_threads = True
    print('shutter-mac http://localhost:%d  settings=%s' % (PORT, SETTINGS), flush=True)
    srv_.serve_forever()
