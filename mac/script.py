#!/usr/bin/env python3
"""화 파일 (docs/script.md): the drama script in the form the machine reads — the `## 씬` lines and the ``` blocks
(frame · cut · gen · drama) of the .md files in a project's script/ folder; everything else is notes for people.

  python3 mac/script.py check <file or folder> …   → errors / warnings by file:line, exit 1 on errors
  python3 mac/script.py json <file or folder> …    → the parsed script as JSON (what the UI gets)

Standard library only (macOS stock python3 3.9), as mac.py which imports it."""
import json, os, re, sys

KINDS = ('frame', 'cut', 'gen', 'drama')
FRAME_TYPES = ('compose', 'plate', 'box', 't2i', 'edit', 'file')
MULTI = ('person', 'thing', 'act', 'image', 'say', 'text', 'key')   # keys that may repeat
MODELS = {'ltx': ('ltx25', None), 'pih3': ('pih3', 2), 'pih3 2': ('pih3', 2), 'pih3 4': ('pih3', 4), 'h3': ('h3', None)}
MAX_SEC = {'ltx25': 20, 'pih3': 15, 'h3': 15}                # mac/ui/models.js sec options
EXPR = {'평정': 'ex_calm', '호기심': 'ex_curious', '긴장': 'ex_tense', '놀람': 'ex_surprise', '두려움': 'ex_fear', '슬픔': 'ex_sad',
        '결연': 'ex_resolve', '이완': 'ex_relax', '말하는입': 'ex_talk', '말하는 입': 'ex_talk', '얼굴': 'sh_head_front', '대표': 'ref'}
CHAR_RE = r'[FM][A-Z]\d{2}(?:-[a-z0-9]+)?'                  # FL01, MS01, FE01, ML01-lab (옷장)
PLACE_RE, PROP_RE = r'L\d{2}', r'P\d{2}'
ID_RE = r'[A-Za-z0-9][A-Za-z0-9-]{0,29}'
SCENE_RE = re.compile(r'^##\s+씬\s+(\d+)\s*(?:[·:.\-—]\s*(.*?))?\s*$')
TEXT_KINDS = ('sub', 'name', 'count', 'live', 'chat', 'card', 'on')
POS = {'위': 'top', '가운데': 'middle', '아래': 'bottom', '왼쪽위': 'top-left', '오른쪽위': 'top-right', '왼쪽아래': 'bottom-left',
       '오른쪽아래': 'bottom-right', 'top': 'top', 'middle': 'middle', 'bottom': 'bottom', 'top-left': 'top-left',
       'top-right': 'top-right', 'bottom-left': 'bottom-left', 'bottom-right': 'bottom-right'}
SHOTS = {'compose': ('wide', 'medium', 'close'), 'box': ('ms', 'mcu', 'cu', 'full')}
TOL = 0.05   # seconds: a borrowed sound range must be as long as the cut
EP_MIN, EP_MAX = 64, 96   # an episode's running time (docs/shortdrama.md §1)


class Doc:
    def __init__(self):
        self.orient, self.scenes, self.errors, self.warnings, self.files = 'v', [], [], [], []

    def err(self, where, msg):
        self.errors.append({'file': where[0], 'line': where[1], 'msg': msg})

    def warn(self, where, msg):
        self.warnings.append({'file': where[0], 'line': where[1], 'msg': msg})


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def span(s):
    """'0.4-2.6' (or 0.4~2.6) → (0.4, 2.6) or None"""
    m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*', s or '')
    return (float(m.group(1)), float(m.group(2))) if m else None


def ref(s):
    """A reference (docs/script.md §2) → dict, or None when it is not one."""
    s = ' '.join((s or '').split())
    m = re.fullmatch(r'(%s)(?: (.+))?' % CHAR_RE, s)
    if m:   # the 9 fixed expressions by their Korean names, a custom one (기능 12) by its English id
        x = m.group(2) or '대표'
        pic = EXPR.get(x) or ('ex_' + x if re.fullmatch(r'[a-z0-9]+(?:_[a-z0-9]+)*', x) else None)
        return {'kind': 'char', 'code': m.group(1), 'pic': pic} if pic else None
    m = re.fullmatch(r'(%s|%s)(?: (.+))?' % (PLACE_RE, PROP_RE), s)
    if m:
        return {'kind': 'place' if m.group(1)[0] == 'L' else 'prop', 'code': m.group(1), 'view': m.group(2) or ''}
    m = re.fullmatch(r'(보관함|frame|file) (.+)', s)
    if m:
        return {'kind': {'보관함': 'lib', 'frame': 'frame', 'file': 'file'}[m.group(1)], 'name': m.group(2)}
    return None


def text_item(v):
    """text: <kind> [@a-b] [pos] | content … → dict, or a str error"""
    head, _, rest = v.partition('|')
    kind = (head.split() or [''])[0]
    if kind not in TEXT_KINDS:
        return '화면 글자 종류가 아님 (%s)' % ' · '.join(TEXT_KINDS)
    out = {'kind': kind, 'at': None, 'pos': None, 'box': None, 'parts': [p.strip() for p in rest.split('|')] if rest else [],
           'src': v.strip()}
    head = head.strip()[len(kind):]
    if kind == 'count':
        m = re.search(r'([\d,]+)\s*(?:→|->)\s*([\d,]+)', head)
        if not m:
            return 'count 뒤에 <시작> → <끝> 필요'
        out['from'], out['to'] = int(m.group(1).replace(',', '')), int(m.group(2).replace(',', ''))
        head = head[:m.start()] + ' ' + head[m.end():]
    w = head.split()
    if kind == 'on':
        box = [num(x) for x in w[:4]]
        if len(box) < 4 or None in box or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            return 'on 뒤에 칸 x0 y0 x1 y1 (0~1) 필요'
        out['box'], w = box, w[4:]
    while w:
        t = w.pop(0)
        if t.startswith('@'):
            out['at'] = span(t[1:])
            if not out['at'] or out['at'][1] <= out['at'][0]:
                return '시간은 @a-b'
        elif t == 'at' and len(w) >= 2 and num(w[0]) is not None and num(w[1]) is not None:
            out['pos'] = [num(w.pop(0)), num(w.pop(0))]
        elif t in POS:
            out['pos'] = POS[t]
        else:
            return '알 수 없는 말: %s' % t
    if not any(out['parts']) and out['kind'] != 'live':
        return '내용 없음 ( | 뒤)'
    if out['kind'] == 'count' and not any('{n}' in p or '{n억}' in p for p in out['parts']):
        out['parts'] = [(out['parts'][0] + ' ' if out['parts'] and out['parts'][0] else '') + '{n}']
    return out


def blocks(text, fname, doc):
    """Split one file into scenes and blocks → [(scene dict or None, kind, id, {key: value|[values]}, prompt, line)]"""
    out, scene, lines = [], None, text.split('\n')
    i, notes = 0, []
    while i < len(lines):
        line = lines[i]
        m = SCENE_RE.match(line)
        if m:
            if scene:
                scene['notes'] = '\n'.join(notes).strip()
            scene = {'no': int(m.group(1)), 'title': (m.group(2) or '').strip(), 'file': fname, 'line': i + 1,
                     'frames': [], 'cuts': [], 'gens': []}
            doc.scenes.append(scene)
            notes, i = [], i + 1
            continue
        fm = re.match(r'^(`{3,})\s*(\S*)\s*(.*?)\s*$', line)
        if not fm:
            notes.append(line)
            i += 1
            continue
        fence, kind, bid, start = fm.group(1), fm.group(2), fm.group(3), i + 1
        j = i + 1
        while j < len(lines) and not re.match(r'^%s\s*$' % fence, lines[j]):
            j += 1
        body = lines[i + 1:j]
        if j >= len(lines):
            doc.err((fname, start), '블록이 닫히지 않음 (```)')
        i = j + 1
        if kind not in KINDS:   # a code block of the notes (```text …)
            notes += lines[start - 1:j + 1]
            continue
        if kind != 'drama' and not re.fullmatch(ID_RE, bid):
            doc.err((fname, start), '블록 ID 필요 (글자·숫자·-): ```%s <ID>' % kind)
            continue
        kv, prompt = {}, None
        for n, b in enumerate(body):
            if prompt is not None:
                prompt.append(b)
                continue
            if not b.strip():
                continue
            km = re.match(r'^([a-z]+)\s*:\s?(.*)$', b)
            if not km:
                doc.err((fname, start + 1 + n), '"키: 값" 꼴이 아님: %s' % b.strip()[:40])
                continue
            k, v = km.group(1), km.group(2).strip()
            if k == 'prompt':
                prompt = [v] if v else []
            elif k in MULTI:
                kv.setdefault(k, []).append((v, start + 1 + n))
            elif k in kv:
                doc.err((fname, start + 1 + n), '키가 두 번: %s' % k)
            else:
                kv[k] = (v, start + 1 + n)
        if kind == 'drama':
            o = kv.get('orient', ('v', start))[0]
            if o not in ('v', 'h'):
                doc.err((fname, start), 'orient 는 v 또는 h')
            doc.orient = o
            continue
        if not scene:
            doc.err((fname, start), '씬 밖 블록 (앞에 ## 씬 <번호> 줄 필요)')
            continue
        out.append((scene, kind, bid, kv, '\n'.join(prompt).strip() if prompt is not None else '', start))
    if scene:
        scene['notes'] = '\n'.join(notes).strip()
    return out


def val(kv, k):
    v = kv.get(k)
    return '' if v is None or v[0] in ('', '-') else v[0]


def build(doc, raw):
    """Typed frames / cuts / gens from the raw blocks, with per-block checks."""
    ids = {}
    for scene, kind, bid, kv, prompt, line in raw:
        where = (scene['file'], line)
        if bid in ids:
            doc.err(where, 'ID 가 겹침: %s (%s:%d)' % (bid, ids[bid][0], ids[bid][1]))
            continue
        ids[bid] = where
        known = {'frame': {'type', 'orient', 'place', 'who', 'shot', 'frame', 'action', 'clothing', 'behind', 'light', 'extra', 'blur',
                           'plate', 'person', 'thing', 'on', 'box', 'act', 'aspect', 'mp', 'image', 'file'},
                 'cut': {'shot', 'note', 'say', 'frame', 'last', 'model', 'who', 'gen', 'use', 'video', 'sound', 'break', 'text', 'fade'},
                 'gen': {'frame', 'last', 'model', 'who', 'gen', 'say', 'note', 'key'}}[kind]
        for k in kv:
            if k not in known:
                doc.err((scene['file'], (kv[k][0][1] if isinstance(kv[k], list) else kv[k][1])), '%s 블록에 없는 키: %s' % (kind, k))
        if kind == 'frame':
            f = frame(doc, kv, prompt, where)
            if f:
                f.update(id=bid, line=line, scene=scene['no'])
                scene['frames'].append(f)
        else:
            c = cut(doc, kind, kv, prompt, where)
            c.update(id=bid, line=line, scene=scene['no'])
            scene['gens' if kind == 'gen' else 'cuts'].append(c)
    return ids


def frame(doc, kv, prompt, where):
    t = val(kv, 'type')
    if val(kv, 'blur') not in ('', 'yes', 'no'):
        doc.err(where, 'blur 는 yes · no (비우면 CU·MCU만 흐림)')
    if t not in FRAME_TYPES:
        doc.err(where, 'frame type 은 %s' % ' · '.join(FRAME_TYPES))
        return None
    f = {'type': t, 'orient': val(kv, 'orient') or ('h' if t == 'plate' else doc.orient), 'prompt': prompt}
    if f['orient'] not in ('v', 'h'):
        doc.err(where, 'orient 는 v 또는 h')
    for k in ('frame', 'action', 'clothing', 'behind', 'light', 'extra', 'plate', 'aspect'):
        if val(kv, k):
            f[k] = val(kv, k)

    def need(*keys):
        for k in keys:
            if not val(kv, k):
                doc.err(where, '%s frame 에 %s 필요' % (t, k))

    def place():
        r = ref(val(kv, 'place'))
        if not r or r['kind'] != 'place':
            doc.err(where, 'place 는 장소 코드 (L01 또는 L01 <시점>)')
        return r
    if t == 'compose':
        need('place', 'who', 'action', 'behind')
        f['place'], f['who'] = place(), ref(val(kv, 'who'))
        if not f['who'] or f['who']['kind'] != 'char':
            doc.err(where, 'who 는 인물 그림 (FL01 · FL01 긴장 …)')
        f['shot'] = val(kv, 'shot') or 'medium'
        f['blur'] = {'yes': True, 'no': False}.get(val(kv, 'blur'))   # None = by shot size (close → blurred, design §2)
        if f['shot'] not in SHOTS['compose']:
            doc.err(where, 'shot 은 %s' % ' · '.join(SHOTS['compose']))
    elif t == 'plate':
        need('place', 'plate')
        f['place'], f['people'] = place(), []
        for v, ln in kv.get('person', []):
            p = [x.strip() for x in v.split('|')]
            r = ref(p[0])
            if len(p) != 4 or not r or r['kind'] != 'char':
                doc.err((where[0], ln), 'person = <인물 그림> | <보이는 범위> | <자세·표정> | <옷: card · - · 글>')
                continue
            f['people'].append({'pic': r, 'frame': p[1], 'action': p[2], 'clothing': '' if p[3] == '-' else p[3]})
        f['things'] = []   # objects on the plate (기능 10): thing: <P01 · P01 옆> | <where · what it does>
        for v, ln in kv.get('thing', []):
            p = [x.strip() for x in v.split('|')]
            r = ref(p[0])
            if len(p) != 2 or not r or r['kind'] != 'prop' or not p[1]:
                doc.err((where[0], ln), 'thing = <사물 P01 · P01 옆> | <자리 · 하는 것>')
                continue
            f['things'].append({'pic': r, 'action': p[1]})
        if not f['people']:
            doc.err(where, 'plate frame 에 person 한 줄 이상')
        if len(f['people']) > 3:
            doc.warn(where, '판은 지금 최대 3명까지 확인됨 (기능 11)')
    elif t == 'box':
        need('plate', 'on', 'frame')
        f['on'], f['things'] = [], []
        for x in val(kv, 'on').split(','):
            r = ref(x)
            if not r or r['kind'] not in ('char', 'prop'):
                doc.err(where, 'on 은 인물 그림 · 사물을 쉼표로 (FL01, ML01 긴장, P03)')
            else:
                (f['on'] if r['kind'] == 'char' else f['things']).append(r)
        if not f['on']:
            doc.err(where, 'on 에 인물 한 명 이상 (상자는 얼굴로 잡음)')
        f['shot'] = val(kv, 'shot')
        f['blur'] = {'yes': True, 'no': False}.get(val(kv, 'blur'))   # None = by shot size (cu · mcu → blurred)
        if val(kv, 'box'):
            b = val(kv, 'box')
            box = [num(x) for x in b.split()]
            if b != 'full' and (len(box) != 4 or None in box or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                doc.err(where, 'box 는 x0 y0 x1 y1 (0~1) 또는 full')
            f['box'] = 'full' if b == 'full' else box
        elif f['shot'] not in SHOTS['box']:
            doc.err(where, 'box frame 에 shot (%s) 또는 box 필요' % ' · '.join(SHOTS['box']))
        f['acts'] = {}
        for v, ln in kv.get('act', []):
            code, _, a = v.partition('|')
            if not re.fullmatch(CHAR_RE, code.strip()) or not a.strip():
                doc.err((where[0], ln), 'act = <코드> | <자세·표정>')
            f['acts'][code.strip()] = a.strip()
    elif t == 't2i':
        if not prompt:
            doc.err(where, 't2i frame 에 prompt 필요')
        f['mp'] = num(val(kv, 'mp')) or 1
        f.setdefault('aspect', '9:16' if f['orient'] == 'v' else '16:9')
    elif t == 'edit':
        if not prompt:
            doc.err(where, 'edit frame 에 prompt 필요')
        f['images'] = []
        for v, ln in kv.get('image', []):
            r = ref(v)
            if not r:
                doc.err((where[0], ln), 'image 는 참조 꼴 (인물 · 장소 · P01 · 보관함 … · frame …)')
            else:
                f['images'].append(r)
        if not f['images']:
            doc.err(where, 'edit frame 에 image 한 줄 이상')
        f.setdefault('aspect', '9:16' if f['orient'] == 'v' else '16:9')
    elif t == 'file':
        f['file'] = ref(val(kv, 'file'))
        if not f['file']:
            doc.err(where, 'file 은 참조 꼴')
    return f


def cut(doc, kind, kv, prompt, where):
    c = {'kind': kind, 'shot': val(kv, 'shot'), 'note': val(kv, 'note'), 'frame': val(kv, 'frame'), 'last': val(kv, 'last'),
         'say': [], 'text': [], 'prompt': prompt, 'break': val(kv, 'break') == 'yes', 'video': None,
         'sound': {'mode': 'self'}}
    for v, ln in kv.get('say', []):
        code, _, line = v.partition('|')
        code = code.strip()
        if not line.strip() or not (re.fullmatch(CHAR_RE, code) or code in ('-', 'P01') or re.fullmatch(PROP_RE, code)):
            doc.err((where[0], ln), 'say = <코드 · P01 · -> | <대사>')
        c['say'].append({'code': code, 'line': line.strip()})
    for v, ln in kv.get('text', []):
        t = text_item(v)
        if isinstance(t, str):
            doc.err((where[0], ln), 'text: ' + t)
        else:
            c['text'].append(t)
    if val(kv, 'break') not in ('', 'yes'):
        doc.err(where, 'break 는 yes')
    if kind == 'cut' and val(kv, 'video'):   # <ID> a-b, or <gen ID> <shot> a-b: a span inside that shot (기능 8)
        m = re.fullmatch(r'(%s)\s+(?:(\d+)\s+)?(.+)' % ID_RE, val(kv, 'video'))
        r = m and span(m.group(3))
        if not r or r[1] <= r[0] or (m.group(2) and int(m.group(2)) < 1):
            doc.err(where, 'video = <컷 · gen ID> a-b · <gen ID> <샷 번호> a-b')
        else:
            c['video'] = {'src': m.group(1), 'a': r[0], 'b': r[1], 'shot': int(m.group(2)) if m.group(2) else None}
        for k in ('frame', 'last', 'model', 'who', 'gen', 'use'):
            if val(kv, k):
                doc.err(where, 'video 를 쓰는 컷엔 %s 안 씀 (생성 안 함)' % k)
        if prompt:
            doc.err(where, 'video 를 쓰는 컷엔 prompt 안 씀')
        c['len'] = r[1] - r[0] if r else 0
    else:
        mv = ' '.join(val(kv, 'model').split())
        if mv not in MODELS:
            doc.err(where, 'model 은 ltx · pih3 · pih3 4 · h3')
        c['model'], c['steps'] = MODELS.get(mv, (None, None))
        c['who'] = [] if not val(kv, 'who') else val(kv, 'who').split()
        for w in c['who']:
            if not re.fullmatch(CHAR_RE, w):
                doc.err(where, 'who 는 인물 코드 (공백으로, 최대 2): %s' % w)
        if len(c['who']) > 2:
            doc.err(where, 'who 는 최대 2명')
        c['gen'] = num(val(kv, 'gen'))
        if not c['gen'] or c['gen'] <= 0:
            doc.err(where, 'gen (생성 길이, 초) 필요')
        elif c['model'] and c['gen'] > MAX_SEC[c['model']]:
            doc.err(where, '%s 생성은 %d초까지' % (val(kv, 'model'), MAX_SEC[c['model']]))
        if not prompt:
            doc.err(where, 'prompt 필요')
        if c['model'] == 'ltx25' and (not c['frame'] or c['last']):
            doc.err(where, 'LTX 는 첫 프레임만 받음 (frame 필요, last 없음)')
        if c['model'] == 'h3' and not c['who']:
            doc.err(where, 'H3 는 who 필요 (시트 참조)')
        if kind == 'cut':
            u = span(val(kv, 'use'))
            if not u or u[1] <= u[0]:
                doc.err(where, 'use (쓸 구간 a-b) 필요')
                u = (0.0, c['gen'] or 0.0)
            elif c['gen'] and u[1] > c['gen'] + 1e-6:
                doc.err(where, 'use 끝 %.2f 가 생성 길이 %.2f 보다 김' % (u[1], c['gen']))
            c['use'], c['len'] = list(u), u[1] - u[0]
    if kind == 'cut':
        c['fade'] = fade(val(kv, 'fade'), c.get('len', 0), doc, where)
    c['keys'] = []
    for v, ln in kv.get('key', []):   # gen: a later shot's first frame at its cut time (H3, official multiframe)
        m = re.fullmatch(r'(%s)\s*@\s*(\d+(?:\.\d+)?)' % ID_RE, v.strip())
        if not m:
            doc.err((where[0], ln), 'key = <frame ID> @<초>')
            continue
        at = float(m.group(2))
        if c.get('gen') and not 0 < at < c['gen']:
            doc.err((where[0], ln), 'key 시각 %.2f 는 0 과 생성 길이 %.2f 사이' % (at, c['gen']))
        if c['keys'] and at <= c['keys'][-1]['at']:
            doc.err((where[0], ln), 'key 시각은 차례로 커짐')
        c['keys'].append({'frame': m.group(1), 'at': at})
    if c['keys'] and c.get('model') != 'h3':
        doc.err(where, 'key (샷 첫 프레임) 는 model h3 만')
    sv = val(kv, 'sound')
    if kind == 'cut' and sv and sv != 'self':
        if sv == 'none':
            c['sound'] = {'mode': 'none'}
        else:
            m = re.fullmatch(r'(%s)\s+(.+)' % ID_RE, sv)
            r = m and span(m.group(2))
            if not r or r[1] <= r[0]:
                doc.err(where, 'sound = self · none · <컷 · gen ID> a-b')
            else:
                c['sound'] = {'mode': 'src', 'src': m.group(1), 'a': r[0], 'b': r[1]}
                if abs((r[1] - r[0]) - c.get('len', 0)) > TOL:
                    doc.err(where, '소리 구간 %.2f초 ≠ 컷 길이 %.2f초' % (r[1] - r[0], c.get('len', 0)))
    return c


def fade(v, length, doc, where):
    """`fade: in <s>` · `out <s>` · `in <s> out <s>` — from / to black, picture and sound → {in, out} or None."""
    if not v:
        return None
    t, f = v.split(), {}
    for k in range(0, len(t), 2):
        x = num(t[k + 1]) if k + 1 < len(t) else None
        if t[k] not in ('in', 'out') or t[k] in f or x is None or x <= 0:
            doc.err(where, 'fade = in <초> · out <초> · in <초> out <초>')
            return None
        f[t[k]] = x
    if length and f.get('in', 0) + f.get('out', 0) > length + 1e-6:
        doc.err(where, '페이드 %.2f초가 컷 길이 %.2f초보다 김' % (f.get('in', 0) + f.get('out', 0), length))
    return {'in': f.get('in', 0.0), 'out': f.get('out', 0.0)}


def link(doc, ids):
    """References between blocks: frames, plates, borrowed video / sound, ranges inside the source generation."""
    frames = {f['id']: f for s in doc.scenes for f in s['frames']}
    gens = {g['id']: g for s in doc.scenes for g in s['cuts'] + s['gens'] if g.get('gen')}
    seen = set()
    for s in doc.scenes:
        if s['no'] in seen:
            doc.err((s['file'], s['line']), '씬 번호가 겹침: %d' % s['no'])
        seen.add(s['no'])
        for f in s['frames']:
            w = (s['file'], f['line'])
            refs = ([f['file']] if f['type'] == 'file' and f.get('file') else []) + f.get('images', [])
            for r in refs:
                if r and r['kind'] == 'frame' and r['name'] not in frames:
                    doc.err(w, '없는 frame: %s' % r['name'])
            if f['type'] == 'box':
                p = frames.get(f.get('plate', ''))
                if not p or p['type'] != 'plate':
                    doc.err(w, 'plate 는 plate frame ID: %s' % f.get('plate', ''))
                else:
                    on = {r['code'] for r in f['on'] + f['things']}
                    have = {q['pic']['code'] for q in p['people'] + p.get('things', [])}
                    if on - have:
                        doc.err(w, '판에 없는 사람: %s' % ' '.join(sorted(on - have)))
                    for code in f['acts']:
                        if code not in on:
                            doc.err(w, 'act 의 %s 가 on 에 없음' % code)
        for c in s['cuts'] + s['gens']:
            w = (s['file'], c['line'])
            for k in ('frame', 'last'):
                if c.get(k) and c[k] not in frames:
                    doc.err(w, '없는 frame: %s' % c[k])
            for x in c.get('keys', []):
                if x['frame'] not in frames:
                    doc.err(w, '없는 frame: %s' % x['frame'])
            for k in ('video', 'sound'):
                x = c.get(k)
                if not x or (k == 'sound' and x['mode'] != 'src'):
                    continue
                g = gens.get(x['src'])
                if not g:
                    doc.err(w, '%s 출처는 생성이 있는 컷 · gen: %s' % (k, x['src']))
                elif x.get('shot') and g['kind'] != 'gen':
                    doc.err(w, '샷 번호는 gen 출처만: %s' % x['src'])
                elif x['b'] > g['gen'] + 1e-6:
                    doc.err(w, '%s 구간 끝 %.2f 가 %s 생성 길이 %.2f 보다 김' % (k, x['b'], x['src'], g['gen']))
    return frames


def load(paths):
    """Parse the files (or the .md files of the folders, in name order) → dict for JSON."""
    doc, raw = Doc(), []
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, n) for n in sorted(os.listdir(p)) if n.endswith('.md') and not n.startswith('.')]
        else:
            files.append(p)
    for p in files:
        with open(p, encoding='utf-8') as f:
            text = f.read()
        doc.files.append(os.path.basename(p))
        raw += blocks(text, os.path.basename(p), doc)
    ids = build(doc, raw)
    link(doc, ids)
    eps, run, start = [], 0.0, None
    for s in doc.scenes:
        if not s['cuts']:
            doc.warn((s['file'], s['line']), '컷 없는 씬')
        s['sec'] = round(sum(c.get('len', 0) for c in s['cuts']), 2)
        for c in s['cuts']:   # episodes: the running time between breaks (shortdrama §1: about 80 s, 64~96)
            start = start or c['id']
            run += c.get('len', 0)
            if c['break']:
                eps.append({'from': start, 'to': c['id'], 'sec': round(run, 1), 'file': s['file'], 'line': c['line']})
                run, start = 0.0, None
    if start:
        eps.append({'from': start, 'to': doc.scenes[-1]['cuts'][-1]['id'], 'sec': round(run, 1), 'file': None, 'line': 0})
    for e in eps[:-1] if len(eps) > 1 else []:
        if not EP_MIN <= e['sec'] <= EP_MAX:
            doc.warn((e['file'], e['line']), '화 길이 %.1f초 (%s ~ %s, 보통 %d~%d초)' % (e['sec'], e['from'], e['to'], EP_MIN, EP_MAX))
    return {'orient': doc.orient, 'files': doc.files, 'scenes': doc.scenes, 'episodes': eps, 'errors': doc.errors, 'warnings': doc.warnings}


def main(argv):
    if len(argv) < 2 or argv[0] not in ('check', 'json'):
        print(__doc__.strip().split('\n\n')[1], file=sys.stderr)
        return 2
    d = load(argv[1:])
    if argv[0] == 'json':
        print(json.dumps(d, ensure_ascii=False, indent=1))
        return 1 if d['errors'] else 0
    for e in d['errors']:
        print('%s:%d: 오류: %s' % (e['file'], e['line'], e['msg']))
    for e in d['warnings']:
        print('%s:%d: 주의: %s' % (e['file'], e['line'], e['msg']))
    n = sum(len(s['cuts']) for s in d['scenes'])
    eps = d['episodes']
    print('씬 %d · 컷 %d · 첫 프레임 %d · %.1f초 · %d화%s · 오류 %d · 주의 %d' % (
        len(d['scenes']), n, sum(len(s['frames']) for s in d['scenes']), sum(s['sec'] for s in d['scenes']), len(eps),
        ' (%.0f~%.0f초)' % (min(e['sec'] for e in eps), max(e['sec'] for e in eps)) if eps else '', len(d['errors']), len(d['warnings'])))
    return 1 if d['errors'] else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
