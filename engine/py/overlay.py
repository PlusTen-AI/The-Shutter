# Screen text (design §12 기능 18, docs/script.md §5): drawn on the frames by video.edit, in the one encode of the
# scene. Kinds: sub (subtitle line), name (name card), count (a number running from → to), live (LIVE badge), chat
# (live comments rising one by one), card (a big caption), on (text printed inside a box of the picture — a prop).
# Look from the reference short dramas (docs/shortdrama.md §4): white bold gothic with a black outline, subtitles at
# 70 % from the top, name cards and comments above them. Sizes follow the frame height, so an upscaled part keeps the
# layout. Sprites are drawn once per item (per value for a counter) and alpha-blended with numpy.
import glob, math, os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 24
FADE = 4   # frames of fade in / out
FONT_DIRS = [os.environ.get('SHUTTER_FONTS', ''), '/opt/shutter/fonts']
MAC_FONT = '/System/Library/Fonts/AppleSDGothicNeo.ttc'   # Mac (replay test bed): the system Korean gothic
POS = {'top': (0.5, 0.08), 'middle': (0.5, 0.45), 'bottom': (0.5, 0.70), 'top-left': (0.05, 0.06), 'top-right': (0.95, 0.06),
       'bottom-left': (0.05, 0.64), 'bottom-right': (0.95, 0.64)}
DEFAULT = {'sub': 'bottom', 'name': 'bottom-left', 'count': 'top-right', 'live': 'top-left', 'chat': 'bottom-left', 'card': 'middle'}


def _faces(path=None):
    """(bold, regular) font files (+ face index for a .ttc): the given file or folder, the image's fonts folder, the
    Mac system gothic; None when no Korean font is there."""
    files = []
    for d in [path] + FONT_DIRS:
        if d and os.path.isfile(d):
            files.append(d)
        elif d and os.path.isdir(d):
            files += sorted(glob.glob(d + '/*.[ot]t[fc]'))
    if not files and os.path.isfile(MAC_FONT):
        files = [MAC_FONT]
    found = {}
    for f in files:
        for i in range(16 if f.endswith('.ttc') else 1):
            try:
                style = ImageFont.truetype(f, 20, index=i).getname()[1].lower()
            except (OSError, IndexError):
                break
            for want, keys in (('bold', ('bold', 'heavy')), ('regular', ('regular', 'medium', 'book'))):
                if want not in found and any(k in style for k in keys) and 'extra' not in style:
                    found[want] = (f, i)
    if not found and files:
        found['regular'] = (files[0], 0)
    if not found:
        return None
    return found.get('bold') or found['regular'], found.get('regular') or found['bold']


def won(eok):
    """An amount in 억 written the Korean way: 31000 → '3조 1,000억', 5000 → '5,000억', 20000 → '2조'."""
    jo, rest = divmod(int(eok), 10000)
    return ' '.join(x for x in (('%d조' % jo) if jo else '', ('{:,}억'.format(rest)) if rest or not jo else '') if x)


class Overlay:
    def __init__(self, font=None):
        self.faces = _faces(font)
        if not self.faces:   # PIL's own font has no Hangul: fail clearly instead of drawing boxes
            raise RuntimeError('no Korean font for screen text (put one in /opt/shutter/fonts or set SHUTTER_FONTS)')
        self.fonts, self.cache, self.top = {}, {}, 0

    def font(self, size, bold=True):
        key = (int(size), bold)
        if key not in self.fonts:
            f, i = self.faces[0 if bold else 1]
            self.fonts[key] = ImageFont.truetype(f, int(size), index=i)
        return self.fonts[key]

    # ---- sprites (RGBA) ----
    def text_img(self, text, size, fill=(255, 255, 255, 255), stroke=0, bold=True, shadow=False):
        f = self.font(size, bold)
        sw = int(round(size * stroke))
        l, t, r, b = ImageDraw.Draw(Image.new('L', (1, 1))).textbbox((0, 0), text, font=f, stroke_width=sw)
        pad = sw + (int(size * 0.12) if shadow else 0) + 2
        im = Image.new('RGBA', (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        if shadow:
            o = max(1, int(size * 0.06))
            d.text((pad - l + o, pad - t + o), text, font=f, fill=(0, 0, 0, 150))
        d.text((pad - l, pad - t), text, font=f, fill=fill, stroke_width=sw, stroke_fill=(0, 0, 0, 255))
        return im

    def pill(self, parts, pad, radius, bg=(0, 0, 0, 150)):
        """Images stacked (or side by side when parts is a ('row', [...]) tuple) on a rounded dark box."""
        row = isinstance(parts, tuple)
        ims = parts[1] if row else parts
        w = sum(i.width for i in ims) + (len(ims) - 1) * pad if row else max(i.width for i in ims)
        h = max(i.height for i in ims) if row else sum(i.height for i in ims)
        im = Image.new('RGBA', (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(im).rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius=radius, fill=bg)
        x = y = pad
        for i in ims:
            im.alpha_composite(i, (x, y if not row else pad + (h - i.height) // 2))
            if row:
                x += i.width + pad
            else:
                y += i.height
        return im

    def sprite(self, it, H, W, value=None):
        k, p = it['kind'], [x for x in it.get('parts') or []]
        if k == 'sub':
            return self.fit(self.text_img(' '.join(p), 0.034 * H, stroke=0.12), 0.92 * W)
        if k == 'card':
            lines = [self.text_img(x, 0.058 * H, stroke=0.05, shadow=True) for x in p if x]
            return self.fit(self.stack(lines), 0.9 * W)
        if k == 'name':
            ims = [self.text_img(p[0], 0.036 * H)] + ([self.text_img(p[1], 0.022 * H, fill=(225, 225, 225, 255), bold=False)] if len(p) > 1 and p[1] else [])
            return self.fit(self.pill(ims, int(0.012 * H), int(0.012 * H), (12, 12, 16, 170)), 0.8 * W)
        if k == 'count':
            fmt = p[0] if p else '{n}'
            return self.fit(self.pill([self.text_img(fmt.replace('{n억}', won(int(value))).replace('{n}', '{:,}'.format(int(value))), 0.024 * H)],
                                      int(0.009 * H), int(0.02 * H)), 0.9 * W)
        if k == 'live':
            badge = self.pill([self.text_img('LIVE', 0.022 * H)], int(0.007 * H), int(0.008 * H), (225, 30, 45, 235))
            if p and p[0]:
                return self.pill(('row', [badge, self.text_img(p[0], 0.022 * H)]), int(0.006 * H), int(0.012 * H), (0, 0, 0, 120))
            return badge
        if k == 'on':
            x0, y0, x1, y1 = it['box']
            bw, bh = (x1 - x0) * W, (y1 - y0) * H
            lines = [x for x in p if x] or [' ']
            size = bh / len(lines) * 0.8
            ims = [self.text_img(x, size, fill=(25, 22, 20, 255)) for x in lines]
            return self.fit(self.stack(ims), bw, bh)
        raise ValueError('text kind %s' % k)

    def chat_sprite(self, line, H, W):
        who, _, msg = line.partition(':')
        a = self.text_img(who.strip(), 0.02 * H, fill=(255, 214, 90, 255)) if msg else None
        b = self.text_img((msg or who).strip(), 0.022 * H, bold=False)
        return self.fit(self.pill(('row', [a, b]) if a else [b], int(0.007 * H), int(0.014 * H), (0, 0, 0, 130)), 0.72 * W)

    @staticmethod
    def stack(ims):
        w = max(i.width for i in ims)
        out = Image.new('RGBA', (w, sum(i.height for i in ims)), (0, 0, 0, 0))
        y = 0
        for i in ims:
            out.alpha_composite(i, ((w - i.width) // 2, y))
            y += i.height
        return out

    @staticmethod
    def fit(im, max_w, max_h=None):
        s = min(1.0, max_w / im.width, (max_h / im.height) if max_h else 1.0)
        return im if s >= 1 else im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.Resampling.LANCZOS)

    # ---- placing ----
    @staticmethod
    def place(it, im, W, H):
        """Top-left corner of the sprite: named spots anchor by their side (left spots by the left edge, right by the
        right, others centred); 'at x y' and 'on' boxes centre it."""
        if it['kind'] == 'on':
            x0, y0, x1, y1 = it['box']
            return int((x0 + x1) / 2 * W - im.width / 2), int((y0 + y1) / 2 * H - im.height / 2)
        pos = it.get('pos') or DEFAULT[it['kind']]
        if isinstance(pos, (list, tuple)):
            return int(pos[0] * W - im.width / 2), int(pos[1] * H - im.height / 2)
        fx, fy = POS[pos]
        x = fx * W if pos.endswith('left') else fx * W - im.width if pos.endswith('right') else fx * W - im.width / 2
        y = fy * H if pos.startswith('top') else fy * H - im.height / 2
        return int(x), int(y)

    @staticmethod
    def blend(img, im, x, y, alpha=1.0):
        H, W = img.shape[:2]
        a = np.asarray(im, dtype=np.float32)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + im.width), min(H, y + im.height)
        if x1 <= x0 or y1 <= y0:
            return
        a = a[y0 - y:y1 - y, x0 - x:x1 - x]
        m = a[..., 3:4] / 255.0 * alpha
        reg = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (reg * (1 - m) + a[..., :3] * m + 0.5).astype(np.uint8)

    def draw(self, rgb, cut, k):
        """Frame k of the cut (cut['n'] frames, cut['text'] items) with its screen text."""
        H, W = rgb.shape[:2]
        img = np.array(rgb)
        n = cut['n']
        self.top = H   # the highest name card top on this frame: comments stack above it
        items = sorted(enumerate(cut.get('text') or []), key=lambda x: x[1]['kind'] == 'chat')
        for i, it in items:
            a, b = (int(round(it['at'][0] * FPS)), min(n, int(round(it['at'][1] * FPS)))) if it.get('at') else (0, n)
            if not a <= k < b:
                continue
            fade = 1.0 if it['kind'] == 'on' or (a == 0 and b == n) else min(1.0, (k - a + 1) / FADE, (b - k) / FADE)
            if it['kind'] == 'chat':
                self.draw_chat(img, it, k - a, b - a, W, H, fade, (id(cut), i))
                continue
            value = None
            if it['kind'] == 'count':
                u = min(1.0, (k - a) / max(1, b - a - 1))
                value = round(it['from'] + (it['to'] - it['from']) * (1 - (1 - u) ** 3))   # ease out
            key = (id(cut), i, value, W, H)
            if key not in self.cache:
                if len(self.cache) > 512:
                    self.cache.clear()
                self.cache[key] = self.sprite(it, H, W, value)
            im = self.cache[key]
            x, y = self.place(it, im, W, H)
            if it['kind'] == 'name' and not it.get('pos'):
                self.top = min(self.top, y)
            self.blend(img, im, x, y, alpha=fade)
        return img

    def draw_chat(self, img, it, k, n, W, H, fade, key):
        """Comments appear one by one (evenly over the item's time), each new one pushing the stack up; the last 4 shown."""
        lines = [x for x in it.get('parts') or [] if x]
        if not lines:
            return
        every = max(1, n // (len(lines) + 1))
        shown = min(len(lines), k // every + 1)
        ims = []
        for j in range(max(0, shown - 4), shown):
            ck = key + (j, W, H)
            if ck not in self.cache:
                self.cache[ck] = self.chat_sprite(lines[j], H, W)
            ims.append(self.cache[ck])
        pos = it.get('pos') or 'bottom-left'
        fx, fy = POS[pos] if isinstance(pos, str) else pos
        gap = int(0.008 * H)
        bottom = min(fy * H, self.top - gap) if not it.get('pos') else fy * H
        rise = min(1.0, (k % every + 1) / FADE)   # the newest slides up into place
        y = bottom + (1 - rise) * (ims[-1].height + gap)
        for j, im in enumerate(reversed(ims)):
            y -= im.height + (gap if j else 0)
            x = fx * W if (isinstance(pos, str) and not pos.endswith('right')) else fx * W - im.width
            self.blend(img, im, int(x), int(y), alpha=fade * (rise if j == 0 else 1.0) * (0.55 if j == 3 else 1.0))
