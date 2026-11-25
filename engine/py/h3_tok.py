# MiniMax H3 text-encoder input (Ref2VA), same as ComfyUI's MiniMaxH3ReferenceToVideo + MiniMaxH3Tokenizer.
# Needs `tokenizers`, numpy, PIL. Tokenizer files = ComfyUI qwen25_tokenizer (same files as the Qwen-Image 2.1 copy).
#   tok = H3Tokenizer(dir)
#   refs, vis = prepare_refs(width, height, length, images=[u8 HxWx3], videos=[u8 FxHxWx3 at 24 fps])
#   ids = tok(prompt, refs)            -> list[int], -1 = one vision block (vis[k], in order)
#   tags = token_tags(ids, vis)        -> per output row: 0 = vision block incl. <|vision_start|>/<|vision_end|>, 1 = text
# The engine (h3_te_encode) expands each -1 to (H/32)*(W/32) rows and returns every row (no template, nothing dropped).
import json
import math
import os

import numpy as np
from PIL import Image
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

VISION_START, VISION_END, PAD = 151652, 151653, 151643
MINIMAX_EXTRA = ["<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>", "<|caption_start|>", "<|caption_end|>"]
PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
FPS, CANVAS, BASE_SHORT, MAX_PIXELS, REF_SHORT = 24, 32, 768, 768 * 1344, 2048


class H3Tokenizer:
    def __init__(self, path):
        t = Tokenizer(models.BPE.from_file(os.path.join(path, "vocab.json"), os.path.join(path, "merges.txt")))
        t.normalizer = normalizers.NFC()
        t.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(PAT), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        t.decoder = decoders.ByteLevel()
        added = json.load(open(os.path.join(path, "tokenizer_config.json")))["added_tokens_decoder"]
        want = {a["content"]: int(i) for i, a in added.items()}
        for i in sorted(added, key=int):
            a = added[i]
            t.add_special_tokens([AddedToken(a["content"], special=a["special"], normalized=False,
                                             lstrip=a["lstrip"], rstrip=a["rstrip"], single_word=a["single_word"])])
        nxt = max(want.values()) + 1  # MiniMaxQwenSDTokenizer appends these: ids 151669..151675
        for s in MINIMAX_EXTRA:
            t.add_special_tokens([AddedToken(s, special=True, normalized=False)])
            want[s] = nxt
            nxt += 1
        for s, i in want.items():
            if t.token_to_id(s) != i:
                raise ValueError(f"token {s} got id {t.token_to_id(s)}, expected {i}")
        self.t = t

    def text(self, s):
        # sd1_clip with weights disabled: only the "\(" "\)" escapes change the text
        s = s.replace("\\)", ")").replace("\\(", "(")
        return self.t.encode(s, add_special_tokens=False).ids if s else []

    def __call__(self, prompt, refs=()):
        """refs in request order: {"type": "image"} | {"type": "audio"} | {"type": "video", "frames": n, ["timestamps"]}
        (n = frames Qwen sees, 2 fps). Each image is one vision block, each video ceil(n/2) blocks."""
        ids, cnt = [], {"image": 0, "audio": 0, "video": 0}
        vis = lambda: ids.extend([VISION_START, -1, VISION_END])  # noqa: E731
        for r in refs:
            k = r["type"]
            cnt[k] += 1
            if k == "image":
                ids += self.text("<Picture %d>: " % cnt[k])
                vis()
            elif k == "audio":
                ids += self.text("<Audio %d>: " % cnt[k])
            else:
                n = r["frames"]
                ts = list(r.get("timestamps") or [i / 2.0 for i in range(n)])
                if n % 2:
                    ts.append(ts[-1])
                ids += self.text("<Video %d>: " % cnt[k])
                for i in range(0, len(ts), 2):
                    ids += self.text("<%.1f seconds>" % ((ts[i] + ts[i + 1]) / 2.0))
                    vis()
        ids += self.text(prompt)
        return ids or [PAD]


def token_tags(ids, vis):
    """modality tag per expanded row (minimax token_tags_from_embeds_info)"""
    tags, spans, k = [], [], 0
    for i in ids:
        if i == -1:
            h, w = vis[k].shape[1:3]
            spans.append((len(tags), len(tags) + (h // 32) * (w // 32)))
            tags += [0] * (spans[-1][1] - spans[-1][0])
            k += 1
        else:
            tags.append(1)
    for s, e in spans:  # the whole block is video modality, flanking tokens included
        tags[max(0, s - 1):e + 1] = [0] * (min(len(tags), e + 1) - max(0, s - 1))
    return tags


def _lanczos(frames_f32, w, h):
    """comfy.utils.lanczos: float -> uint8 (truncating) -> PIL LANCZOS -> /255"""
    out = []
    for f in frames_f32:
        im = Image.fromarray(np.clip(255. * f, 0, 255).astype(np.uint8)).resize((w, h), resample=Image.Resampling.LANCZOS)
        out.append(np.array(im).astype(np.float32) / 255.0)
    return np.stack(out)


def prepare_refs(width, height, length, images=(), videos=(), ref_image_size="match"):
    """MiniMaxH3ReferenceToVideo's Qwen side: images (u8 HxWx3) and videos (u8 FxHxWx3, 24 fps) ->
    (refs for H3Tokenizer, vision inputs float32 [T, H, W, 3] in [0, 1], T = 1 per image, 2 per video block)"""
    frame_count = length
    while frame_count % 17 != 5:
        frame_count += 1
    refs, vis = [], []
    f32 = lambda a: np.asarray(a).astype(np.float32) / 255.0  # noqa: E731  LoadImage
    for img in images:
        hh, ww = img.shape[:2]
        scale = min(1.0, math.sqrt((width * height) / (ww * hh))) if ref_image_size == "match" else min(1.0, REF_SHORT / min(ww, hh))
        tw = max(CANVAS, round(ww * scale / CANVAS) * CANVAS)
        th = max(CANVAS, round(hh * scale / CANVAS) * CANVAS)
        refs.append({"type": "image"})
        vis.append(_lanczos(f32(img)[None], tw, th))
    for v in videos:
        vh, vw = v.shape[1:3]
        r = vw / vh
        nw, nh = (BASE_SHORT * r, BASE_SHORT) if r >= 1.0 else (BASE_SHORT, BASE_SHORT / r)
        if nw * nh > MAX_PIXELS:
            s = math.sqrt(MAX_PIXELS / (nw * nh))
            nw, nh = nw * s, nh * s
        cw, ch = max(CANVAS, round(nw / CANVAS) * CANVAS), max(CANVAS, round(nh / CANVAS) * CANVAS)
        if vw * vh < cw * ch:
            cw, ch = max(CANVAS, round(vw / CANVAS) * CANVAS), max(CANVAS, round(vh / CANVAS) * CANVAS)
        n = min(len(v), frame_count)
        if n < 5:
            raise ValueError("H3 reference videos need at least 5 frames")
        while n % 17 != 5:
            n -= 1
        q = _lanczos(f32(v[:n:FPS // 2]), cw, ch)  # 2 fps; resizing frame by frame = resizing all, then sampling
        if len(q) % 2:
            q = np.concatenate([q, q[-1:]])
        refs.append({"type": "video", "frames": (n + FPS // 2 - 1) // (FPS // 2)})
        vis += [q[i:i + 2] for i in range(0, len(q), 2)]
    return refs, vis
