# Face pass after the H3 swap (session #7): per-frame face swap onto the character's identity with FaceFusion
# hyperswap_1a_256 (ONNX, run with torch by face.OnnxTorch; inswapper_128 was tried and dropped — boss: its lips come
# out too big and dark). Pre/post-processing follows FaceFusion processors/modules/face_swapper/core.py:
# 5-point similarity warp to the arcface_128 template, model, box mask (FaceFusion default: blur 0.3, no padding),
# colour match inside the mask, inverse warp + blend. Frames where no face is found are left as they are.
# Occlusion (FaceFusion face_occluder, default model xseg_1): XSeg on the target crop gives where the face is visible;
# the paste mask is min(box, occlusion), so an arm or hand in front of the face is kept, not covered by the new face.
#   fs = FaceSwap(fc); ident, sims = fs.identity([master, *head_panels]); out, st = fs.video(frames, ident)
import time

import numpy as np
import torch
import torch.nn.functional as F

import face
import models

DIR = models.path(models.FACESWAP)
MODEL, SIZE = "hyperswap_1a_256.onnx", 256
OCCLUDER = "xseg_1.onnx"   # 256x256, NHWC BGR in [0, 1] -> [N, 256, 256, 1]
TEMPLATE = face.ARCFACE_DST + np.array([8.0, 0.0], np.float32)   # arcface_128 (insightface estimate_norm, size 128)


def _box_mask(size, blur=0.3):
    """FaceFusion create_box_mask: ones with a zeroed border of blur_amount/2, Gaussian sigma blur_amount/4."""
    amount = int(size * 0.5 * blur)
    area = max(amount // 2, 1)
    m = torch.ones(1, 1, size, size)
    m[..., :area, :] = 0
    m[..., -area:, :] = 0
    m[..., :, :area] = 0
    m[..., :, -area:] = 0
    s = amount * 0.25
    r = int(3 * s)
    k = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32) ** 2) / (2 * s * s))
    k = k / k.sum()
    m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(m, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def _grid(M, out_h, out_w, in_h, in_w, x0=0, y0=0, device="cuda"):
    """Sampling grid for output pixels (x0.., y0..) through the 2x3 map M (output px -> input px), align_corners=True."""
    xs, ys = np.meshgrid(np.arange(x0, x0 + out_w), np.arange(y0, y0 + out_h))
    src = np.stack([xs, ys, np.ones_like(xs)], -1).reshape(-1, 3) @ np.asarray(M, np.float64).T
    g = np.stack([src[:, 0] * 2 / (in_w - 1) - 1, src[:, 1] * 2 / (in_h - 1) - 1], -1)
    return torch.from_numpy(g.reshape(1, out_h, out_w, 2)).float().to(device)


class FaceSwap:
    def __init__(self, fc=None, device="cuda", occlusion=True):
        self.size, self.device = SIZE, device
        self.net = face.OnnxTorch(f"{DIR}/{MODEL}", device)
        self.occ = face.OnnxTorch(f"{DIR}/{OCCLUDER}", device) if occlusion else None
        # cv2.GaussianBlur(m, (0, 0), 5) on float32: ksize = round(5 * 8 + 1) | 1 = 41, reflect-101 border
        r = 20
        k = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32) ** 2) / (2 * 5.0 ** 2))
        self.gk, self.gr = (k / k.sum()).to(device), r
        self.fc = fc or face.FaceCheck(device)
        self.dst = TEMPLATE * (self.size / 128.0)
        self.mask = _box_mask(self.size).to(device)

    def identity(self, imgs, min_sim=0.4):
        """Mean ArcFace embedding of the character's images (imgs[0] = master). Other images whose face is far from
        the master (profiles, failed detections) are left out. Returns (embedding, [similarity to master or None])."""
        e0 = self.fc.embed(np.ascontiguousarray(imgs[0]))
        es, sims = [e0], [1.0]
        for im in imgs[1:]:
            e = self.fc.embed(np.ascontiguousarray(im))
            s = None if e is None else self.fc.similarity(e0, e)
            sims.append(None if s is None else round(s, 3))
            if s is not None and s >= min_sim:
                es.append(e)
        m = torch.stack(es).mean(0)
        return m / m.norm(), sims

    def _source(self, ident):
        return ident.to(self.device).float().reshape(1, -1)

    def _occlusion(self, crop):
        """FaceFusion create_occlusion_mask on the target crop ([1, 3, 256, 256] RGB in [0, 1]): XSeg (BGR, NHWC),
        clip, Gaussian sigma 5, then (clip(0.5, 1) - 0.5) * 2 → [1, 1, 256, 256], 1 = face visible."""
        m = self.occ(crop[:, [2, 1, 0]].permute(0, 2, 3, 1).contiguous())[0][..., 0].clamp(0, 1)[:, None]
        k, r = self.gk, self.gr
        m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        m = F.conv2d(F.pad(m, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
        return (m.clamp(0, 1).clamp(0.5, 1) - 0.5) * 2

    @torch.no_grad()
    def frame(self, img, det, src):
        """Swap one face (a FaceCheck.detect entry) in img (HxWx3 RGB uint8). Returns a new uint8 array."""
        h, w = img.shape[:2]
        S = self.size
        M = face._similarity(det[2].astype(np.float64), self.dst.astype(np.float64))   # frame px -> crop px
        Minv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]                            # crop px -> frame px
        t = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float()
        crop = F.grid_sample(t, _grid(Minv, S, S, h, w, device=self.device), mode="bilinear",
                             padding_mode="border", align_corners=True) / 255.0
        paste = self.mask if self.occ is None else torch.minimum(self.mask, self._occlusion(crop))
        out = (self.net(target=(crop - 0.5) / 0.5, source=src)[0] * 0.5 + 0.5).clamp(0, 1)
        # colour match (mean / std per channel) inside the mask, std ratio limited so contrast can't blow up
        m = (self.mask > 0.5).float()
        n = m.sum()
        mo, mc = (out * m).sum((2, 3), keepdim=True) / n, (crop * m).sum((2, 3), keepdim=True) / n
        so = (((out - mo) * m) ** 2).sum((2, 3), keepdim=True).div(n).sqrt()
        sc = (((crop - mc) * m) ** 2).sum((2, 3), keepdim=True).div(n).sqrt()
        out = ((out - mo) * (sc / so.clamp_min(1e-4)).clamp(0.8, 1.25) + mc).clamp(0, 1)
        # paste back over the crop's bounding box in the frame
        corners = np.array([[0, 0, 1], [S - 1, 0, 1], [0, S - 1, 1], [S - 1, S - 1, 1]], np.float64) @ Minv.T
        x0, y0 = max(0, int(np.floor(corners[:, 0].min()))), max(0, int(np.floor(corners[:, 1].min())))
        x1, y1 = min(w, int(np.ceil(corners[:, 0].max())) + 1), min(h, int(np.ceil(corners[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return img
        g = _grid(M, y1 - y0, x1 - x0, S, S, x0, y0, self.device)
        fo = F.grid_sample(out, g, mode="bilinear", padding_mode="border", align_corners=True) * 255.0
        fm = F.grid_sample(paste, g, mode="bilinear", padding_mode="zeros", align_corners=True)
        reg = t[:, :, y0:y1, x0:x1]
        t[:, :, y0:y1, x0:x1] = reg * (1 - fm) + fo * fm
        return t[0].permute(1, 2, 0).round().clamp(0, 255).byte().cpu().numpy()

    def _detect(self, f, size):
        """Faces at the native resolution (small faces of wide swap shots) plus, when that is larger than 640, the
        faces found at 640 that overlap none of them: SCRFD is trained at 640 and misses a big face at native size —
        final test (session #18): 15 of 49 face passes lost the face from a camera push-in to the clip's end, which
        switched back to the generated face while the checks (at 640) still saw it. Largest first."""
        d = self.fc.detect(f, size=size)
        if size > 640:
            d = d + [x for x in self.fc.detect(f, size=640) if all(_iou(x[0], y[0]) < 0.3 for y in d)]
        return sorted(d, key=lambda x: -(x[0][2] - x[0][0]) * (x[0][3] - x[0][1]))

    def _tracks(self, frames, idents, size, gap, progress):
        """Detect every frame and link its faces into tracks (box overlap with a face of the last `gap` frames); a
        track's score per character is the mean similarity of its faces. Returns (detections per frame, track id per
        detection, {track: [mean similarity per character]})."""
        dets, tid, sims, last = [], [], {}, {}   # last: track -> (frame, box)
        for i, f in enumerate(frames):
            d = self._detect(f, size)
            ids = []
            for box, _, _ in d:
                best, bi = 0.2, None
                for t, (j, b) in last.items():
                    if i - j <= gap and t not in ids:
                        ix = max(0, min(box[2], b[2]) - max(box[0], b[0])) * max(0, min(box[3], b[3]) - max(box[1], b[1]))
                        iou = ix / ((box[2] - box[0]) * (box[3] - box[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix)
                        if iou > best:
                            best, bi = iou, t
                bi = len(sims) if bi is None else bi
                ids.append(bi)
                last[bi] = (i, box)
            for (box, sc, kp), t in zip(d, ids):
                e = self.fc.embed(f, (box, sc, kp))
                sims.setdefault(t, []).append([self.fc.similarity(e, x.cpu()) for x in idents])
            dets.append(d)
            tid.append(ids)
            if progress:
                progress((i + 1) / len(frames))
        return dets, tid, {t: np.mean(v, 0) for t, v in sims.items()}

    def video_multi(self, frames, idents, det_size=None, progress=None, gap=6):
        """Several characters (two-person swap): every face of every frame goes to the character it resembles most.
        Faces are linked into tracks (_tracks); a whole track takes one character by the mean similarity of its faces,
        so a face that briefly looks like the other person doesn't flip, and two faces in one frame never take the
        same character (the larger margin keeps it). Returns (frames, stats)."""
        srcs = [self._source(i) for i in idents]
        size = det_size or int(np.ceil(max(frames[0].shape[:2]) / 32) * 32)
        t0 = time.perf_counter()
        dets, tid, mean = self._tracks(frames, idents, size, gap, progress and (lambda p: progress(0.5 * p)))
        out, missed, flips = [], [], 0
        for i, f in enumerate(frames):
            if not dets[i]:
                missed.append(i)
                out.append(f)
                continue
            pick = [int(np.argmax(mean[t])) for t in tid[i]]
            if len(pick) == 2 == len(idents) and pick[0] == pick[1]:   # both faces chose the same character
                m = [mean[t][pick[0]] - mean[t][1 - pick[0]] for t in tid[i]]
                pick[int(np.argmin(m))] = 1 - pick[0]
                flips += 1
            img = f
            for dd, c in zip(dets[i], pick):
                img = self.frame(img, dd, srcs[c])
            out.append(img)
            if progress:
                progress(0.5 + 0.5 * (i + 1) / len(frames))
        torch.cuda.synchronize()
        return np.stack(out), {"missed": missed, "tracks": len(mean), "conflicts": flips,
                               "track_char": {int(t): int(np.argmax(v)) for t, v in mean.items()},
                               "seconds": round(time.perf_counter() - t0, 2)}

    def video(self, frames, ident, det_size=None, progress=None, gap=6):
        """One character: in every frame only the face whose track resembles the character most is swapped (the only
        face in a one-person shot, as before); everyone else stays. It used to be the largest face — in a two-shot
        that was the other person (swap 17, session #12). Returns (frames, stats: missed frame indices, seconds)."""
        src = self._source(ident)
        size = det_size or int(np.ceil(max(frames[0].shape[:2]) / 32) * 32)   # detect at native resolution
        t0 = time.perf_counter()
        dets, tid, mean = self._tracks(frames, [ident], size, gap, progress and (lambda p: progress(0.5 * p)))
        out, missed = [], []
        for i, f in enumerate(frames):
            if not dets[i]:
                missed.append(i)
                out.append(f)
            else:
                k = max(range(len(dets[i])), key=lambda j: mean[tid[i][j]][0])
                out.append(self.frame(f, dets[i][k], src))
            if progress:
                progress(0.5 + 0.5 * (i + 1) / len(frames))
        torch.cuda.synchronize()
        return np.stack(out), {"missed": missed, "seconds": round(time.perf_counter() - t0, 2)}
