# Body keypoints for the swap pose check (video.swap boundary edits): DWPose dw-ll_ucoco_384 (TorchScript, the
# comfyui_controlnet_aux model and pre/post-processing: BGR, mean/std, 288x384 top-down crop with 1.25 padding, SimCC
# split ratio 2). No person detector (yolox not ported): swaps are single-person, so pass 1 runs on the whole image and
# pass 2 on the box around the pass-1 body keypoints.
#   P = Pose(); kp = P.body([img, ...])       [N, 17, 3] COCO body keypoints (x, y px, score), RGB uint8 inputs
#   arm, leg = limb_diff(kp_a, kp_b)          mean limb-angle difference (deg) of the arms / legs, None if unseen
import numpy as np
import torch
from PIL import Image

import models

PATH = models.path(models.POSE)
SIZE = (288, 384)   # model input w, h
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
BS = 5              # the TorchScript file was traced with batch 5
CONF = 0.3          # keypoint score below this = not seen (DWPose's own threshold)
LIMBS = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16))   # arms (upper, fore) x2, legs x2


class Pose:
    def __init__(self, path=PATH):
        self.m = torch.jit.load(path, map_location='cuda').eval()
        self.dtype = next(self.m.parameters()).dtype

    @staticmethod
    def _crop(img, box):
        x0, y0, x1, y1 = box
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        s = np.array([x1 - x0, y1 - y0], np.float32) * 1.25
        w, h = SIZE
        s = np.array([s[0], s[0] * h / w] if s[0] > s[1] * w / h else [s[1] * w / h, s[1]], np.float32)
        ext = (c[0] - s[0] / 2, c[1] - s[1] / 2, c[0] + s[0] / 2, c[1] + s[1] / 2)
        a = np.asarray(Image.fromarray(img).transform(SIZE, Image.Transform.EXTENT, ext, Image.Resampling.BILINEAR),
                       np.float32)
        return (a[..., ::-1] - MEAN) / STD, c, s          # the model expects BGR

    @torch.no_grad()
    def _run(self, imgs, boxes):
        crops = [self._crop(i, b) for i, b in zip(imgs, boxes)]
        x = np.stack([c[0] for c in crops]).transpose(0, 3, 1, 2)
        n = len(x)
        if n % BS:
            x = np.concatenate([x, np.zeros((BS - n % BS,) + x.shape[1:], x.dtype)])
        x = torch.from_numpy(np.ascontiguousarray(x)).to('cuda', self.dtype)
        sx, sy = [], []
        for i in range(0, len(x), BS):
            o = self.m(x[i:i + BS])
            sx.append(o[0].float())
            sy.append(o[1].float())
        vx, ix = torch.cat(sx)[:n].max(-1)
        vy, iy = torch.cat(sy)[:n].max(-1)
        kp = torch.stack([ix, iy], -1).float().cpu().numpy() / 2.0
        sc = torch.minimum(vx, vy).cpu().numpy()
        return [(kp[k] / np.array(SIZE) * s + c - s / 2, sc[k]) for k, (_, c, s) in enumerate(crops)]

    def body(self, imgs):
        H, W = imgs[0].shape[:2]
        boxes = []
        for k, s in self._run(imgs, [(0, 0, W, H)] * len(imgs)):
            ok = s[:17] > CONF
            if ok.sum() < 4:
                boxes.append((0, 0, W, H))
                continue
            (x0, y0), (x1, y1) = k[:17][ok].min(0), k[:17][ok].max(0)
            bw, bh = x1 - x0, y1 - y0
            boxes.append((x0 - 0.1 * bw, y0 - 0.15 * bh, x1 + 0.1 * bw, y1 + 0.05 * bh))   # head top, feet
        return np.stack([np.concatenate([k[:17], s[:17, None]], 1) for k, s in self._run(imgs, boxes)])


def ankles_seen(kp):
    """True if either ankle (COCO 15, 16) of one [17, 3] keypoint set is seen — a master meant to end below the hips
    went full length (char.ref framing check)."""
    return bool((kp[[15, 16], 2] > CONF).any())


def shot_size(kp, h):
    """Shot size of one [17, 3] keypoint set in an image h px high: 'full' if an ankle (COCO 15, 16) is seen inside the
    frame, 'medium' if a hip (11, 12), else 'close' (video.swap picks the boundary edit's character picture by it). Knees
    don't count: a frame cut at mid-thigh gave a knee on the bottom edge (swap 16 retest, session #12)."""
    s = kp[:, 2] > CONF
    return 'full' if any(s[i] and kp[i, 1] < 0.97 * h for i in (15, 16)) else 'medium' if s[11:13].any() else 'close'


def leg_ratio(kp):
    """(hip → ankle) / (nose → ankle) on the vertical axis of one standing [17, 3] keypoint set, None if the nose, a hip or
    an ankle is not seen. Normal adult ~0.55; a body squeezed head-to-shoes into the frame gives short legs (char.sheet
    full-body panel check)."""
    s = kp[:, 2] > CONF
    hips, ank = [i for i in (11, 12) if s[i]], [i for i in (15, 16) if s[i]]
    if not (s[0] and hips and ank):
        return None
    ya, yh = kp[ank, 1].mean(), kp[hips, 1].mean()
    return round(float((ya - yh) / max(ya - kp[0, 1], 1.0)), 3)


def limb_angles(kp):
    """[..., 8] limb directions (deg) of [..., 17, 3] keypoints, NaN where an end is not seen."""
    i, j = np.array(LIMBS).T
    d = kp[..., j, :2] - kp[..., i, :2]
    a = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
    return np.where((kp[..., i, 2] > CONF) & (kp[..., j, 2] > CONF), a, np.nan)


def limb_diff(a, b):
    """(arm, leg) mean absolute limb-angle difference in degrees between two keypoint sets; None if no limb is seen in
    both."""
    d = np.abs((limb_angles(b) - limb_angles(a) + 180) % 360 - 180)
    m = lambda x: round(float(np.nanmean(x)), 1) if np.isfinite(x).any() else None
    return m(d[..., :4]), m(d[..., 4:])
