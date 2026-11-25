# Face identity check (#18): InsightFace buffalo_l (SCRFD det_10g + ArcFace w600k_r50) run with torch.
# The ONNX graphs are executed by a tiny interpreter (onnx + torch only; no onnxruntime/cv2 in the worker image).
# Pre/post-processing follows insightface's scrfd.py / arcface_onnx.py / face_align.py. Images are RGB.

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

import models

FACE_DIR = models.path(models.FACE)
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float32)


def _f32(t):
    """fp16 weights/constants (hyperswap is stored in fp16) are computed in fp32."""
    return t.float() if t.dtype == torch.float16 else t


class OnnxTorch:
    def __init__(self, path, device="cuda"):
        g = onnx.load(path).graph
        self.v0 = {t.name: _f32(torch.from_numpy(numpy_helper.to_array(t).copy())).to(device) for t in g.initializer}
        self.nodes = [(n.op_type, list(n.input), list(n.output),
                       {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}) for n in g.node]
        self.inps = [i.name for i in g.input if i.name not in self.v0]
        self.inp = self.inps[0]
        self.device = device
        self.out = [o.name for o in g.output]

    @torch.no_grad()
    def __call__(self, x=None, **feeds):
        """One input positionally, or several by graph input name (face swappers: target=..., source=...)."""
        v = dict(self.v0)
        v.update(feeds if feeds else {self.inp: x})
        for op, ins, outs, a in self.nodes:
            r = getattr(self, "op_" + op)([v[i] if i else None for i in ins], a)
            for name, t in zip(outs, r if isinstance(r, tuple) else (r,)):
                v[name] = t
        return [v[o] for o in self.out]

    # ---- ops used by det_10g / w600k_r50 (opset 11) ----
    def op_Conv(self, I, a):
        p = a.get("pads", [0, 0, 0, 0])
        return F.conv2d(I[0], I[1], I[2] if len(I) > 2 else None, a.get("strides", 1), (p[0], p[1]),
                        a.get("dilations", 1), a.get("group", 1))
    def op_Relu(self, I, a): return F.relu(I[0])
    def op_Sigmoid(self, I, a): return torch.sigmoid(I[0])
    @staticmethod
    def _pair(I):
        """Binary-op operands on one device: a 0-d (cpu scalar constant) operand follows the other one."""
        x, y = I[0], I[1]
        if x.device != y.device:
            if x.dim() == 0:
                x = x.to(y.device)
            else:
                y = y.to(x.device)
        return x, y
    def op_Add(self, I, a): x, y = self._pair(I); return x + y
    def op_Mul(self, I, a): x, y = self._pair(I); return x * y
    def op_MaxPool(self, I, a):
        return F.max_pool2d(I[0], a["kernel_shape"], a.get("strides"), a.get("pads", [0, 0])[:2], ceil_mode=bool(a.get("ceil_mode", 0)))
    def op_AveragePool(self, I, a):
        return F.avg_pool2d(I[0], a["kernel_shape"], a.get("strides"), a.get("pads", [0, 0])[:2],
                            ceil_mode=bool(a.get("ceil_mode", 0)), count_include_pad=bool(a.get("count_include_pad", 0)))
    def op_Shape(self, I, a): return torch.tensor(list(I[0].shape), dtype=torch.int64)
    def op_Gather(self, I, a):
        ax, idx = a.get("axis", 0), I[1].to(I[0].device)
        idx = torch.where(idx < 0, idx + I[0].shape[ax], idx)
        out = torch.index_select(I[0], ax, idx.reshape(-1))
        return out.reshape(I[0].shape[:ax % I[0].dim()] + idx.shape + I[0].shape[ax % I[0].dim() + 1:])
    def op_Unsqueeze(self, I, a):
        x = I[0]
        axes = a["axes"] if "axes" in a else I[1].tolist()
        for ax in sorted(ax % (x.dim() + len(axes)) for ax in axes):
            x = x.unsqueeze(ax)
        return x
    # ---- more ops: yolov10m (person detection) and ViTPose-H wholebody (Wan Animate preprocessing) ----
    def op_Sub(self, I, a): x, y = self._pair(I); return x - y
    def op_Div(self, I, a):
        x, y = self._pair(I)
        return torch.div(x, y, rounding_mode="trunc") if not (x.is_floating_point() or y.is_floating_point()) else x / y
    def op_Mod(self, I, a):
        x, y = I[0], I[1].to(I[0].device)
        return torch.fmod(x, y) if a.get("fmod", 0) else torch.remainder(x, y)
    def op_Pow(self, I, a): x, y = self._pair(I); return torch.pow(x, y)
    def op_Sqrt(self, I, a): return torch.sqrt(I[0])
    def op_Erf(self, I, a): return torch.erf(I[0])
    def op_MatMul(self, I, a): return torch.matmul(I[0], I[1].to(I[0].device))
    def op_Softmax(self, I, a): return torch.softmax(I[0], a.get("axis", -1))
    def op_Constant(self, I, a):
        t = a["value"]
        return _f32(torch.from_numpy(numpy_helper.to_array(t).copy())).to(self.device if t.dims else "cpu")
    def op_Cast(self, I, a):
        if a["to"] == 10 and I[0].is_floating_point():   # fp16 islands (hyperswap) stay fp32: weights are fp32
            return I[0]
        return I[0].to({1: torch.float32, 6: torch.int32, 7: torch.int64, 9: torch.bool, 10: torch.float16, 11: torch.float64}[a["to"]])
    def op_ReduceMean(self, I, a):
        axes = a["axes"] if "axes" in a else I[1].tolist()
        return I[0].mean(dim=axes, keepdim=bool(a.get("keepdims", 1)))
    def op_ReduceMax(self, I, a):
        axes = a["axes"] if "axes" in a else (I[1].tolist() if len(I) > 1 and I[1] is not None else list(range(I[0].dim())))
        return torch.amax(I[0], dim=axes, keepdim=bool(a.get("keepdims", 1)))
    def op_Split(self, I, a):
        ax = a.get("axis", 0)
        sizes = a["split"] if "split" in a else (I[1].tolist() if len(I) > 1 and I[1] is not None else None)
        if sizes is None:
            n = a["num_outputs"]
            sizes = [-(-I[0].shape[ax] // n)] * n
        return tuple(torch.split(I[0], sizes, ax))
    def op_Tile(self, I, a): return I[0].repeat(*I[1].tolist())
    def op_TopK(self, I, a):
        r = torch.topk(I[0], int(I[1].reshape(-1)[0]), dim=a.get("axis", -1), largest=bool(a.get("largest", 1)),
                       sorted=bool(a.get("sorted", 1)))
        return r.values, r.indices
    def op_GatherElements(self, I, a): return torch.gather(I[0], a.get("axis", 0), I[1].to(I[0].device))
    def op_ConvTranspose(self, I, a):
        p = a.get("pads", [0, 0, 0, 0])
        if p[0] == p[2] and p[1] == p[3]:
            return F.conv_transpose2d(I[0], I[1], I[2] if len(I) > 2 else None, a.get("strides", 1), (p[0], p[1]),
                                      a.get("output_padding", 0), a.get("group", 1), a.get("dilations", 1))
        # asymmetric pads (XSeg: 0,0,1,1): full transpose conv, then crop begin/end pads per axis
        y = F.conv_transpose2d(I[0], I[1], I[2] if len(I) > 2 else None, a.get("strides", 1), 0,
                               a.get("output_padding", 0), a.get("group", 1), a.get("dilations", 1))
        return y[..., p[0]:y.shape[-2] - p[2], p[1]:y.shape[-1] - p[3]]
    def op_Slice(self, I, a):
        x, st, en = I[0], I[1].tolist(), I[2].tolist()
        axes = I[3].tolist() if len(I) > 3 and I[3] is not None else list(range(len(st)))
        steps = I[4].tolist() if len(I) > 4 and I[4] is not None else [1] * len(st)
        sl = [slice(None)] * x.dim()
        for s, e, ax, sp in zip(st, en, axes, steps):
            sl[ax] = slice(s, min(e, x.shape[ax]), sp)
        return x[tuple(sl)]
    def op_Concat(self, I, a): return torch.cat([t.to(I[0].device) for t in I], a["axis"])
    def op_Resize(self, I, a):
        x = I[0]
        mode = a.get("mode", b"nearest").decode()
        kw = {"mode": "nearest"} if mode == "nearest" else \
             {"mode": "bilinear", "align_corners": a.get("coordinate_transformation_mode", b"") == b"align_corners"}
        if I[2] is not None and I[2].numel():
            return F.interpolate(x, scale_factor=tuple(I[2].tolist()[2:]), **kw)
        return F.interpolate(x, size=tuple(int(s) for s in I[3].tolist()[2:]), **kw)
    def op_Transpose(self, I, a): return I[0].permute(*a["perm"])
    def op_Reshape(self, I, a):
        shp = [int(s) if s != 0 else I[0].shape[i] for i, s in enumerate(I[1].tolist())]
        return I[0].reshape(shp)
    def op_PRelu(self, I, a): return torch.where(I[0] >= 0, I[0], I[0] * I[1])
    def op_BatchNormalization(self, I, a):
        return F.batch_norm(I[0], I[3], I[4], I[1], I[2], False, 0.0, a.get("epsilon", 1e-5))
    def op_Flatten(self, I, a): return I[0].flatten(a.get("axis", 1))
    def op_Gemm(self, I, a):
        x = I[0].t() if a.get("transA", 0) else I[0]
        w = I[1].t() if a.get("transB", 0) else I[1]
        y = a.get("alpha", 1.0) * (x @ w)
        return y + a.get("beta", 1.0) * I[2] if len(I) > 2 and I[2] is not None else y
    # ---- more ops: face swappers (inswapper_128 opset 11, hyperswap_1a_256 opset 15) ----
    def op_LeakyRelu(self, I, a): return F.leaky_relu(I[0], a.get("alpha", 0.01))
    def op_Tanh(self, I, a): return torch.tanh(I[0])
    def op_Pad(self, I, a):
        p = I[1].tolist() if len(I) > 1 and I[1] is not None else a["pads"]
        n = len(p) // 2
        tp = [v for d in reversed(range(n)) for v in (p[d], p[d + n])]   # ONNX begins..ends -> torch last dim first
        mode = {"constant": "constant", "reflect": "reflect", "edge": "replicate"}[a.get("mode", b"constant").decode()]
        while tp and tp[-2:] == [0, 0]:
            tp = tp[:-2]                                                  # torch reflect pads only trailing dims
        if mode == "constant" and len(I) > 2 and I[2] is not None:
            return F.pad(I[0], tp, mode, float(I[2]))
        return F.pad(I[0], tp, mode)
    # ---- ops used by FaceFusion xseg_1 (filter response normalization + TLU) ----
    def op_GlobalAveragePool(self, I, a): return I[0].mean((2, 3), keepdim=True)
    def op_Reciprocal(self, I, a): return 1.0 / I[0]
    def op_Max(self, I, a):
        y = I[0]
        for t in I[1:]:
            y = torch.maximum(*self._pair([y, t]))
        return y
    def op_InstanceNormalization(self, I, a): return F.instance_norm(I[0], weight=I[1], bias=I[2], eps=a.get("epsilon", 1e-5))
    def op_Expand(self, I, a):
        x = I[0].to(self.device) if I[0].is_floating_point() else I[0]
        return x.expand(torch.broadcast_shapes(tuple(x.shape), tuple(I[1].tolist())))
    def op_DepthToSpace(self, I, a):
        assert a.get("mode", b"DCR") == b"CRD", "DepthToSpace DCR not needed so far"
        return F.pixel_shuffle(I[0], a["blocksize"])


def _nms(boxes, scores, thr):
    order, keep = scores.argsort()[::-1], []
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    while order.size:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter) <= thr]
    return keep


def _similarity(src, dst):
    """Umeyama similarity transform src->dst (2x3), as skimage SimilarityTransform.estimate."""
    n = src.shape[0]
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    A = d.T @ s / n
    U, S, Vt = np.linalg.svd(A)
    D = np.eye(2)
    if np.linalg.det(A) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = (S * np.diag(D)).sum() / (s ** 2).sum() * n
    t = md - scale * R @ ms
    return np.hstack([scale * R, t[:, None]])


class FaceCheck:
    def __init__(self, device="cuda"):
        self.det = OnnxTorch(f"{FACE_DIR}/det_10g.onnx", device)
        self.rec = OnnxTorch(f"{FACE_DIR}/w600k_r50.onnx", device)
        self.device = device

    def detect(self, img, thresh=0.5, size=640):
        """img: HxWx3 RGB uint8 numpy. Returns [(box[4], score, kps[5,2])] sorted by area, largest first."""
        h, w = img.shape[:2]
        if h / w > 1:
            nh, nw = size, int(size / (h / w))
        else:
            nw, nh = size, int(size * (h / w))
        scale = nh / h
        t = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float()
        t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False, antialias=False)
        blob = torch.zeros(1, 3, size, size, device=self.device)
        blob[:, :, :nh, :nw] = (t - 127.5) / 128.0
        outs = [o.float().cpu().numpy() for o in self.det(blob)]
        boxes, scores, kpss = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            sc, bb, kp = outs[i][:, 0], outs[i + 3] * stride, outs[i + 6] * stride
            g = size // stride
            c = np.stack(np.mgrid[:g, :g][::-1], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            c = np.repeat(c, 2, axis=0)
            m = sc >= thresh
            b = np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1], c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], -1)
            k = (np.tile(c, 5) + kp).reshape(-1, 5, 2)
            boxes.append(b[m]), scores.append(sc[m]), kpss.append(k[m])
        boxes, scores, kpss = np.concatenate(boxes) / scale, np.concatenate(scores), np.concatenate(kpss) / scale
        keep = _nms(boxes, scores, 0.4) if len(scores) else []
        faces = [(boxes[i], float(scores[i]), kpss[i]) for i in keep]
        return sorted(faces, key=lambda f: -(f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))

    def visible(self, img, min_luma=60):
        """Faces that are lit: a silhouette's face box averages 34~39, a lit face 99~117 (measured)."""
        return [f for f in self.detect(img)
                if img[max(0, int(f[0][1])):int(f[0][3]), max(0, int(f[0][0])):int(f[0][2])].mean() >= min_luma]

    def embed(self, img, face=None):
        """Largest face (or the given detect() face) -> L2-normalized 512-d embedding (torch, cpu), or None if no face."""
        faces = [face] if face is not None else self.detect(img)
        if not faces:
            return None
        M = _similarity(faces[0][2].astype(np.float64), ARCFACE_DST.astype(np.float64))
        Minv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]           # output pixel -> source pixel
        h, w = img.shape[:2]
        uu, vv = np.meshgrid(np.arange(112), np.arange(112))
        src = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3) @ Minv.T
        grid = torch.from_numpy(np.stack([src[:, 0] * 2 / (w - 1) - 1, src[:, 1] * 2 / (h - 1) - 1], -1)
                                .reshape(1, 112, 112, 2)).float().to(self.device)
        t = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float()
        crop = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        e = self.rec((crop - 127.5) / 127.5)[0][0].float().cpu()
        return e / e.norm()

    @staticmethod
    def similarity(a, b):
        return float((a * b).sum())
