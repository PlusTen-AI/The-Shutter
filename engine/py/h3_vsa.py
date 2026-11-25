# Option vsa (session #12): block-sparse attention for the pi-H3 DiT, the way ComfyUI's "Model Sparse Attention" node
# runs it with method vsa (comfy_extras/nodes_sparse_attention.py, ComfyUI 0.45.0 + comfy-kitchen 0.2.41 — the same
# comfy-kitchen build is in our image). Blocks outside dense_blocks hand their qkv projection rows to the engine's
# attention hook (h3_dit.cu); here they are put in VSA tile order (prefix segments in 64-row tiles, video in 4x4x4
# cubes), comfy-kitchen's sol_attn_chunked does q/k norm + rope + top-k block routing + attention (the call the node
# makes), and the rows go back to packed order. Settings = the boss's screenshot (drama 40·41): keep 10 %, start 0 /
# end 1 (every pi-H3 sigma), dense_blocks 0, 1, min_tokens 12288; extra_tokens is ignored for vsa, sink_conditioning
# too (every prefix tile is exact as key and query). Our FL2VA weights have no to_gate_compress layers: no coarse branch.
# Per block, the K mean / V scale of the last step feed the next one (reset per job, as the node's ON_CLEANUP).
import ctypes

import torch

import comfy_kitchen as ck

import h3

BLOCK, CUBE, CHUNK = 64, (4, 4, 4), 4096
NB, NH, HD, ROT = 50, 56, 128, 48
QKV, IN = 3 * NH * HD, NH * HD
EPS = 1e-5            # q / k RMSNorm eps of the H3 blocks (h3_dit.cu EPS)
TAU = 1.3             # the node's tau for non-sol methods (selection.get("tau", 1.3)); ignored with topk_ratio
CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)


def _hook_api():
    L = h3.lib()
    if not getattr(L, "_vsa", False):
        P = ctypes.c_void_p
        L.h3_dit_attn_hook.argtypes = [P, CB, P, P, ctypes.POINTER(ctypes.c_ubyte)]
        L.h3_dit_rope_copy.argtypes = [P, P]
        L.h3_dit_qk_norm_copy.argtypes = [P, ctypes.c_int, P, P]
        L._vsa = True
    return L


def plan(segs, grid, S, device="cuda"):
    """The node's vsa_plan: padded tile order. src [n] = packed row per padded row (-1 = pad), inv [S] = padded row per
    packed row, block_len [n / 64] = live rows per tile, n_prefix = tiles before the video cubes."""
    tiles, n_prefix = [], 0
    for a, b, kind in segs:
        n = b - a
        if kind != "video":
            m = (n + BLOCK - 1) // BLOCK
            seg = torch.full((m * BLOCK,), -1, dtype=torch.int64, device=device)
            seg[:n] = torch.arange(a, b, device=device)
            tiles.append(seg.view(m, BLOCK))
            n_prefix += m
            continue
        if grid[0] * grid[1] * grid[2] != n:
            raise RuntimeError(f"vsa: video segment of {n} rows does not match the latent grid {grid}")
        ct, ch, cw = CUBE
        pt, ph, pw = ((g + c - 1) // c * c for g, c in zip(grid, CUBE))
        padded = torch.full((pt, ph, pw), -1, dtype=torch.int64, device=device)
        padded[:grid[0], :grid[1], :grid[2]] = torch.arange(a, b, device=device).view(*grid)
        cubes = padded.view(pt // ct, ct, ph // ch, ch, pw // cw, cw).permute(0, 2, 4, 1, 3, 5).reshape(-1, BLOCK)
        order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
        tiles.append(torch.gather(cubes, 1, order))
    tiles = torch.cat(tiles)
    src = tiles.reshape(-1)
    live = src >= 0
    inv = torch.empty(S, dtype=torch.int64, device=device)
    inv[src[live]] = torch.nonzero(live).flatten()
    return {"n": int(src.numel()), "n_prefix": n_prefix, "src": src, "inv": inv,
            "block_len": (tiles >= 0).sum(1).to(torch.int32)}


class Vsa:
    """Attach to a prepared h3.Dit (after Dit.prepare, before sampling); detach() restores the normal attention."""

    def __init__(self, dit, keep=0.10, dense_blocks=(0, 1), min_tokens=12288):
        self.dit, self.keep, self.err = dit, float(keep), None
        L = _hook_api()
        S, (T, H, W) = dit.S, dit.shape
        on = [int(S >= min_tokens and i not in dense_blocks) for i in range(NB)]
        self.blocks = sum(on)
        if not self.blocks:
            L.h3_dit_attn_hook(dit.h, CB(), None, None, None)
            return
        self.p = plan(dit.segs, (T, (H + 1) // 2, (W + 1) // 2), S)
        n, inv = self.p["n"], self.p["inv"]
        pe = torch.empty(S, ROT, 2, device="cuda")
        L.h3_dit_rope_copy(dit.h, h3._ptr(pe))
        c, s = pe[..., 0], pe[..., 1]
        f = torch.zeros(1, n, 1, ROT, 2, 2, device="cuda")   # ComfyUI's rope_freqs rows [[cos, -sin], [sin, cos]]
        f[0, inv, 0] = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
        self.freqs = f
        self.norms = []
        for i in range(NB):
            q, k = (torch.empty(HD, dtype=torch.bfloat16, device="cuda") for _ in range(2))
            if on[i]:
                L.h3_dit_qk_norm_copy(dit.h, i, h3._ptr(q), h3._ptr(k))
            self.norms.append((q, k))
        self.qkv = torch.empty(S, QKV, dtype=torch.bfloat16, device="cuda")
        self.att = torch.empty(S, IN, dtype=torch.bfloat16, device="cuda")
        self.state = {}
        self._cb = CB(self._call)                     # kept alive while the hook is set
        torch.cuda.synchronize()
        L.h3_dit_attn_hook(dit.h, self._cb, h3._ptr(self.qkv), h3._ptr(self.att), (ctypes.c_ubyte * NB)(*on))

    def _call(self, i):
        try:
            p, src = self.p, self.p["src"]
            n = p["n"]

            def chunks():   # padded tile order; pad rows are zero (the projection has no bias, as the node's x * 0)
                for a in range(0, n, CHUNK):
                    idx = src[a:a + CHUNK]
                    yield self.qkv[idx.clamp_min(0)] * (idx >= 0).unsqueeze(1).to(torch.bfloat16)

            km, vs = self.state.get(i, (None, None))
            out, km, vs = ck.sol_attn_chunked(
                chunks, n, NH, self.freqs, self.norms[i], kmean=km, vscale=vs, tau=TAU, topk_ratio=self.keep,
                token_aug=0, sink_blocks=[0, p["n_prefix"]], sink_q=[0, p["n_prefix"]], rope_eps=EPS, tail=False,
                block_len=p["block_len"])
            self.state[i] = (km, vs)
            torch.index_select(out.view(n, IN), 0, p["inv"], out=self.att)
            torch.cuda.synchronize()
            return 0
        except BaseException as e:   # surfaced by check() after the forward
            self.err = self.err or e
            return 1

    def check(self):
        if self.err is not None:
            e, self.err = self.err, None
            raise RuntimeError("vsa attention failed") from e

    def detach(self):
        _hook_api().h3_dit_attn_hook(self.dit.h, CB(), None, None, None)
        self.qkv = self.att = self.freqs = None
        self.state = {}
