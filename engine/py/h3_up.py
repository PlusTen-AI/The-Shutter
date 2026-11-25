# MiniMax H3 latent upscaler (LBH-123-AI "3D conv v1"), plain-PyTorch port of ComfyUI comfy/ldm/minimax/latent_upscaler.py.
# Same module names as the checkpoint. ComfyUI runs GroupNorm with clip-wide statistics in slabs (kitchen kernels);
# nn.GroupNorm over the whole [B, C, T, H, W] clip computes the same statistics in one pass.
#   up = Upscaler(path); v2 = up(v, (T, H2, W2))   # v: sampler-space H3 video latent [1, 24, T, H, W]
# The 3D model is best at exactly 2x (model card: 3D wins only at 2x / 2.25x / 2.5x / 4x).
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

MEAN = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
        -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
        -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
        -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
        -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
        -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
STD = [1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
       1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
       0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
       0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
       2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
       3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525]


class ResBlock(nn.Module):
    def __init__(self, c, e):
        super().__init__()
        self.in_layers = nn.Sequential(nn.GroupNorm(32, c), nn.Identity(), nn.Conv3d(c, c, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(e, 2 * c))
        self.out_norm = nn.GroupNorm(32, c)
        self.out_layers = nn.Sequential(nn.Identity(), nn.Identity(), nn.Conv3d(c, c, 3, padding=1))

    def forward(self, x, emb):
        scale, shift = (t.view(1, -1, 1, 1, 1) for t in self.emb_layers(emb)[0].chunk(2))
        h = self.in_layers[2](F.silu(self.in_layers[0](x)))
        h = F.silu(self.out_norm(h) * (1 + scale) + shift)   # the modulation folded into the norm affine
        return x + self.out_layers[2](h)


class TemporalConv(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.norm = nn.GroupNorm(32, c)
        self.dwconv = nn.Conv3d(c, c, (k, 1, 1), padding=(k // 2, 0, 0), groups=c)
        self.pwconv = nn.Conv3d(c, c, 1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class Upscaler(nn.Module):
    def __init__(self, path, device="cuda", dtype=torch.float16):
        super().__init__()
        sd = load_file(path)
        ch, cin = sd["conv_in.weight"].shape[:2]
        nb = {p: len({k.split(".")[1] for k in sd if k.startswith(p + ".") and ".in_layers." in k}) for p in ("in_blocks", "out_blocks")}
        k = sd["in_blocks.1.dwconv.weight"].shape[2]
        self.conv_in = nn.Conv3d(cin, ch, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        for p in ("in_blocks", "out_blocks"):
            blocks = nn.ModuleList()
            for i in range(nb[p]):
                blocks.append(ResBlock(ch, 64))
                if i % 2 == 0:
                    blocks.append(TemporalConv(ch, k))
            setattr(self, p, blocks)
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv3d(ch, cin, 3, padding=1)
        self.load_state_dict(sd)
        self.to(device=device, dtype=dtype).eval()
        self.dtype = dtype

    @staticmethod
    def _run(blocks, x, emb):
        for b in blocks:
            x = b(x, emb) if isinstance(b, ResBlock) else b(x)
        return x

    @torch.inference_mode()
    def forward(self, x, size):
        """sampler-space H3 video latent [B, 24, T, H, W] -> the same at size (T, H', W'), float32."""
        dt, dev = self.dtype, self.conv_in.weight.device
        x = x.to(device=dev, dtype=dt)
        mean = torch.tensor(MEAN, device=dev, dtype=dt).view(1, -1, 1, 1, 1)
        std = torch.tensor(STD, device=dev, dtype=dt).view(1, -1, 1, 1, 1)
        scale = (size[-1] / x.shape[-1] + size[-2] / x.shape[-2]) / 2
        x = (x - mean) / std                     # trained on latents normalized a second time with the same stats
        emb = self.embed(torch.full((1, 1), scale - 1, device=dev, dtype=dt))
        x = self._run(self.in_blocks, self.conv_in(x), emb)
        x = F.interpolate(x, size=tuple(size), mode="trilinear", align_corners=False)
        x = self._run(self.out_blocks, x, emb)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return torch.addcmul(mean, x, std).float()
