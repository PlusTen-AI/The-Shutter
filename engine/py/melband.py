# Mel-Band RoFormer, inference only — Kim's vocal model (KimberleyJensen/Mel-Band-Roformer-Vocal-Model,
# fp16 safetensors from Kijai/MelBandRoFormer_comfy). The module tree (and so the state-dict keys) is that of
# kijai/ComfyUI-MelBandRoFormer model/mel_band_roformer.py (= Kim's file with torch SDPA in place of Attend), which is
# lucidrains/BS-RoFormer's MelBandRoformer; the rotary embedding is a minimal copy of lucidrains/rotary-embedding-torch
# (only `freqs` is a stored parameter). Training loss, beartype checks and the librosa import of the mel filter stay out.
#
# MIT License
#
# Copyright (c) 2023 Phil Wang (BS-RoFormer)  ·  Copyright (c) 2021 Phil Wang (rotary-embedding-torch)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from torch import nn

# configs/config_vocals_mel_band_roformer.yaml (Kim) = the dict kijai's loader builds (dropout 0, stereo, 44.1 kHz)
KIM = dict(dim=384, depth=6, stereo=True, num_stems=1, time_transformer_depth=1, freq_transformer_depth=1, num_bands=60,
           dim_head=64, heads=8, sample_rate=44100, stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048,
           mask_estimator_depth=2)
RATE = 44100
CHUNK = 352800        # inference.chunk_size (8 s)
OVERLAP = 2           # inference.num_overlap


def mel_filter(sr, n_fft, n_mels):
    """librosa.filters.mel(sr=, n_fft=, n_mels=) with its defaults (Slaney mel, slaney norm, fmin 0, fmax sr/2)."""
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class Rotary(nn.Module):
    """rotary_embedding_torch.RotaryEmbedding(dim) as used here: freqs_for 'lang', theta 10000, interleaved pairs,
    rotate_queries_or_keys on the sequence dim -2, positions 0..n-1."""

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.freqs = nn.Parameter(1. / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)), requires_grad=False)

    def rotate_queries_or_keys(self, t):
        pos = torch.arange(t.shape[-2], device=t.device, dtype=t.dtype)
        f = torch.einsum('n,f->nf', pos.type(self.freqs.dtype), self.freqs)
        f = repeat(f, 'n f -> n (f r)', r=2).to(t)
        x1, x2 = rearrange(t, '... (d r) -> ... d r', r=2).unbind(dim=-1)
        rot = rearrange(torch.stack((-x2, x1), dim=-1), '... d r -> ... (d r)')
        return t * f.cos() + rot * f.sin()


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, inner), nn.GELU(), nn.Dropout(0.), nn.Linear(inner, dim),
                                 nn.Dropout(0.))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, rotary_embed=None):
        super().__init__()
        self.heads = heads
        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, heads * dim_head * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, dim, bias=False), nn.Dropout(0.))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out * rearrange(self.to_gates(x), 'b n h -> b h n 1').sigmoid()
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class Transformer(nn.Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, rotary_embed=None):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads, dim_head, rotary_embed), FeedForward(dim)])
                                     for _ in range(depth)])
        self.norm = RMSNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in dim_inputs])

    def forward(self, x):
        return torch.stack([f(s) for s, f in zip(x.split(self.dim_inputs, dim=-1), self.to_features)], dim=-2)


def mlp(dim_in, dim_out, dim_hidden, depth):
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)
    net = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        net.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            net.append(nn.Tanh())
    return nn.Sequential(*net)


class MaskEstimator(nn.Module):
    def __init__(self, dim, dim_inputs, depth, mlp_expansion_factor=4):
        super().__init__()
        self.to_freqs = nn.ModuleList([nn.Sequential(mlp(dim, d * 2, dim * mlp_expansion_factor, depth), nn.GLU(dim=-1))
                                       for d in dim_inputs])

    def forward(self, x):
        return torch.cat([f(b) for b, f in zip(x.unbind(dim=-2), self.to_freqs)], dim=-1)


class MelBandRoformer(nn.Module):
    def __init__(self, dim, *, depth, stereo=False, num_stems=1, time_transformer_depth=2, freq_transformer_depth=2,
                 num_bands=60, dim_head=64, heads=8, sample_rate=44100, stft_n_fft=2048, stft_hop_length=512,
                 stft_win_length=2048, mask_estimator_depth=1):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        kw = dict(dim=dim, heads=heads, dim_head=dim_head)
        time_rot, freq_rot = Rotary(dim_head), Rotary(dim_head)
        self.layers = nn.ModuleList([nn.ModuleList([Transformer(depth=time_transformer_depth, rotary_embed=time_rot, **kw),
                                                    Transformer(depth=freq_transformer_depth, rotary_embed=freq_rot, **kw)])
                                     for _ in range(depth)])
        self.stft_win_length = stft_win_length
        self.stft_kwargs = dict(n_fft=stft_n_fft, hop_length=stft_hop_length, win_length=stft_win_length, normalized=False)
        freqs = stft_n_fft // 2 + 1

        fb = torch.from_numpy(mel_filter(sample_rate, stft_n_fft, num_bands))
        fb[0][0] = 1.     # as upstream: the first bin is not covered otherwise
        fb[-1, -1] = 1.   # some envs give 0.0 instead of ~1.9e-18 here
        freqs_per_band = fb > 0
        assert freqs_per_band.any(dim=0).all()
        freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)[freqs_per_band]
        if stereo:
            freq_indices = rearrange(repeat(freq_indices, 'f -> f s', s=2) * 2 + torch.arange(2), 'f s -> (f s)')
        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('num_bands_per_freq', reduce(freqs_per_band, 'b f -> f', 'sum'), persistent=False)
        dims = tuple(2 * f * self.audio_channels for f in reduce(freqs_per_band, 'b f -> b', 'sum').tolist())
        self.band_split = BandSplit(dim, dims)
        self.mask_estimators = nn.ModuleList([MaskEstimator(dim, dims, mask_estimator_depth) for _ in range(num_stems)])

    def forward(self, raw_audio):
        """raw_audio [b, s, t] → separated stem [b, s, t'] (t' = t for t a multiple of the hop)."""
        device = raw_audio.device
        batch, channels, _ = raw_audio.shape
        raw_audio, ps = pack([raw_audio], '* t')
        window = torch.hann_window(self.stft_win_length, device=device)
        stft = torch.view_as_real(torch.stft(raw_audio, **self.stft_kwargs, window=window, return_complex=True))
        stft = rearrange(unpack(stft, ps, '* f t c')[0], 'b s f t c -> b (f s) t c')
        x = stft[torch.arange(batch, device=device)[..., None], self.freq_indices]
        x = self.band_split(rearrange(x, 'b f t c -> b t (f c)'))
        for time_tf, freq_tf in self.layers:   # axial attention: time, then frequency
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')
            x = unpack(time_tf(x), ps, '* t d')[0]
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')
            x = unpack(freq_tf(x), ps, '* f d')[0]
        n = len(self.mask_estimators)
        masks = rearrange(torch.stack([fn(x) for fn in self.mask_estimators], dim=1), 'b n t (f c) -> b n f t c', c=2)
        stft = torch.view_as_complex(rearrange(stft, 'b f t c -> b 1 f t c'))
        masks = torch.view_as_complex(masks).type(stft.dtype)
        idx = repeat(self.freq_indices, 'f -> b n f t', b=batch, n=n, t=stft.shape[-1])
        summed = torch.zeros_like(repeat(stft, 'b 1 ... -> b n ...', n=n)).scatter_add_(2, idx, masks)
        stft = stft * (summed / repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels).clamp(min=1e-8))
        stft = rearrange(stft, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)
        out = torch.istft(stft, **self.stft_kwargs, window=window, return_complex=False)
        out = rearrange(out, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=n)
        return out[:, 0] if n == 1 else out


def load(path, device='cuda'):
    """Kim's model from the fp16 safetensors (strict, as kijai's loader); runs in fp32 like kijai's node."""
    from safetensors.torch import load_file
    m = MelBandRoformer(**KIM).eval()
    m.load_state_dict(load_file(path), strict=True)
    return m.to(device)


def window(size, fade, device):
    w = torch.ones(size)
    w[-fade:] *= torch.linspace(1, 0, fade)
    w[:fade] *= torch.linspace(0, 1, fade)
    return w.to(device)


@torch.inference_mode()
def demix(model, mix, chunk=CHUNK, overlap=OVERLAP, autocast=False):
    """Kim's demix_track = kijai's sampler, step for step: 8 s chunks every 4 s, linear fades of chunk/10, reflect border
    padding when longer than 8 s, short tails padded (reflect if over half a chunk, else zeros). mix [2, t] float32 at
    44.1 kHz on the model's device → vocals [2, t]. kijai runs fp32 (autocast False, our default); Kim's script runs
    under torch.cuda.amp.autocast (autocast True). NaN → 0 as in Kim's script."""
    step = chunk // overlap
    fade = chunk // 10
    border = chunk - step
    padded = mix.shape[1] > 2 * border and border > 0
    if padded:
        mix = F.pad(mix, (border, border), mode='reflect')
    win = window(chunk, fade, mix.device)
    res = torch.zeros_like(mix, dtype=torch.float32)
    cnt = torch.zeros_like(mix, dtype=torch.float32)
    total = mix.shape[1]
    for i in range(0, total, step):
        part = mix[:, i:i + chunk]
        n = part.shape[-1]
        if n < chunk:
            part = F.pad(part, (0, chunk - n), mode='reflect') if n > chunk // 2 + 1 else F.pad(part, (0, chunk - n))
        with torch.autocast('cuda', enabled=autocast and mix.is_cuda):
            x = model(part.unsqueeze(0))[0].float()
        w = win.clone()
        if i == 0:
            w[:fade] = 1
        elif i + chunk >= total:
            w[-fade:] = 1
        res[..., i:i + n] += x[..., :n] * w[..., :n]
        cnt[..., i:i + n] += w[..., :n]
    out = torch.nan_to_num(res / cnt, nan=0.0)
    return out[..., border:-border] if padded else out
