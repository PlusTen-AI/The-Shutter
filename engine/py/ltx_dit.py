# LTX-2.5 22B audio+video DiT on the shutter engine (libshutter.so): text connectors, RoPE tables, timestep inputs and the
# euler_ancestral RF sampler with the I2V inpaint mask blend, ported from ComfyUI e9afbccc2 (comfy/ldm/lightricks/
# av_model.py, model.py, embeddings_connector.py, symmetric_patchifier.py; model_base.LTXAV; samplers.KSamplerX0Inpaint;
# k_diffusion sample_euler_ancestral_RF). torch does the small fp32 math (same ops as ComfyUI, so the same bits) and the
# ComfyUI-identical noise; the engine does the network.
import ctypes
import math

import numpy as np
import torch

import models

LIB = models.LIB
# template "LTX-2.5 I2V" (video_ltx2_5_i2v.json): stage 1 at half size, latent x2 upsampler, stage 2 at full size
SIGMAS_S1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS_S2 = "0.85, 0.7250, 0.4219, 0.0"
SCALE = (8, 32, 32)                      # vae_scale_factors (t, h, w)
MAX_POS, AUDIO_MAX_POS, CONN_MAX_POS = [20, 2048, 2048], [20], [4096]
THETA = 10000.0
DV, DA, CTX, CTX_MAX, NREG = 4096, 2048, 1024, 4096, 128
bf16 = torch.bfloat16

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(LIB)
        P, I = ctypes.c_void_p, ctypes.c_int
        IP = ctypes.POINTER(I)
        L.ltx_dit_new.restype = P
        L.ltx_dit_new.argtypes = [ctypes.c_char_p]
        L.ltx_dit_connector.argtypes = [P, I, P, I, P, P, P, P]
        L.ltx_dit_prepare.argtypes = [P, P, P, I, I, I, I] + [P] * 8
        L.ltx_dit_forward.argtypes = [P, P, P, P, I, IP, IP, IP, P, P]
        L.ltx_dit_forward_tok.argtypes = [P, P, P, P, I, IP, IP, IP, P, P]
        L.ltx_dit_block.argtypes = [P, I, P, P, P, P]
        L.ltx_dit_tap.argtypes = [P, I, P]
        L.ltx_dit_tap_blocks.argtypes = [P, P, P]
        L.ltx_dit_temb.argtypes = [P, I, P]
        L.ltx_dit_attn_q8.argtypes = [P, I]
        L.ltx_dit_qnoise.argtypes = [P, I]
        L.ltx_dit_attention.argtypes = [P] * 5 + [I] * 3 + [ctypes.c_float]
        L.ltx_dit_free.argtypes = [P]
        _lib = L
    return _lib


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr() if t is not None else 0)


def _ints(v):
    v = [int(x) for x in v]
    return (ctypes.c_int * max(1, len(v)))(*v)


def parse_sigmas(s):
    return torch.FloatTensor([float(x) for x in s.split(",")])


# ------------------------------------------------------------------------------------------------- RoPE (ComfyUI order)
def _log_base(x, base):
    return np.log(x) / np.log(base)


def freq_grid(n_pos, inner_dim, theta=THETA):
    """generate_freq_grid_np: fp64 numpy grid, cast to fp32."""
    pow_indices = np.power(theta, np.linspace(_log_base(1, theta), _log_base(theta, theta), inner_dim // (2 * n_pos),
                                              dtype=np.float64))
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


def rope_tables(grid, dim, max_pos, middle=True):
    """grid [1, n_pos, N, 2] (start, end) or [1, n_pos, N] fp32 -> (cos, sin) bf16 [N, dim/2]: generate_freqs +
    freqs_cis_matrix (split mode, identity padding at the front). Head h owns entries [h * hd/2, (h + 1) * hd/2)."""
    indices = freq_grid(grid.shape[1], dim).to(grid.device)
    if grid.dim() == 4:
        grid = (grid[..., 0] + grid[..., 1]) / 2.0 if middle else grid[..., 0]
    frac = torch.stack([grid[:, i] / max_pos[i] for i in range(grid.shape[1])], dim=-1)
    freqs = (indices * (frac.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    cos, sin = freqs.cos().to(bf16), freqs.sin().to(bf16)
    pad = dim // 2 - freqs.shape[-1]
    if pad:
        cos = torch.cat((torch.ones_like(cos[:, :, :pad]), cos), dim=-1)
        sin = torch.cat((torch.zeros_like(sin[:, :, :pad]), sin), dim=-1)
    return cos[0].contiguous(), sin[0].contiguous()


def video_pixel_coords(F, H, W, device="cuda"):
    """SymmetricPatchifier(1, start_end) latent coords -> latent_to_pixel_coords(causal_fix=True): [1, 3, F*H*W, 2] int64."""
    g = torch.meshgrid(torch.arange(F, device=device), torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    start = torch.stack(g, dim=0)
    end = start + torch.ones(3, dtype=start.dtype, device=device)[:, None, None, None]
    coords = torch.stack((start.reshape(1, 3, -1), end.reshape(1, 3, -1)), dim=-1)
    pix = coords * torch.tensor(SCALE, device=device).view(1, -1, 1, 1)
    pix[:, 0, ...] = (pix[:, 0, ...] + 1 - SCALE[0]).clamp(min=0)
    return pix


def audio_coords(T, device="cuda"):
    """AudioPatchifier(start_end): latent times in seconds [1, 1, T, 2] fp32 (16 kHz, hop 160, x4 causal)."""
    def sec(a, b):
        f = torch.arange(a, b, dtype=torch.float32, device=device) * 4
        f = (f + 1 - 4).clip(min=0)
        return f * 160 / 16000
    return torch.stack([sec(0, T)[None, None], sec(1, T + 1)[None, None]], dim=-1)


def av_rope(F, H, W, T, frame_rate, device="cuda"):
    """_prepare_positional_embeddings: video self (4096), audio self (2048), cross video / audio side (2048, time only)."""
    pix = video_pixel_coords(F, H, W, device)
    fc = pix.to(torch.float32)
    fc[:, 0] = fc[:, 0] * (1.0 / frame_rate)
    v = rope_tables(fc, DV, MAX_POS)
    ac = audio_coords(T, device)
    a = rope_tables(ac, DA, AUDIO_MAX_POS)
    vc = pix.to(torch.float32)
    vc[:, 0] = vc[:, 0] * (1.0 / frame_rate)
    mp = [max(MAX_POS[0], AUDIO_MAX_POS[0])]
    xv = rope_tables(vc[:, 0:1, :], DA, mp)
    xa = rope_tables(ac[:, 0:1, :], DA, mp)
    return v, a, xv, xa


def ctx_len(n):
    """Text context rows for n prompt tokens: Embeddings1DConnector tiles its 128 registers to max(1024, n) rounded up."""
    return max(CTX, -(-n // NREG) * NREG)


def connector_rope(dim, L=CTX, device="cuda"):
    """Embeddings1DConnector.precompute_freqs_cis: positions 0..L-1 over max_pos 4096 (double-precision grid)."""
    grid = torch.arange(L, dtype=torch.float32, device=device)[None, None, :]
    return rope_tables(grid, dim, CONN_MAX_POS)


# ------------------------------------------------------------------------------------------------- timesteps
def sinusoid(t):
    """get_timestep_embedding(t, 256, flip_sin_to_cos=True, downscale_freq_shift=0), cast to bf16 like the time_proj input."""
    half = 128
    exponent = -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
    exponent = exponent / (half - 0)
    emb = torch.exp(exponent)
    emb = t[:, None].float() * emb[None, :]
    emb = 1 * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    return emb.to(bf16).contiguous()


def token_timesteps(mask_v, mask_a, sigma):
    """LTXAV.process_timestep: per-token sigma * denoise_mask (mask as the DiT gets it: bf16), video [1, N, 1] / audio [1, T, 1]."""
    t = sigma.float().reshape(1)
    v = (mask_v.to(bf16) * t.view([1] * mask_v.dim()))[:, :1]
    a = (mask_a.to(bf16) * t.view([1] * mask_a.dim()))[:, :1, :, :1]
    return v.reshape(1, -1, 1), a.reshape(1, -1, 1)


def patchify_video(z):          # [1, C, F, H, W] -> [F*H*W, C]
    return z[0].permute(1, 2, 3, 0).reshape(-1, z.shape[1])


def unpatchify_video(rows, F, H, W):
    return rows.reshape(F, H, W, -1).permute(3, 0, 1, 2).unsqueeze(0)


def patchify_audio(z):          # [1, 8, T, 16] -> [T, 128] ("b c t f -> b t (c f)")
    return z[0].permute(1, 0, 2).reshape(z.shape[2], -1)


def unpatchify_audio(rows, c=8, f=16):
    return rows.reshape(rows.shape[0], c, f).permute(1, 0, 2).unsqueeze(0)


def pack(*parts):
    """Packed sampler state: the parts (video, audio) flattened and concatenated -> [1, 1, n]."""
    return torch.cat([t.reshape(1, 1, -1) for t in parts], dim=-1)


def unpack(x, shapes):
    out, n0 = [], 0
    for s in shapes:
        n = math.prod(s)
        out.append(x[..., n0:n0 + n].reshape(s))
        n0 += n
    return out


def init_noise(seed, vshape, ashape):
    """comfy.sample.prepare_noise on the nested AV latent: one CPU generator, video then audio."""
    gen = torch.manual_seed(seed)
    return (torch.randn(vshape, dtype=torch.float32, generator=gen, device="cpu"),
            torch.randn(ashape, dtype=torch.float32, generator=gen, device="cpu"))


def i2v_mask(vshape, ashape, strength, device="cuda"):
    """LTXVImgToVideoInplace noise mask broadcast to the latents: frame 0 = 1 - strength (fp32), the rest and audio 1."""
    mv = torch.ones(vshape, dtype=torch.float32, device=device)
    mv[:, :, 0] = 1.0 - strength
    return mv, torch.ones(ashape, dtype=torch.float32, device=device)


# ------------------------------------------------------------------------------------------------- DiT
class Dit:
    def __init__(self, path=None, attn_q8=False):
        path = path or models.path(models.LTX_DIT)
        self.path = path
        self.h = ctypes.c_void_p(lib().ltx_dit_new(path.encode()))
        self.set_attn_q8(attn_q8)
        self.last = None

    def set_attn_q8(self, on):
        """lossy switch: SageAttention2-style int8 QK / e4m3 PV for the video self-attention (default off)."""
        lib().ltx_dit_attn_q8(self.h, int(bool(on)))

    def set_qnoise(self, rate):
        """debug: flip int8 activations by +-1 at rate per 1000 (noise floor of the network)."""
        lib().ltx_dit_qnoise(self.h, int(rate))

    def close(self):
        if self.h:
            lib().ltx_dit_free(self.h)
            self.h = None

    def connect(self, cond, tap_blocks=None):
        """preprocess_text_embeds: TE projection output [n, 6144] (video 4096 | audio 2048) -> context [L, 6144] bf16,
        L = ctx_len(n) (1024 up to 1024 tokens; video and audio connector, learnable registers fill the rows after the
        n prompt tokens). tap_blocks: optional (video [8, L, 4096], audio [8, L, 2048]) bf16 destinations."""
        x = cond.reshape(-1, DV + DA).to(device="cuda", dtype=bf16)
        n = x.shape[0]
        L = ctx_len(n)
        if L > CTX_MAX:
            raise ValueError(f"prompt too long: {n} text tokens (at most {CTX_MAX})")
        outs = []
        for k, (c, lo, hi) in enumerate(((DV, 0, DV), (DA, DV, DV + DA))):
            xi = x[:, lo:hi].contiguous()
            cs, sn = connector_rope(c, L)
            o = torch.empty(L, c, dtype=bf16, device="cuda")
            torch.cuda.synchronize()
            lib().ltx_dit_connector(self.h, k, _ptr(xi), n, _ptr(cs), _ptr(sn), _ptr(o), _ptr(tap_blocks[k] if tap_blocks else None))
            outs.append(o)
        return torch.cat(outs, dim=-1)

    def prepare(self, context, vshape, ta, frame_rate=24.0):
        """context [L, 6144] bf16 (connect()); vshape (F, H, W) of the video latent; ta audio latent frames."""
        F, H, W = vshape
        ctx = context.reshape(-1, DV + DA).to(device="cuda", dtype=bf16)
        vctx, actx = ctx[:, :DV].contiguous(), ctx[:, DV:].contiguous()
        self.rope = av_rope(F, H, W, ta, frame_rate)
        (vc, vs), (ac, as_), (xvc, xvs), (xac, xas) = self.rope
        self.shape, self.ta = (F, H, W), ta
        torch.cuda.synchronize()
        lib().ltx_dit_prepare(self.h, _ptr(vctx), _ptr(actx), ctx.shape[0], F * H * W, H * W, ta, _ptr(vc), _ptr(vs), _ptr(ac),
                              _ptr(as_), _ptr(xvc), _ptr(xvs), _ptr(xac), _ptr(xas))

    def forward(self, xv, xa, tv, ta):
        """LTXAVModel.forward: xv [1, 128, F, H, W] / xa [1, 8, T, 16] bf16, per-token timesteps tv [1, N, 1] / ta [1, T, 1] fp32
        (sigma * mask) -> velocity (video, audio) bf16."""
        F, H, W = self.shape
        tsv = tv.reshape(-1).float() * 1000.0                      # timestep_scaled (per token)
        tsa = ta.reshape(-1).float() * 1000.0
        ga = tsa.max() * 1.0                                       # a2v gate input (av_ca_factor 1.0)
        gv = tsv.max() * 1.0                                       # v2a gate / prompt input
        vals = torch.cat([tsv, tsa, ga.view(1), gv.view(1)])
        uniq = torch.unique(vals)
        idx = lambda v: torch.searchsorted(uniq, v.contiguous())
        vtu, atu = idx(tsv).tolist(), idx(tsa).tolist()
        sel = [int(idx(ga.view(1))), int(idx(gv.view(1))), int(idx(tsv.max().view(1))), int(idx(tsa.max().view(1)))]
        sin = sinusoid(uniq)
        vr = patchify_video(xv.to(bf16)).contiguous()
        if vr.shape[0] != F * H * W:
            raise ValueError(f"forward: {vr.shape[0]} video tokens, prepared for {F * H * W}")
        ar = patchify_audio(xa.to(bf16)).contiguous()
        vo = torch.empty(vr.shape[0], 128, dtype=bf16, device="cuda")
        ao = torch.empty(ar.shape[0], 128, dtype=bf16, device="cuda")
        torch.cuda.synchronize()
        lib().ltx_dit_forward_tok(self.h, _ptr(vr), _ptr(ar), _ptr(sin), uniq.numel(), _ints(vtu), _ints(atu), _ints(sel),
                                  _ptr(vo), _ptr(ao))
        self.last = dict(uniq=uniq, vtu=vtu, atu=atu, sel=sel, vfu=vtu[:F * H * W:H * W])
        return unpatchify_video(vo[:F * H * W], F, H, W), unpatchify_audio(ao)

    def temb(self, k):
        """debug: adaLN rows of the last forward (module k, see ltx_dit_temb) as [U, n] bf16."""
        U = self.last["uniq"].numel()
        n = {0: 9 * DV, 1: 2 * DV, 2: 9 * DA, 3: 2 * DA, 4: 4 * DV, 5: DV, 6: 4 * DA, 7: DA, 8: DV, 9: DA}[k]
        out = torch.empty(U, n, dtype=bf16, device="cuda")
        torch.cuda.synchronize()
        lib().ltx_dit_temb(self.h, k, _ptr(out))
        return out

    def denoise(self, x, sigma, lat, mask, mask_bf, shapes):
        """KSamplerX0Inpaint + BaseModel._apply_model on the packed fp32 latent x (parts: video, audio, shapes as
        given): returns (denoised, model_in, model_out)."""
        lm = 1.0 - mask
        model_in = x * mask + lat * lm
        tv = token_timesteps(mask_bf[0], mask_bf[0], sigma)[0]
        ta = token_timesteps(mask_bf[1], mask_bf[1], sigma)[1]
        xs = unpack(model_in.to(bf16), shapes)
        o = self.forward(xs[0], xs[1], tv, ta)
        out = pack(o[0], o[1]).float()
        raw = model_in - out * sigma.reshape(1, 1, 1)
        return raw * mask + lat * lm, model_in, out

    def sample(self, latent, mask, sigmas, noise, seed=None, anc=None, callback=None, eta=1.0, s_noise=1.0):
        """SamplerCustomAdvanced (cfg 1) + euler_ancestral (CONST -> sample_euler_ancestral_RF) on one stage.
        latent: (video [1,128,F,H,W], audio [1,8,T,16]) fp32 clean latents (I2V: frame 0 = image latent); mask: fp32
        denoise masks broadcast to them (i2v_mask); sigmas fp32 [S + 1]; noise: initial noise per part (init_noise);
        anc: list of per-step injected noise per part, or None -> torch.Generator('cuda').manual_seed(seed) over the
        packed state as ComfyUI does. Returns the stage output (video, audio) fp32."""
        shapes = [tuple(t.shape) for t in latent]
        lat = pack(*[t.to("cuda", torch.float32) for t in latent])
        m = pack(*[t.to("cuda", torch.float32).expand_as(l) for t, l in zip(mask, latent)])
        mbf = tuple(t.to("cuda", torch.float32).expand_as(l).to(bf16) for t, l in zip(mask, latent))
        sig = sigmas.to("cuda", torch.float32)
        n = pack(*[t.to("cuda", torch.float32) for t in noise])
        x = sig[0] * (1.0 * n) + (1.0 - sig[0]) * lat          # CONST.noise_scaling
        gen = None
        if anc is None:
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seed)
        for i in range(len(sig) - 1):
            den, _, _ = self.denoise(x, sig[i:i + 1], lat, m, mbf, shapes)
            if callback:
                callback(i, x, den)
            if sig[i + 1] == 0:
                x = den
            else:
                downstep_ratio = 1 + (sig[i + 1] / sig[i] - 1) * eta
                sigma_down = sig[i + 1] * downstep_ratio
                alpha_ip1 = 1 - sig[i + 1]
                alpha_down = 1 - sigma_down
                renoise_coeff = (sig[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
                sigma_down_i_ratio = sigma_down / sig[i]
                x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * den
                if eta > 0:
                    if anc is not None:
                        nz = pack(*[t.to("cuda", torch.float32) for t in anc[i]])
                    else:
                        nz = torch.randn((1, 1, x.shape[-1]), dtype=x.dtype, device=x.device, generator=gen)
                    x = (alpha_ip1 / alpha_down) * x + nz * s_noise * renoise_coeff
        x = x / (1.0 - sig[-1])                                   # inverse_noise_scaling
        return tuple(unpack(x, shapes))
