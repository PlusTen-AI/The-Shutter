# Voice pass (기능 17, design §0 1안): a generated cut's sound → speech + rest (Mel-Band RoFormer, Kim's vocal model,
# melband.py) → each speaker's lines converted to that character's reference voice with Seed-VC v1 (offline small:
# whisper-small content + CAM++ style + DiT CFM + BigVGAN 22 kHz; the length is kept, so the lips stay in sync) → mixed
# back with the rest. Reference voices: Qwen3-TTS 1.7B VoiceDesign (a text description → a Korean line).
#
# Seed-VC is GPL-3.0: none of its code is in this repo. The image fetches its modules/ at a pinned commit into
# /opt/seed-vc (docker/Dockerfile) and Converter imports them from there. What inference.py does is reproduced here
# from its public behaviour (Plachtaa/seed-vc @51383ef inference.py, load_models + main, f0_condition False).
# No torchaudio in the image (a pip torchaudio would replace the base image's torch): its resample and kaldi fbank are
# ported below (pytorch/audio v2.8.0, BSD-2-Clause) and the qwen_tts import gets empty stand-ins (_import_qwen_tts).
# All weights are local (models.py, /models/voice/...), nothing is fetched at run time.

import hashlib, math, os, sys, types
import importlib.machinery, importlib.util

os.environ.setdefault('HF_HUB_OFFLINE', '1')          # before any huggingface_hub / transformers import
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch
import torch.nn.functional as F

import models

SEED_VC = os.environ.get('SHUTTER_SEED_VC', '/opt/seed-vc')
SR_SEP = 44100       # Mel-Band RoFormer
SR_VC = 22050        # Seed-VC v1 small (config preprocess_params.sr; BigVGAN 22 kHz 80 band 256x)
SR_TTS = 24000       # Qwen3-TTS 12 Hz tokenizer output_sample_rate


# ---------------------------------------------------------------- torchaudio ports
# resample and fbank80 follow torchaudio v2.8.0 functional/functional.py (_get_sinc_resample_kernel,
# _apply_sinc_resample_kernel) and compliance/kaldi.py (fbank and helpers); same results bit for bit (CPU check).
#   BSD 2-Clause License. Copyright (c) 2017 Facebook Inc. (Soumith Chintala), All rights reserved.
#   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
#   following conditions are met: * Redistributions of source code must retain the above copyright notice, this list
#   of conditions and the following disclaimer. * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided
#   with the distribution.
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
#   USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def resample(x, sr0, sr1, width=6, rolloff=0.99):
    """torchaudio.functional.resample(x, sr0, sr1) with its defaults (sinc_interp_hann, lowpass_filter_width 6,
    rolloff 0.99), same arithmetic in x's dtype. x: float tensor [..., n]."""
    if sr0 == sr1:
        return x
    g = math.gcd(int(sr0), int(sr1))
    a, b = int(sr0) // g, int(sr1) // g
    base = min(a, b) * rolloff
    w = math.ceil(width * a / base)
    dt, dev = x.dtype, x.device
    idx = torch.arange(-w, w + a, dtype=dt, device=dev)[None, None] / a
    t = torch.arange(0, -b, -1, dtype=dt, device=dev)[:, None, None] / b + idx
    t *= base
    t = t.clamp_(-width, width)
    win = torch.cos(t * math.pi / width / 2) ** 2
    t *= math.pi
    k = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    k *= win * (base / a)
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    n = x.shape[1]
    y = F.conv1d(F.pad(x, (w, w + a))[:, None], k, stride=a).transpose(1, 2).reshape(x.shape[0], -1)
    y = y[..., :int(torch.ceil(torch.as_tensor(b * n / a)).long())]
    return y.view(shape[:-1] + y.shape[-1:])


def fbank80(w):
    """torchaudio.compliance.kaldi.fbank(w[None], num_mel_bins=80, dither=0, sample_frequency=16000), the other options
    at their defaults (25/10 ms povey frames padded to 512, DC removal, preemphasis 0.97, power, log, 20 Hz..Nyquist,
    snip_edges). w: float tensor [n] at 16 kHz → [frames, 80]."""
    dev, dt = w.device, w.dtype
    w = w.contiguous()
    m = 1 + (w.numel() - 400) // 160
    x = w.as_strided((m, 400), (160, 1))
    x = x - x.mean(dim=1).unsqueeze(1)
    x = x - 0.97 * F.pad(x.unsqueeze(0), (1, 0), mode='replicate').squeeze(0)[:, :-1]
    x = x * torch.hann_window(400, periodic=False, device=dev, dtype=dt).pow(0.85).unsqueeze(0)
    x = F.pad(x.unsqueeze(0), (0, 112), mode='constant', value=0).squeeze(0)
    spec = torch.fft.rfft(x).abs().pow(2.0)
    lo, hi = 1127.0 * math.log(1.0 + 20.0 / 700.0), 1127.0 * math.log(1.0 + 8000.0 / 700.0)
    d = (hi - lo) / 81
    i = torch.arange(80).unsqueeze(1)
    left, center, right = lo + i * d, lo + (i + 1.0) * d, lo + (i + 2.0) * d
    mel = (1127.0 * (1.0 + 16000 / 512 * torch.arange(256.0) / 700.0).log()).unsqueeze(0)
    banks = torch.max(torch.zeros(1), torch.min((mel - left) / (center - left), (right - mel) / (right - center)))
    banks = F.pad(banks.to(device=dev, dtype=dt), (0, 1), mode='constant', value=0)
    eps = torch.tensor(torch.finfo(torch.float).eps).to(device=dev, dtype=dt)
    return torch.max(torch.mm(spec, banks.T), eps).log()


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)   # find_spec() on a stub must not raise
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _fit(y, n):
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


# ---------------------------------------------------------------- separation

class Separator:
    """Speech / rest split with Kim's Mel-Band RoFormer (melband.py; 8 s chunks at 44.1 kHz stereo, kijai's sampler)."""

    def __init__(self, device='cuda'):
        import melband
        self.melband, self.dev = melband, device
        self.model = melband.load(models.path(models.SEP), device)

    @torch.inference_mode()
    def __call__(self, audio, sr):
        """audio [ch, n] (or [n]) float32 at sr → (vocals, rest), same shape and rate, vocals + rest == audio.
        As kijai's node: mono is duplicated to stereo, the input resampled to 44.1 kHz (torchaudio's resample)."""
        x = np.asarray(audio, np.float32)
        x2 = np.stack([x, x]) if x.ndim == 1 else (np.repeat(x, 2, 0) if x.shape[0] == 1 else x[:2])
        t = resample(torch.from_numpy(np.ascontiguousarray(x2)).to(self.dev), sr, SR_SEP)
        v = resample(self.melband.demix(self.model, t), SR_SEP, sr)
        v = v.float().cpu().numpy()
        n = x.shape[-1]
        v = np.stack([_fit(c, n) for c in v])
        if x.ndim == 1 or x.shape[0] == 1:
            v = v.mean(0).reshape(x.shape)
        elif x.shape[0] > 2:   # surround etc.: speech taken from the first two channels only
            v = np.concatenate([v, np.zeros((x.shape[0] - 2, n), np.float32)])
        return v, x - v

    def close(self):
        self.model = None


# ---------------------------------------------------------------- voice conversion

def _import_seed_vc():
    """Seed-VC's modules from /opt/seed-vc. modules/length_regulator.py imports dac.nn.quantize.VectorQuantize
    (descript-audio-codec, which pulls in audiotools and torchaudio) at import; with vector_quantize False (build_model
    never passes it) the class is never built, so an empty stand-in replaces the package when it is absent."""
    if SEED_VC not in sys.path:
        sys.path.insert(0, SEED_VC)
    added = []
    if importlib.util.find_spec('dac') is None:
        class VectorQuantize:   # never instantiated (InterpolateRegulator(vector_quantize=False))
            def __init__(self, *a, **k):
                raise RuntimeError('descript-audio-codec is not installed')
        q = _stub('dac.nn.quantize', VectorQuantize=VectorQuantize)
        _stub('dac', nn=_stub('dac.nn', quantize=q))
        added = ['dac', 'dac.nn', 'dac.nn.quantize']
    try:
        import modules.length_regulator, modules.flow_matching   # noqa: F401  (build_model imports them inside)
        from modules.commons import build_model, recursive_munch
        from modules.audio import mel_spectrogram
        from modules.campplus.DTDNN import CAMPPlus
        from modules.bigvgan import bigvgan
    finally:
        for k in added:
            sys.modules.pop(k, None)
    return build_model, recursive_munch, mel_spectrogram, CAMPPlus, bigvgan


def _load_seed_vc(model, path):
    """modules.commons.load_checkpoint(model, None, path, load_only_params=True, ignore_modules=[]): per sub-model,
    'module.' prefixes stripped, only keys present with the same shape, strict=False. The pruned checkpoint holds
    net.{cfm, style_encoder, vq, length_regulator}; cfm skips f0_embedder.weight and input_pos (8192 vs 16384) and keeps
    its two t_embedder freqs buffers at init, as upstream. Its pickle has only OrderedDict + tensors: weights_only works."""
    params = torch.load(path, map_location='cpu', weights_only=True)['net']
    for key in model:
        if key not in params:
            continue
        p = {(k[7:] if k.startswith('module.') else k): v for k, v in params[key].items()}
        own = model[key].state_dict()
        keep = {k: v for k, v in p.items() if k in own and v.shape == own[k].shape}
        if len(keep) != len(p):
            print('seed-vc %s: skipped %s' % (key, sorted(set(p) - set(keep))), flush=True)
        model[key].load_state_dict(keep, strict=False)


class Converter:
    """Seed-VC v1 offline, the whisper-small + wavenet + BigVGAN 22 kHz preset (DiT_seed_v2_uvit_whisper_small_wavenet
    _bigvgan_pruned.pth), as inference.py runs it with f0_condition False: source and reference at 22.05 kHz
    (librosa soxr), reference cut to 25 s, whisper-small encoder features at 16 kHz (30 s windows, 5 s overlap past
    30 s), CAM++ style from the reference's kaldi fbank, length regulator at length_adjust 1.0 (target length = source
    mel length), CFM euler with inference_cfg_rate 0.7 under fp16 autocast, BigVGAN (torch activations, weight norm
    removed), long sources in windows of 30 s minus the reference with a 16-frame crossfade."""

    HOP, OVERLAP, REF_MAX = 256, 16, 25

    def __init__(self, device='cuda', fp16=True):
        import json, yaml
        from transformers import AutoFeatureExtractor, WhisperModel
        build_model, recursive_munch, mel_spectrogram, CAMPPlus, bigvgan = _import_seed_vc()
        self.dev, self.fp16 = torch.device(device), fp16
        cfg = yaml.safe_load(open(models.path(models.VC_CFG)))
        mp = recursive_munch(cfg['model_params'])
        mp.dit_type = 'DiT'
        self.model = build_model(mp, stage='DiT')
        _load_seed_vc(self.model, models.path(models.VC_DIT))
        for k in self.model:
            self.model[k].eval().to(self.dev)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus.load_state_dict(torch.load(models.path(models.VC_CAMPP), map_location='cpu', weights_only=True))
        self.campplus.eval().to(self.dev)

        # BigVGAN.from_pretrained(<dir>, use_cuda_kernel=False) for a local folder = these steps (_from_pretrained:
        # load_hparams_from_json → BigVGAN(h) → state dict 'generator', saved with weight norm) + remove_weight_norm()
        voc = models.path(models.VC_BIGVGAN)
        with open(os.path.join(voc, 'config.json')) as f:
            h = bigvgan.AttrDict(json.loads(f.read()))
        self.vocoder = bigvgan.BigVGAN(h, use_cuda_kernel=False)
        self.vocoder.load_state_dict(torch.load(os.path.join(voc, 'bigvgan_generator.pt'), map_location='cpu',
                                                weights_only=True)['generator'])
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.dev)

        wd = models.path(models.VC_WHISPER)
        self.whisper = WhisperModel.from_pretrained(wd, dtype=torch.float16).to(self.dev)   # upstream: torch_dtype=
        del self.whisper.decoder
        self.fe = AutoFeatureExtractor.from_pretrained(wd)

        sp = cfg['preprocess_params']['spect_params']
        self.sr = cfg['preprocess_params']['sr']
        assert self.sr == SR_VC and sp['hop_length'] == self.HOP
        args = dict(n_fft=sp['n_fft'], win_size=sp['win_length'], hop_size=sp['hop_length'], num_mels=sp['n_mels'],
                    sampling_rate=self.sr, fmin=sp.get('fmin', 0),
                    fmax=None if sp.get('fmax', 'None') == 'None' else 8000, center=False)
        self.to_mel = lambda x: mel_spectrogram(x, **args)
        self.refs, self.embs = {}, {}

    # -- pieces of inference.py

    def _semantic(self, w16):
        """semantic_fn for speech_tokenizer.type 'whisper': [1, n] at 16 kHz → [1, n // 320 + 1, 768]."""
        inp = self.fe([w16.squeeze(0).cpu().numpy()], return_tensors='pt', return_attention_mask=True,
                      sampling_rate=16000)   # upstream passes none (same features, one warning per call)
        feats = self.whisper._mask_input_features(inp.input_features, attention_mask=inp.attention_mask).to(self.dev)
        out = self.whisper.encoder(feats.to(self.whisper.encoder.dtype), head_mask=None, output_attentions=False,
                                   output_hidden_states=False, return_dict=True)
        return out.last_hidden_state.to(torch.float32)[:, :w16.size(-1) // 320 + 1]

    def _semantic_long(self, w16):
        if w16.size(-1) <= 16000 * 30:
            return self._semantic(w16)
        ov, outs, buf, done = 5, [], None, 0
        while done < w16.size(-1):
            if buf is None:
                chunk = w16[:, done:done + 16000 * 30]
            else:
                chunk = torch.cat([buf, w16[:, done:done + 16000 * (30 - ov)]], dim=-1)
            s = self._semantic(chunk)
            outs.append(s if done == 0 else s[:, 50 * ov:])
            buf = chunk[:, -16000 * ov:]
            done += 30 * 16000 if done == 0 else chunk.size(-1) - 16000 * ov
        return torch.cat(outs, dim=1)

    def _style(self, w16):
        feat = fbank80(w16[0])
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.campplus(feat.unsqueeze(0))

    @staticmethod
    def _key(w, sr):
        return '%d:%d:%s' % (sr, len(w), hashlib.sha1(np.ascontiguousarray(w, np.float32).tobytes()).hexdigest())

    def _to22k(self, w, sr):
        import librosa
        w = np.asarray(w, np.float32)
        return w if sr == self.sr else librosa.resample(w, orig_sr=sr, target_sr=self.sr)   # librosa.load(sr=22050)

    def _ref(self, ref, sr):
        """(mel2, style2, prompt_condition) of a reference voice, cached (the same voice serves many lines)."""
        k = self._key(ref, sr)
        if k not in self.refs:
            r = torch.tensor(self._to22k(ref, sr)[:self.sr * self.REF_MAX]).unsqueeze(0).float().to(self.dev)
            r16 = resample(r, self.sr, 16000)
            mel2 = self.to_mel(r)
            prompt = self.model.length_regulator(self._semantic(r16), ylens=torch.LongTensor([mel2.size(2)]).to(self.dev),
                                                 n_quantizers=3, f0=None)[0]
            self.refs[k] = (mel2, self._style(r16), prompt)
        return self.refs[k]

    @torch.inference_mode()
    def convert(self, src, ref, sr, ref_sr=None, steps=30, cfg_rate=0.7, seed=0):
        """src [n] float32 at sr (one speaker's speech), ref [m] float32 at ref_sr (default sr) → [n] float32 at sr.
        inference.py defaults: diffusion_steps 30, length_adjust 1.0, inference_cfg_rate 0.7, fp16 True."""
        n = len(src)
        mel2, style2, prompt = self._ref(ref, ref_sr or sr)
        s = torch.tensor(self._to22k(src, sr)).unsqueeze(0).float().to(self.dev)
        cond = self.model.length_regulator(self._semantic_long(resample(s, self.sr, 16000)),
                                           ylens=torch.LongTensor([int(self.to_mel(s).size(2) * 1.0)]).to(self.dev),
                                           n_quantizers=3, f0=None)[0]
        window = self.sr // self.HOP * 30 - mel2.size(2)    # max_context_window - reference frames
        ovw = self.OVERLAP * self.HOP
        torch.manual_seed(seed)                              # CFM starts from torch.randn
        done, parts, prev = 0, [], None
        while done < cond.size(1):
            chunk = cond[:, done:done + window]
            last = done + window >= cond.size(1)
            cat = torch.cat([prompt, chunk], dim=1)
            with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=self.fp16):
                mel = self.model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(self.dev), mel2, style2, None,
                                               steps, inference_cfg_rate=cfg_rate)[:, :, mel2.size(-1):]
            wav = self.vocoder(mel.float()).squeeze()[None]
            if done == 0 and last:
                parts.append(wav[0].cpu().numpy())
                break
            if done == 0:
                parts.append(wav[0, :-ovw].cpu().numpy())
            else:
                cur = wav[0].cpu().numpy() if last else wav[0, :-ovw].cpu().numpy()
                parts.append(_crossfade(prev, cur, ovw))
            if last:
                break
            prev = wav[0, -ovw:].cpu().numpy()
            done += mel.size(2) - self.OVERLAP
        y = torch.from_numpy(np.concatenate(parts)).float()
        return _fit(resample(y, self.sr, sr).numpy(), n).astype(np.float32)   # 22.05 kHz mel frames = floor(n/256)

    @torch.inference_mode()
    def embed(self, w, sr):
        """CAM++ speaker embedding [192] (unit length) — the style vector Seed-VC conditions on."""
        k = self._key(w, sr)
        if k not in self.embs:
            w16 = resample(torch.tensor(np.asarray(w, np.float32)).unsqueeze(0).to(self.dev), sr, 16000)
            self.embs[k] = F.normalize(self._style(w16)[0].float(), dim=0)
        return self.embs[k]

    def speaker_sim(self, a, b, sr, sr_b=None):
        """Cosine of CAM++ embeddings: the voice check (like the face check's ArcFace cosine)."""
        return round(float(torch.dot(self.embed(a, sr), self.embed(b, sr_b or sr))), 4)

    def close(self):
        self.model = self.campplus = self.vocoder = self.whisper = None
        self.refs, self.embs = {}, {}


def _crossfade(a, b, n):
    """inference.py crossfade: cos² fades over n samples (b's head over a's tail)."""
    out = np.cos(np.linspace(0, np.pi / 2, n)) ** 2
    inn = np.cos(np.linspace(np.pi / 2, 0, n)) ** 2
    b = b.copy()
    if len(b) < n:
        b[:n] = b[:n] * inn[:len(b)] + (a[-n:] * out)[:len(b)]
    else:
        b[:n] = b[:n] * inn + a[-n:] * out
    return b


# ---------------------------------------------------------------- reference voices

def _import_qwen_tts():
    """qwen_tts imports sox, onnxruntime and torchaudio.compliance.kaldi at package import (core/tokenizer_25hz/vq/
    speech_vq.py, via qwen_tts/core/__init__.py), used only by the 25 Hz tokenizer; the 12 Hz VoiceDesign model never
    calls them. Absent ones get empty stand-ins for the import, removed afterwards. transformers is imported first so its
    availability flags see the real (absent) packages."""
    import transformers  # noqa: F401
    added = []
    for name in ('sox', 'onnxruntime', 'torchaudio'):
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            if name == 'torchaudio':
                k = _stub('torchaudio.compliance.kaldi')
                _stub('torchaudio', compliance=_stub('torchaudio.compliance', kaldi=k))
                added += ['torchaudio', 'torchaudio.compliance', 'torchaudio.compliance.kaldi']
            else:
                _stub(name)
                added.append(name)
    try:
        from qwen_tts import Qwen3TTSModel
    finally:
        for k in added:
            sys.modules.pop(k, None)
    return Qwen3TTSModel


class VoiceDesign:
    """Qwen3-TTS 12 Hz 1.7B VoiceDesign (qwen-tts 0.1.1): a voice description (instruct) + a line → 24 kHz speech.
    Loaded from the local folder (speech_tokenizer/ inside it) in bf16 with torch SDPA attention (no flash-attn)."""

    def __init__(self, device='cuda'):
        Qwen3TTSModel = _import_qwen_tts()
        self.tts = Qwen3TTSModel.from_pretrained(models.path(models.TTS_VD), device_map=device, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa')

    @torch.inference_mode()
    def generate(self, text, instruct, language='Korean', seed=0, **gen):
        """→ (wav [n] float32, 24000). gen: do_sample, top_k, top_p, temperature, repetition_penalty, subtalker_*,
        max_new_tokens (defaults from the model's generation_config.json: sampling on, top_k 50, top_p 1.0,
        temperature 0.9, repetition_penalty 1.05, max_new_tokens 8192)."""
        torch.manual_seed(seed)
        wavs, sr = self.tts.generate_voice_design(text=text, instruct=instruct, language=language, **gen)
        return np.asarray(wavs[0], np.float32), int(sr)

    def close(self):
        self.tts = None


# ---------------------------------------------------------------- the cut's sound

def speech_segments(vocals, sr, n, frame=0.02, hop=0.01, rel_db=35.0):
    """Split a speech track into n utterances at its n-1 longest pauses (energy VAD: 20 ms frames, voiced = within
    rel_db of the loudest frame). vocals [ch, m] or [m] → [(t0, t1)] seconds in order, voiced extents only; fewer than n
    when there are not enough pauses, [] when silent. The caller pairs them with the cut's say-lines in order."""
    x = np.asarray(vocals, np.float32)
    x = x.mean(0) if x.ndim == 2 else x
    fl, h = int(frame * sr), int(hop * sr)
    if len(x) < fl or n < 1:
        return []
    m = 1 + (len(x) - fl) // h
    fr = np.lib.stride_tricks.as_strided(x, (m, fl), (x.strides[0] * h, x.strides[0]))
    db = 10 * np.log10(np.mean(np.square(fr, dtype=np.float64), axis=1) + 1e-12)
    on = db > db.max() - rel_db
    if not on.any():
        return []
    idx = np.flatnonzero(on)
    s, e = idx[0], idx[-1]
    gaps, k = [], s
    while k <= e:   # runs of silent frames between s and e
        if not on[k]:
            j = k
            while not on[j]:
                j += 1
            gaps.append((j - k, k, j))   # (length, first silent, next voiced)
            k = j
        else:
            k += 1
    cut = sorted(sorted(gaps, reverse=True)[:n - 1], key=lambda g: g[1])
    starts = [s] + [g[2] for g in cut]
    ends = [g[1] - 1 for g in cut] + [e]
    return [(round(float(a * h / sr), 3), round(float((b * h + fl) / sr), 3)) for a, b in zip(starts, ends)]


def pipeline(video_audio, sr, segments, refs, sep=None, vc=None, pad=0.2, fade=0.03, steps=30, cfg_rate=0.7, seed=0,
             progress=None, separated=None):
    """The cut's audio [2, n] float32 (48 kHz from the clip), speaker segments [(t0, t1, speaker)], refs {speaker:
    (wav, rate)} → (audio [2, n] float32, report). Speech is separated, each segment converted with pad s of context
    on both sides and blended in with fade s linear ramps outside [t0, t1] (later segments over earlier ones), its RMS
    matched to the source segment's; the converted voice is mono on both channels, unconverted speech stays as
    separated; then speech + rest. report: per segment speaker similarity to its reference after (sim) and before
    (src_sim) conversion. separated: (speech, rest) already split (Voice.clip splits once for the turns too)."""
    x = np.asarray(video_audio, np.float32)
    x = np.stack([x, x]) if x.ndim == 1 else x
    n = x.shape[1]
    voc, rest = separated if separated is not None else (sep or Separator())(x, sr)
    vc = vc or Converter()
    mono = voc.mean(0)
    speech = voc.copy()
    p, f = int(pad * sr), max(1, int(fade * sr))
    report = []
    for i, (t0, t1, spk) in enumerate(segments):
        a, b = max(0, int(round(t0 * sr))), min(n, int(round(t1 * sr)))
        if b - a < int(0.25 * sr):   # too short for whisper / CAM++ windows: left as generated
            report.append(dict(i=i, speaker=spk, t0=t0, t1=t1, skipped=True))
            continue
        a0, b0 = max(0, a - p), min(n, b + p)
        ref, rsr = refs[spk]
        y = vc.convert(mono[a0:b0], ref, sr, ref_sr=rsr, steps=steps, cfg_rate=cfg_rate, seed=seed + i)
        y *= float(np.clip(_rms(mono[a:b]) / max(_rms(y[a - a0:b - a0]), 1e-6), 0.25, 4.0))
        w = np.zeros(b0 - a0, np.float32)
        w[a - a0:b - a0] = 1
        r = min(f, a - a0)
        if r:
            w[a - a0 - r:a - a0] = np.linspace(0, 1, r + 1, dtype=np.float32)[:-1]
        r = min(f, b0 - b)
        if r:
            w[b - a0:b - a0 + r] = np.linspace(1, 0, r + 1, dtype=np.float32)[1:]
        speech[:, a0:b0] = speech[:, a0:b0] * (1 - w) + y * w
        report.append(dict(i=i, speaker=spk, t0=t0, t1=t1, sim=vc.speaker_sim(y[a - a0:b - a0], ref, sr, rsr),
                           src_sim=vc.speaker_sim(mono[a:b], ref, sr, rsr)))
        if progress:
            progress((i + 1) / len(segments))
    return (speech + rest).astype(np.float32), report


# ---------------------------------------------------------------- the worker's side (worker.run_voice / run_tts)

class Voice:
    """audio.voice: separation + Seed-VC, kept loaded next to whatever else is (small, like the face pass)."""

    def __init__(self, device='cuda'):
        self.sep, self.vc = Separator(device), Converter(device)

    def clip(self, pcm, sr, turns, refs, progress=None):
        """pcm [2, n] float32 (the clip's sound), turns [speaker index or -1 per spoken turn, in say-line order],
        refs {speaker: (wav, rate)} → (pcm [2, n], report). One speaker for every turn: the whole voiced extent is one
        segment. Several: the speech is split into as many utterances as turns (the longest pauses) and paired in order;
        when the pauses do not give that many, nothing is converted (reported) rather than a line in the wrong voice."""
        x = np.asarray(pcm, np.float32)
        voc, rest = self.sep(x, sr)
        segs = speech_segments(voc, sr, 1 if len(set(turns)) == 1 else len(turns))
        if len(set(turns)) == 1:
            pairs = [(a, b, turns[0]) for a, b in segs] if turns[0] >= 0 else []
        else:
            pairs = [(a, b, t) for (a, b), t in zip(segs, turns) if t >= 0] if len(segs) == len(turns) else []
        out, rep = pipeline(x, sr, pairs, refs, vc=self.vc, separated=(voc, rest), progress=progress)
        return out, {'turns': turns, 'segments': segs, 'converted': rep,
                     'mismatch': len(set(turns)) > 1 and len(segs) != len(turns),
                     'speech_rms': round(_rms(voc), 5), 'rest_rms': round(_rms(rest), 5)}

    def close(self):
        self.sep.close()
        self.vc.close()


def load_wav(path):
    """A reference voice file → (mono float32, rate): wav / flac / ogg by soundfile, anything else (a phone's m4a,
    mp3, aac) through PyAV at 48 kHz."""
    import soundfile
    try:
        w, sr = soundfile.read(path, dtype='float32', always_2d=True)
        return w.mean(1), int(sr)
    except RuntimeError:   # soundfile.LibsndfileError: not a format libsndfile reads
        import edit
        a = edit.audio48(path)
        if a is None:
            raise ValueError('the voice file has no sound')
        return a.mean(0), edit.RATE


def save_wav(path, wav, sr):
    import soundfile
    soundfile.write(path, np.asarray(wav, np.float32), int(sr), subtype='PCM_16')


def remux(src, pcm, sr, dst):
    """The clip's video packets copied as they are, its sound replaced by pcm [2, n] at sr (48 kHz, edit.RATE) → dst."""
    import av, edit
    assert sr == edit.RATE
    i, o = av.open(src), av.open(dst, 'w')
    try:
        ivs = i.streams.video[0]
        ovs = o.add_stream_from_template(ivs)
        ast = edit.add_audio_stream(o)
        for pkt in i.demux(ivs):
            if pkt.dts is None:
                continue
            pkt.stream = ovs
            o.mux(pkt)
        edit.encode_audio(o, ast, np.ascontiguousarray(np.asarray(pcm, np.float32)[:2]))
        for p in ast.encode():
            o.mux(p)
    finally:
        o.close()
        i.close()
