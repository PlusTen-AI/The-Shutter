# Load-time LoRA / adapter merge for the H3 DiT, computed the way ComfyUI e9afbccc2 computes the weights it runs with
# (dynamic VRAM path, comfy/ops.py post_cast), so shutter reads the original files and needs no baked copy:
#   int8 ConvRot linear: x = dequantize(w) as bf16; x = patch(x); w' = requantize_from_float(x, scale="recalculate",
#                        stochastic_rounding=string_to_seed(module path))    (ComfyUI comfy/ops.py, comfy/lora.py)
#   LoRA patch:          x += bf16((strength * alpha / rank) * mm(up, down)), up/down cast to bf16 first
#                        (comfy/weight_adapter/lora.py; alpha absent -> alpha / rank = 1)
#   set patch:           x = the adapter's value (an int8 adapter weight is dequantized on the CPU first, as
#                        comfyui-piflow 1.4.1 loader_utils.patch_model_with_adapter does)
#   float tensors:       bf16 linears take x as is; the pi-H3 adapter's float "set" tensors are its F16 values (returned
#                        as fp32; a bf16 engine tensor such as final_layer.norm.weight takes them rounded to nearest)
# The int8 ConvRot (de)quantization and the stochastic rounding are comfy-kitchen's (the package ComfyUI calls; its
# CUDA kernels draw the rounding noise), so the result matches ComfyUI bit for bit (checked against the old baked
# files, golden/merge_check.py).
#   lora_patch(base_path, lora_path)        H3 turbo LoRA on Singularity
#   adapter_patch(base_path, adapter_path)  pi-H3 DWD adapter on FL2VA (dwd_* heads are read by h3_dwd.py)
# Both yield (bare tensor name, tensor on the GPU) one layer at a time in the base file's layout (int8 + f32 row scale,
# bf16, or f32 for the float set tensors); the caller writes each into the engine (h3.Dit) and drops it.
import json
import zlib

import torch
from safetensors import safe_open

import comfy_kitchen as ck
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

ck.registry.disable("triton")   # ComfyUI's default (comfy/quant_ops.py): CUDA backend, triton off
PFX = "model.diffusion_model."  # some fine-tunes (Singularity) keep ComfyUI's full key prefix; the engine reads bare names
bf16 = torch.bfloat16


def seed(module):
    """comfy.utils.string_to_seed of the module path ComfyUI uses ('diffusion_model.blocks.0.attn.qkv_proj') = CRC-32."""
    return zlib.crc32(("diffusion_model." + module).encode())


class Base:
    """The base DiT file: bare names, int8 ConvRot layers as comfy-kitchen QuantizedTensors on the GPU."""

    def __init__(self, path):
        self.f = safe_open(path, "pt", device="cuda")
        self.keys = {k[len(PFX):] if k.startswith(PFX) else k: k for k in self.f.keys()}

    def get(self, name):
        return self.f.get_tensor(self.keys[name])

    def has(self, name):
        return name in self.keys

    def int8(self, module):
        conf = json.loads(bytes(self.get(module + ".comfy_quant").cpu().tolist()))
        return qtensor(self.get(module + ".weight"), self.get(module + ".weight_scale"), conf, bf16)


def qtensor(q, scale, conf, dtype):
    """comfy/ops.py: an int8_tensorwise layer as ComfyUI loads it (orig_dtype = the compute dtype)."""
    if conf.get("format") != "int8_tensorwise":
        raise ValueError("unexpected quant config %s" % conf)
    p = conf.get("params", {}) if isinstance(conf.get("params"), dict) else {}
    kw = {"scale": scale}
    if conf.get("convrot", p.get("convrot", False)):
        kw.update(convrot=True, convrot_groupsize=int(conf.get("convrot_groupsize", p.get("convrot_groupsize", 256))))
    params = TensorWiseINT8Layout.Params(**kw, orig_dtype=dtype, orig_shape=tuple(q.shape))
    return QuantizedTensor(q.to(torch.int8), "TensorWiseINT8Layout", params)


def requant(w, x, module):
    """requantize_from_float(x, scale="recalculate", stochastic_rounding=seed) -> int8 weight + f32 row scales."""
    y = w.requantize_from_float(x, scale="recalculate", stochastic_rounding=seed(module))
    return y._qdata.contiguous(), y._params.scale.float().reshape(-1).contiguous()


def lora_add(x, up, down, alpha, strength=1.0):
    """comfy/weight_adapter/lora.py calculate_weight with intermediate dtype = x.dtype (bf16), in place."""
    up, down = up.to(device=x.device, dtype=x.dtype), down.to(device=x.device, dtype=x.dtype)
    a = alpha / down.shape[0] if alpha is not None else 1.0
    x += ((strength * a) * torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1)).reshape(x.shape)).type(x.dtype)
    return x


def lora_layers(f, prefix=""):
    """{module: (up key, down key, alpha key or None)} for every '<module>.lora_A/B.weight' pair in a LoRA file."""
    keys = set(f.keys())
    out = {}
    for k in keys:
        if k.endswith(".lora_A.weight") and k.startswith(prefix):
            m = k[len(prefix):-len(".lora_A.weight")]
            a = k[:-len(".lora_A.weight")] + ".alpha"
            out[m] = (k[:-len("lora_A.weight")] + "lora_B.weight", k, a if a in keys else None)
    return out


def apply_lora(base, f, layers, strength=1.0):
    for m, (ku, kd, ka) in sorted(layers.items()):
        alpha = f.get_tensor(ka).item() if ka else None
        up, down = f.get_tensor(ku), f.get_tensor(kd)
        if base.has(m + ".comfy_quant"):
            w = base.int8(m)
            q, s = requant(w, lora_add(w.dequantize(), up, down, alpha, strength), m)
            yield m + ".weight", q
            yield m + ".weight_scale", s
        else:   # bf16 linear (token refiner): the patched weight is used as is
            yield m + ".weight", lora_add(base.get(m + ".weight").to(bf16), up, down, alpha, strength)


@torch.inference_mode()
def lora_patch(base_path, lora_path, strength=1.0):
    """ComfyUI LoraLoaderModelOnly (keys 'diffusion_model.<module>.lora_A/B.weight' + '.alpha'). Yields (name, tensor)."""
    base = Base(base_path)
    with safe_open(lora_path, "pt", device="cuda") as f:
        yield from apply_lora(base, f, lora_layers(f, "diffusion_model."), strength)


@torch.inference_mode()
def adapter_patch(base_path, adapter_path, strength=1.0):
    """comfyui-piflow 'Load LakonLab Adapter' on an H3 DiT. The adapter's keys are already ComfyUI's module names:
    '<module>.lora_A/B.weight' = LoRA (alpha absent), int8 '<module>.weight' + scale + comfy_quant = set (dequantized on
    the CPU, as piflow loads it), any other tensor = float set; dwd_* = the DWD head (not a DiT weight). Yields
    (name, tensor)."""
    base = Base(base_path)
    with safe_open(adapter_path, "pt", device="cpu") as f:
        keys = [k for k in f.keys() if not k.startswith("dwd_")]
        layers = lora_layers(f)
        lora_keys = {k for v in layers.values() for k in v if k}
        quant = sorted(k[:-len(".comfy_quant")] for k in keys if k.endswith(".comfy_quant"))
        for m in quant:   # int8 set: piflow dequantizes the adapter weight on the CPU, ComfyUI casts it into x (bf16)
            conf = json.loads(bytes(f.get_tensor(m + ".comfy_quant").tolist()))
            v = qtensor(f.get_tensor(m + ".weight"), f.get_tensor(m + ".weight_scale"), conf, bf16)  # mixed_precision_ops
            q, s = requant(base.int8(m), v.dequantize().to("cuda"), m)
            yield m + ".weight", q
            yield m + ".weight_scale", s
        done = lora_keys | {m + s for m in quant for s in (".weight", ".weight_scale", ".comfy_quant")}
        for k in keys:    # float set: the adapter's value (F16) as the forward's fp32 input; the engine rounds it
            if k not in done:
                yield k, f.get_tensor(k).to(device="cuda", dtype=torch.float32)
    with safe_open(adapter_path, "pt", device="cuda") as fg:
        yield from apply_lora(base, fg, layers, strength)
