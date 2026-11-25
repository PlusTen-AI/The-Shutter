# Where shutter finds its weights: one shared models folder in ComfyUI's folder layout (design §0b), mounted read-only
# at /models — the same files ComfyUI uses, no copies. LoRAs and the pi-H3 adapter are merged at load (merge.py).
# SOURCES says where each file comes from (group, Hugging Face repo, path in the repo) for docker/fetch_models.py; the
# repo commit, size and sha256 of each file shutter was tested with are pinned in docker/weights.lock.json.
import os

ROOT = os.environ.get("SHUTTER_MODELS", "/models")
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ.get("SHUTTER_LIB", os.path.normpath(os.path.join(HERE, "..", "..", "build", "libshutter.so")))
QWEN_TOKENIZER = os.path.normpath(os.path.join(HERE, "..", "tokenizers", "qwen25"))   # Qwen-Image 2.1 TE and H3 TE

# Qwen-Image 2.1
QI_DIT = "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"
QI_TURBO = "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"
QI_TE = "text_encoders/qwen3vl_8b_int8_convrot.safetensors"
QI_VAE = "vae/qwen_image_2.1_vae_bf16.safetensors"
# MiniMax H3: Singularity v1.3 + ref2v turbo 4-step LoRA @1.0 (video.r2v / swap), FL2VA + pi-H3 adapter (video.pih3),
# official ref2va (golden reference only)
H3_SINGULARITY = "diffusion_models/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
H3_TURBO_LORA = "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
H3_FL2VA = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
PIH3_ADAPTER = "loras/pi_h3_%dstep_dwd_adapter_int8_convrot.safetensors"
H3_REF2VA = "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
H3_TE = "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VAE = "vae/minimax_h3_video_vae_int8_convrot.safetensors"
H3_AUDIO = "vae/minimax_h3_audio_vae_fp32.safetensors"
H3_UP = "latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
# LTX-2.5
LTX_DIT = "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
LTX_TE = "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
LTX_VAE = "vae/ltx-2.5-video-vae-bf16.safetensors"
LTX_AUDIO = "vae/ltx-2.5-audio-vae-bf16.safetensors"
LTX_UP = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
# SeedVR2 7B
SV2_DIT = "diffusion_models/seedvr2_7b_int8_convrot.safetensors"
SV2_VAE = "vae/seedvr2_ema_vae_fp16.safetensors"
# faces and pose
FACE = "insightface/models/buffalo_l"                  # det_10g.onnx, w600k_r50.onnx
FACESWAP = "facefusion"                                # hyperswap_1a_256.onnx, xseg_1.onnx
POSE = "annotators/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt"
# voice consistency (기능 17, boss session #18): speech / background separation, voice conversion (Seed-VC v1 small:
# DiT + whisper-small content encoder + BigVGAN 22 kHz + CAM++ speaker embedding), reference voices (Qwen3-TTS)
VC_DIR = "voice/seed_vc"                             # DiT_…_pruned.pth + its config yml
VC_DIT = VC_DIR + "/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
VC_CFG = VC_DIR + "/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
VC_WHISPER = "voice/whisper-small"                   # config.json, preprocessor_config.json, model.safetensors
VC_BIGVGAN = "voice/bigvgan_v2_22khz_80band_256x"    # bigvgan_generator.pt, config.json
VC_CAMPP = "voice/campplus/campplus_cn_common.bin"
TTS_VD = "voice/Qwen3-TTS-12Hz-1.7B-VoiceDesign"     # the repo's files (+ speech_tokenizer/)
SEP = "voice/separation/MelBandRoformer_fp16.safetensors"

# groups (fetch_models.py fetches the groups asked for, base always): base = face check / face pass / pose check,
# image = Qwen-Image 2.1, h3 = MiniMax H3 (r2v, swap, pi-H3), ltx = LTX-2.5, upscale = SeedVR2, voice = 기능 17
SOURCES = [   # (group, file in ROOT, Hugging Face repo, path in the repo)
    ("base", FACE + "/det_10g.onnx", "public-data/insightface", "models/buffalo_l/det_10g.onnx"),
    ("base", FACE + "/w600k_r50.onnx", "public-data/insightface", "models/buffalo_l/w600k_r50.onnx"),
    ("base", FACESWAP + "/hyperswap_1a_256.onnx", "facefusion/models-3.3.0", "hyperswap_1a_256.onnx"),
    ("base", FACESWAP + "/xseg_1.onnx", "facefusion/models-3.1.0", "xseg_1.onnx"),
    ("base", POSE, "hr16/DWPose-TorchScript-BatchSize5", os.path.basename(POSE)),
    ("image", QI_DIT, "Comfy-Org/Qwen-Image-2.1", QI_DIT),
    ("image", QI_TURBO, "Comfy-Org/Qwen-Image-2.1", QI_TURBO),
    ("image", QI_TE, "Comfy-Org/Qwen-Image-2.1", QI_TE),
    ("image", QI_VAE, "Comfy-Org/Qwen-Image-2.1", QI_VAE),
    ("h3", H3_SINGULARITY, "WarmBloodAban/Minimax-h3_Singularity", os.path.basename(H3_SINGULARITY)),
    ("h3", H3_TURBO_LORA, "Comfy-Org/MiniMax-H3", H3_TURBO_LORA),
    ("h3", H3_FL2VA, "Comfy-Org/MiniMax-H3", H3_FL2VA),
    ("h3", PIH3_ADAPTER % 2, "Lakonik/pi-H3", os.path.basename(PIH3_ADAPTER % 2)),
    ("h3", PIH3_ADAPTER % 4, "Lakonik/pi-H3", os.path.basename(PIH3_ADAPTER % 4)),
    ("h3", H3_TE, "Comfy-Org/MiniMax-H3", H3_TE),
    ("h3", H3_VAE, "Comfy-Org/MiniMax-H3", H3_VAE),
    ("h3", H3_AUDIO, "Comfy-Org/MiniMax-H3", H3_AUDIO),
    ("h3", H3_UP, "LBH-123-AI/Minimax_h3_latent_Upscaler", "minimax_h3_latent_upscaler_3d_conv_v1/" + os.path.basename(H3_UP)),
    ("ltx", LTX_DIT, "Lightricks/LTX-2.5", LTX_DIT),
    ("ltx", LTX_TE, "Lightricks/LTX-2.5", LTX_TE),
    ("ltx", LTX_VAE, "Lightricks/LTX-2.5", LTX_VAE),
    ("ltx", LTX_AUDIO, "Lightricks/LTX-2.5", LTX_AUDIO),
    ("ltx", LTX_UP, "Lightricks/LTX-2.5", LTX_UP),
    ("upscale", SV2_DIT, "Comfy-Org/SeedVR2", SV2_DIT),
    ("upscale", SV2_VAE, "Comfy-Org/SeedVR2", SV2_VAE),
    ("voice", VC_DIT, "Plachta/Seed-VC", os.path.basename(VC_DIT)),
    ("voice", VC_CFG, "Plachta/Seed-VC", os.path.basename(VC_CFG)),
] + [("voice", VC_WHISPER + "/" + f, "openai/whisper-small", f) for f in ("config.json", "preprocessor_config.json", "model.safetensors")] + [
    ("voice", VC_BIGVGAN + "/" + f, "nvidia/bigvgan_v2_22khz_80band_256x", f) for f in ("bigvgan_generator.pt", "config.json")] + [
    ("voice", VC_CAMPP, "funasr/campplus", os.path.basename(VC_CAMPP)),
    ("voice", SEP, "Kijai/MelBandRoFormer_comfy", os.path.basename(SEP)),
] + [("voice", TTS_VD + "/" + f, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", f) for f in (
    "model.safetensors", "config.json", "generation_config.json", "merges.txt", "vocab.json", "tokenizer_config.json",
    "preprocessor_config.json", "speech_tokenizer/model.safetensors", "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json", "speech_tokenizer/preprocessor_config.json")]

def path(rel):
    return os.path.join(ROOT, rel)
