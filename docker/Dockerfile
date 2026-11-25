# shutter: one image for the server and the worker (design §0b). Built on each node (deploy.sh build):
#   docker build --label project=shutter -t shutter:<tag> -f docker/Dockerfile <checkout or GitHub URL#tag>
# Run: entrypoint.sh main (server + worker) or worker; weights mounted read-only at /models (ComfyUI folder layout,
# engine/py/models.py), data (queue, input cache, results) at /work.
FROM nvcr.io/nvidia/pytorch:26.07-py3
# PyAV: video/audio I/O. comfy-kitchen: int8 ConvRot (de)quantization + stochastic rounding for the load-time LoRA merge
# (engine/py/merge.py), the same package ComfyUI calls. huggingface_hub: fetch_models.py (<1.0 for transformers 4.57).
RUN pip install --no-cache-dir av==19.0.2 comfy-kitchen==0.2.41 huggingface_hub==0.36.2
# Voice (기능 17, engine/py/voice.py · melband.py). The base image's torch / numpy / scipy / triton are held as
# constraints (pip fails instead of swapping them; no torchaudio — a PyPI one would replace the NGC torch, voice.py
# ports what it needs). qwen-tts without its deps (torchaudio, gradio, sox, onnxruntime: unused by the 12 Hz
# VoiceDesign model, stand-ins at import); its real needs are pinned here with Seed-VC's (librosa, munch, einops …).
RUN pip list --format=freeze | grep -iE '^(torch|numpy|scipy|triton)==' > /tmp/base-pins.txt && \
    pip install --no-cache-dir -c /tmp/base-pins.txt \
      transformers==4.57.3 tokenizers==0.22.2 huggingface_hub==0.36.2 accelerate==1.12.0 \
      librosa==0.11.0 soundfile==0.13.1 einops==0.8.1 munch==4.0.0 && \
    pip install --no-cache-dir --no-deps qwen-tts==0.1.1 && \
    (python -c "import matplotlib" 2>/dev/null || pip install --no-cache-dir -c /tmp/base-pins.txt matplotlib==3.10.9) && \
    python -c "import yaml, tqdm, scipy, safetensors" && rm /tmp/base-pins.txt
# Seed-VC (GPL-3.0, archived) at its last commit — not in this repository: only modules/ and LICENSE, by a sparse,
# blob-filtered fetch (no checkpoints or sample audio); voice.py puts /opt/seed-vc on sys.path itself.
ARG SEED_VC_COMMIT=51383efd921027683c89e5348211d93ff12ac2a8
RUN git init -q /opt/seed-vc && cd /opt/seed-vc && \
    git remote add origin https://github.com/Plachtaa/seed-vc.git && \
    git sparse-checkout set --no-cone /modules/ /LICENSE && \
    git fetch -q --depth 1 --filter=blob:none origin ${SEED_VC_COMMIT} && \
    git checkout -q FETCH_HEAD && rm -rf .git && echo ${SEED_VC_COMMIT} > COMMIT && \
    test -f modules/campplus/DTDNN.py && test -f modules/bigvgan/bigvgan.py
ENV SHUTTER_SEED_VC=/opt/seed-vc NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /opt/shutter
# the CUDA engine first: rebuilt only when its sources change (CUTLASS: headers shipped in the base image)
COPY engine/*.cu engine/*.cuh engine/*.cpp engine/*.h engine/
RUN mkdir -p build && cd engine && nvcc -O3 -std=c++17 -arch=sm_121 -shared -Xcompiler -fPIC -lineinfo \
      -I/opt/pytorch/pytorch/third_party/cutlass/include --expt-relaxed-constexpr \
      -o /opt/shutter/build/libshutter.so *.cpp *.cu -lcublasLt 2>&1 | grep -v 'NOTE\|^$'; \
    test -f /opt/shutter/build/libshutter.so
COPY engine/py engine/py
COPY engine/tokenizers engine/tokenizers
COPY engine/fonts /opt/shutter/fonts
COPY server server
COPY worker worker
COPY docker/entrypoint.sh docker/fetch_models.py docker/weights.lock.json ./
ENV PYTHONPATH=/opt/shutter/engine/py PYTHONDONTWRITEBYTECODE=1 SHUTTER_WORK=/work SHUTTER_MODELS=/models
LABEL project=shutter
ENTRYPOINT ["/opt/shutter/entrypoint.sh"]
CMD ["worker"]
