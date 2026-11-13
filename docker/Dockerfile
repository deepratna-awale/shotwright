# syntax=docker/dockerfile:1.7
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

FROM ${CUDA_IMAGE} AS base
ARG PYTHON_VERSION=3.12
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    SHOTWRIGHT_RUNS_DIR=/workspace/runs

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg libgl1 libglib2.0-0 openssh-server tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
RUN uv venv --python ${PYTHON_VERSION} ${VIRTUAL_ENV} \
    && uv pip install --index-url ${TORCH_INDEX} torch torchvision \
    && uv pip install hf_transfer

WORKDIR /opt/shotwright
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install ".[diffusers,serverless]"

COPY docker/start.sh /start.sh
RUN chmod +x /start.sh
WORKDIR /workspace
EXPOSE 22
ENTRYPOINT ["/usr/bin/tini", "--", "/start.sh"]
CMD ["sleep", "infinity"]

FROM base AS comfyui
ARG COMFYUI_REF=master
ENV COMFYUI_DIR=/opt/ComfyUI \
    COMFYUI_MODELS=/workspace/models \
    COMFYUI_PORT=8188
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_DIR} \
    && git -C ${COMFYUI_DIR} checkout ${COMFYUI_REF} \
    && uv pip install -r ${COMFYUI_DIR}/requirements.txt
EXPOSE 8188
ENV START_COMFYUI=1

FROM base AS serverless
ENV HF_HOME=/runpod-volume/hf \
    SHOTWRIGHT_RUNS_DIR=/runpod-volume/shotwright/runs
CMD ["python", "-m", "shotwright.serverless"]
