#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/hf /workspace/runs

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
    ssh-keygen -A >/dev/null
    service ssh start >/dev/null
fi

if [[ "${START_COMFYUI:-0}" == "1" ]]; then
    mkdir -p "$COMFYUI_MODELS"
    if [[ ! -L "$COMFYUI_DIR/models" ]]; then
        cp -rn "$COMFYUI_DIR/models/." "$COMFYUI_MODELS/" 2>/dev/null || true
        rm -rf "$COMFYUI_DIR/models"
        ln -s "$COMFYUI_MODELS" "$COMFYUI_DIR/models"
    fi
    if [[ -n "${SHOTWRIGHT_PRELOAD:-}" ]]; then
        for model in ${SHOTWRIGHT_PRELOAD//,/ }; do
            shotwright download "$model" --for comfyui --comfy-models "$COMFYUI_MODELS"
        done
    fi
    python "$COMFYUI_DIR/main.py" --listen 0.0.0.0 --port "$COMFYUI_PORT" \
        ${COMFYUI_ARGS:-} >/workspace/comfyui.log 2>&1 &
elif [[ -n "${SHOTWRIGHT_PRELOAD:-}" ]]; then
    for model in ${SHOTWRIGHT_PRELOAD//,/ }; do
        shotwright download "$model"
    done
fi

exec "$@"
