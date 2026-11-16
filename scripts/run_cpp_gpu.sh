#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ORT_ROOT="${ORT_ROOT:-/home/zenglei/opt/onnx-gpu/onnxruntime-linux-x64-gpu-1.15.1}"
export CUDNN_ROOT="${CUDNN_ROOT:-/home/zenglei/opt/onnx-gpu/cudnn-linux-x86_64-8.7.0.84_cuda11-archive}"
# CUDA_ROOT is optional. The installed distro runtime already uses ld.so paths.
runtime_paths="$ORT_ROOT/lib:$CUDNN_ROOT/lib:$CUDNN_ROOT/lib64"
if [[ -n "${CUDA_ROOT:-}" ]]; then
  runtime_paths="$runtime_paths:$CUDA_ROOT/lib64:$CUDA_ROOT/targets/x86_64-linux/lib"
fi
export LD_LIBRARY_PATH="$runtime_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
binary="${UFLDV2_BINARY:-$PROJECT_ROOT/build/ufldv2_gpu}"
if [[ ! -x "$binary" ]]; then
  echo "Executable not found: $binary; see README_CPP_GPU.md for build commands" >&2
  exit 1
fi
exec "$binary" "$@"
