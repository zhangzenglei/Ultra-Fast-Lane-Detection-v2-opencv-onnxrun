#!/usr/bin/env bash
# Read-only diagnostics; no installations or global configuration writes.
set -uo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ORT_ROOT="${ORT_ROOT:-/home/zenglei/opt/onnx-gpu/onnxruntime-linux-x64-gpu-1.15.1}"
export CUDNN_ROOT="${CUDNN_ROOT:-/home/zenglei/opt/onnx-gpu/cudnn-linux-x86_64-8.7.0.84_cuda11-archive}"
runtime_paths="$ORT_ROOT/lib:$CUDNN_ROOT/lib:$CUDNN_ROOT/lib64"
if [[ -n "${CUDA_ROOT:-}" ]]; then
  runtime_paths="$runtime_paths:$CUDA_ROOT/lib64:$CUDA_ROOT/targets/x86_64-linux/lib"
fi
export LD_LIBRARY_PATH="$runtime_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
failures=0
check() {
  echo "+ $*"
  if ! "$@"; then failures=$((failures + 1)); echo "FAILED: $*" >&2; fi
}
echo "ORT_ROOT=$ORT_ROOT"
echo "CUDNN_ROOT=$CUDNN_ROOT"
echo "CUDA_ROOT=${CUDA_ROOT:-<system runtime search paths>}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
check cat /etc/os-release
check nvidia-smi
check nvcc --version
check cmake --version
check g++ --version
check pkg-config --modversion opencv4
check test -f /usr/include/nlohmann/json.hpp
check test -f "$ORT_ROOT/include/onnxruntime_cxx_api.h"
check test -f "$ORT_ROOT/lib/libonnxruntime.so"
check test -f "$CUDNN_ROOT/lib/libcudnn.so.8"
provider="$ORT_ROOT/lib/libonnxruntime_providers_cuda.so"
check test -f "$provider"
if [[ -f "$provider" ]]; then
  echo "+ ldd $provider"
  if dependencies="$(ldd "$provider" 2>&1)"; then
    echo "$dependencies"
    if [[ "$dependencies" == *"not found"* ]]; then
      echo "MISSING PROVIDER DEPENDENCIES (not found)" >&2
      failures=$((failures + 1))
    fi
  else
    echo "$dependencies" >&2
    failures=$((failures + 1))
  fi
fi
model="${1:-$PROJECT_ROOT/weights/ufldv2_culane_res18_320x1600.onnx}"
echo "Model=$model"
check test -s "$model"
if [[ "$model" != *.onnx ]]; then
  echo "Model must be a completed .onnx file" >&2
  failures=$((failures + 1))
fi
echo "Diagnostics: failures=$failures (a compiler/SDK check is not a GPU inference test)"
(( failures == 0 ))
