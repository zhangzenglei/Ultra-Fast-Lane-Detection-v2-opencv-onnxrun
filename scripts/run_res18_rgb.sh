#!/usr/bin/env bash
# Full RGB comparison, with the original model/crop/normalization/gates fixed.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image_dir="/home/zenglei/Data/v6453/mcapWriter1_record_20260916-102329/sensor_camera_front_mono_image_encode"
output_dir="$PROJECT_ROOT/outputs/v6453_res18_rgb_r50"
gpu_id=0
resume=false
dry_run=false

usage() {
  cat <<'HELP'
Usage: ./scripts/run_res18_rgb.sh [options]
  --image-dir DIR    Original JPG/JPEG/PNG directory
  --output-dir DIR   Separate RGB output directory
  --gpu-id N         CUDA device (default 0)
  --resume           Skip matching, completed RGB outputs
  --dry-run          Print the fixed command without running or writing outputs
  --help            Show help without running inference

Fixed: CULane Res18, CUDA, RGB, no CLAHE, row ratio 0.5, col ratio 0.25.
Saves result JPGs, keypoint/diagnostic JSON and a separate run log.
Relative paths are resolved from the project root. Existing outputs are
refused unless --resume is explicitly provided. No automatic build/install.
HELP
}

while (( $# )); do
  case "$1" in
    --help) usage; exit 0 ;;
    --resume) resume=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --image-dir|--output-dir|--gpu-id)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      case "$1" in
        --image-dir) image_dir="$2" ;;
        --output-dir) output_dir="$2" ;;
        --gpu-id) gpu_id="$2" ;;
      esac
      shift 2
      ;;
    *) echo "Unknown option: $1 (use --help)" >&2; exit 2 ;;
  esac
done

cd -- "$PROJECT_ROOT"
export UFLDV2_BINARY="$PROJECT_ROOT/build/prepost/ufldv2_gpu"
model="$PROJECT_ROOT/model/weights/ufldv2_culane_res18_320x1600.onnx"
if [[ ! -x "$UFLDV2_BINARY" ]]; then
  echo "RGB executable not found: $UFLDV2_BINARY; build with README_PREPOST.md commands" >&2
  exit 1
fi
if [[ ! -f "$model" || ! -d "$image_dir" ]]; then
  echo "Missing model or input directory: $model ; $image_dir" >&2
  exit 1
fi
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "--gpu-id must be a non-negative integer" >&2
  exit 2
fi
if [[ -e "$output_dir" && "$resume" != true ]]; then
  echo "Output already exists: $output_dir; use --resume or a new directory" >&2
  exit 1
fi
command=("$PROJECT_ROOT/scripts/run_cpp_gpu.sh"
  --model "$model" --image-dir "$image_dir" --output-dir "$output_dir"
  --device cuda --gpu-id "$gpu_id" --save-points
  --input-color rgb --enhance none --row-min-ratio 0.5 --col-min-ratio 0.25)
if [[ "$resume" == true ]]; then
  command+=(--skip-existing)
fi
if [[ "$dry_run" == true ]]; then
  printf 'UFLDV2_BINARY=%q ' "$UFLDV2_BINARY"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
mkdir -p -- "$output_dir"
log="$output_dir/run_rgb_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$.log"
{
  echo "RGB-only comparison: same Res18, no CLAHE, row=0.5, col=0.25"
  echo "Executable: $UFLDV2_BINARY"
  echo "Output: $output_dir"
  echo "Log: $log"
  printf 'Command: '
  printf '%q ' "${command[@]}"
  printf '\n'
} | tee "$log"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${command[@]}" 2>&1 | tee -a "$log"
else
  "${command[@]}" 2>&1 | tee -a "$log"
fi
