#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON:-python3}"
image_checkpoint="${IMAGE_CHECKPOINT:-${project_dir}/checkpoints/image_residual_epoch043.pt}"
backbone="${BACKBONE_CHECKPOINT:-${project_dir}/checkpoints/e2fai_backbone.ckpt}"
[ -f "${image_checkpoint}" ] || image_checkpoint="${project_dir}/epoch_043.pt"
[ -f "${backbone}" ] || backbone="${project_dir}/e2fai_backbone.ckpt"
port="${PORT:-8765}"
window_ms="${WINDOW_MS:-100}"
gpu="${GPU:-0}"
sensor_width="${SENSOR_WIDTH:-960}"
sensor_height="${SENSOR_HEIGHT:-720}"
resolution="${RESOLUTION:-${sensor_width}x${sensor_height}}"
if [[ "${resolution}" =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]]; then
  input_width="${BASH_REMATCH[1]}"
  input_height="${BASH_REMATCH[2]}"
else
  echo "Invalid RESOLUTION: ${resolution}. Use WIDTHxHEIGHT, e.g. 640x480." >&2
  exit 1
fi
strict_windows="${STRICT_WINDOWS:-true}"

command -v "${python_bin}" >/dev/null || { echo "Python not found: ${python_bin}" >&2; exit 1; }
[ -d "${project_dir}/runtime/nrv_e2fai" ] || { echo 'Missing bundled E2FAI runtime.' >&2; exit 1; }
[ -f "${image_checkpoint}" ] || { echo "Missing image checkpoint: ${image_checkpoint}" >&2; exit 1; }
[ -f "${backbone}" ] || { echo "Missing E2FAI checkpoint: ${backbone}" >&2; exit 1; }
if [ "${HEADLESS:-false}" != true ] && [ -z "${DISPLAY:-}" ]; then
  echo 'DISPLAY is empty. Run with HEADLESS=true or from a desktop terminal.' >&2
  exit 1
fi

mkdir -p "${project_dir}/output"
run_dir="$(mktemp -d "${project_dir}/output/e2fai_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
host_args=(
  --port "${port}" --window-ms "${window_ms}"
  --sensor-width "${sensor_width}" --sensor-height "${sensor_height}"
  --input-width "${input_width}" --input-height "${input_height}"
  --image-checkpoint "${image_checkpoint}" --backbone "${backbone}"
  --device cuda:0 --output-dir "${run_dir}"
)
[ "${HEADLESS:-false}" = true ] && host_args+=(--headless)
[ "${RECORD:-false}" = true ] && host_args+=(--record)
[ -n "${MAX_EVENTS:-}" ] && host_args+=(--max-events "${MAX_EVENTS}")
[ "${strict_windows}" != true ] && host_args+=(--latest-batch)

host_pid=""
cleanup() {
  if [ -n "${host_pid}" ] && kill -0 "${host_pid}" 2>/dev/null; then
    kill -TERM "${host_pid}" 2>/dev/null || true
    wait "${host_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Output: ${run_dir}"
echo "Starting E2FAI on physical GPU ${gpu}; Ctrl+C stops and saves the last frame."
echo "Camera: ${sensor_width}x${sensor_height}; model/image/flow: ${resolution}"
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${project_dir}/runtime${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" "${project_dir}/examples/e2fai_realtime.py" "${host_args[@]}" &
host_pid=$!

for _ in $(seq 1 120); do
  [ -f "${run_dir}/ready" ] && break
  if ! kill -0 "${host_pid}" 2>/dev/null; then
    wait "${host_pid}"
    echo 'E2FAI process exited before opening the event bridge.' >&2
    exit 1
  fi
  sleep 0.5
done
[ -f "${run_dir}/ready" ] || { echo 'Timed out waiting for E2FAI startup.' >&2; exit 1; }

RUN_DIR="${run_dir}" SHOW_GUI=false DECODE_EVENTS=true \
  "${project_dir}/scripts/run.sh" \
  "bridge_host:=127.0.0.1" "bridge_port:=${port}" \
  "bridge_compact:=${BRIDGE_COMPACT:-true}" \
  "run_algorithm:=false" "run_renderer:=false" "${@}"

cleanup
host_pid=""
trap - EXIT INT TERM
echo "Saved real-time results in ${run_dir}"
