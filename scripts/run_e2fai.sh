#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
learning_root="${LEARNING_EVERYTHING_ROOT:-/home/shuang/codes/learning_everything}"
python_bin="${PYTHON:-/home/shuang/miniconda3/envs/e2fai_pp/bin/python}"
image_checkpoint="${IMAGE_CHECKPOINT:-/raid/shuang/learning_everything/learning_everything_reproduction/fresh_e2fai_seed42_rerun_20260824T172241Z/run/image/checkpoints/epoch_043.pt}"
backbone="${BACKBONE_CHECKPOINT:-${learning_root}/checkpoints/e2fai_backbone.ckpt}"
port="${PORT:-8765}"
window_ms="${WINDOW_MS:-100}"
gpu="${GPU:-0}"
sensor_width="${SENSOR_WIDTH:-960}"
sensor_height="${SENSOR_HEIGHT:-720}"

[ -x "${python_bin}" ] || { echo "Python not executable: ${python_bin}" >&2; exit 1; }
[ -d "${learning_root}/src/learning_everything" ] || { echo "Missing learning_everything: ${learning_root}" >&2; exit 1; }
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
  --image-checkpoint "${image_checkpoint}" --backbone "${backbone}"
  --device cuda:0 --output-dir "${run_dir}"
)
[ "${HEADLESS:-false}" = true ] && host_args+=(--headless)
[ "${RECORD:-false}" = true ] && host_args+=(--record)

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
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${learning_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
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
  "bridge_host:=127.0.0.1" "bridge_port:=${port}" "${@}"

cleanup
host_pid=""
trap - EXIT INT TERM
echo "Saved real-time results in ${run_dir}"
