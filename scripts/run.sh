#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
SHOW_GUI="${SHOW_GUI:-true}"
DECODE_EVENTS="${DECODE_EVENTS:-true}"
E2FAI_ENABLED="${E2FAI_ENABLED:-true}"
PERF_ENABLED="${PERF_ENABLED:-false}"
[ "${DECODE_EVENTS}" = true ] || E2FAI_ENABLED=false

if ! docker image inspect nrv-demo:noetic >/dev/null 2>&1; then
  "${project_dir}/scripts/build.sh"
fi
mkdir -p "${project_dir}/output"
run_dir="${RUN_DIR:-$(mktemp -d "${project_dir}/output/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")}"
mkdir -p "${run_dir}"
run_dir="$(realpath "${run_dir}")"
case "${run_dir}" in "${project_dir}/output/"*) ;; *) echo 'RUN_DIR must be inside this project output directory.' >&2; exit 1;; esac
container_output="/output/${run_dir#"${project_dir}/output/"}"
host_pid=""
container_name="nrv-demo-run-$$"
xhost_granted=false
cleanup() {
  if docker container inspect "${container_name}" >/dev/null 2>&1; then
    docker stop -t 10 "${container_name}" >/dev/null 2>&1 || true
  fi
  if [ -n "${host_pid}" ] && kill -0 "${host_pid}" 2>/dev/null; then
    kill -TERM "${host_pid}" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "${host_pid}" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "${host_pid}" 2>/dev/null || true
    wait "${host_pid}" 2>/dev/null || true
  fi
  if [ "${xhost_granted}" = true ]; then xhost -si:localuser:root >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "${SHOW_GUI}" = true ] && [ "${DECODE_EVENTS}" = true ]; then
  [ -n "${DISPLAY:-}" ] || { echo 'DISPLAY is empty. Set SHOW_GUI=false for a headless run.' >&2; exit 1; }
  xhost +si:localuser:root >/dev/null
  xhost_granted=true
fi

if [ "${E2FAI_ENABLED}" = true ]; then
  python_bin="${E2FAI_PYTHON:-${HOME}/miniconda3/envs/nrv-e2fai/bin/python}"
  [ -x "${python_bin}" ] || { echo 'Model environment missing. Run scripts/setup_e2fai.sh or set E2FAI_PYTHON.' >&2; exit 1; }
  host_args=(--host 127.0.0.1 --port "${E2FAI_PORT:-8765}" --window-ms "${WINDOW_MS:-250}"
    --result-mode "${E2FAI_RESULT_MODE:-thread}"
    --queue-batches "${E2FAI_QUEUE_BATCHES:-256}"
    --backbone "${BACKBONE_CHECKPOINT:-${project_dir}/checkpoints/e2fai_backbone.ckpt}"
    --image-checkpoint "${IMAGE_CHECKPOINT:-${project_dir}/checkpoints/image_residual_epoch043.pt}"
    --device "${E2FAI_DEVICE:-cuda:0}" --output-dir "${run_dir}")
  [ "${PERF_ENABLED}" != true ] || host_args+=(--perf-enabled)
  [ "${E2FAI_FRESHNESS:-true}" != false ] || host_args+=(--no-freshness)
  PYTHONNOUSERSITE=1 "${python_bin}" "${project_dir}/examples/e2fai_worker.py" "${host_args[@]}" >"${run_dir}/host.log" 2>&1 &
  host_pid=$!
  for _ in $(seq 1 480); do
    [ ! -f "${run_dir}/ready" ] || break
    if ! kill -0 "${host_pid}" 2>/dev/null; then cat "${run_dir}/host.log" >&2; exit 1; fi
    sleep 0.5
  done
  [ -f "${run_dir}/ready" ] || { echo "Model startup timed out: ${run_dir}/host.log" >&2; exit 1; }
fi

echo "Output: ${run_dir}"
echo 'Move an object in front of the camera. Ctrl+C stops the demo and writes the summary.'
launch_file=demo.launch
if [ "${SHOW_GUI}" = true ] && [ "${DECODE_EVENTS}" = true ]; then
  launch_file=gui.launch
fi
launch_status=0
docker compose -f "${project_dir}/compose.yaml" run --rm --name "${container_name}" nrv-demo \
  roslaunch nrv_demo "${launch_file}" \
  "serial_number:=${CAMERA_SERIAL:-}" "device_index:=${CAMERA_INDEX:-0}" \
  "show_gui:=${SHOW_GUI}" "decode_events:=${DECODE_EVENTS}" \
  "duration:=${DURATION:-0}" "output_dir:=${container_output}" \
  "e2fai_enabled:=${E2FAI_ENABLED}" "e2fai_port:=${E2FAI_PORT:-8765}" "perf_enabled:=${PERF_ENABLED}" \
  "$@" || launch_status=$?

# Flush the model session before evaluating the complete integration.
cleanup
host_pid=""
xhost_granted=false

# roslaunch may return zero when a required child fails; use the receiver result.
if [ ! -f "${run_dir}/summary.json" ]; then
  echo "FAIL: no receiver summary (launcher exit ${launch_status}). See the ROS log above." >&2
  exit 1
fi
check_args=("${run_dir}")
[ "${E2FAI_ENABLED}" = true ] || check_args+=(--model-disabled)
python3 "${project_dir}/scripts/check_run.py" "${check_args[@]}"
