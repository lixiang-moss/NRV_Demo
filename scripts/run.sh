#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
SHOW_GUI="${SHOW_GUI:-true}"
DECODE_EVENTS="${DECODE_EVENTS:-true}"

if ! docker image inspect nrv-demo:noetic >/dev/null 2>&1; then
  "${project_dir}/scripts/build.sh"
fi
mkdir -p "${project_dir}/output"
run_dir="$(mktemp -d "${project_dir}/output/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"

if [ "${SHOW_GUI}" = true ] && [ "${DECODE_EVENTS}" = true ]; then
  [ -n "${DISPLAY:-}" ] || { echo 'DISPLAY is empty. Set SHOW_GUI=false for a headless run.' >&2; exit 1; }
  xhost +si:localuser:root >/dev/null
  trap 'xhost -si:localuser:root >/dev/null 2>&1 || true' EXIT
fi

echo "Output: ${run_dir}"
echo 'Move an object in front of the camera. Ctrl+C stops the demo and writes the summary.'
launch_file=demo.launch
if [ "${SHOW_GUI}" = true ] && [ "${DECODE_EVENTS}" = true ]; then
  launch_file=gui.launch
fi
launch_status=0
docker compose -f "${project_dir}/compose.yaml" run --rm nrv-demo \
  roslaunch nrv_demo "${launch_file}" \
  "serial_number:=${CAMERA_SERIAL:-}" "device_index:=${CAMERA_INDEX:-0}" \
  "show_gui:=${SHOW_GUI}" "decode_events:=${DECODE_EVENTS}" \
  "duration:=${DURATION:-0}" "output_dir:=/output/$(basename "${run_dir}")" \
  "$@" || launch_status=$?

# roslaunch may return zero when a required child fails; use the receiver result.
if [ ! -f "${run_dir}/summary.json" ]; then
  echo "FAIL: no receiver summary (launcher exit ${launch_status}). See the ROS log above." >&2
  exit 1
fi
python3 - "${run_dir}/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    result = json.load(handle)
print(json.dumps(result, indent=2))
print("Summary:", sys.argv[1])
sys.exit(0 if result["status"] == "PASS" else 1)
PY
