#!/usr/bin/env bash
set -e
source /opt/ros/noetic/setup.bash
source /opt/nrv_demo_ws/devel/setup.bash
export XDG_RUNTIME_DIR=/tmp/nrv-runtime
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
exec "$@"
