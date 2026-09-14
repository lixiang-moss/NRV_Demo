#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

project_dir="$(cd "$(dirname "$0")" && pwd)"
python_bin="${E2FAI_PYTHON:-${HOME}/miniconda3/envs/nrv-e2fai/bin/python}"

command -v docker >/dev/null 2>&1 || {
  echo "Error: Docker was not found. Install Docker Engine and Compose v2 first." >&2
  exit 1
}
docker info >/dev/null 2>&1 || {
  echo "Error: Docker is not running, or the current user cannot access it." >&2
  exit 1
}
docker compose version >/dev/null 2>&1 || {
  echo "Error: Docker Compose v2 was not found." >&2
  exit 1
}

echo "Verifying model checkpoints..."
(cd "${project_dir}/checkpoints" && sha256sum --check SHA256SUMS)

[ -x "${python_bin}" ] || {
  echo "Error: the E2FAI runtime was not found at ${python_bin}." >&2
  echo "Run ${project_dir}/scripts/setup_e2fai.sh once before using this launcher." >&2
  exit 1
}

echo "Starting the NRV E2FAI demo..."
exec "${project_dir}/scripts/run.sh" "$@"
