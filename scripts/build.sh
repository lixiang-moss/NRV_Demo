#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
docker compose -f "${project_dir}/compose.yaml" build "$@"
