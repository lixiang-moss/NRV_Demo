#!/usr/bin/env bash
# Independent host environment; ROS and Qt remain in Docker.
set -euo pipefail
export PYTHONNOUSERSITE=1
conda_bin="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"
[ -x "${conda_bin}" ] || conda_bin="$(command -v conda)"
"${conda_bin}" create -n nrv-e2fai python=3.10 pip -y
"${conda_bin}" run -n nrv-e2fai python -m pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
"${conda_bin}" run -n nrv-e2fai python -m pip install numpy==1.26.4 opencv-python==4.7.0.72 psutil pytest
"${conda_bin}" run -n nrv-e2fai python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
