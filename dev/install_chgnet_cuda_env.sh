#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "numpy<2"

# Pinned stack verified to work with CHGNet + DGL backend on Python 3.12.
python -m pip install \
  torch==2.4.0+cu121 \
  torchvision==0.19.0+cu121 \
  torchaudio==2.4.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install torchdata==0.8.0
python -m pip install "https://data.dgl.ai/wheels/torch-2.4/dgl-2.4.0-cp312-cp312-manylinux1_x86_64.whl"
python -m pip install lightning<=2.6.0 pymatgen ase pydantic boto3 torch-geometric
python -m pip install -e . --no-deps

echo
echo "[OK] CHGNet environment install complete."
echo "[INFO] Run check:"
echo "  source .venv/bin/activate"
echo "  export MATGL_BACKEND=DGL"
echo "  export MATGL_CACHE='${ROOT_DIR}/.matgl_cache'"
echo "  python dev/chgnet_env_check.py --device auto"
