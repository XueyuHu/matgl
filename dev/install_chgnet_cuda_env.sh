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

# Pinned stack for modern NVIDIA GPUs (e.g., RTX 50xx).
python -m pip install \
  torch==2.9.1 \
  torchvision==0.24.1 \
  torchaudio==2.9.1

python -m pip install torchdata==0.8.0
python -m pip install "https://data.dgl.ai/wheels/cu121/dgl-2.1.0%2Bcu121-cp312-cp312-manylinux1_x86_64.whl"
python -m pip install lightning<=2.6.0 pymatgen ase pydantic boto3 torch-geometric
python -m pip install -e . --no-deps

# DGL 2.1.0 graphbolt does not ship a binary for torch 2.9.x, but graphbolt is
# not required for current CHGNet training path. Patch to skip hard failure.
python - <<'PY'
from pathlib import Path
p = Path(".venv/lib/python3.12/site-packages/dgl/graphbolt/__init__.py")
txt = p.read_text()
txt = txt.replace(
    "    if not os.path.exists(path):\n        raise FileNotFoundError(\n            f\"Cannot find DGL C++ graphbolt library at {path}\"\n        )\n",
    "    if not os.path.exists(path):\n        return\n",
)
txt = txt.replace(
    "    except Exception:  # pylint: disable=W0703\n        raise ImportError(\"Cannot load Graphbolt C++ library\")\n",
    "    except Exception:  # pylint: disable=W0703\n        return\n",
)
p.write_text(txt)
print(f\"[OK] Patched graphbolt loader: {p}\")
PY

echo
echo "[OK] CHGNet environment install complete."
echo "[INFO] Run check:"
echo "  source .venv/bin/activate"
echo "  export MATGL_BACKEND=DGL"
echo "  export MATGL_CACHE='${ROOT_DIR}/.matgl_cache'"
echo "  python dev/chgnet_env_check.py --device auto"
