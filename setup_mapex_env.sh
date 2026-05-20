#!/bin/bash -l
#SBATCH --job-name=setup_mapex
#SBATCH --output=logs/setup_mapex_%j.out
#SBATCH --error=logs/setup_mapex_%j.err
#SBATCH --mem-per-cpu=16g
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --account=ls_polle

# Set up the MapEx baseline environment.
#
# Strategy: clone ftnet_inference_env_2 (which already has open3d, pywavemap,
# ompl, unik3d, hdbscan, etc.) and layer the LaMa/MapEx-specific packages
# (hydra-core, omegaconf, pytorch-lightning, kornia, …) on top.
# pywavemap has no Python-3.10 wheel so this avoids rebuilding from scratch.
#
# Usage (interactive):  bash setup_mapex_env.sh
# Usage (sbatch):       sbatch setup_mapex_env.sh
#
# After this script, manually download the pretrained LaMa ensemble weights
# (see "Model download" note at the bottom of this script's output).

module load eth_proxy

set -euo pipefail

MAPEX_DIR=/cluster/project/cvg/students/shangwu/MapEx
BASE_ENV=/cluster/project/cvg/students/shangwu/ftnet_inference_env_2
ENV_DIR=/cluster/project/cvg/students/shangwu/mapex_env

# ---------------------------------------------------------------------------
# 1. Clone MapEx with submodules
# ---------------------------------------------------------------------------
echo "================================================================"
echo " Step 1: Clone MapEx"
echo "================================================================"

if [[ -d "${MAPEX_DIR}/.git" ]]; then
    echo "MapEx already cloned at ${MAPEX_DIR} — updating submodules"
    git -C "${MAPEX_DIR}" submodule update --init --recursive
else
    git clone --recurse-submodules https://github.com/castacks/MapEx.git "${MAPEX_DIR}"
fi

echo "MapEx at: $(git -C "${MAPEX_DIR}" rev-parse --short HEAD)"

# ---------------------------------------------------------------------------
# 2. Create conda environment (clone from ftnet_inference_env_2)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Step 2: Clone ${BASE_ENV} → ${ENV_DIR}"
echo "================================================================"

source ~/miniconda3/etc/profile.d/conda.sh

if [[ -d "${ENV_DIR}" ]]; then
    echo "Environment already exists at ${ENV_DIR} — skipping clone"
else
    conda create --clone "${BASE_ENV}" -p "${ENV_DIR}"
fi

conda activate "${ENV_DIR}"

# ---------------------------------------------------------------------------
# 3. Install LaMa inference dependencies
#
# The cloned env already provides: torch, numpy, open3d, pywavemap, ompl,
# unik3d, hdbscan, segmentation-models-pytorch, albumentations, wandb, etc.
#
# We add only what LaMa's inference pipeline additionally requires:
#   hydra-core 1.1.2       — LaMa conf/ uses Hydra 1.x Structured Configs
#   omegaconf  2.1.2       — required by hydra-core 1.1.x (>=2.1.1,<2.2)
#   pytorch-lightning 1.9.5 — checkpoint dict matches LaMa training (1.2.9)
#   kornia 0.6.12          — LaMa uses kornia; cloned env doesn't have it
#   setuptools<71          — pkg_resources still needed by lightning_fabric
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Step 3: Install LaMa inference dependencies"
echo "================================================================"

pip install --no-cache-dir \
    "hydra-core==1.1.2" \
    "omegaconf==2.1.2" \
    "pytorch-lightning==1.9.5" \
    "kornia==0.6.12" \
    "scikit-image" \
    "imageio" \
    "braceexpand" \
    "webdataset" \
    "easydict" \
    "setuptools<71"

# lama has no setup.py — add it to sys.path via a .pth file
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
echo "${MAPEX_DIR}/lama" > "${SITE_PKG}/mapex_lama.pth"
echo "Added lama to sys.path via ${SITE_PKG}/mapex_lama.pth"

# ---------------------------------------------------------------------------
# 4. Patch LaMa source for compatibility with modern Python packages
#
#  aug.py        : albumentations 1.2+ removed imgaug-based DualIAATransform
#                  (training-only; stub it out so inference still works)
#  fake_fakes.py : kornia 0.6.x moved SamplePadding to kornia.constants
#  trainers/__init__.py : torch 2.6+ changed torch.load default to
#                  weights_only=True, which rejects LaMa's checkpoint format
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Step 4: Patch LaMa source"
echo "================================================================"

AUG_PY="${MAPEX_DIR}/lama/saicinpainting/training/data/aug.py"
if ! grep -q "_HAVE_IAA" "${AUG_PY}"; then
python - "${AUG_PY}" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
old = "from albumentations import DualIAATransform, to_tuple\nimport imgaug.augmenters as iaa"
new = (
    "try:\n"
    "    from albumentations import DualIAATransform, to_tuple\n"
    "    import imgaug.augmenters as iaa\n"
    "    _HAVE_IAA = True\n"
    "except ImportError:\n"
    "    _HAVE_IAA = False\n"
    "    class DualIAATransform:\n"
    "        def __init__(self, *a, **kw):\n"
    "            raise NotImplementedError('imgaug transforms unavailable')\n"
    "    def to_tuple(x, low=None):\n"
    "        if isinstance(x, (tuple, list)):\n"
    "            return tuple(x)\n"
    "        return (x, x) if low is None else (low, x)"
)
with open(path, 'w') as f:
    f.write(src.replace(old, new, 1))
print(f"Patched {path}")
PYEOF
else
    echo "aug.py already patched"
fi

FF_PY="${MAPEX_DIR}/lama/saicinpainting/training/modules/fake_fakes.py"
if ! grep -q "kornia.constants" "${FF_PY}"; then
python - "${FF_PY}" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
old = "from kornia import SamplePadding"
new = (
    "try:\n"
    "    from kornia import SamplePadding\n"
    "except ImportError:\n"
    "    from kornia.constants import SamplePadding"
)
with open(path, 'w') as f:
    f.write(src.replace(old, new, 1))
print(f"Patched {path}")
PYEOF
else
    echo "fake_fakes.py already patched"
fi

TRAINERS_PY="${MAPEX_DIR}/lama/saicinpainting/training/trainers/__init__.py"
if ! grep -q "weights_only=False" "${TRAINERS_PY}"; then
python - "${TRAINERS_PY}" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
old = "state = torch.load(path, map_location=map_location)"
new = "state = torch.load(path, map_location=map_location, weights_only=False)"
with open(path, 'w') as f:
    f.write(src.replace(old, new, 1))
print(f"Patched {path}")
PYEOF
else
    echo "trainers/__init__.py already patched"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Setup complete!  Activate with:"
echo "   conda activate ${ENV_DIR}"
echo "================================================================"
echo ""
echo " *** Model download (manual step) ***"
echo " Download the pretrained LaMa weights from the Google Drive link"
echo " in the MapEx README (https://github.com/castacks/MapEx) and"
echo " extract them so the directory looks like:"
echo ""
echo "   ${MAPEX_DIR}/pretrained_models/weights/big_lama/models/best.ckpt"
echo "   ${MAPEX_DIR}/pretrained_models/weights/lama_ensemble/train_1-3/models/best.ckpt"
echo "================================================================"
