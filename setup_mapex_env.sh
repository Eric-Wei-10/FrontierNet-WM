#!/bin/bash -l
#SBATCH --job-name=setup_mapex
#SBATCH --output=logs/setup_mapex_%j.out
#SBATCH --error=logs/setup_mapex_%j.err
#SBATCH --mem-per-cpu=16g
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --account=ls_polle

# Set up MapEx (https://github.com/castacks/MapEx) as a baseline.
#
# Usage (interactive):  bash setup_mapex_env.sh
# Usage (sbatch):       sbatch setup_mapex_env.sh
#
# What this does:
#   1. Clones MapEx + lama submodule to MAPEX_DIR
#   2. Creates a conda env at ENV_DIR (Python 3.10 + CUDA 12.1-compatible PyTorch)
#   3. Installs all LaMa inference dependencies
#   4. Builds range_libc from source (needed for MapEx raycasting)
#   5. Installs pyastar2d and numba
#
# After this script, manually download the pretrained LaMa weights
# (see "Model download" section at the bottom of this script's output).

module load eth_proxy

set -euo pipefail

MAPEX_DIR=/cluster/project/cvg/students/shangwu/MapEx
ENV_DIR=/cluster/project/cvg/students/shangwu/mapex_env

# ---------------------------------------------------------------------------
# 1. Clone MapEx with submodules
# ---------------------------------------------------------------------------
echo "================================================================"
echo " Step 1: Clone MapEx"
echo "================================================================"

if [[ -d "${MAPEX_DIR}/.git" ]]; then
    echo "MapEx already cloned at ${MAPEX_DIR} — updating submodules"
    cd "${MAPEX_DIR}"
    git submodule update --init --recursive
else
    git clone --recurse-submodules https://github.com/castacks/MapEx.git "${MAPEX_DIR}"
    cd "${MAPEX_DIR}"
fi

echo "MapEx at: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

# ---------------------------------------------------------------------------
# 2. Create conda environment
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Step 2: Create conda environment at ${ENV_DIR}"
echo "================================================================"

source ~/miniconda3/etc/profile.d/conda.sh

if [[ -d "${ENV_DIR}" ]]; then
    echo "Environment already exists — skipping creation"
else
    conda create -y -p "${ENV_DIR}" python=3.10
fi

conda activate "${ENV_DIR}"

# ---------------------------------------------------------------------------
# 3. Install PyTorch (CUDA 12.1 wheels are forward-compatible with CUDA 12.8)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Step 3: Install PyTorch 2.1 + CUDA 12.1"
echo "================================================================"

pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 4. Install LaMa inference dependencies
#
# Version notes vs the original lama conda_env.yml (Python 3.6 / CUDA 10.2):
#   hydra-core 1.1.2  — keeps lama's conf/ format (1.x Structured Configs)
#   omegaconf  2.1.2  — required by hydra-core 1.1.x (range: >=2.1.1, <2.2)
#   pytorch-lightning 1.9.5  — last 1.x release; checkpoint dict compatible
#                              with weights trained on 1.2.9
#   kornia 0.6.12     — last 0.6.x; API stable vs 0.5.0, works with torch 2.x
#   albumentations 1.3.1  — modern Python-compatible; original 0.5.2 API is
#                           only used in LaMa training augments, not inference
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Step 4: Install LaMa inference dependencies"
echo "================================================================"

pip install \
    "hydra-core==1.1.2" \
    "omegaconf==2.1.2" \
    "pytorch-lightning==1.9.5" \
    "kornia==0.6.12" \
    "albumentations==1.3.1" \
    "imageio==2.31.6" \
    "scikit-image==0.22.0" \
    "scikit-learn==1.3.2" \
    "scipy==1.11.4" \
    "pandas==2.1.4" \
    "matplotlib==3.8.2" \
    "tqdm" \
    "pillow>=10.0.0" \
    "pyyaml" \
    "tabulate" \
    "braceexpand" \
    "opencv-python-headless==4.9.0.80" \
    "webdataset" \
    "easydict"

# lama has no setup.py — add it to sys.path via a .pth file so
# 'import saicinpainting' resolves in this env
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
echo "${MAPEX_DIR}/lama" > "${SITE_PKG}/mapex_lama.pth"
echo "Added lama to sys.path via ${SITE_PKG}/mapex_lama.pth"

# ---------------------------------------------------------------------------
# 5. MapEx-specific: range_libc, pyastar2d, numba
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Step 5: Build range_libc + install pyastar2d / numba"
echo "================================================================"

# range_libc needs Cython at build time
pip install "cython==3.0.10"

cd "${MAPEX_DIR}/range_libc/pywrapper"
python setup.py install

pip install pyastar2d
conda install -y -c conda-forge numba

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Setup complete!"
echo " Activate with:  conda activate ${ENV_DIR}"
echo "================================================================"
echo ""
echo " *** Model download (manual step) ***"
echo " Download the pretrained LaMa weights from the Google Drive link"
echo " in the MapEx README (https://github.com/castacks/MapEx) and"
echo " extract them so the directory looks like:"
echo ""
echo "   ${MAPEX_DIR}/pretrained_models/weights/big_lama/models/best.ckpt"
echo "   ${MAPEX_DIR}/pretrained_models/weights/lama_ensemble/train_1-3/models/best.ckpt"
echo ""
echo " Then symlink or copy into the expected location:"
echo "   ln -s ${MAPEX_DIR}/pretrained_models/weights  ${MAPEX_DIR}/weights"
echo "================================================================"
