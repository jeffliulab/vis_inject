#!/bin/bash
#SBATCH -J laion_art_download
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/logs/download_%j.out
#SBATCH --error=/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/logs/download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pliu07@tufts.edu

set -euo pipefail

############################
# Configuration
############################
ENV_DIR="/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject"
PYTHON="${ENV_DIR}/bin/python"

DATA_DIR="/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART"
LOG_DIR="${DATA_DIR}/logs"

# SLURM copies the .sh to a temp spool dir on the compute node,
# so BASH_SOURCE points there instead of the original location.
# $SLURM_SUBMIT_DIR is the directory where sbatch was invoked.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DOWNLOADER="${SCRIPT_DIR}/download_images.py"

# Select mode: test | full | resume
MODE="${1:-full}"

# HuggingFace token: auto-detect from huggingface-cli login cache,
# or override with HF_TOKEN env var before running sbatch.
# Example: export HF_TOKEN=hf_xxxxx && sbatch download_laion_art.sh test
HF_TOKEN_ARGS=""
if [ -n "${HF_TOKEN:-}" ]; then
    HF_TOKEN_ARGS="--hf-token ${HF_TOKEN}"
    echo "[INFO] HF_TOKEN set via environment variable"
elif [ -f "${HOME}/.cache/huggingface/token" ]; then
    echo "[INFO] HF token found at ~/.cache/huggingface/token"
else
    echo "[WARN] No HuggingFace token detected."
    echo "       If LAION-Art requires auth, run on login node first:"
    echo "         pip install huggingface_hub && huggingface-cli login"
fi

############################
# Modules
############################
module load class || true
module load ee110/2025fall-1 || true

############################
# Setup
############################
mkdir -p "${LOG_DIR}"

echo "===== LAION-Art Download Job ====="
echo "JobID  : ${SLURM_JOB_ID:-local}"
echo "Node   : ${SLURM_NODELIST:-$(hostname)}"
echo "Mode   : ${MODE}"
echo "Script : ${DOWNLOADER}"
echo "=================================="

if [ ! -x "${PYTHON}" ]; then
    echo "[FATAL] Python not found at: ${PYTHON}"
    exit 2
fi
${PYTHON} -V

# Check required Python packages (all standard or commonly available)
echo "[INFO] Checking dependencies..."
${PYTHON} -c "import pyarrow; print(f'  pyarrow {pyarrow.__version__}')" || {
    echo "[WARN] pyarrow not found. Installing..."
    ${ENV_DIR}/bin/pip install pyarrow --quiet
}
${PYTHON} -c "from PIL import Image; import PIL; print(f'  Pillow {PIL.__version__}')" || {
    echo "[WARN] Pillow not found. Installing..."
    ${ENV_DIR}/bin/pip install Pillow --quiet
}
echo "[OK] Dependencies ready."

############################
# Run
############################
case "${MODE}" in
    test)
        echo ""
        echo "[MODE] Test run: downloading 100 images to verify setup"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-16} \
            ${PYTHON} "${DOWNLOADER}" \
                --test-run \
                --test-count 100 \
                --workers 8 \
                --output-dir "${DATA_DIR}/webdataset_test" \
                ${HF_TOKEN_ARGS}
        echo ""
        echo "[INFO] Test images saved to: ${DATA_DIR}/webdataset_test"
        echo "[INFO] If this worked, run: sbatch download_laion_art.sh full"
        ;;
    full)
        echo ""
        echo "[MODE] Full download: ~8M images"
        echo "[INFO] This is resumable. Resubmit if it times out:"
        echo "       sbatch download_laion_art.sh resume"
        echo ""
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-16} \
            ${PYTHON} "${DOWNLOADER}" \
                --workers 32 \
                --output-dir "${DATA_DIR}/webdataset" \
                ${HF_TOKEN_ARGS}
        ;;
    resume)
        echo ""
        echo "[MODE] Resuming interrupted download"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-16} \
            ${PYTHON} "${DOWNLOADER}" \
                --resume \
                --workers 32 \
                --output-dir "${DATA_DIR}/webdataset" \
                ${HF_TOKEN_ARGS}
        ;;
    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        echo ""
        echo "Usage:"
        echo "  sbatch download_laion_art.sh test     # Test with 100 images first"
        echo "  sbatch download_laion_art.sh full     # Full download (~8M images)"
        echo "  sbatch download_laion_art.sh resume   # Resume interrupted download"
        exit 1
        ;;
esac

echo ""
echo "[DONE] Download job finished at $(date)"
