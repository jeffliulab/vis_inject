#!/bin/bash
#SBATCH -J visinject
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=00-12:00:00
#SBATCH --output=/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/visinject/logs/slurm_%j.out
#SBATCH --error=/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/visinject/logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pliu07@tufts.edu

set -euo pipefail

############################
# Configuration
############################
ENV_DIR="/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject"
PYTHON="${ENV_DIR}/bin/python"
WORK_DIR="/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/visinject"
LOG_DIR="${WORK_DIR}/logs"

HF_HOME="/cluster/tufts/c26sp1ee0141/pliu07/model_cache"

# Select mode: full | inject | eval | compare
MODE="${1:-full}"

# Default clean image (override with $2)
CLEAN_IMAGE="${2:-../demos/demo_images/ORIGIN_dog.png}"

############################
# Modules
############################
module load class || true
module load ee110/2025fall-1 || true

############################
# Environment
############################
mkdir -p "${LOG_DIR}" "${WORK_DIR}/outputs/universal" "${WORK_DIR}/outputs/adversarial" "${WORK_DIR}/outputs/results" "${WORK_DIR}/outputs/checkpoints"
export HF_HOME="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "===== VisInject Pipeline ====="
echo "JobID : ${SLURM_JOB_ID}"
echo "Node  : ${SLURM_NODELIST}"
echo "Mode  : ${MODE}"
echo "Image : ${CLEAN_IMAGE}"
echo "=============================="
${PYTHON} -V
nvidia-smi || true

cd "${WORK_DIR}"

############################
# Run
############################
case "${MODE}" in
    full)
        echo "[MODE] Full pipeline: UniversalAttack + AnyAttack fusion + evaluation"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} pipeline.py \
                --clean-images "${CLEAN_IMAGE}" \
                --num-steps 3000 \
                --evaluate
        ;;
    inject)
        echo "[MODE] AnyAttack fusion only (requires existing universal image)"
        UNIVERSAL_IMG=$(ls -t outputs/universal/universal_*.png 2>/dev/null | head -1)
        if [ -z "${UNIVERSAL_IMG}" ]; then
            echo "[ERROR] No universal image found. Run 'full' mode first."
            exit 1
        fi
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} generate.py \
                --universal-image "${UNIVERSAL_IMG}" \
                --clean-images "${CLEAN_IMAGE}"
        ;;
    eval)
        echo "[MODE] Evaluation only"
        ADV_IMG=$(ls -t outputs/adversarial/adv_*.png 2>/dev/null | head -1)
        UNIVERSAL_IMG=$(ls -t outputs/universal/universal_*.png 2>/dev/null | head -1)
        if [ -z "${ADV_IMG}" ]; then
            echo "[ERROR] No adversarial image found. Run 'full' or 'inject' mode first."
            exit 1
        fi
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} evaluate.py \
                --adv-images "${ADV_IMG}" \
                --clean-images "${CLEAN_IMAGE}" \
                --universal-image "${UNIVERSAL_IMG}"
        ;;
    compare)
        echo "[MODE] Decoder comparison (LAION400M vs LAIONArt)"
        ADV_IMG=$(ls -t outputs/adversarial/adv_*.png 2>/dev/null | head -1)
        UNIVERSAL_IMG=$(ls -t outputs/universal/universal_*.png 2>/dev/null | head -1)
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} evaluate.py \
                --adv-images "${ADV_IMG}" \
                --clean-images "${CLEAN_IMAGE}" \
                --universal-image "${UNIVERSAL_IMG}" \
                --compare-decoders
        ;;
    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        echo "Usage: sbatch hpc_pipeline.sh [full|inject|eval|compare] [clean_image_path]"
        exit 1
        ;;
esac

echo "[DONE] Mode=${MODE} finished at $(date)"
