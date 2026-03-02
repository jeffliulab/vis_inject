#!/bin/bash
#SBATCH -J anyattack_s2
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02-00:00:00
#SBATCH --output=/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_S2_AnyAttack/logs/slurm_%j.out
#SBATCH --error=/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_S2_AnyAttack/logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pliu07@tufts.edu

set -euo pipefail

############################
# Configuration
############################
ENV_DIR="/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject"
PYTHON="${ENV_DIR}/bin/python"
DEMO_DIR="/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_S2_AnyAttack"
LOG_DIR="${DEMO_DIR}/logs"

HF_HOME="/cluster/tufts/c26sp1ee0141/pliu07/model_cache"
LAION_DIR="/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset"

# Select mode: pretrain | finetune | generate | evaluate
MODE="${1:-pretrain}"

############################
# Modules
############################
module load class || true
module load ee110/2025fall-1 || true

############################
# Environment
############################
mkdir -p "${LOG_DIR}" "${DEMO_DIR}/checkpoints"
export HF_HOME="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "===== AnyAttack S2 ====="
echo "JobID : ${SLURM_JOB_ID}"
echo "Node  : ${SLURM_NODELIST}"
echo "Mode  : ${MODE}"
echo "========================"
${PYTHON} -V
nvidia-smi || true

cd "${DEMO_DIR}"

############################
# Run
# max-shards: maximum dataset support, delete this row if dataset is complete 
############################
case "${MODE}" in
    pretrain)
        echo "[MODE] Self-supervised pre-training on LAION-Art"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} pretrain.py \
                --tar-dir "${LAION_DIR}" \
                --batch-size 600 \
                --epochs 5 \
                --max-shards 120
        ;;
    finetune)
        echo "[MODE] Fine-tuning on COCO"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} finetune.py \
                --pretrain-checkpoint checkpoints/pre-trained.pt \
                --use-auxiliary \
                --epochs 20
        ;;
    generate)
        echo "[MODE] Generate adversarial images"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} generate_adv.py \
                --decoder-path checkpoints/finetuned.pt
        ;;
    evaluate)
        echo "[MODE] Evaluate against VLMs"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} evaluate.py \
                --target-vlms qwen2_5_vl_3b blip2_opt_2_7b
        ;;
    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        echo "Usage: sbatch hpc_train.sh [pretrain|finetune|generate|evaluate]"
        exit 1
        ;;
esac

echo "[DONE] Mode=${MODE} finished at $(date)"
