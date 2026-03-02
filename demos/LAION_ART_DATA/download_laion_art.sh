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
PIP="${ENV_DIR}/bin/pip"

DATA_DIR="/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART"
PARQUET_DIR="${DATA_DIR}/metadata"
OUTPUT_DIR="${DATA_DIR}/webdataset"
LOG_DIR="${DATA_DIR}/logs"

PARQUET_URL="https://huggingface.co/datasets/laion/laion-art/resolve/main/laion-art.parquet"

IMAGE_SIZE=224
PROCESSES=16
THREADS=64

############################
# Modules
############################
module load class || true
module load ee110/2025fall-1 || true

############################
# Directory setup
############################
mkdir -p "${PARQUET_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

############################
# Environment check
############################
echo "===== LAION-Art Download Job ====="
echo "JobID     : ${SLURM_JOB_ID:-local}"
echo "Node      : ${SLURM_NODELIST:-$(hostname)}"
echo "DATA_DIR  : ${DATA_DIR}"
echo "IMAGE_SIZE: ${IMAGE_SIZE}"
echo "PROCESSES : ${PROCESSES}"
echo "THREADS   : ${THREADS}"
echo "=================================="

if [ ! -x "${PYTHON}" ]; then
    echo "[FATAL] Python not found at: ${PYTHON}"
    exit 2
fi
${PYTHON} -V

############################
# Step 1: Install img2dataset if missing
############################
if ! ${PYTHON} -c "import img2dataset" 2>/dev/null; then
    echo "[INFO] Installing img2dataset..."
    ${PIP} install img2dataset --quiet
fi
echo "[OK] img2dataset version: $(${PYTHON} -c 'import img2dataset; print(img2dataset.__version__)')"

############################
# Step 2: Download parquet metadata
############################
PARQUET_FILE="${PARQUET_DIR}/laion-art.parquet"

if [ -f "${PARQUET_FILE}" ]; then
    FILE_SIZE=$(stat -c%s "${PARQUET_FILE}" 2>/dev/null || stat -f%z "${PARQUET_FILE}" 2>/dev/null || echo "0")
    if [ "${FILE_SIZE}" -gt 100000000 ]; then
        echo "[OK] Parquet metadata already exists ($(numfmt --to=iec ${FILE_SIZE})). Skipping download."
    else
        echo "[WARN] Parquet file too small (${FILE_SIZE} bytes). Re-downloading..."
        rm -f "${PARQUET_FILE}"
    fi
fi

if [ ! -f "${PARQUET_FILE}" ]; then
    echo "[INFO] Downloading LAION-Art parquet metadata..."
    wget -q --show-progress -O "${PARQUET_FILE}" "${PARQUET_URL}"
    echo "[OK] Parquet download complete."
fi

############################
# Step 3: Show dataset stats
############################
echo "[INFO] Dataset statistics:"
${PYTHON} -c "
import pyarrow.parquet as pq
table = pq.read_table('${PARQUET_FILE}')
print(f'  Total rows   : {table.num_rows:,}')
print(f'  Columns      : {table.column_names}')
print(f'  File size    : {table.nbytes / 1e9:.2f} GB (in-memory)')
"

############################
# Step 4: Download images via img2dataset
#
# img2dataset is RESUMABLE by design:
#   - It tracks completed shards in the output directory
#   - Rerunning the same command skips already-downloaded shards
#   - Safe to resubmit this script if the job times out
############################
echo "[INFO] Starting img2dataset download..."
echo "[INFO] This is resumable. Resubmit this script if it times out."
echo "[INFO] Target: ${OUTPUT_DIR}"

${PYTHON} -m img2dataset \
    --url_list "${PARQUET_FILE}" \
    --input_format "parquet" \
    --url_col "URL" \
    --caption_col "TEXT" \
    --output_format "webdataset" \
    --output_folder "${OUTPUT_DIR}" \
    --image_size ${IMAGE_SIZE} \
    --resize_mode "center_crop" \
    --resize_only_if_bigger True \
    --processes_count ${PROCESSES} \
    --thread_count ${THREADS} \
    --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic","LANGUAGE"]' \
    --number_sample_per_shard 10000 \
    --encode_format "jpg" \
    --encode_quality 95 \
    --retries 3 \
    --timeout 30 \
    --disallowed_header_directives '[]'

############################
# Step 5: Post-download summary
############################
echo ""
echo "===== Download Summary ====="
SHARD_COUNT=$(find "${OUTPUT_DIR}" -name "*.tar" 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" 2>/dev/null | cut -f1)
echo "  Shards downloaded : ${SHARD_COUNT}"
echo "  Total size        : ${TOTAL_SIZE}"

${PYTHON} -c "
import glob, tarfile, os

tar_files = sorted(glob.glob('${OUTPUT_DIR}/*.tar'))
if not tar_files:
    print('  No tar files found yet.')
else:
    total_images = 0
    for tf in tar_files:
        try:
            with tarfile.open(tf, 'r') as t:
                jpg_count = sum(1 for m in t.getmembers() if m.name.endswith('.jpg'))
                total_images += jpg_count
        except Exception:
            pass
    print(f'  Total images      : {total_images:,}')
    print(f'  Avg per shard     : {total_images // max(len(tar_files), 1):,}')
"
echo "============================="
echo "[DONE] LAION-Art download job finished at $(date)"
