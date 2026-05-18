#!/bin/bash
#SBATCH -A nvr_elm_llm
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -J movie-batch
#SBATCH --array=1-10%1
#SBATCH --output=slurm_out/movie_batch_%A_%a.out
#SBATCH -p batch_block1
#SBATCH --gpus-per-node 8
#SBATCH --exclusive

set -euo pipefail

# Slurm wrapper for the Hugging Face movie batch pipeline.
# Each array task runs one resumable batch pass. If a task is stopped by the
# 4-hour wall time, the next array task continues from the saved state and
# existing outputs.

export LOGLEVEL=INFO

# REQUIRED: conda environment bin path.
export PATH="/home/yuchaog/workplace/miniconda3/envs/data_process/bin:${PATH}"

# REQUIRED: TransNetV2 project root.
PROJECT_ROOT="/lustre/fs11/portfolios/nvr/projects/nvr_elm_llm/users/yuchaog/workplace/code/chenlan/data_process/movie/TransNetV2"

# OPTIONAL: set before sbatch if the dataset requires auth:
#   export HF_TOKEN="..."
export HF_TOKEN="${HF_TOKEN:-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}"

cd "${PROJECT_ROOT}"
mkdir -p slurm_out

echo "[$(date '+%F %T')] job=${SLURM_JOB_ID:-unknown} array=${SLURM_ARRAY_TASK_ID:-0}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "python=$(which python)"

cmd=$(cat <<'EOF'
set -euo pipefail
cd "${PROJECT_ROOT}"
INTERVAL_SECONDS=7200 bash multishot_pipeline/run_batch_hf_multishot_pipeline_every_2h.sh
EOF
)

export PROJECT_ROOT

srun bash -c "${cmd}"
