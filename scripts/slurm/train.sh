#!/bin/bash

#SBATCH --job-name=wm-toy
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH -t 47:59:00
#SBATCH --output=logs/wm-toy-train_%j.out
#SBATCH --error=logs/wm-toy-train_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/slurm/constants.sh"

WORK=/$DATA_PARTITION/$USER

export SINGULARITYENV_TFDS_DATA_DIR=$WORK/tfds
export SINGULARITYENV_XDG_CACHE_HOME=$WORK/.cache
export SINGULARITYENV_HF_HOME=$WORK/.cache/huggingface
export SINGULARITYENV_TORCH_HOME=$WORK/.cache/torch

export SINGULARITYENV_SSL_CERT_FILE=/home/ms16518/ws/WM-playground/cacert.pem

export SINGULARITYENV_EXEC_PATH="${SLURM_SUBMIT_DIR}/scripts/run_train.sh"

singularity exec --nv \
  --bind "$WORK:$WORK" \
  --overlay "$SINGULARITY_OVERLAY:ro" \
  "$SINGULARITY_IMG" bash -lc '
    set -euo pipefail
    source /ext3/env.sh
    bash "${EXEC_PATH}"
  '