#!/bin/bash
#SBATCH --job-name=tb_prod
#SBATCH --account=project_2010751
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_prod_%j.out
#SBATCH --error=logs/train_prod_%j.err
#SBATCH --signal=TERM@300

set -Eeuo pipefail

echo "============================================================"
echo "TB CLASSIFIER - PRODUCTION TRAINING"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
echo "============================================================"

cd /scratch/project_2010751/TB_Classifier

module --force purge || true
module load python-data/3.10-24.04

export PROJECT_DIR="/scratch/project_2010751/TB_Classifier"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME" logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .venv/bin/activate

echo "Environment:"
python -V
python - <<'PY'
import torch, os
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

trap 'echo "[SIGTERM] Saving & exiting..."; sleep 10' TERM

# Kick off training
srun -u python -u src/train_production.py --config config_production.yaml

RC=$?
echo "Job finished $(date)  Exit code: $RC"
exit $RC
