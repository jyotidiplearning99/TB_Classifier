#!/bin/bash
#SBATCH --job-name=tb_classifier
#SBATCH --account=project_2010751
#SBATCH --partition=gpusmall          # ✅ CHANGED from 'gpu' to 'gpusmall'
#SBATCH --gres=gpu:a100:1             # ✅ CHANGED from 'v100' to 'a100'
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # ✅ CHANGED from 10 to 8
#SBATCH --mem=64G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --signal=TERM@300

set -Eeuo pipefail
set -x

echo "============================================================"
echo "TB CLASSIFIER TRAINING JOB"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Start time:   $(date)"
echo "============================================================"

#-------------------------------
# Change to project directory
#-------------------------------
cd /scratch/project_2010751/TB_Classifier || exit 1

#-------------------------------
# Load modules
#-------------------------------
module --force purge || true
module load python-data/3.10-24.04

#-------------------------------
# Setup directories & caches
#-------------------------------
mkdir -p logs outputs .cache

export PROJECT_DIR="/scratch/project_2010751/TB_Classifier"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$PROJECT_DIR/.cache/huggingface/hub"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export TMPDIR="$PROJECT_DIR/.cache/tmp"

# Create all cache directories
for dir in "$HF_HOME" "$TORCH_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"; do
    mkdir -p "$dir"
done

#-------------------------------
# Environment variables
#-------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=1337
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#-------------------------------
# Activate virtual environment
#-------------------------------
source .venv/bin/activate

#-------------------------------
# Print diagnostics
#-------------------------------
echo ""
echo "Environment Information:"
echo "  Python version:   $(python --version)"
echo "  PyTorch version:  $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available:   $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU device:       $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "  Cache directory:  $HF_HOME"
echo "  Working dir:      $(pwd)"
echo ""

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo ""
echo "============================================================"
echo "STARTING TRAINING"
echo "============================================================"
echo ""

#-------------------------------
# Graceful shutdown handler
#-------------------------------
trap 'echo "[$(date)] Caught SIGTERM, saving checkpoint..."; sleep 10' TERM

#-------------------------------
# Run training
#-------------------------------
srun -u python -u src/train.py

RC=$?

echo ""
echo "============================================================"
echo "Job finished at: $(date)"
echo "Exit code: $RC"
echo "============================================================"

exit $RC
