#!/bin/bash
#SBATCH --job-name=tb_v2_robust
#SBATCH --account=project_2010751
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_v2_%j.out
#SBATCH --error=logs/train_v2_%j.err
#SBATCH --signal=TERM@300

set -Eeuo pipefail
set -x

echo "============================================================"
echo "TB CLASSIFIER V2 - ROBUST TRAINING WITH DA+DB"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Start time:   $(date)"
echo "============================================================"

#-------------------------------
# Navigate to project
#-------------------------------
cd /scratch/project_2010751/TB_Classifier || exit 1

#-------------------------------
# Load modules
#-------------------------------
module --force purge || true
module load python-data/3.10-24.04

#-------------------------------
# Setup cache
#-------------------------------
export PROJECT_DIR="/scratch/project_2010751/TB_Classifier"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$PROJECT_DIR/.cache/huggingface/hub"
export TMPDIR="$PROJECT_DIR/.cache/tmp"

for dir in "$HF_HOME" "$TORCH_HOME" "$TRANSFORMERS_CACHE" "$TMPDIR"; do
    mkdir -p "$dir"
done

#-------------------------------
# Environment
#-------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=1337
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#-------------------------------
# Activate environment
#-------------------------------
source .venv/bin/activate

#-------------------------------
# Diagnostics
#-------------------------------
echo ""
echo "Environment:"
echo "  Python:   $(python --version)"
echo "  PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU:      $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

#-------------------------------
# Prepare data (if not done)
#-------------------------------
if [ ! -f "metadata/train_v2.csv" ]; then
    echo ""
    echo "Preparing diverse training data..."
    python src/prepare_training_v2.py
fi

echo ""
echo "============================================================"
echo "STARTING TRAINING V2"
echo "============================================================"
echo ""

#-------------------------------
# Graceful shutdown
#-------------------------------
trap 'echo "[$(date)] Caught SIGTERM, saving checkpoint..."; sleep 10' TERM

#-------------------------------
# Run training
#-------------------------------
srun -u python -u src/train_v2.py

RC=$?

echo ""
echo "============================================================"
echo "Job finished at: $(date)"
echo "Exit code: $RC"
echo "============================================================"

# Show results if successful
if [ $RC -eq 0 ]; then
    echo ""
    echo "Training complete! Results in outputs/tb_v2_robust/"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate on test set V2: python src/evaluate_test_v2.py"
    echo "  2. Evaluate on DA+DB held-out: python src/evaluate_dadb_final.py"
fi

exit $RC
