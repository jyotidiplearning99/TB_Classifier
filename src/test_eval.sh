#!/bin/bash
#SBATCH --job-name=tb_test_eval
#SBATCH --account=project_2010751
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/test_eval_%j.out
#SBATCH --error=logs/test_eval_%j.err

set -Eeuo pipefail
set -x

echo "============================================================"
echo "TB CLASSIFIER - TEST SET EVALUATION"
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
# Setup cache directories
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
# Environment variables
#-------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

#-------------------------------
# Activate environment
#-------------------------------
source .venv/bin/activate

#-------------------------------
# Print diagnostics
#-------------------------------
echo ""
echo "Environment:"
echo "  Python:   $(python --version)"
echo "  PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU:      $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo ""
echo "============================================================"
echo "RUNNING TEST EVALUATION"
echo "============================================================"
echo ""

#-------------------------------
# Run evaluation
#-------------------------------
srun -u python -u src/test_evaluation.py

RC=$?

echo ""
echo "============================================================"
echo "Evaluation finished at: $(date)"
echo "Exit code: $RC"
echo "============================================================"

# Show results location
if [ $RC -eq 0 ]; then
    echo ""
    echo "Results saved to:"
    ls -lh outputs/test_evaluation/
    echo ""
    echo "View predictions:"
    echo "  head outputs/test_evaluation/test_predictions.csv"
fi

exit $RC
