#!/bin/bash
#SBATCH --job-name=tb_independent
#SBATCH --account=project_2010751
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/independent_%j.out

cd /scratch/project_2010751/TB_Classifier
source .venv/bin/activate

# Set cache
export HF_HOME="/scratch/project_2010751/TB_Classifier/.cache/huggingface"
export TORCH_HOME="/scratch/project_2010751/TB_Classifier/.cache/torch"

python src/evaluate_dadb_v2.py
