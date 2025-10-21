# activate_env.sh
#!/bin/bash

# Load module
module purge
module load python-data/3.10-24.04

# Activate virtual environment
source .venv/bin/activate

# Set cache directories to scratch
export PROJECT_DIR="/scratch/project_2010376/JDs_Project/TB_Classifier"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/huggingface/datasets"

# Create directories if they don't exist
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

echo "✓ Environment activated"
echo "  Python: $(python --version)"
echo "  Cache location: $HF_HOME"
