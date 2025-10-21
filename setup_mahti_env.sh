# setup_mahti_env.sh
#!/bin/bash

echo "🧹 Cleaning old environment..."
deactivate 2>/dev/null
rm -rf .venv

echo "📦 Loading Python module..."
module purge
module load python-data/3.10-24.04

echo "✓ Python version:"
python3 --version

echo "🏗️  Creating virtual environment..."
python3 -m venv .venv

echo "🔌 Activating environment..."
source .venv/bin/activate

echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "📥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "📥 Installing required packages..."
pip install timm==0.9.12
pip install albumentations
pip install opencv-python-headless
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install pillow
pip install kaggle

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate in future sessions, run:"
echo "  module load python-data/3.10-24.04"
echo "  source .venv/bin/activate"
echo ""
echo "Now test with:"
echo "  python src/model.py"
