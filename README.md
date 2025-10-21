# TB Classifier - Research Demo

Tuberculosis screening classifier using ConvNeXt architecture.

⚠️ **RESEARCH USE ONLY** - Not for clinical diagnosis

## Features
- High-sensitivity screening (optimized for ~98% NPV)
- FastAPI backend with CLAHE preprocessing
- Streamlit web interface
- Domain-specific thresholds
- Supports 8/16-bit medical images

## Setup

### Installation
\`\`\`bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
\`\`\`

### Download Model
Place trained model in `outputs/tb_production/`:
- `best_model.pth`
- `thresholds.json`
- `model_meta.json`

### Run Service
\`\`\`bash
# Terminal 1: Backend
python src/service/main.py

# Terminal 2: Frontend
streamlit run src/service/frontend_app.py
\`\`\`

## Project Structure
\`\`\`
TB_Classifier/
├── src/
│   ├── model.py              # ConvNeXt architecture
│   ├── dataset_fixed.py      # Data pipeline
│   ├── train.py             # Training script
│   └── service/
│       ├── main.py          # FastAPI backend
│       └── frontend_app.py  # Streamlit UI
├── outputs/                 # Model checkpoints (gitignored)
└── data/                    # Training data (gitignored)
\`\`\`

## Model Performance
- AUC: 0.95+ on validation set
- Sensitivity: ~98% at operating threshold
- Optimized for screening (minimize false negatives)

## License
Research use only - not approved for clinical use
\`\`\`

Add and commit:
```bash
git add README.md
git commit -m "Add comprehensive README"
git push
