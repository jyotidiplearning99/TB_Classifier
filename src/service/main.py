import os, io, json, time, gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Configuration ----------------------------------------------------
MODEL_DIR = Path("outputs/tb_production")
MODEL_PATH = MODEL_DIR / "best_model.pth"
META_PATH = MODEL_DIR / "model_meta.json"

from src.model import MedicalTBClassifier
from src.dataset_fixed import get_valid_transforms

# ---- App --------------------------------------------------------------
app = FastAPI(title="TB Classifier (Research Demo)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- Device & Memory Management ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ CRITICAL: Limit threads on CPU to prevent memory explosion
if device.type == "cpu":
    torch.set_num_threads(2)  # Reduce from default (usually 8-16)
    torch.set_num_interop_threads(1)

print(f"🔧 Device: {device}")
print(f"🔧 PyTorch threads: {torch.get_num_threads()}")

# ---- Load Model ONCE --------------------------------------------------
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

print(f"📥 Loading model from {MODEL_PATH}...")
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model = MedicalTBClassifier(
    model_name=ckpt["config"]["model_name"],
    pretrained=False,
    dropout=ckpt["config"].get("dropout", 0.3),
).to(device).eval()

model.load_state_dict(ckpt["model_state_dict"], strict=True)

# ✅ Only use channels_last on CUDA (can cause issues on CPU)
if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

# ✅ Set inference mode (reduces memory)
torch.set_grad_enabled(False)

print(f"✅ Model loaded successfully")

# ---- Clear checkpoint from memory -------------------------------------
del ckpt
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---- Metadata ---------------------------------------------------------
meta = json.load(open(META_PATH)) if META_PATH.exists() else {}

# ---- Threshold --------------------------------------------------------
SAFE_DEFAULT_THRESHOLD = float(os.getenv("GLOBAL_THRESHOLD", "0.5"))

# ---- Preprocessing ----------------------------------------------------
tfm = get_valid_transforms(512)

def preprocess(image_bytes: bytes) -> torch.Tensor:
    """Robust image loader with memory-efficient processing."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)  # ✅ Read as grayscale directly
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # ✅ Resize BEFORE processing to save memory
    if img.shape[0] > 1024 or img.shape[1] > 1024:
        scale = 1024 / max(img.shape)
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to 8-bit
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Stack to 3 channels
    img = np.stack([img, img, img], axis=-1)
    
    # Apply transforms
    tens = tfm(image=img)["image"].unsqueeze(0)
    
    return tens

# ---- Response Schema --------------------------------------------------
class PredictResponse(BaseModel):
    probability: float
    label: str
    threshold_used: float
    domain: str
    model_version: Optional[str] = None
    model_checksum: Optional[str] = None
    latency_ms: int

# ---- Routes -----------------------------------------------------------
@app.get("/")
def root():
    return {"service": "tb-classifier", "status": "ok"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "model_epoch": meta.get("model", {}).get("epoch", "unknown"),
        "default_threshold": SAFE_DEFAULT_THRESHOLD,
        "torch_threads": torch.get_num_threads(),
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    domain: str = Form(default="default")
):
    start = time.time()
    
    try:
        raw = await file.read()
        
        # Size check
        if len(raw) > 30 * 1024 * 1024:
            raise HTTPException(413, detail="File too large (>30MB)")
        
        # ✅ Preprocess with error handling
        try:
            tens = preprocess(raw)
        except Exception as e:
            raise HTTPException(400, detail=f"Image preprocessing failed: {str(e)}")
        
        # ✅ Move to device with non-blocking
        tens = tens.to(device, non_blocking=False)  # blocking=False can cause issues on CPU
        
        if torch.cuda.is_available():
            tens = tens.to(memory_format=torch.channels_last)
        
        # ✅ Inference with explicit no_grad (redundant but safe)
        with torch.no_grad():
            logit = model(tens)
            prob = torch.sigmoid(logit).float().item()
        
        # ✅ CRITICAL: Clean up tensors immediately
        del tens, logit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Threshold logic
        thr = SAFE_DEFAULT_THRESHOLD
        label = "TB" if prob >= thr else "Normal"
        
        latency = int((time.time() - start) * 1000)
        
        return PredictResponse(
            probability=round(prob, 6),
            label=label,
            threshold_used=thr,
            domain=domain,
            model_version=str(meta.get("version")),
            model_checksum=(meta.get("model", {}) or {}).get("checksum"),
            latency_ms=latency,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # ✅ Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise HTTPException(500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_cam")
async def predict_cam_disabled():
    raise HTTPException(503, detail="Grad-CAM disabled in this demo build.")

# ✅ Cleanup on shutdown
@app.on_event("shutdown")
def shutdown_event():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
