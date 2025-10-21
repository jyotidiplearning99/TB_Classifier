# src/service/frontend_app.py
import os, io, time, json
from typing import Dict, Any, List, Optional
from pathlib import Path

import requests
from PIL import Image
import streamlit as st
import pandas as pd

PREDICT_URL = os.getenv("PREDICT_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="TB Classifier", page_icon="🫁", layout="centered")
st.title("🫁 TB Classifier — Upload & Predict")
st.caption(f"Backend: `{PREDICT_URL}`")

with st.sidebar:
    st.header("About")
    st.write("- Research demo")
    st.write("- Global threshold is set on the backend")
    st.divider()
    st.write("Upload one or more chest X-rays (JPG/PNG).")

def call_backend(img_bytes: bytes, filename: str, domain: Optional[str] = "default") -> Dict[str, Any]:
    files = {"file": (filename, img_bytes, "application/octet-stream")}
    data = {"domain": domain or "default"}
    start = time.time()
    r = requests.post(PREDICT_URL, files=files, data=data, timeout=120)
    r.raise_for_status()
    resp = r.json()
    resp.setdefault("client_latency_ms", int((time.time() - start) * 1000))
    return resp

def show_result(img: Image.Image, resp: Dict[str, Any]):
    c1, c2 = st.columns([2, 3])
    with c1:
        st.image(img, caption="Uploaded image", use_container_width=True)
    with c2:
        st.subheader("Prediction")
        st.metric("Probability (TB)", f"{resp.get('probability', 0):.4f}")
        st.write(f"**Label:** {resp.get('label','?')}")
        st.write(f"**Domain (echo):** {resp.get('domain','default')}")
        if "threshold_used" in resp:
            st.write(f"**Threshold used:** {resp['threshold_used']:.4f}")
        if resp.get("model_version"):
            st.caption(f"Model v{resp['model_version']}")
        if resp.get("model_checksum"):
            st.code(resp["model_checksum"], language="text")
        lat = resp.get("latency_ms") or resp.get("client_latency_ms")
        if lat:
            st.caption(f"Latency: {lat} ms")

files = st.file_uploader(
    "Upload chest X-ray(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if files:
    rows: List[Dict[str, Any]] = []
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            resp = call_backend(buf.getvalue(), f.name)
            with st.expander(f"📄 {f.name}", expanded=len(files) == 1):
                show_result(img, resp)
            rows.append({
                "filename": f.name,
                "probability": resp.get("probability"),
                "label": resp.get("label"),
                "threshold_used": resp.get("threshold_used"),
                "latency_ms": resp.get("latency_ms") or resp.get("client_latency_ms"),
            })
        except requests.HTTPError as e:
            st.error(f"{f.name}: HTTP {e.response.status_code} - {e.response.text}")
        except Exception as e:
            st.error(f"{f.name}: {e}")

    if len(rows) > 1:
        st.subheader("Batch results")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="tb_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("Drop one or more images above to get predictions.")
