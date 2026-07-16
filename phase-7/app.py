"""
Phase 7 — Chest X-Ray Multi-Label Classifier: Streamlit deployment app.

Serves the Phase 4 weighted DenseNet-121 with the Phase 5 per-class thresholds and
Phase 6 Grad-CAM. Upload a chest X-ray -> per-disease probabilities, threshold flags,
and a Grad-CAM overlay for the top finding.

Research demonstration only. NOT a medical device.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# MUST be the first Streamlit command — before any cached loader triggers a spinner.
st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🫁", layout="wide")

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _find(candidates):
    """Return the first existing path from a list of candidates."""
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


MODEL_PATH = _find([
    os.environ.get("MODEL_PATH"),
    "models/densenet_finetune_weighted.pt",
    _ROOT / "models" / "densenet_finetune_weighted.pt",
    _HERE / "densenet_finetune_weighted.pt",
])
THRESH_PATH = _find([
    os.environ.get("THRESH_PATH"),
    "thresholds.json",
    _HERE / "thresholds.json",
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# --------------------------------------------------------------------------------------
# Load model / thresholds / Grad-CAM (cached so they load once per session)
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_cam(_model):
    # target layer = last conv-stage BN before global pooling (DenseNet-121)
    return GradCAM(model=_model, target_layers=[_model.features.norm5])


@st.cache_data
def load_thresholds():
    return json.loads(Path(THRESH_PATH).read_text())


model = load_model()
cam = load_cam(model)
thresholds = load_thresholds()


# --------------------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------------------
def denorm(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    rgb = (img_tensor.cpu() * std + mean).permute(1, 2, 0).numpy()
    return np.clip(rgb, 0, 1)


def predict(pil_img):
    x = transform(pil_img.convert("RGB"))
    inp = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(inp))[0].cpu().numpy()
    return x, inp, probs


def gradcam_overlay(x, inp, disease_idx):
    grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(disease_idx)])[0]
    return show_cam_on_image(denorm(x), grayscale_cam, use_rgb=True)


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.title("🫁 Chest X-Ray Multi-Label Classifier")
st.error(
    "**Research demonstration only — NOT a medical device.** "
    "Trained on the public NIH ChestX-ray14 dataset for a portfolio project. "
    "It must **not** be used for diagnosis or any clinical decision."
)

with st.expander("How this works", expanded=False):
    st.markdown(
        "- **Model:** DenseNet-121 fine-tuned on NIH ChestX-ray14 with class-weighted loss "
        "(test macro-AUROC **0.797**).\n"
        "- **Flags:** each finding is flagged when its probability crosses a **per-disease threshold "
        "tuned on validation data** (Phase 5) — not a single global 0.5 cutoff.\n"
        "- **Grad-CAM:** the overlay shows where the model looked for the highest-probability finding.\n"
        "- A prior audit found the model attends to plausible chest anatomy, not image-border artifacts."
    )

# input: uploader + optional bundled samples
uploaded = st.file_uploader("Upload a chest X-ray (PNG / JPG)", type=["png", "jpg", "jpeg"])

sample_dir = _HERE / "samples"
img_src = uploaded
if sample_dir.exists():
    names = sorted(p.name for p in sample_dir.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if names and uploaded is None:
        choice = st.selectbox("…or try a bundled sample image", ["(none)"] + names)
        if choice != "(none)":
            img_src = sample_dir / choice

if img_src is None:
    st.info("Upload a chest X-ray (or pick a sample) to run the model.")
    st.stop()

# run
pil = Image.open(img_src)
x, inp, probs = predict(pil)
top_idx = int(np.argmax(probs))

col_img, col_cam = st.columns(2)
with col_img:
    st.subheader("Input X-ray")
    st.image(pil, use_column_width=True)
with col_cam:
    st.subheader(f"Grad-CAM — {LABELS[top_idx]} (top finding)")
    st.image(
        gradcam_overlay(x, inp, top_idx),
        caption=f"{LABELS[top_idx]}: p = {probs[top_idx]:.2f}",
        use_column_width=True,
    )

# per-finding table
rows = [
    {
        "Finding": lab,
        "Probability": round(float(probs[i]), 3),
        "Threshold": thresholds[lab],
        "Flagged": "🔴" if probs[i] >= thresholds[lab] else "",
    }
    for i, lab in enumerate(LABELS)
]
df = pd.DataFrame(rows).sort_values("Probability", ascending=False).reset_index(drop=True)

st.subheader("Per-finding predictions")
flagged = [r["Finding"] for r in rows if r["Flagged"]]
if flagged:
    st.warning("**Findings above their tuned thresholds:** " + ", ".join(flagged))
else:
    st.success("No findings crossed their tuned thresholds.")
st.dataframe(df, use_container_width=True, hide_index=True)

st.caption(
    "Probabilities are independent per-finding sigmoids (multi-label). Thresholds were selected on the "
    "validation split by maximizing per-class F1 (Phase 5). This is a research artifact, not a diagnostic tool."
)
