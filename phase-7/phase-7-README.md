# Phase 7: Deployment — Streamlit App on Streamlit Community Cloud

The capstone phase. It does not train a model. It wraps the Phase 4 weighted DenseNet-121, the Phase 5 per-class thresholds, and the Phase 6 Grad-CAM into a single interactive web app and ships it as a public, always-on demo on Streamlit Community Cloud.

> **Research demonstration only — NOT a medical device.** Trained on the public NIH ChestX-ray14 dataset for a portfolio project. It must **not** be used for diagnosis, screening, or any clinical decision.

---

## Table of Contents

- [Overview](#overview)
- [Phase Questions](#phase-questions)
- [What Gets Served](#what-gets-served)
- [Application Architecture](#application-architecture)
- [Methodology](#methodology)
- [Key Decisions & Findings](#key-decisions--findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Deploy](#how-to-deploy)
- [Run Locally](#run-locally)
- [Results Summary](#results-summary)
- [Ideas for Extension](#ideas-for-extension)

---

## Overview

Phases 1–6 produced a model and understood it. Phase 7 makes it *usable*: a visitor uploads a frontal chest X-ray (or picks a bundled sample) and gets, in one screen —

1. **Per-finding probabilities** for all 14 thoracic findings (independent sigmoids, multi-label).
2. **Threshold flags** — each finding is flagged only when its probability crosses its **own validation-tuned threshold** (Phase 5), not a naive global `0.5`.
3. **A Grad-CAM overlay** (Phase 6) showing where the model looked for the highest-probability finding.

The app loads exactly the deployed checkpoint from earlier phases:

```text
models/densenet_finetune_weighted.pt   # fine-tuned + class-weighted DenseNet-121
phase-7/thresholds.json                # 14 per-class thresholds tuned on validation
```

The deployment target is **Streamlit Community Cloud** — free, permanent, and it redeploys automatically on every `git push`. No servers to manage, no credits to burn.

---

## Phase Questions

- How do you turn three separate research artifacts (a checkpoint, a threshold table, an interpretability method) into one coherent user-facing tool?
- What does an *honest* medical-ML demo surface — a single label, or the calibrated probabilities, thresholds, and the disclaimer around them?
- How do you keep inference cheap enough to run on a free CPU tier (no GPU)?
- What breaks when you move from a local notebook to a managed cloud build, and how do you make the environment reproducible?

---

## What Gets Served

| Artifact | Source phase | Role in the app |
|---|---|---|
| `densenet_finetune_weighted.pt` | Phase 4 | The model — fine-tuned DenseNet-121 continued with class-weighted `BCEWithLogitsLoss` |
| `thresholds.json` | Phase 5 | Per-class decision thresholds; drives the 🔴 flags, not a global 0.5 |
| Grad-CAM (`features.norm5` target) | Phase 6 | Heatmap overlay for the top predicted finding |
| `MODEL_CARD.md` | Phase 7 | Metrics, limitations, and the shortcut audit summary |

---

## Application Architecture

`phase-7/app.py`, top to bottom:

```text
imports
st.set_page_config(...)            # MUST be the first Streamlit call
_find([...candidate paths...])     # locate model + thresholds locally OR on the cloud
@st.cache_resource load_model()    # DenseNet-121, head -> Linear(1024, 14), load_state_dict, eval
@st.cache_resource load_cam(model) # GradCAM(target_layers=[model.features.norm5])
@st.cache_data     load_thresholds()
UI: disclaimer -> uploader/sample -> predict() -> 2-col (input | Grad-CAM) -> per-finding table
```

Two design details that make it robust:

- **`set_page_config` is the very first Streamlit command.** The cached loaders trigger a spinner (a Streamlit call) at import time; if `set_page_config` runs after them, Streamlit throws `set_page_config() can only be called once`. It sits immediately after the imports.
- **`_find()` path resolution.** The model and thresholds are located by trying a list of candidate paths (env var → repo-relative → next-to-`app.py`). The same `app.py` therefore works when run locally from the repo root *and* when Streamlit Cloud sets the working directory to the repo root — no code change between environments.

The model is loaded once per session (`@st.cache_resource`), so only the first request pays the ~1–2 s load; later predictions are fast even on CPU.

---

## Methodology

### 1. Assemble the artifacts
Copy the deployed checkpoint, the tuned thresholds, and the Grad-CAM configuration into a single Streamlit app. Nothing is retrained; the app is a thin, honest wrapper over Phases 4–6.

### 2. Make inference CPU-only
The free tier has no GPU. `requirements.txt` installs **CPU-only PyTorch** (`torch==2.2.2+cpu`) via the PyTorch CPU wheel index. This keeps the install small and matches how the app runs in production — `DEVICE` auto-selects `cpu`.

### 3. Pin a reproducible environment
- **Python 3.12** (set in Streamlit Cloud → *Advanced settings*). The pinned stack (`torch 2.2.2`, `numpy 1.26.4`, `pandas 2.2.2`, …) has no wheels for Python 3.13/3.14, which is the platform default — so the version is pinned explicitly.
- **`packages.txt` → `libgl1`.** OpenCV (pulled in by `grad-cam`) needs `libGL.so.1`, which isn't in the base image. Only `libgl1` is added; the earlier `libglib2.0-0` was already present in the base image and forced a broken cross-release apt resolution, so it was removed.

### 4. Deploy from GitHub
Point Streamlit Cloud at the repo, branch `main`, main file `phase-7/app.py`. It clones the repo (the 28 MB checkpoint is committed), installs apt + pip deps, and launches. Every subsequent `git push` triggers an automatic redeploy.

---

## Key Decisions & Findings

### 1. The demo shows the operating point, not a verdict
A naive medical demo prints "positive / negative." This one surfaces the **probability, the tuned threshold, and the flag together**, plus a red disclaimer banner. That is the honest representation of what the model is — a ranker with a chosen operating point (Phase 5), not a diagnosis. Flags use the per-class thresholds, so a finding like Pneumonia (low threshold, weak class) and Cardiomegaly (high threshold) are judged on their own terms.

### 2. Free-tier deployment is an environment-engineering problem, not an ML one
The model was never the obstacle — the build was. Three concrete lessons, each now encoded in the repo:
- **Python version drift:** the cloud defaulted to 3.14, which had no wheels for the pinned stack → pin 3.12.
- **apt dependency conflicts:** over-specifying system libs (`libglib2.0-0`) triggered a bullseye/trixie conflict → install only what's actually missing (`libgl1`).
- **CPU vs CUDA torch:** an unqualified `torch==2.2.2` can pull the ~2 GB CUDA build → pin `+cpu` explicitly.

### 3. Committing the checkpoint makes deploys trivial
The 28 MB weighted checkpoint lives in the git repo, so Streamlit Cloud gets it on clone — no external model hosting, no LFS, no download step at boot. For a model this size that is the simplest reliable option.

### 4. Streamlit Cloud over Docker/EC2 for the always-on link
An EC2 + Docker path was scoped and then dropped: it burns credits, needs a running instance, and the public IP changes on restart. For a portfolio demo that must simply *stay up*, Streamlit Community Cloud is the better fit — zero cost, zero maintenance, auto-redeploy on push.

---

## Project Structure

```text
Chest-XRay-Project/
|
|-- models/
|   |-- densenet_finetune_weighted.pt   # Deployed checkpoint (committed, ~28 MB)
|
|-- phase-7/
|   |-- app.py                          # Streamlit application
|   |-- thresholds.json                 # 14 per-class thresholds (Phase 5)
|   |-- samples/                        # Optional bundled sample X-rays
|   |-- phase-7-README.md               # This file
|
|-- requirements.txt                    # Python deps (Streamlit Cloud reads this at repo root)
|-- packages.txt                        # apt deps: libgl1 (for OpenCV)
|-- MODEL_CARD.md                       # Model card: metrics, limitations, shortcut audit
```

---

## Requirements

```text
Python 3.12          # required — pinned stack has no 3.13/3.14 wheels
torch==2.2.2+cpu
torchvision==0.17.2+cpu
streamlit==1.36.0
grad-cam==1.5.0
pillow==10.3.0
numpy==1.26.4
pandas==2.2.2
opencv-python-headless==4.9.0.80
```

System (Linux, via `packages.txt`): `libgl1`.

---

## How to Deploy

Streamlit Community Cloud, from this GitHub repo:

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub.**
3. Set:
   - **Repository:** `aarshdesai-ds/chest-xray-classifier`
   - **Branch:** `main`
   - **Main file path:** `phase-7/app.py`
4. Open **Advanced settings → Python version → `3.12`**. *(Required — this is the single most important setting.)*
5. **Deploy.** The build installs `libgl1` (apt), then the pinned Python deps, then launches.

Redeploys happen automatically on every `git push` to `main`.

---

## Run Locally

```bash
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell (use source .venv/bin/activate on macOS/Linux)
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run phase-7/app.py
```

The app opens at `http://localhost:8501`. Locally, `_find()` resolves the checkpoint at `models/densenet_finetune_weighted.pt` relative to the repo root.

---

## Results Summary

The app serves the deployed model, whose held-out performance is:

| Model | Val macro-AUROC | Test macro-AUROC | Val→Test gap |
|---|---:|---:|---:|
| From-scratch CNN (baseline) | 0.694 | 0.604 | 0.090 |
| DenseNet-121 frozen | 0.739 | 0.697 | 0.042 |
| DenseNet-121 fine-tuned | 0.821 | 0.794 | 0.028 |
| **Fine-tuned + weighted (deployed)** | **0.822** | **0.797** | **0.025** |

At the tuned thresholds (test, macro): recall ≈ 0.39, F1 ≈ 0.32. Per-class AUROC spans ~0.71 (Pneumonia) to ~0.90 (Emphysema). **0.797 is the honest test estimate** — the NIH test split is distribution-shifted (~2× prevalence), and real-world performance on other sites is unknown and likely lower. Full details in [`MODEL_CARD.md`](../MODEL_CARD.md).

---

## Ideas for Extension

### 1. Show all flagged findings' Grad-CAMs
Currently only the top finding gets an overlay. Render a small Grad-CAM per flagged finding.

### 2. Add example galleries and a confidence legend
Bundle a few public NIH samples with known labels so a first-time visitor can see the flow without their own image.

### 3. Report memory headroom
The free tier caps ~1 GB RAM. Add a lightweight check (or free the Grad-CAM object between runs) if the app approaches the ceiling under load.

### 4. Serve a REST endpoint alongside the UI
Expose a `/predict` API (FastAPI) for programmatic use, with the Streamlit UI as the human front-end.

### 5. Add input validation
Reject non-radiograph images (e.g. a quick "is this plausibly a chest X-ray?" gate) so the disclaimer is backed by a guardrail.

---

*Phase 7 closes the loop: the model that Phases 1–6 built, tuned, and audited is now a public, reproducible, honestly-framed web app. The engineering lessons here — pinning the interpreter, installing only the system libs you need, and forcing CPU wheels — are the difference between "runs on my machine" and "runs on a clean cloud build."*
