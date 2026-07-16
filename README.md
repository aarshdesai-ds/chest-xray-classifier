# Chest X-Ray Multi-Label Classifier (NIH ChestX-ray14)

An end-to-end deep-learning project that detects **14 thoracic findings** from a single frontal chest X-ray — built in seven phases, from raw-data preprocessing through a from-scratch baseline, transfer learning, operating-point tuning, an interpretability audit, and a deployed public web app.

> **Research demonstration only — NOT a medical device.** Trained on the public NIH ChestX-ray14 dataset for a portfolio project. It must **not** be used for diagnosis, screening, or any clinical decision. See [`MODEL_CARD.md`](MODEL_CARD.md).

**🔗 Live demo:** `https://<your-app>.streamlit.app`  ·  **Model card:** [`MODEL_CARD.md`](MODEL_CARD.md)

<!-- Replace <your-app> above with the deployed Streamlit Community Cloud URL. -->

---

## TL;DR

- **Task:** multi-label classification of 14 findings (independent per-finding sigmoids — findings co-occur).
- **Headline result:** transfer learning lifted held-out **test macro-AUROC from 0.604 → 0.797** and shrank the validation→test generalization gap from **0.090 → 0.025**.
- **Honest metric:** macro-AUROC, not accuracy — only ~4.5% of label slots are positive, so an all-negative predictor scores ~95% accuracy while being useless.
- **Beyond the score:** per-class thresholds tuned to the operating point (Phase 5), a Grad-CAM shortcut audit that found the model attends to real anatomy (Phase 6), and a deployed Streamlit app (Phase 7).

---

## What this project demonstrates

This is deliberately a *depth* project, not a leaderboard chase. It shows the full lifecycle of a medical-imaging model built and reasoned about honestly:

- **Custom PyTorch training loops** (no `.fit()`, no Lightning) — full control over the loop, AMP, schedulers, and checkpointing.
- **Correct handling of class imbalance** — square-root-scaled `pos_weight` in `BCEWithLogitsLoss`, and macro-AUROC as the headline metric.
- **Leakage-safe evaluation** — patient-level splits (no patient in two splits) and honest reporting of the distribution-shifted NIH test set.
- **Operating-point engineering** — per-class threshold tuning, and the finding that it does *not* stack with weighted loss.
- **Interpretability & shortcut auditing** — Grad-CAM plus a quantitative border-activation check.
- **Reproducible deployment** — a CPU-only Streamlit app on a clean cloud build.

---

## The seven phases

| Phase | Title | What it delivers |
|---|---|---|
| **1** | [Dataset Preprocessing](phase-1/phase-1-README.md) | Patient-level train/val/test splits (76,277 / 10,247 / 25,596), prevalence analysis |
| **2** | [PyTorch Dataset Loading](phase-2/phase-2-README.md) | `Dataset`/`DataLoader` pipeline, ImageNet normalization, 224×224 |
| **3** | [Baseline CNN Training](phase-3/phase-3-README.md) | From-scratch CNN, custom training loop — **test macro-AUROC 0.604** |
| **4** | [Transfer Learning (DenseNet-121)](phase-4/phase-4-README.md) | Frozen → fine-tuned → class-weighted ablation — **0.797**, gap 0.090→0.025 |
| **5** | [Threshold Tuning](phase-5/phase-5-README.md) | Per-class thresholds (max-F1 on val); threshold tuning ≈ weighted loss, and they don't stack |
| **6** | [Grad-CAM & Shortcut Audit](phase-6/phase-6-README.md) | No border shortcut; attention tracks anatomy (Cardiomegaly central, Emphysema peripheral) |
| **7** | [Deployment](phase-7/phase-7-README.md) | Streamlit app on Streamlit Community Cloud, serving the model + thresholds + Grad-CAM |

---

## Key results

Judged on **macro-AUROC** (prevalence-invariant):

| Model | Val macro-AUROC | Test macro-AUROC | Val→Test gap |
|---|---:|---:|---:|
| From-scratch CNN (baseline) | 0.694 | 0.604 | 0.090 |
| DenseNet-121 frozen | 0.739 | 0.697 | 0.042 |
| DenseNet-121 fine-tuned | 0.821 | 0.794 | 0.028 |
| **Fine-tuned + weighted (deployed)** | **0.822** | **0.797** | **0.025** |

At the tuned thresholds (test, macro): recall ≈ 0.39, F1 ≈ 0.32. Per-class AUROC spans ~0.71 (Pneumonia, weakest) to ~0.90 (Emphysema, strongest).

**Why 0.797 is the number to trust:** the official NIH test split is distribution-shifted (~2× the disease prevalence of train/val), so the validation score (0.822) is optimistic. 0.797 is the honest held-out estimate; performance on other sites/scanners is unknown and likely lower.

Three findings worth more than the headline number:

1. **Transfer learning was the single biggest lever** — +0.19 test AUROC over the from-scratch baseline, *and* it more than halved the generalization gap.
2. **Threshold tuning and weighted loss are substitutes, not complements** — each raised macro-F1 from ~0.13 to ~0.30, but combining them didn't stack. They fix the same problem (the decision boundary), from two directions.
3. **No shortcut detected** — every finding's Grad-CAM activation sat below the uniform-attention baseline, and the per-class ordering tracked anatomy, evidence the model learned chest structure rather than framing artifacts.

---

## The 14 findings

```text
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis,
Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax
```

---

## Repository structure

```text
Chest-XRay-Project/
|
|-- phase-1/ ... phase-6/     # One notebook + findings-focused README per phase
|-- phase-7/                  # Deployment
|   |-- app.py                # Streamlit application
|   |-- thresholds.json       # 14 per-class thresholds (Phase 5)
|   |-- samples/              # Optional bundled sample X-rays
|   |-- phase-7-README.md
|
|-- models/
|   |-- densenet_finetune_weighted.pt   # Deployed checkpoint (committed)
|   |-- densenet_finetune.pt / densenet_frozen.pt / baseline_model.pt
|
|-- results/                  # Split CSVs, per-class metric tables, Grad-CAM gallery
|-- requirements.txt          # Deployment/dev dependencies (Python 3.12)
|-- packages.txt              # apt deps for the cloud build (libgl1)
|-- MODEL_CARD.md             # Metrics, limitations, shortcut audit, intended use
|-- README.md                 # This file
```

---

## Run the app

**Live:** the Streamlit Community Cloud demo (link at the top).

**Locally:**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell; source .venv/bin/activate on macOS/Linux
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run phase-7/app.py        # opens http://localhost:8501
```
Requires **Python 3.12** (the pinned stack has no wheels for 3.13/3.14). Deployment details in [`phase-7/phase-7-README.md`](phase-7/phase-7-README.md).

**Reproduce the modeling** (Phases 1–6): each phase's notebook and README stand alone. The pipeline expects the NIH ChestX-ray14 images and the Phase 1 split CSVs; a GPU is recommended for training and Grad-CAM.

---

## Tech stack

`PyTorch` · `torchvision` (DenseNet-121 / CheXNet) · `scikit-learn` (AUROC, PR curves) · `pytorch-grad-cam` · `pandas` / `NumPy` · `Streamlit` · `Streamlit Community Cloud`

---

## Data & ethics

- **Dataset:** [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) — 112,120 frontal chest X-rays from 30,805 patients, public and de-identified. Labels are NLP-mined from radiology reports (a known source of label noise).
- **Intended use:** education, portfolio demonstration, and ML-engineering discussion of imbalance, operating points, transfer learning, and interpretability.
- **Out of scope:** any clinical, diagnostic, triage, or screening use; use on populations or equipment unlike NIH ChestX-ray14. Full statement in [`MODEL_CARD.md`](MODEL_CARD.md).
