# Phase 4: Transfer Learning with DenseNet-121

Transfer-learning study that replaces the Phase 3 from-scratch CNN with an ImageNet-pretrained DenseNet-121, run as a controlled ablation — frozen feature extraction, full fine-tuning, and weighted-loss fine-tuning — to measure how much pretraining helps, whether it narrows the validation-to-test generalization gap, and whether class-weighting can recover usable operating points from the imbalance.

---

## Table of Contents

- [Overview](#overview)
- [Phase Questions](#phase-questions)
- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Ideas for Extension](#ideas-for-extension)

---

## Overview

Phase 4 introduces transfer learning and measures its effect **as a controlled ablation** against the Phase 3 baseline. The same data splits, the same evaluation code, and the same macro-AUROC metric are reused so that the only thing that changes between "before" and "after" is the model itself — which makes any improvement cleanly attributable to transfer learning.

Three DenseNet-121 configurations are trained with the **same custom PyTorch training loop** used in Phase 3 (no high-level framework):

1. **Frozen** — the pretrained convolutional backbone is frozen and only a new 14-output head is trained (feature extraction).
2. **Fine-tuned** — the entire network is unfrozen and trained end-to-end with mixed precision (AMP).
3. **Weighted fine-tuned** — continues from the fine-tuned checkpoint with a class-weighted loss (`sqrt`-scaled `pos_weight`) to address the operating-point failure caused by class imbalance.

Every model is evaluated per disease on validation and once on the official test split, and stacked against the from-scratch baseline in a single comparison table.

---

## Phase Questions

- Does an ImageNet-pretrained DenseNet-121 beat the from-scratch baseline?
- How much comes from frozen features alone versus full fine-tuning?
- Does transfer learning **narrow the validation-to-test generalization gap** found in Phase 3?
- Which of the 14 diseases benefit most from pretraining?
- Can **class-weighted loss recover usable recall/F1** from the near-zero operating point, and at what cost to ranking (AUROC)?

---

## Dataset Description

Phase 4 reuses the exact patient-safe splits from Phase 1 — unchanged, on purpose, so the comparison against Phase 3 is apples-to-apples.

| File | Rows | Role |
|---|---|---|
| `train_data.csv` | 76,277 | Model fitting |
| `val_data.csv` | 10,247 | Model selection (early stopping) |
| `test_data.csv` | 25,596 | Final held-out evaluation (used once per model) |

The 14 disease targets and the multi-label (independent-sigmoid) formulation are identical to Phase 3. The DenseNet input contract — `[B, 3, 224, 224]`, ImageNet-normalized — is also identical, so no data-pipeline changes were required.

---

## Model Architecture

**DenseNet-121**, pretrained on ImageNet, with its 1000-class classifier replaced by a 14-logit head:

```text
Input                          [B, 3, 224, 224]
DenseNet-121 feature extractor (dense blocks + transitions, ImageNet-pretrained)
                               -> [B, 1024, 7, 7] -> global pool -> [B, 1024]
classifier = Linear(1024 -> 14)                 -> [B, 14] logits
```

Notes:

- **`model.classifier` is swapped** to `Linear(in_features, 14)`; `in_features` is read off the layer (1024), not hardcoded.
- **No sigmoid in the model** — raw logits, with `BCEWithLogitsLoss` applying the sigmoid internally (same convention as Phase 3).
- DenseNet-121 is the **CheXNet** architecture — the established baseline on this dataset — so results are comparable to published work.

The three training configurations share this architecture:

| Config | Trainable params | Loss | Optimizer |
|---|---|---|---|
| Frozen | ~14K (head only) | `BCEWithLogitsLoss` | `Adam(lr=1e-3)` |
| Fine-tuned | ~7M (all) | `BCEWithLogitsLoss` | `Adam(lr=1e-4)` + AMP |
| Weighted fine-tuned | ~7M (all) | `BCEWithLogitsLoss(pos_weight=sqrt(...))` | `Adam(lr=1e-4)` + AMP |

The learning rate tracks how much the starting weights are trusted: a fresh head can take large steps (`1e-3`); good pretrained features need gentle updates (`1e-4`).

---

## Methodology

### 1. Reuse the Phase 2/3 data pipeline

The same `ChestImageDataset` and loaders (`batch_size=64`, `num_workers=4`) are reused. Transforms are split into a `train_transform` and an `eval_transform`; for these controlled runs they are identical (no augmentation) so the only variable is the model.

### 2. Load the pretrained backbone

`densenet121(weights=IMAGENET1K_V1)` (requires internet enabled on Kaggle), then the classifier is swapped to 14 outputs.

### 3. Frozen run

The backbone (`model.features`) is frozen (`requires_grad = False`); only the new head trains. The optimizer receives just the trainable parameters. Checkpointed to `densenet_frozen.pt`.

### 4. Fine-tuned run

The whole network is trainable. Training uses **mixed precision** via `torch.amp.autocast()` + `GradScaler` inside the same loop, checkpointed to `densenet_finetune.pt`.

### 5. Weighted fine-tuned run

Continues from `densenet_finetune.pt` with a class-weighted loss. The raw imbalance ratios (up to 662:1 for Hernia) are **square-root-scaled** — `pos_weight = sqrt(neg/pos)` — so rare classes get emphasis (~26x for Hernia) without the instability and precision collapse that raw weights would cause. Checkpointed to `densenet_finetune_weighted.pt`.

### 6. Early stopping with `min_delta`

To stop noise-level "improvements" from resetting patience (a lesson from Phase 3), the early-stop check requires improvement of at least `min_delta = 0.005` in validation macro-AUROC, with patience 5.

### 7. Evaluation and comparison

Each model's best checkpoint is evaluated per class on validation and once on the test split (via a shared `per_class_table` helper, rounded to 4 dp). Macro results are stacked in `phase4_comparison.csv`, and the unweighted-vs-weighted models are compared per disease to isolate the effect of class-weighting.

---

## Key Findings

### 1. Transfer learning wins decisively

The fine-tuned DenseNet-121 reaches **0.8214 validation** and **0.7939 test** macro-AUROC — versus the from-scratch baseline's 0.694 / 0.604. That is a **+0.19 lift on the test set**.

### 2. A clean, monotonic ablation

Each controlled step improves both splits:

| Step | Val | Test |
|---|---|---|
| From-scratch CNN | 0.694 | 0.604 |
| DenseNet frozen | 0.739 | 0.697 |
| DenseNet fine-tuned | 0.821 | 0.794 |
| DenseNet fine-tuned + weighted | 0.822 | 0.797 |

Because only the model changed between rows, the gains are cleanly attributable to transfer learning.

### 3. The generalization gap shrinks (the standout result)

The validation-to-test gap **narrows monotonically**:

- From-scratch CNN: **0.090**
- DenseNet frozen: **0.042**
- DenseNet fine-tuned: **0.028**
- DenseNet fine-tuned + weighted: **0.025**

Since AUROC is prevalence-invariant, this is not a base-rate effect — it is evidence that **ImageNet-pretrained features generalize more robustly under the NIH split's distribution shift**. Notably, full fine-tuning did **not** re-widen the gap; it narrowed it further (a 3.6x reduction versus the baseline).

### 4. Frozen features alone already beat the baseline

The frozen model surpassed the baseline's best validation score at **epoch 0**, in a third of the training time — confirming that generic ImageNet features carry real signal for chest X-rays before any adaptation.

### 5. Weighted loss recovers usable operating points at ~zero AUROC cost

The frozen and fine-tuned models both showed ~0 precision/recall/F1 at the 0.5 threshold — a calibration artifact of the imbalance, not a ranking failure. Weighted BCE (sqrt-scaled `pos_weight`) fixes this on the test set:

| Metric (test, macro) | Fine-tuned | Weighted | Δ |
|---|---|---|---|
| AUROC | 0.794 | 0.797 | +0.003 (flat) |
| Recall @0.5 | 0.09 | 0.35 | **+0.26** |
| F1 @0.5 | 0.13 | 0.31 | **+0.18** |

Weighting shifted the **operating point** (probabilities cross 0.5) without changing the **ranking** (AUROC held). The rarest classes gained most — Hernia F1 went **0.00 → 0.37**. The gain is bounded by AUROC, however: weakly-ranked classes such as Pneumonia (AUROC 0.71) barely moved, because an operating-point fix can only exploit an existing ranking.

### 6. A limit worth naming

Recall rose more than F1, i.e. **precision fell** — the expected recall/precision tradeoff of class-weighting. This confirms weighted loss and per-class threshold tuning (Phase 5) are two complementary levers on the *same* problem; threshold tuning achieves the same recovery without retraining and with per-class control.

---

## Project Structure

```text
Chest-XRay-Project/
|
|-- models/
|   |-- baseline_model.pt                   # Phase 3 from-scratch CNN
|   |-- densenet_frozen.pt                  # DenseNet-121, frozen backbone + linear head
|   |-- densenet_finetune.pt                # DenseNet-121, fully fine-tuned
|   |-- densenet_finetune_weighted.pt       # DenseNet-121, weighted fine-tune (deployment model)
|
|-- phase-1/
|-- phase-2/
|-- phase-3/
|
|-- phase-4/
|   |-- chest-xray-detection-phase4.ipynb   # Transfer-learning training + evaluation
|   |-- phase-4-README.md                     # This file
|
|-- results/
|   |-- train_data.csv
|   |-- val_data.csv
|   |-- test_data.csv
|   |-- phase4_comparison.csv                 # Baseline vs frozen vs fine-tuned vs weighted
|   |-- per_class_val_densenet_frozen.csv
|   |-- per_class_test_densenet_frozen.csv
|   |-- per_class_val_densenet_finetune.csv
|   |-- per_class_test_densenet_finetune.csv
|   |-- per_class_val_densenet_weighted.csv
|   |-- per_class_test_densenet_weighted.csv
```

---

## Requirements

```text
Python 3.8+
pandas
numpy
torch
torchvision
scikit-learn
Pillow
kagglehub
jupyter
```

Install dependencies:

```bash
pip install pandas numpy torch torchvision scikit-learn pillow kagglehub jupyter
```

- **Internet must be enabled** to download the pretrained DenseNet-121 weights (or attach a weights dataset and load offline).
- A CUDA GPU is strongly recommended; the fine-tune runs use mixed precision (AMP).

---

## How to Run

1. Attach the NIH Chest X-ray image dataset and the Phase 1 CSV dataset as Kaggle inputs.
2. Enable **GPU** and **Internet** in the notebook settings.
3. Open `phase-4/chest-xray-detection-phase4.ipynb`.
4. Run the data cells (reused pipeline, two transforms).
5. Run the **frozen**, then **fine-tune**, then **weighted fine-tune** configurations.
6. Evaluate each best checkpoint on validation and once on test; export per-class tables.
7. Assemble `phase4_comparison.csv` and the weighting comparison; save the checkpoints to `models/`.

The training loops are gated as "run once" — after the checkpoints exist, run only the setup, model-definition, and loader cells, then load the `.pt` files and run the evaluation cells to reproduce all tables without retraining.

---

## Results Summary

### Headline comparison

| Model | Val AUROC | Test AUROC | Val→Test Gap | Best epoch | Train time |
|---|---|---|---|---|---|
| From-scratch CNN (Phase 3) | 0.6940 | 0.6043 | 0.0897 | 11 | ~247 min |
| DenseNet-121 frozen | 0.7389 | 0.6973 | 0.0416 | 1 | ~93 min |
| DenseNet-121 fine-tuned | 0.8214 | 0.7939 | 0.0275 | 1 | ~102 min |
| **DenseNet-121 fine-tuned + weighted** | **0.8221** | **0.7970** | **0.0251** | 1 | — |

### Operating-point recovery (weighted vs. fine-tuned, test macro)

| Metric @0.5 | Fine-tuned | Weighted |
|---|---|---|
| Recall | 0.09 | 0.35 |
| F1 | 0.13 | 0.31 |

### Interpretation

- **+0.190 test macro-AUROC** over the from-scratch baseline; the generalization gap reduced **3.6x** (0.090 → 0.025).
- **Weighted BCE recovered recall/F1 from ~0 to usable levels at flat AUROC**, confirming the near-zero F1 was an operating-point artifact of imbalance, not a modeling failure.
- Per-class tables (saved in `results/`) show the largest AUROC gains on subtle findings, and the largest F1 gains from weighting on the rarest classes.
- The weighted fine-tuned model is the project's deployment model for Phases 5-7.

---

## Ideas for Extension

### 1. Per-class threshold tuning and calibration (Phase 5)

Weighted loss recovers F1 during training; per-class threshold selection on validation achieves the same recovery **without retraining** and with per-class control, plus calibration curves and sensitivity@specificity operating points.

### 2. Data augmentation

Both DenseNet runs peaked at epoch 1 and overfit afterward. Training-only augmentation (small rotations, mild brightness/contrast, and — with a laterality caveat — horizontal flip) may delay overfitting and squeeze out additional AUROC. `eval_transform` stays clean.

### 3. Architecture depth ablation

Compare DenseNet-121 against DenseNet-169/201 to test whether more depth helps enough to justify the extra compute (expected to be marginal).

### 4. Interpretability (Phase 6)

Grad-CAM overlays on the fine-tuned model to confirm it attends to lung anatomy rather than dataset shortcuts.

### 5. Deployment (Phase 7)

Package the weighted fine-tuned model with tuned thresholds and Grad-CAM into a demo app with a model card.

---

*Phase 4 establishes transfer learning as a large, cleanly-attributable improvement: a fine-tuned DenseNet-121 lifts test macro-AUROC from 0.60 to 0.79 (+0.19) over the from-scratch baseline while shrinking the validation-to-test generalization gap from 0.090 to 0.025. A follow-up weighted-loss stage then recovers usable operating points (test macro recall 0.09 → 0.35, F1 0.13 → 0.31) at essentially zero AUROC cost, showing the near-zero F1 was a calibration artifact of class imbalance. The weighted fine-tuned checkpoint becomes the deployment model for the remaining phases.*
