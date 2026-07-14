# Phase 5: Per-Class Threshold Tuning

Threshold-tuning workflow for the Phase 4 weighted DenseNet-121 model. This phase does not train a new network. It loads the saved weighted checkpoint, selects one validation-optimized decision threshold per disease label, evaluates those thresholds on validation and test data, and exports the threshold-tuned metric tables.

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

Phase 5 addresses the operating-point problem left by Phase 4. The weighted DenseNet-121 already ranks images well, but a single global `0.5` threshold is not ideal for every disease label in a highly imbalanced multi-label task.

The notebook uses the validation split to choose a separate threshold for each of the 14 disease labels by maximizing F1 score from the precision-recall curve. Those fixed thresholds are then applied to both validation and test predictions. This changes precision, recall, and F1 without changing AUROC, because the raw probability scores and ranking remain the same.

The phase produces three file artifacts:

```text
results/per_class_val_threshold_tuned.csv
results/per_class_test_threshold_tuned.csv
results/phase5_comparison.csv
```

`phase5_comparison.csv` is the saved `cmp_tuning_test` table. The validation comparison table, `cmp_tuning_val`, is useful for inspection inside the notebook but is intentionally not exported as a separate file.

---

## Phase Questions

- Can per-class thresholds improve recall and F1 without retraining the DenseNet model?
- Does AUROC stay unchanged after threshold tuning, confirming that only binary decisions changed?
- Which disease labels benefit most from validation-selected thresholds?
- Which labels trade away precision or F1 after threshold tuning?
- What artifacts should later deployment phases use: tuned per-class metrics, the test comparison table, and the threshold vector stored on the model during notebook execution?

---

## Dataset Description

Phase 5 reuses the same patient-safe split files created in Phase 1 and consumed throughout Phases 2-4.

| File | Rows | Role |
|---|---:|---|
| `train_data.csv` | 76,277 | Loaded for data-pipeline consistency and sanity checks |
| `val_data.csv` | 10,247 | Threshold selection |
| `test_data.csv` | 25,596 | Final held-out threshold-tuned evaluation |

The 14 disease targets are unchanged:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening,
Pneumonia, Pneumothorax
```

Phase 5 also loads Phase 4 weighted-model metric tables as the prior baseline:

```text
results/per_class_val_densenet_weighted.csv
results/per_class_test_densenet_weighted.csv
```

---

## Model Architecture

The model is the Phase 4 deployment checkpoint: DenseNet-121 with the ImageNet-pretrained backbone fine-tuned using square-root-scaled class weights.

```text
Input X-ray image                  [B, 3, 224, 224]
DenseNet-121 feature extractor     -> [B, 1024]
classifier = Linear(1024 -> 14)    -> [B, 14] logits
sigmoid(logits)                    -> [B, 14] probabilities
per-class threshold vector         -> [B, 14] binary predictions
```

The notebook rebuilds the architecture with `weights=None`, replaces the classifier head, and loads `densenet_finetune_weighted.pt`. This avoids downloading ImageNet weights during Phase 5 because the saved checkpoint already contains the trained parameters.

No new model checkpoint is exported in this phase. The tuned thresholds are registered on the in-memory model as `column_thresholds`.

---

## Methodology

### 1. Reuse the earlier data pipeline

The notebook uses the same `ChestImageDataset`, target label list, image resizing, tensor conversion, and ImageNet normalization conventions from the previous modeling phases.

### 2. Build train, validation, and test loaders

All three split datasets are constructed and sanity checked. Validation is used for threshold selection; test is reserved for final evaluation.

### 3. Load the saved weighted DenseNet checkpoint

The Phase 4 weighted checkpoint is loaded directly, and the model is switched to evaluation mode. No training loop, optimizer, or scheduler is used in Phase 5.

### 4. Generate validation and test probabilities

The `evaluate` helper runs inference and collects sigmoid probabilities plus true labels for validation and test data.

### 5. Select one threshold per disease

For each label, the notebook computes precision, recall, and threshold candidates from the validation predictions. The selected threshold is the one that maximizes validation F1.

### 6. Re-evaluate with tuned thresholds

The tuned threshold vector is applied label by label. The notebook then creates validation and test per-class tables with threshold, AUROC, precision, recall, F1, and support.

### 7. Compare prior and tuned performance

The prior weighted-model metric tables are joined against the tuned tables. The exported test comparison reports prior values, tuned values, and deltas for AUROC, precision, recall, and F1.

---

## Key Findings

### 1. Threshold tuning improves the test operating point without retraining

On the test set, macro recall improves from `0.3454` to `0.3882`, and macro F1 improves from `0.3061` to `0.3169`. Macro precision changes only slightly, from `0.3108` to `0.3068`.

The gain is deliberately modest, and that is itself the finding: because tuning is applied *on top of* the Phase 4 weighted model — whose loss already shifted probabilities toward a usable 0.5 operating point — there is little left for thresholds to recover. **Weighted loss and threshold tuning fix the same imbalance-induced problem, so they are alternatives, not additive levers.** Applied to the unweighted fine-tuned model, threshold tuning would show a much larger F1 recovery.

### 2. AUROC stays fixed, as expected

Test macro-AUROC remains `0.7970` before and after threshold tuning. This is the expected sanity check: threshold tuning changes binary predictions, not the underlying probability ranking.

### 3. The gains are label-specific

The largest test F1 gains are:

| Disease | F1 delta |
|---|---:|
| Pneumonia | +0.0704 |
| Consolidation | +0.0365 |
| Infiltration | +0.0319 |
| Mass | +0.0227 |
| Atelectasis | +0.0187 |

The largest F1 declines are:

| Disease | F1 delta |
|---|---:|
| Cardiomegaly | -0.0341 |
| Hernia | -0.0108 |
| Edema | -0.0030 |

The `Cardiomegaly` regression is instructive: its validation-selected threshold (0.69) was too high for the test set, where Cardiomegaly is roughly 2x more prevalent. This is the **Phase 1 distribution shift affecting the thresholds themselves** — a cutoff optimal on validation need not be optimal on a differently-distributed test set, so the per-class thresholds carry their own generalization risk.

### 4. Validation comparison is diagnostic, not an exported artifact

`cmp_tuning_val` is created inside the notebook to inspect validation behavior, but only `cmp_tuning_test` is exported as `phase5_comparison.csv`.

---

## Project Structure

```text
Chest-XRay-Project/
|
|-- models/
|   |-- densenet_finetune_weighted.pt          # Phase 4 weighted DenseNet checkpoint
|
|-- phase-1/
|-- phase-2/
|-- phase-3/
|-- phase-4/
|
|-- phase-5/
|   |-- chest-xray-detection-phase5.ipynb      # Threshold tuning notebook
|   |-- phase-5-README.md                      # This file
|
|-- results/
|   |-- train_data.csv
|   |-- val_data.csv
|   |-- test_data.csv
|   |-- per_class_val_densenet_weighted.csv    # Prior Phase 4 validation metrics
|   |-- per_class_test_densenet_weighted.csv   # Prior Phase 4 test metrics
|   |-- per_class_val_threshold_tuned.csv      # Phase 5 validation metrics
|   |-- per_class_test_threshold_tuned.csv     # Phase 5 test metrics
|   |-- phase5_comparison.csv                  # Saved cmp_tuning_test table
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

A GPU is helpful for faster inference, but Phase 5 does not train and can run on CPU. The DenseNet architecture is rebuilt without downloading pretrained weights, so the weighted checkpoint must be attached or available locally.

---

## How to Run

1. Attach the NIH Chest X-ray image dataset, the Phase 1 split CSV files, the Phase 4 weighted checkpoint, and the Phase 4 weighted per-class metric tables.
2. Open `phase-5/chest-xray-detection-phase5.ipynb`.
3. Run the setup, dataset, transform, and loader cells.
4. Load `densenet_finetune_weighted.pt`.
5. Generate validation and test probability scores.
6. Select validation-optimized thresholds for all 14 disease labels.
7. Evaluate validation and test data using the tuned threshold vector.
8. Export the following files:

```text
per_class_val_threshold_tuned.csv
per_class_test_threshold_tuned.csv
phase5_comparison.csv
```

Do not export `cmp_tuning_val`; it is intentionally kept as an in-notebook diagnostic table only.

---

## Results Summary

### Macro metrics

| Split / Table | AUROC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Validation threshold-tuned | 0.8221 | 0.2921 | 0.3497 | 0.3010 |
| Test weighted prior | 0.7970 | 0.3108 | 0.3454 | 0.3061 |
| Test threshold-tuned | 0.7970 | 0.3068 | 0.3882 | 0.3169 |

### Test change from prior weighted model

| Metric | Prior | Tuned | Change |
|---|---:|---:|---:|
| AUROC | 0.7970 | 0.7970 | +0.0000 |
| Precision | 0.3108 | 0.3068 | -0.0040 |
| Recall | 0.3454 | 0.3882 | +0.0428 |
| F1 | 0.3061 | 0.3169 | +0.0108 |

### Interpretation

- Threshold tuning acts on the decision rule, not the learned representation.
- The main benefit is higher recall and modestly higher F1 on the held-out test set.
- AUROC remains constant, confirming the model's ranking ability did not change.
- Per-label thresholds are a deployment-ready complement to the Phase 4 weighted checkpoint.

---

## Ideas for Extension

### 1. Save the threshold vector for deployment

Export the 14 tuned thresholds to JSON or YAML alongside the model card so Phase 7 can load the exact operating point without rerunning validation tuning.

### 2. Add calibration curves

Reliability diagrams and expected calibration error would show whether the weighted DenseNet probabilities are well calibrated or only useful for ranking.

### 3. Tune for clinical constraints

Instead of maximizing F1, choose thresholds for sensitivity at a fixed specificity, or optimize different metrics for high-risk labels.

### 4. Bootstrap uncertainty intervals

Add confidence intervals for per-class F1, recall, precision, and AUROC, especially for rare labels such as Hernia and Pneumonia.

### 5. Validate on an external dataset

The tuned thresholds are selected on NIH validation data. A later phase should test whether they transfer to another chest X-ray dataset or need site-specific recalibration.

---

*Phase 5 turns the Phase 4 weighted DenseNet into a more usable decision system by replacing one global threshold with 14 validation-selected thresholds. The model is not retrained, AUROC stays fixed at 0.7970 on test, and the test operating point improves from 0.3061 to 0.3169 macro F1 while recall rises from 0.3454 to 0.3882.*
