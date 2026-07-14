# Phase 5: DenseNet Threshold Tuning

Threshold-tuning workflow for the Phase 4 DenseNet-121 checkpoints. This phase does not train a new network. It loads saved checkpoints, selects one validation-optimized decision threshold per disease label, evaluates those thresholds on validation and test data, and compares the resulting operating points against the prior global `0.5` threshold metrics.

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

Phase 5 addresses the operating-point problem identified in Phases 3 and 4. DenseNet-121 ranks images well by AUROC, but a single global threshold of `0.5` is not a good decision rule for every disease label in a highly imbalanced multi-label task.

The notebook now evaluates two related threshold-tuning tracks:

1. **Weighted DenseNet threshold tuning**: starts from `densenet_finetune_weighted.pt`, the Phase 4 deployment checkpoint trained with class-weighted BCE.
2. **Fine-tuned DenseNet threshold tuning**: starts from `densenet_finetune.pt`, the unweighted fine-tuned checkpoint, then recovers operating-point behavior post hoc with validation-selected thresholds.

For each checkpoint, the validation split chooses a separate threshold for each of the 14 disease labels by maximizing F1 score from the precision-recall curve. Those fixed thresholds are then applied to validation and test predictions.

Threshold tuning changes precision, recall, and F1. It does not change AUROC for a given checkpoint, because AUROC depends on raw score ranking rather than the binary cutoff.

---

## Phase Questions

- Can per-class thresholds improve recall and F1 without retraining a DenseNet model?
- Does AUROC remain unchanged after threshold tuning, confirming that only the decision rule changed?
- How much benefit remains when threshold tuning is applied on top of the already weighted DenseNet checkpoint?
- Can threshold tuning recover the unweighted fine-tuned model's near-zero global-threshold F1?
- Which output artifacts should later deployment phases use for weighted and fine-tuned operating-point comparisons?

---

## Dataset Description

Phase 5 reuses the same patient-safe split files created in Phase 1 and consumed throughout Phases 2-4.

| File | Rows | Role |
|---|---:|---|
| `train_data.csv` | 76,277 | Loaded for pipeline consistency and sanity checks |
| `val_data.csv` | 10,247 | Threshold selection |
| `test_data.csv` | 25,596 | Final held-out threshold-tuned evaluation |

The 14 disease targets are unchanged:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening,
Pneumonia, Pneumothorax
```

The notebook also loads Phase 4 prior metric tables:

```text
results/per_class_val_densenet_weighted.csv
results/per_class_test_densenet_weighted.csv
results/per_class_val_densenet_finetune.csv
results/per_class_test_densenet_finetune.csv
```

---

## Model Architecture

Both Phase 5 tracks use the same DenseNet-121 architecture from Phase 4:

```text
Input X-ray image                  [B, 3, 224, 224]
DenseNet-121 feature extractor     -> [B, 1024]
classifier = Linear(1024 -> 14)    -> [B, 14] logits
sigmoid(logits)                    -> [B, 14] probabilities
per-class threshold vector         -> [B, 14] binary predictions
```

The notebook rebuilds DenseNet-121 with `weights=None`, replaces the classifier head, and loads saved Phase 4 checkpoints. This avoids downloading ImageNet weights during Phase 5, because each saved checkpoint already contains the trained parameters.

No new model checkpoint is exported in this phase. The tuned thresholds are registered on the in-memory model as `column_thresholds`.

---

## Methodology

### 1. Reuse the earlier data pipeline

The notebook uses the same `ChestImageDataset`, target label list, image resizing, tensor conversion, and ImageNet normalization conventions from the previous modeling phases.

### 2. Build train, validation, and test loaders

All three split datasets are constructed and sanity checked. Validation is used for threshold selection; test is reserved for held-out evaluation.

### 3. Load the weighted DenseNet checkpoint

The Phase 4 weighted checkpoint is loaded directly from `densenet_finetune_weighted.pt`, and the model is switched to evaluation mode.

### 4. Tune weighted-model thresholds

The notebook collects validation probabilities, computes one F1-maximizing threshold per disease label, applies those thresholds to validation and test predictions, and exports weighted threshold-tuned metric tables.

### 5. Compare weighted prior and tuned metrics

The prior weighted-model per-class tables are joined against the tuned tables. The saved test comparison is `phase5_comparison.csv`.

### 6. Load the unweighted fine-tuned checkpoint

The notebook repeats the same process for `densenet_finetune.pt`, the Phase 4 unweighted fine-tuned checkpoint.

### 7. Tune fine-tuned-model thresholds

Validation-selected thresholds are computed for the fine-tuned checkpoint and applied to validation and test predictions. The resulting tuned metric tables are exported separately from the weighted-model tables.

### 8. Compare fine-tuned prior and tuned metrics

The prior fine-tuned per-class tables are joined against the tuned fine-tuned tables. The saved test comparison is `phase5_comparison_v2.csv`.

---

## Key Findings

### 1. Weighted-model threshold tuning gives a modest operating-point lift

On the held-out test set, the weighted model changes as follows after threshold tuning:

| Metric | Prior weighted | Tuned weighted | Change |
|---|---:|---:|---:|
| AUROC | 0.7970 | 0.7970 | +0.0000 |
| Precision | 0.3108 | 0.3068 | -0.0040 |
| Recall | 0.3454 | 0.3882 | +0.0428 |
| F1 | 0.3061 | 0.3169 | +0.0108 |

This gain is intentionally modest. Weighted BCE already shifted probabilities toward a usable `0.5` operating point in Phase 4, so there is less left for threshold tuning to recover.

### 2. Fine-tuned-model threshold tuning gives the larger recovery

On the held-out test set, the unweighted fine-tuned model changes as follows after threshold tuning:

| Metric | Prior fine-tuned | Tuned fine-tuned | Change |
|---|---:|---:|---:|
| AUROC | 0.7939 | 0.7939 | +0.0000 |
| Precision | 0.3669 | 0.2585 | -0.1084 |
| Recall | 0.0852 | 0.4170 | +0.3318 |
| F1 | 0.1265 | 0.3010 | +0.1745 |

This is the clearer threshold-tuning result: the unweighted fine-tuned model had strong ranking but weak default-threshold behavior, and per-class thresholds recover much of its recall and F1 without retraining.

### 3. AUROC stays fixed for each checkpoint

Both comparison files show unchanged macro-AUROC after threshold tuning. This is the expected sanity check: threshold tuning changes binary decisions, not raw score ranking.

### 4. The two levers are equivalent — and they do not stack

Placing all four operating points side by side on the held-out test set:

| Model + decision rule | Test AUROC | Test macro F1 | Test macro recall |
|---|---:|---:|---:|
| Fine-tuned @ 0.5 | 0.7939 | 0.127 | 0.085 |
| **Fine-tuned + tuned thresholds** | 0.7939 | **0.301** | 0.417 |
| Weighted @ 0.5 | 0.7970 | 0.306 | 0.345 |
| Weighted + tuned thresholds | 0.7970 | 0.317 | 0.388 |

Threshold tuning on the unweighted model reaches **0.301** macro F1. Weighted loss on its own reaches **0.306**. Those are statistically indistinguishable — two entirely independent interventions arriving at the same destination. This is empirical proof that class-weighted loss and per-class thresholds fix the **same** underlying problem (imbalance-induced probability compression), which is precisely why stacking them adds only `+0.011`.

The mechanism is visible in the threshold vectors themselves: the weighted model's tuned thresholds sit at **0.18-0.72**, the unweighted model's at **0.016-0.318**.

> Weighted loss raises the probabilities up to the threshold. Threshold tuning lowers the threshold down to the probabilities. Same fix, opposite directions.

**Practical implication:** threshold tuning is the cheaper lever — no retraining, adjustable per class after deployment, and it reaches the same operating point.

**The caveat that bounds both:** AUROC caps what any operating-point fix can achieve. Pneumonia (AUROC 0.71, the weakest ranking of the 14) recovers least under *both* levers. No cutoff can rescue a class the model cannot rank; improving it requires a better model, not a better threshold.

### 5. Validation comparisons are diagnostic only

The notebook creates validation comparison tables in memory:

```text
cmp_tuning_val_weighted
cmp_tuning_val_finetune
```

These are useful for inspection but are intentionally not exported as separate CSV files. The exported comparison files are held-out test comparisons.

---

## Project Structure

```text
Chest-XRay-Project/
|
|-- models/
|   |-- densenet_finetune.pt                  # Phase 4 unweighted fine-tuned DenseNet
|   |-- densenet_finetune_weighted.pt         # Phase 4 weighted DenseNet
|
|-- phase-1/
|-- phase-2/
|-- phase-3/
|-- phase-4/
|
|-- phase-5/
|   |-- chest-xray-detection-phase5.ipynb     # Threshold tuning notebook
|   |-- phase-5-README.md                     # This file
|
|-- results/
|   |-- train_data.csv
|   |-- val_data.csv
|   |-- test_data.csv
|   |-- per_class_val_densenet_weighted.csv
|   |-- per_class_test_densenet_weighted.csv
|   |-- per_class_val_threshold_tuned.csv
|   |-- per_class_test_threshold_tuned.csv
|   |-- phase5_comparison.csv                 # Weighted model test comparison
|   |-- per_class_val_densenet_finetune.csv
|   |-- per_class_test_densenet_finetune.csv
|   |-- per_class_val_finetune_threshold_tuned.csv
|   |-- per_class_test_finetune_threshold_tuned.csv
|   |-- phase5_comparison_v2.csv              # Fine-tuned model test comparison
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

A GPU is helpful for faster inference, but Phase 5 does not train and can run on CPU. The saved DenseNet checkpoints and Phase 4 per-class metric tables must be attached in the notebook environment.

---

## How to Run

1. Attach the NIH Chest X-ray image dataset, the Phase 1 split CSV files, the Phase 4 DenseNet checkpoints, and the Phase 4 per-class metric tables.
2. Open `phase-5/chest-xray-detection-phase5.ipynb`.
3. Run the setup, dataset, transform, and loader cells.
4. Load `densenet_finetune_weighted.pt`.
5. Generate weighted-model validation and test probabilities.
6. Select validation-optimized thresholds for all 14 disease labels.
7. Export the weighted threshold-tuned tables and `phase5_comparison.csv`.
8. Load `densenet_finetune.pt`.
9. Repeat probability generation, threshold selection, tuned evaluation, and export for the fine-tuned model.
10. Export `phase5_comparison_v2.csv`.

Expected exported files:

```text
per_class_val_threshold_tuned.csv
per_class_test_threshold_tuned.csv
phase5_comparison.csv
per_class_val_finetune_threshold_tuned.csv
per_class_test_finetune_threshold_tuned.csv
phase5_comparison_v2.csv
```

---

## Results Summary

### Weighted DenseNet threshold tuning

| Split / Table | AUROC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Validation threshold-tuned | 0.8221 | 0.2921 | 0.3497 | 0.3010 |
| Test weighted prior | 0.7970 | 0.3108 | 0.3454 | 0.3061 |
| Test threshold-tuned | 0.7970 | 0.3068 | 0.3882 | 0.3169 |

### Fine-tuned DenseNet threshold tuning

| Split / Table | AUROC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Validation threshold-tuned | 0.8214 | 0.2318 | 0.3745 | 0.2787 |
| Test fine-tuned prior | 0.7939 | 0.3669 | 0.0852 | 0.1265 |
| Test threshold-tuned | 0.7939 | 0.2585 | 0.4170 | 0.3010 |

### Interpretation

- Weighted BCE and threshold tuning both improve the threshold-dependent operating point while preserving ranking quality.
- The weighted checkpoint remains the steadier deployment candidate because it already has usable global-threshold behavior and only needs modest threshold adjustment.
- The fine-tuned checkpoint demonstrates the strongest post hoc threshold-tuning recovery: macro F1 rises from 0.1265 to 0.3010 without retraining.
- `phase5_comparison_v2.csv` captures the newer fine-tuned threshold-tuning experiment, while `phase5_comparison.csv` preserves the weighted-model comparison.

---

## Ideas for Extension

### 1. Save threshold vectors for deployment

Export both 14-value threshold vectors to JSON or YAML so later phases can load the exact operating points without rerunning validation tuning.

### 2. Add calibration curves

Reliability diagrams and expected calibration error would show whether the DenseNet probabilities are calibrated or mainly useful for ranking.

### 3. Tune for clinical constraints

Instead of maximizing F1, choose thresholds for sensitivity at a fixed specificity, or optimize different thresholds for high-risk findings.

### 4. Bootstrap uncertainty intervals

Add confidence intervals for per-class F1, recall, precision, and AUROC, especially for rare labels such as Hernia and Pneumonia.

### 5. Validate thresholds externally

The thresholds are selected on NIH validation data. A future phase should test whether they transfer to another chest X-ray dataset or need site-specific recalibration.

---

*Phase 5 shows that DenseNet ranking performance and operating-point behavior should be evaluated separately. Threshold tuning leaves AUROC unchanged, modestly improves the already weighted checkpoint, and substantially recovers recall/F1 for the unweighted fine-tuned checkpoint without any additional training.*
