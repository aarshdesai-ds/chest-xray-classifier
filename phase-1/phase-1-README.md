# Phase 1: Chest X-Ray Dataset Preprocessing

Preprocessing workflow for the NIH Chest X-ray dataset, focused on creating patient-safe train, validation, and test metadata files for multi-label disease classification.

---

## Table of Contents

- [Overview](#overview)
- [Phase Questions](#phase-questions)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Class Balance and Metric Choice](#class-balance-and-metric-choice)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Ideas for Extension](#ideas-for-extension)

---

## Overview

This phase prepares the NIH Chest X-ray metadata for downstream image modeling. The notebook loads the official metadata, cleans unused columns, encodes the disease labels, attaches image file paths, creates patient-safe splits, and exports reusable CSV files for later phases.

The core goal is to ensure that later modeling notebooks start from stable data splits where:

1. each image has a valid file path,
2. each image has 14 binary disease labels,
3. patients do not leak across train, validation, and test sets,
4. label imbalance is documented before model selection.

---

## Phase Questions

- How many unique patients and images are represented in the metadata?
- Are there repeated images per patient that could create leakage?
- Can the raw `Finding Labels` string be converted into model-ready binary targets?
- Can each metadata row be linked to a valid X-ray image path?
- Do the train/validation and test splits have any patient overlap?
- How imbalanced are the 14 disease labels across train, validation, and test?
- Which evaluation metrics should Phase 3 prioritize based on this imbalance?

---

## Dataset Description

### Metadata File

`Data_Entry_2017.csv`

This file contains image-level metadata for the NIH Chest X-ray dataset.

| Column Group | Columns | Description |
|---|---|---|
| Image ID | `Image Index` | X-ray image filename |
| Labels | `Finding Labels` | Pipe-separated disease labels |
| Patient Tracking | `Patient ID`, `Follow-up #` | Patient identifier and follow-up sequence |
| Demographics | `Patient Age`, `Patient Gender` | Basic patient metadata |
| Image View | `View Position` | PA or AP view |
| Image Metadata | `OriginalImage[Width`, `Height]`, `OriginalImagePixelSpacing[x`, `y]` | Image dimensions and spacing |
| Empty Field | `Unnamed: 11` | Fully empty column removed during preprocessing |

### Split Files

| File | Description |
|---|---|
| `train_val_list.txt` | Official train/validation image filename list |
| `test_list.txt` | Official test image filename list |

### Image Files

Images are stored across nested Kaggle folders such as `images_001/images/`. The notebook discovers all `.png` files and merges each path into the metadata through `Image Index`.

---

## Methodology

### 1. Metadata Loading and Inspection

The notebook loads `Data_Entry_2017.csv`, previews the first records, checks schema information, and computes summary statistics.

This step confirms the dataset size, column types, missingness, and possible metadata issues such as implausible age values.

### 2. Patient-Level Review

Images are grouped by `Patient ID` to identify repeated imaging. This is necessary because image-level random splitting could place images from the same patient in different splits, causing leakage.

### 3. Metadata Cleaning

The fully empty `Unnamed: 11` column is removed. The remaining columns are checked to confirm the metadata table is cleaner before label engineering.

### 4. Multi-Label Encoding

The raw `Finding Labels` text field is converted into one binary column per disease using pipe-separated label parsing.

The `No Finding` indicator is removed so the final label columns represent the 14 positive disease targets.

### 5. Image Path Mapping

All `.png` image paths are discovered from the Kaggle dataset directory and mapped to filenames. The mapping is merged into the metadata so every row includes a `full_path` value.

An assertion verifies:

```python
assert final_data["full_path"].notna().all()
```

This protects Phase 2 from rows that have labels but no loadable image file.

### 6. Patient-Safe Splitting

The official `train_val_list.txt` and `test_list.txt` files define the train/validation pool and test split.

The notebook checks that there is no patient overlap between these groups, then uses `GroupShuffleSplit` to create a validation split from the train/validation pool while keeping each patient entirely in one split.

### 7. Class Balance Summary

The positive-label ratio is computed for each disease across train, validation, and test. These results are exported as `final_metrics.csv`.

---

## Key Findings

### 1. The dataset contains substantial repeated imaging

- Total image records: `112,120`
- Unique patients: `30,805`
- Patient `10007` has the longest sequence, with follow-up values reaching `183`

This confirms that patient-level grouping is required.

### 2. The official split is patient-safe

The train/validation pool and official test set have no overlapping patient IDs.

### 3. The custom validation split is also patient-safe

`GroupShuffleSplit` is used with `Patient ID` as the grouping variable, and an assertion confirms no train/validation patient overlap.

### 4. Every retained row has an image path

The `full_path` assertion passes, which means Phase 2 can load images directly from the exported CSVs.

### 5. The label distribution is highly imbalanced

Common labels such as `Infiltration`, `Effusion`, and `Atelectasis` occur far more often than rare labels such as `Hernia`, `Pneumonia`, and `Fibrosis`.

---

## Class Balance and Metric Choice

The notebook exports these values to `final_metrics.csv`. The local CSV file is not present in this folder, so the table below documents the saved notebook output.

| Finding | Train Positive Rate | Validation Positive Rate | Test Positive Rate |
|---|---:|---:|---:|
| Atelectasis | 9.49% | 10.19% | 12.81% |
| Cardiomegaly | 1.98% | 1.93% | 4.18% |
| Consolidation | 3.33% | 3.03% | 7.09% |
| Edema | 1.62% | 1.41% | 3.61% |
| Effusion | 10.01% | 9.96% | 18.20% |
| Emphysema | 1.70% | 1.25% | 4.27% |
| Fibrosis | 1.42% | 1.63% | 1.70% |
| Hernia | 0.15% | 0.25% | 0.34% |
| Infiltration | 15.98% | 15.54% | 23.88% |
| Mass | 4.66% | 4.68% | 6.83% |
| Nodule | 5.44% | 5.48% | 6.34% |
| Pleural_Thickening | 2.53% | 3.04% | 4.47% |
| Pneumonia | 1.00% | 1.08% | 2.17% |
| Pneumothorax | 3.06% | 2.93% | 10.41% |

### Metric implications

Plain accuracy should not be the primary metric. Since every label is mostly negative, a model can achieve high accuracy by under-predicting positive disease labels.

Recommended metrics for Phase 3:

- **BCEWithLogitsLoss** for training, because the task is multi-label.
- **Weighted BCE or `pos_weight`** to address rare positive labels.
- **Macro AUROC** to give every disease equal importance.
- **Macro AUPRC** because precision-recall is more informative when positives are rare.
- **Per-class AUROC and AUPRC** so rare labels are not hidden by aggregate scores.
- **F1, precision, and recall** after choosing thresholds on the validation set.

---

## Project Structure

```text
Chest-XRay-Project/
|
|-- models/
|   |-- baseline_model.pt     # Phase 3 checkpoint
|
|-- phase-1/
|   |-- phase-1.ipynb         # Phase 1 preprocessing notebook
|   |-- phase-1-README.md     # Phase 1 documentation
|
|-- phase-2/
|-- phase-3/
|
|-- results/
|   |-- train_data.csv            # Shared output from Phase 1
|   |-- val_data.csv              # Shared output from Phase 1
|   |-- test_data.csv             # Shared output from Phase 1
|   |-- final_metrics.csv         # Shared class-balance summary
```

The CSV files are exported to the `results/` directory because they are shared inputs for later phase folders.

---

## Requirements

```text
Python 3.8+
pandas
numpy
scikit-learn
kagglehub
jupyter
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn kagglehub jupyter
```

---

## How to Run

1. Open `phase-1/phase-1.ipynb` in Kaggle or another environment with access to the NIH Chest X-ray dataset.
2. Run all cells sequentially.
3. Confirm that all assertions pass.
4. Confirm the following CSV files are created in `results/`:

```text
results/train_data.csv
results/val_data.csv
results/test_data.csv
results/final_metrics.csv
```

---

## Results Summary

| Output | Result |
|---|---|
| Image metadata records | `112,120` |
| Unique patients | `30,805` |
| Train/validation image list | `86,524` |
| Test image list | `25,596` |
| Disease labels | `14` |
| Train/test patient overlap | None |
| Train/validation patient overlap | None |
| Missing `full_path` values | None |
| Main exported metric file | `final_metrics.csv` |

---

## Ideas for Extension

### 1. Add age and view-position quality checks

The maximum age appears implausible, and view position may influence model behavior. A future preprocessing phase could clean age outliers and compare AP vs. PA distributions across splits.

### 2. Visualize class imbalance

Convert `final_metrics.csv` into bar charts comparing train, validation, and test positive rates for each disease.

### 3. Add image-level sanity checks

Sample images from each split and confirm that paths, labels, and visual content line up correctly.

### 4. Compute `pos_weight` values

Use the training label counts to compute class-specific `pos_weight` values for `BCEWithLogitsLoss`.

### 5. Track split reproducibility

Export the exact random seed and split configuration to make the validation split fully auditable.

---

*Phase 1 prepares the dataset for multi-label chest X-ray model development. Its most important modeling contribution is the `final_metrics.csv` class-balance summary, which directly motivates imbalance-aware evaluation in later phases.*
