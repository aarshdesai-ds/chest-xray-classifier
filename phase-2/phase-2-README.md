# Phase 2: PyTorch Dataset Loading

Dataset-loading workflow that converts Phase 1 CSV outputs into PyTorch-ready image and label tensors for chest X-ray multi-label classification.

---

## Table of Contents

- [Overview](#overview)
- [Phase Questions](#phase-questions)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Ideas for Extension](#ideas-for-extension)

---

## Overview

Phase 2 bridges preprocessing and model building. It loads the CSV files created in Phase 1, defines a custom PyTorch `Dataset`, applies image transforms, verifies single-sample tensor shapes, and creates a training `DataLoader`.

The core goal is to confirm that the model-building phase can receive batches shaped correctly:

```text
images: [batch_size, 3, 224, 224]
labels: [batch_size, 14]
```

---

## Phase Questions

- Can the Phase 1 CSV files be loaded into PyTorch?
- Can every image be opened through the `full_path` column?
- Do images convert cleanly into 3-channel tensors?
- Do the 14 disease labels load as float tensors?
- Do train, validation, and test samples return the same shape contract?
- Does the training `DataLoader` produce model-ready batches?

---

## Dataset Description

### Phase 1 CSV Inputs

| File | Description |
|---|---|
| `train_data.csv` | Patient-safe training metadata, labels, and image paths |
| `val_data.csv` | Patient-safe validation metadata, labels, and image paths |
| `test_data.csv` | Official test metadata, labels, and image paths |

Each CSV is expected to contain:

| Column Group | Columns | Description |
|---|---|---|
| Image ID | `Image Index` | Image filename |
| Metadata | patient, age, gender, view, dimensions, spacing | Information retained from Phase 1 |
| Labels | 14 disease columns | Multi-label binary targets |
| Image Path | `full_path` | Full path used to open the image |

### Disease Labels

The label tensor has 14 values:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule,
Pleural_Thickening, Pneumonia, Pneumothorax
```

---

## Methodology

### 1. Import and mount data

The notebook imports Kaggle, pandas, PIL, PyTorch, and torchvision utilities, then mounts the NIH Chest X-ray image dataset.

This is required because the Phase 1 CSV files store `full_path` values that must point to accessible image files.

### 2. Define `ChestImageDataset`

The custom dataset reads one CSV file and returns one `(image, labels)` pair for any row index.

Inside `__getitem__`:

1. the row is selected,
2. `full_path` is used to open the image,
3. the X-ray is converted to RGB,
4. the disease label columns are extracted,
5. transforms are applied,
6. the transformed image and label tensor are returned.

### 3. Apply image transforms

Images are resized to `224 x 224`, converted to tensors, and normalized with ImageNet mean and standard deviation values:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

This prepares the input for common pretrained CNN backbones.

### 4. Create split datasets

The notebook creates:

- `train_dataset`
- `val_dataset`
- `test_dataset`

Each split uses the same dataset class and transform pipeline so train, validation, and test preprocessing are consistent.

### 5. Verify sample shapes

One example from each split is loaded and checked. This catches broken paths, transform errors, wrong channel counts, and label-selection mistakes before training begins.

### 6. Create a training `DataLoader`

The training dataset is wrapped with:

```python
DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
```

Shuffling is appropriate for training, and batching is required for efficient model optimization.

---

## Key Findings

### 1. Image mounting works

The Kaggle image dataset is available at:

```text
/kaggle/input/datasets/organizations/nih-chest-xrays/data
```

### 2. Single samples have the expected shape

Each split returns:

```text
image: [3, 224, 224]
label: [14]
```

This confirms that images are transformed correctly and labels match the 14 disease targets.

### 3. The training loader returns model-ready batches

The `DataLoader` returns:

```text
images: [32, 3, 224, 224]
labels: [32, 14]
```

This is the expected input/output shape contract for Phase 3 model building.

---

## Project Structure

```text
Chest-XRay-Project/
|
|-- models/
|   |-- baseline_model.pt     # Phase 3 checkpoint
|
|-- phase-1/
|   |-- phase-1.ipynb
|   |-- phase-1-README.md
|
|-- phase-2/
|   |-- chest-xray-detection-phase2.ipynb  # Phase 2 dataset-loading notebook
|   |-- phase-2-README.md                  # Phase 2 documentation
|
|-- phase-3/
|
|-- results/
|   |-- train_data.csv            # Shared Phase 1 output used here
|   |-- val_data.csv              # Shared Phase 1 output used here
|   |-- test_data.csv             # Shared Phase 1 output used here
|   |-- final_metrics.csv         # Shared class-balance summary for modeling decisions
```

The CSV files live in the `results/` directory so Phase 2 and later modeling phases can read the same shared split files without duplicating them.

---

## Requirements

```text
Python 3.8+
pandas
torch
torchvision
Pillow
kagglehub
jupyter
```

Install dependencies:

```bash
pip install pandas torch torchvision pillow kagglehub jupyter
```

---

## How to Run

1. Run Phase 1 and upload or attach the exported CSV files as a Kaggle dataset.
2. Open `phase-2/chest-xray-detection-phase2.ipynb`.
3. Confirm the CSV paths point to the Phase 1 exported files.
4. Run all cells sequentially.
5. Confirm the sample and batch shape checks match the expected outputs.

---

## Results Summary

| Check | Expected Result |
|---|---|
| Train image sample | `[3, 224, 224]` |
| Train label sample | `[14]` |
| Validation image sample | `[3, 224, 224]` |
| Validation label sample | `[14]` |
| Test image sample | `[3, 224, 224]` |
| Test label sample | `[14]` |
| Training image batch | `[32, 3, 224, 224]` |
| Training label batch | `[32, 14]` |

---

## Ideas for Extension

### 1. Add validation and test DataLoaders

Create `val_data` and `test_data` loaders so Phase 3 can run full validation and test loops cleanly.

### 2. Add visualization checks

Display a few transformed images with their positive labels to confirm that image loading and labels line up.

### 3. Add class-weight calculation

Bring in `final_metrics.csv` or training label counts to compute `pos_weight` values for the Phase 3 loss function.

### 4. Add configurable batch size

Define `BATCH_SIZE`, `NUM_WORKERS`, and `IMAGE_SIZE` as variables at the top of the notebook for easier experimentation.

### 5. Add reproducibility settings

Set seeds for PyTorch, NumPy, and Python random to make training runs easier to compare.

---

*Phase 2 confirms that the cleaned Phase 1 outputs can be loaded as PyTorch tensors. Its main contribution is establishing the data contract for Phase 3: model inputs shaped `[batch_size, 3, 224, 224]` and multi-label targets shaped `[batch_size, 14]`.*
