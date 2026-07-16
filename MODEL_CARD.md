# Model Card — Chest X-Ray Multi-Label Classifier (DenseNet-121)

> **Research demonstration only. NOT a medical device.** This model was built for a
> portfolio project on the public NIH ChestX-ray14 dataset. It must **not** be used for
> diagnosis, screening, or any clinical decision.

---

## Model details

- **Architecture:** DenseNet-121 (the CheXNet architecture), ImageNet-pretrained, with the
  1000-class head replaced by a 14-output linear layer.
- **Task:** multi-label classification of 14 thoracic findings from a single frontal chest X-ray
  (independent per-finding sigmoids — a finding can co-occur with others).
- **Deployment checkpoint:** `densenet_finetune_weighted.pt` — fully fine-tuned, then continued
  with a square-root-scaled class-weighted `BCEWithLogitsLoss` to counter severe class imbalance.
- **Input:** RGB `224 x 224`, ImageNet-normalized.
- **Output:** 14 independent probabilities. Each finding is **flagged** when its probability crosses
  a **per-class threshold tuned on the validation split** (see `phase-7/thresholds.json`).

### Findings
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia,
Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax.

---

## Training data

- **Dataset:** NIH ChestX-ray14 (112,120 frontal chest X-rays, 30,805 patients) — public, de-identified.
- **Splits:** patient-level (no patient appears in more than one split), using the official
  train/val/test partition; 76,277 train / 10,247 validation / 25,596 test images.
- **Labels:** NLP-mined from radiology reports (a known source of label noise in this dataset).

---

## Performance

Judged on **macro-AUROC** (prevalence-invariant) rather than accuracy, because only ~4.5% of
label slots are positive — an all-negative predictor scores ~95% accuracy while being useless.

| Model | Val macro-AUROC | Test macro-AUROC | Val→Test gap |
|---|---:|---:|---:|
| From-scratch CNN (baseline) | 0.694 | 0.604 | 0.090 |
| DenseNet-121 frozen | 0.739 | 0.697 | 0.042 |
| DenseNet-121 fine-tuned | 0.821 | 0.794 | 0.028 |
| **DenseNet-121 fine-tuned + weighted (deployed)** | **0.822** | **0.797** | **0.025** |

**Operating points at the tuned thresholds (test, macro):** recall ≈ 0.39, F1 ≈ 0.32.
Per-class AUROC ranges from ~0.71 (Pneumonia, weakest) to ~0.90 (Emphysema, strongest); see
`results/per_class_test_densenet_weighted.csv`.

---

## Limitations

- **Validation-to-test generalization gap.** The official NIH test set is distribution-shifted
  (~2x higher disease prevalence than train/val). AUROC drops from 0.822 (val) to 0.797 (test);
  **0.797 is the honest estimate**. Real-world performance on other sites/scanners is unknown and
  likely lower.
- **Weak classes.** Pneumonia (AUROC ~0.71) and other subtle findings are near the model's floor.
  No threshold or class-weighting fixes a poorly-ranked class — this is a modeling limit.
- **Threshold transfer.** Per-class thresholds are tuned on validation and may not transfer to a
  differently-distributed population (observed: Cardiomegaly's threshold generalized worse to test).
- **Label noise.** NIH labels are NLP-mined, not radiologist-verified per image.
- **Single view only.** Frontal X-ray; no lateral view, priors, or clinical context.
- **Not calibrated for any clinical operating point** (e.g., a target sensitivity/specificity).

---

## Interpretability & shortcut audit

A Grad-CAM audit (Phase 6) found **no evidence of a border/text shortcut**: for every finding,
mean Grad-CAM activation in the outer image frame was below the ~0.42 uniform baseline (overall
mean 0.199). The per-disease ordering tracked anatomy — most central for Cardiomegaly (0.066,
the enlarged heart) and most peripheral for diffuse/peripheral findings (Emphysema 0.387,
Effusion 0.262). This indicates the model attends to plausible chest anatomy rather than framing
artifacts, though a below-uniform score does not rule out subtler artifact reliance.

---

## Intended use & ethics

- **Intended use:** education, portfolio demonstration, and ML-engineering discussion of imbalance,
  operating-point selection, transfer learning, and interpretability.
- **Out of scope:** any clinical, diagnostic, triage, or screening use; use on populations or
  imaging equipment unlike NIH ChestX-ray14; any use implying medical validity.
- **Privacy:** trained only on public, de-identified images. Demo samples are drawn from the public
  NIH dataset.

---

## How to reproduce

The full pipeline (Phases 1–7) is in this repository, each phase with its own notebook and README:
data preparation and patient-level splits (P1), PyTorch data loading (P2), a from-scratch baseline
(P3), the transfer-learning ablation (P4), per-class threshold tuning (P5), the Grad-CAM/shortcut
audit (P6), and this deployment (P7).
