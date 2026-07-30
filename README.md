# Memory-Augmented Adaptive Deepfake Audio Detection

Hybrid acoustic features + physics-guided voice analysis + XLS-R self-supervised embeddings, combined with a **Speaker Memory Bank** for personalized, reference-free deepfake audio detection.

This repository implements a two-tier detection framework:

1. **General Detector** — a speaker-agnostic classifier trained on a 1,767-dimensional hybrid feature vector.
2. **Personalized Detector** — a cross-attention head that augments the general detector's embedding with speaker-specific history retrieved from a dynamic **Memory Bank**, enabling adaptive, reference-free speaker consistency verification that improves as it observes more genuine speech from a given user.

> 📄 Companion paper: *"Memory-Augmented Adaptive Deepfake Audio Detection Using Hybrid Acoustic Features, Physics-Guided Analysis, and XLS-R Embeddings"* (included in this repo as `Audio_Deepfake_Detection.pdf`).

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Feature Extraction](#feature-extraction)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations & Future Work](#limitations--future-work)


---

## Overview

Most audio deepfake detectors are trained and evaluated as **speaker-independent** classifiers: a single decision boundary is applied uniformly to every speaker. This ignores the fact that many real-world systems (voice assistants, telephone banking, call centers) interact with the *same* user repeatedly, and could exploit that history to improve detection.

This project addresses that gap by combining:

- **Handcrafted acoustic descriptors** (MFCC, CQCC, LFCC, spectral, prosodic features)
- **Physics-guided articulatory features** (jitter, shimmer, harmonics-to-noise ratio) that model biomechanical constraints of human voice production which generative TTS/VC systems tend to violate
- **XLS-R self-supervised embeddings** (`facebook/wav2vec2-xls-r-300m`) for high-level phonetic/speaker information
- A **persistent, per-speaker Memory Bank** that starts empty and is populated online with high-confidence genuine embeddings, retrieved via adaptive Top-K search and fused with the current utterance through cross-attention — removing the need for a separate enrolled reference sample at deployment time.

## Key Features

- **Hybrid 1,767-dim feature vector** per 4-second utterance (MFCC, CQCC, LFCC + deltas, spectral, prosodic, physics-guided, XLS-R)
- **Adaptive gating-based feature fusion**: each feature group is independently layer-normalized and weighted by a learned gate instead of naive concatenation
- **Joint objective**: class-weighted cross-entropy + speaker-aware triplet loss to produce a 256-d discriminative embedding
- **Speaker Memory Bank** (SQLite + FAISS-backed) with:
  - Confidence-guided enrollment (only high-confidence, physics-consistent embeddings are stored)
  - Adaptive complexity-aware Top-K retrieval
  - Memory aging (recency weighting)
  - Prototype compression / clustering to bound memory growth
- **Cross-attention personalization head** that fuses retrieved speaker memory with the current query embedding
- **Deployment-realistic evaluation protocol**: the memory bank starts empty and grows online during evaluation (no pre-populated speaker profiles), simulating real-world cold-start conditions
- Threshold analysis at default (0.5), EER, and F1-optimal operating points

## Architecture

```
Raw audio (16 kHz, 4s)
      │
      ├── MFCC (240) ─┐
      ├── CQCC (240)  │
      ├── LFCC (240)  │
      ├── Spectral(13)├─► Adaptive Feature Fusion (LayerNorm + gating) ─► 256-d embedding
      ├── Prosody (5) │        (CE loss + speaker-aware triplet loss)         │
      ├── Physics (5) │                                                      │
      └── XLS-R (1024)┘                                                      │
                                                                               │
                                        ┌──────────────────────────────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │                            │
                 General Classifier          Speaker Memory Bank (per-speaker)
                 (speaker-agnostic)             confidence gate → aging →
                          │                     prototype compression → FAISS retrieval
                          │                            │
                          │                 Cross-Attention Personalizer
                          │                  (query ⊕ retrieved memory)
                          │                            │
                          └──────────► Final: General decision / Personalized decision
```

Feature layout used internally (`FEATURE_GROUPS`):

| Group     | Dims | Index range |
|-----------|------|--------------|
| MFCC      | 240  | 0–240 |
| CQCC      | 240  | 240–480 |
| LFCC      | 240  | 480–720 |
| Spectral  | 13   | 720–733 |
| Prosody   | 5    | 733–738 |
| Physics   | 5    | 738–743 |
| XLS-R     | 1024 | 743–1767 |
| **Total** | **1767** | |

*(Each of MFCC/CQCC/LFCC is 80 base coefficients + delta + delta-delta = 240 dims.)*

## Feature Extraction

| Feature group | Description | Library |
|---|---|---|
| MFCC | Mel-Frequency Cepstral Coefficients + Δ + ΔΔ | `librosa` |
| CQCC | Constant-Q Cepstral Coefficients (log CQT → DCT) + Δ + ΔΔ | `librosa`, `scipy` |
| LFCC | Linear Frequency Cepstral Coefficients + Δ + ΔΔ | `spafe` |
| Spectral | Centroid, bandwidth, roll-off, flatness, ZCR, RMS, contrast | `librosa` |
| Prosodic | Pitch (mean/std via PYIN), energy (mean/std), voiced ratio | `librosa` |
| Physics-guided | Jitter, shimmer, harmonics-to-noise ratio (HNR) | `praat-parselmouth` |
| XLS-R | Mean-pooled last hidden state of `facebook/wav2vec2-xls-r-300m` | `transformers` |

All handcrafted features are mean-pooled over time to produce a fixed-length vector per clip.

## Dataset

This project is evaluated on two speaker-disjoint benchmarks:

- **ASVspoof 2019 Logical Access (LA)** — the dataset the notebook in this repository is actually set up to train/evaluate on (official protocol splits, downloaded via the Kaggle API). 107 total speakers, 121,461 clips (25,380 train / 24,844 dev / 71,237 eval), spoofed via 19 TTS/VC attack IDs, recorded under controlled studio conditions.
- **In-the-Wild Audio Deepfake Dataset** — a more realistic, uncontrolled benchmark reported in the companion paper. 54 speakers, 31,779 clips (19,963 bona fide / 11,816 spoof), with diverse real-world recording conditions, background noise, and compression artifacts.

Both use a speaker-disjoint train/test protocol so that speakers seen during evaluation are never seen during training.

Audio preprocessing pipeline:
1. Resample to 16 kHz, convert to mono
2. (Optional) trim leading/trailing silence via VAD
3. Peak-normalize amplitude
4. Fix duration to 4 seconds (64,000 samples): zero-pad short clips, random/center-crop long clips
5. Train-time augmentation: Gaussian noise, pitch shift, time stretch, gain, time shift, air absorption (`audiomentations`)

## Repository Structure

```
.
├── Audio_Deepfake_Detection.pdf   # Companion paper describing the method & results
├── notebook.ipynb                 # End-to-end pipeline (this repo's main artifact)
└── README.md
```

The notebook is organized into the following logical stages:

1. Kaggle dataset setup & dependency installation
2. Reading official ASVspoof 2019 LA protocol files into train/dev/eval DataFrames
3. Audio augmentation pipeline
4. `ASVSpoofDataset` (on-the-fly loading, padding/cropping)
5. Handcrafted feature extractors: `extract_mfcc`, `extract_cqcc`, `extract_lfcc`, `extract_spectral`, `extract_prosody`, `extract_physics`
6. XLS-R embedding extraction (`extract_xlsr`, batched variant)
7. Hybrid feature fusion (`extract_all_features`) and checkpointed, resumable batch feature extraction to disk (with optional Kaggle Dataset push for persistence across sessions)
8. `CachedFeatureDataset` + `FeatureFusion` gating module + general embedding/classification model training (`train_general_model`)
9. Evaluation utilities: accuracy, precision/recall/F1, AUC-ROC, EER, confusion matrices
10. `MemoryBank` (SQLite + FAISS) with confidence gating, aging, prototype compression
11. `CrossAttentionPersonalizer` module and personalizer training (`train_personalizer`)
12. Deployment-style online replay evaluation (`run_deployment_replay`) — memory starts empty and grows as the model processes the eval stream — followed by final reporting

## Installation

```bash
git clone <this-repo-url>
cd <this-repo>

pip install librosa==0.10.2 audiomentations transformers soundfile \
            spafe praat-parselmouth faiss-cpu torchmetrics kaggle \
            torch scikit-learn pandas numpy tqdm sentencepiece
```

**Requirements**
- Python 3.9+
- CUDA-capable GPU recommended (XLS-R feature extraction is significantly faster on GPU)
- A Kaggle account/API token if you want to download the ASVspoof 2019 LA dataset directly from Kaggle (`~/.kaggle/kaggle.json`)

## Usage

The pipeline is provided as a single Jupyter notebook designed to run on Kaggle (it references `/kaggle/input` and `/kaggle/working` paths). To run locally:

1. **Get the data.** Download the ASVspoof 2019 LA dataset and its `ASVspoof2019_LA_cm_protocols` folder, and update `ROOT`, `TRAIN_AUDIO`, `DEV_AUDIO`, `EVAL_AUDIO`, and `PROTOCOL_DIR` paths in the notebook to point to your local copy.
2. **Extract hybrid features.** Run the feature-extraction cells to compute the 1,767-dim hybrid vector for every clip and cache it to disk as `.pt` files (this step is resumable/checkpointed since XLS-R + physics feature extraction over tens of thousands of clips is time-consuming).
3. **Train the general detector.** Run `train_general_model(train_dataset, val_dataset)` to train the adaptive feature-fusion + classification model on cached features.
4. **Train the personalizer.** Run `train_personalizer(train_dataset, general_model)` to train the cross-attention memory-personalization head using a simulated online enrollment/replay procedure.
5. **Evaluate.** Run `run_deployment_replay(eval_dataset, general_model, personalizer)` to reproduce the deployment-style evaluation, where the Speaker Memory Bank starts empty and is populated as evaluation utterances stream in.

```python
general_model, history = train_general_model(train_dataset, val_dataset)
personalizer, p_history = train_personalizer(train_dataset, general_model)
replay_df = run_deployment_replay(eval_dataset, general_model, personalizer)
general_metrics, personalized_metrics, submission = report_results(replay_df)
```

## Results

Results are reported on two speaker-disjoint benchmarks: **ASVspoof 2019 LA** (this repository's actual notebook run, official protocol splits) and **In-the-Wild Audio Deepfake Dataset** (from the companion paper).

### ASVspoof 2019 LA (this repository's actual run)

Official speaker-disjoint splits: 25,380 train / 24,844 dev / 71,237 eval clips (20 train/dev speakers, 67 eval speakers). The personalized deployment-replay evaluation streams all 71,237 eval clips across 67 speakers through an initially **empty** memory bank; 46,727 clips (65.6%) were accepted as memory writes, with a dynamic Top-K retrieval size averaging 4.85 (range 3–10).

**General detector** (decision threshold = 0.5)

| Metric | Value |
|---|---|
| Accuracy | 93.00% |
| Precision (spoof) | 0.999 |
| Recall (spoof) | 0.923 |
| F1 (spoof) | 0.959 |
| F1 (bonafide) | 0.746 |
| AUC-ROC | 0.9966 |
| EER | 2.35% |

**Personalized detector** (decision threshold = 0.5, Speaker Memory Bank + cross-attention)

| Metric | Value |
|---|---|
| Accuracy | **97.16%** |
| Precision (spoof) | 0.999 |
| Recall (spoof) | 0.970 |
| F1 (spoof) | **0.984** |
| F1 (bonafide) | **0.878** |
| AUC-ROC | **0.9978** |
| EER | **1.90%** |

At the personalized model's best-F1 operating threshold (0.0251), macro-F1 reaches **0.967**.

Training summary: the general embedding/classification model converged in 6 epochs (early stopping on validation loss, best `val_loss=0.0366`, `val_acc≈98.7–99.2%` across epochs); the cross-attention personalizer was trained for 6 epochs with loss decreasing from 0.071 → 0.013.

> Note: the bonafide class is a small minority in ASVspoof 2019 LA (e.g., only ~10% of eval clips), so bonafide-class F1 is the more informative metric for the memory bank's practical effect — it improves substantially (0.746 → 0.878) with personalization, alongside a ~30% relative reduction in EER (2.35% → 1.90%).

### In-the-Wild Audio Deepfake Dataset (companion paper)

Evaluated on the speaker-disjoint In-the-Wild dataset (54 speakers, 31,779 clips: 19,963 bona fide / 11,816 spoof).

**General detector**

| Metric | Value |
|---|---|
| Accuracy | 95% |
| Precision / Recall / F1 | 0.95 / 0.95 / 0.95 |
| AUC-ROC | 0.989 |
| EER | 4.76% |

**Personalized detector (with Speaker Memory Bank)**

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Default (0.50) | 95% | 0.95 | 0.95 | 0.95 |
| EER (0.4167) | 95% | 0.95 | 0.95 | 0.95 |
| Best F1 (0.1255) | **96%** | **0.96** | **0.96** | **0.96** |

**General vs. Personalized (best-F1 threshold)**

| Metric | General | Personalized |
|---|---|---|
| Accuracy | 95% | 96% |
| Precision | 0.95 | 0.96 |
| Recall | 0.95 | 0.96 |
| F1 | 0.95 | 0.96 |
| AUC | 0.989 | 0.989 |
| EER | 4.76% | 4.72% |

Memory-augmented personalization reduced false positives (117→114) and false negatives (124→117) relative to the general detector under a realistic online-enrollment protocol (memory bank starts empty and grows during evaluation).

### Takeaway across both datasets

On both benchmarks, the Speaker Memory Bank + cross-attention personalization consistently improves over the speaker-agnostic general detector under a realistic online-enrollment protocol (empty memory at the start of evaluation, populated only from high-confidence genuine speech as it streams in) — with a larger relative gain observed on ASVspoof 2019 LA (accuracy +4.2 pts, EER −0.45 pts) than on In-the-Wild (accuracy +1 pt, EER −0.04 pts), plausibly reflecting differences in speaker count, class balance, and recording conditions between the two datasets.

## Limitations & Future Work

- **Cold-start enrollment**: the memory bank has little/no information for newly seen speakers, limiting early-stage personalization gains.
- **Conservative enrollment policy**: confidence-guided gating intentionally restricts what gets stored, trading off contextual richness for reliability.
- **Ceiling effect**: the general detector already performs strongly, leaving limited headroom for personalization to improve raw accuracy.
- Evaluated on a single benchmark with a limited number of speakers; cross-dataset generalization is untested.
- Future directions: end-to-end joint optimization of the embedding network and memory retrieval module, learned (rather than heuristic Top-K) retrieval policies, adaptive per-speaker threshold calibration, and evaluation on multilingual / cross-dataset corpora.

## 📚 Documentation

For a detailed explanation of the hybrid feature extraction pipeline, feature dimensions, preprocessing workflow, and download instructions for the precomputed features, see:

➡️ **[Hybrid Extracted Features](HYBRID_EXTRACTED_FEATURES.md)**
