# Memory-Augmented Adaptive Audio Deepfake Detection

> A hybrid deepfake audio detection framework combining handcrafted acoustic features, physics-guided speech analysis, self-supervised XLS-R embeddings, adaptive feature fusion, and a dynamic speaker memory bank for personalized detection.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Research-Active-success)

---

## Overview

Audio deepfake generation has advanced rapidly through modern Text-to-Speech (TTS) and Voice Conversion (VC) systems, making synthetic voices increasingly difficult to distinguish from genuine human speech.

Most existing deepfake detectors:

- Learn only a **general decision boundary**
- Ignore **speaker-specific characteristics**
- Require **reference audio** during verification
- Do not improve as more user interactions become available

This project proposes a **Memory-Augmented Adaptive Deepfake Detection Framework** that learns both **general spoof detection** and **personalized speaker verification** without requiring stored reference recordings during inference.

Instead of comparing against pre-recorded reference samples, the system gradually builds a **dynamic speaker memory** using verified genuine speech.

---

# Key Features

- Hybrid acoustic feature extraction
- Physics-guided speech analysis
- XLS-R self-supervised embeddings
- Adaptive feature fusion
- Speaker-aware contrastive learning
- Dynamic Speaker Memory Bank
- Cross-attention personalization
- Continual memory updates
- Confidence-guided enrollment
- Memory aging and prototype compression
- Reference-free deployment
- Explainable feature importance

---

# Proposed Architecture

```
                  Audio Input
                       │
              Audio Preprocessing
                       │
      ┌─────────────────────────────────┐
      │                                 │
      │ Hybrid Feature Extraction        │
      │                                 │
      │ • MFCC                          │
      │ • CQCC                          │
      │ • LFCC                          │
      │ • Spectral Features             │
      │ • Prosodic Features             │
      │ • Physics Features              │
      │ • XLS-R Embeddings              │
      └─────────────────────────────────┘
                       │
             Adaptive Feature Fusion
                       │
           256-D Speaker Embedding
                       │
        ┌──────────────┴──────────────┐
        │                             │
 General Detector           Speaker Memory Bank
        │                             │
        │                     Adaptive Retrieval
        │                             │
        └────────── Cross Attention ──┘
                       │
          Personalized Deepfake Detector
                       │
                Real / Fake Prediction
```

---

# Novel Contributions

Unlike conventional systems, this framework introduces:

- Hybrid handcrafted + self-supervised features
- Physics-aware speech modeling
- Adaptive feature weighting
- Speaker memory without retraining
- Continual personalization
- Confidence-based enrollment
- Dynamic Top-K memory retrieval
- Prototype compression
- Memory aging mechanism

---

# Feature Extraction

The system combines handcrafted acoustic descriptors with deep speech representations.

| Feature | Dimension |
|----------|-----------|
| MFCC | 80 |
| CQCC | 80 |
| LFCC | 80 |
| Spectral Features | 14 |
| Prosodic Features | 4 |
| Physics-guided Features | 4 |
| XLS-R Embeddings | 1024 |
| **Total** | **1286** |

---

# Physics-Guided Features

Unlike traditional detectors, this project explicitly models human speech production.

Features include:

- Jitter
- Shimmer
- Harmonics-to-Noise Ratio (HNR)
- Formant continuity
- Breathing cycles
- Voice stability
- Prosodic consistency

These characteristics are difficult for current generative models to perfectly reproduce.

---

# Adaptive Feature Fusion

Instead of concatenating features directly, each feature group receives a learnable attention weight.

Benefits include:

- Dynamic importance assignment
- Better robustness to unseen attacks
- Improved generalization
- Reduced dependence on any single feature type

---

# Speaker Memory Bank

One of the major contributions of this work is the **Speaker Memory Bank**.

Unlike existing methods, the detector remembers previous genuine speech.

The memory stores:

- High-confidence embeddings
- Verified genuine samples only
- Speaker prototypes
- Historical speaker representations

Memory operations include:

- Enrollment
- Retrieval
- Aging
- Compression
- Continual updates

This allows the detector to evolve from a generic detector into a personalized one.

---

# Datasets

The framework was evaluated using:

## In-the-Wild Audio Deepfake Dataset

- 54 speakers
- 31,779 audio clips
- Real-world recording conditions
- Background noise
- Compression artifacts
- Multiple synthesis methods

## ASVspoof 2019 Logical Access

- Standard benchmark
- Controlled recording conditions
- TTS attacks
- Voice Conversion attacks

---

# Audio Preprocessing

Each recording undergoes:

- Resampling to 16 kHz
- Mono conversion
- Peak normalization
- Voice Activity Detection (optional)
- Silence removal
- Padding or center cropping
- Fixed duration of 4 seconds

---

# Model Components

## General Detector

Learns universal spoofing artifacts.

Training objectives:

- Weighted Cross Entropy
- Speaker-aware Triplet Loss

Output:

- 256-dimensional embedding
- Binary classification

---

## Personalized Detector

Uses:

- Speaker Memory Bank
- Adaptive retrieval
- Cross-attention fusion

Produces personalized predictions without retraining.

---

# Explainability

The framework is designed to be interpretable.

Visualization includes:

- Feature attention weights
- Memory retrieval scores
- Cross-attention maps
- Speaker similarity
- Confidence scores

---

# Experimental Results

## General Detector

| Metric | Score |
|---------|-------|
| Accuracy | **95%** |
| Precision | **0.95** |
| Recall | **0.95** |
| F1-score | **0.95** |
| AUC | **0.989** |
| EER | **4.76%** |

---

## Personalized Detector

| Metric | Score |
|---------|-------|
| Accuracy | **96%** |
| Precision | **0.96** |
| Recall | **0.96** |
| F1-score | **0.96** |
| AUC | **0.989** |
| EER | **4.72%** |

The memory-augmented detector consistently reduces both false positives and false negatives while maintaining strong generalization.

---

# Project Structure

```
Audio-Deepfake-Detection/
│
├── datasets/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── preprocessing/
│
├── feature_extraction/
│   ├── mfcc.py
│   ├── cqcc.py
│   ├── lfcc.py
│   ├── spectral.py
│   ├── prosodic.py
│   ├── physics.py
│   └── xlsr.py
│
├── models/
│   ├── adaptive_fusion.py
│   ├── embedding_network.py
│   ├── classifier.py
│   ├── memory_bank.py
│   └── cross_attention.py
│
├── training/
│
├── evaluation/
│
├── utils/
│
├── configs/
│
├── notebooks/
│
├── README.md
└── requirements.txt
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/audio-deepfake-detection.git

cd audio-deepfake-detection
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Training

```bash
python train.py
```

---

# Evaluation

```bash
python evaluate.py
```

---

# Inference

```bash
python predict.py --audio sample.wav
```

---

# Future Improvements

- End-to-end memory optimization
- Cross-dataset evaluation
- Multilingual support
- Transformer-based memory retrieval
- Real-time deployment
- Edge-device optimization
- ASVspoof 5 evaluation
- Federated continual learning

---

# Citation

If you use this work in your research, please cite:

```bibtex
@article{memory_audio_deepfake_detection,
  title={Memory-Augmented Adaptive Deepfake Audio Detection Using Hybrid Acoustic Features, Physics-Guided Analysis, and XLS-R Embeddings},
  author={Author(s)},
  year={2026}
}
```

---

# Acknowledgements

This work builds upon research in:

- ASVspoof Challenge
- XLS-R
- wav2vec 2.0
- RawNet2
- AASIST
- Physics-guided speech analysis
- Speaker representation learning

---

# License

This project is released under the MIT License.

---

# Contact

For questions, suggestions, or collaborations, please open an issue or submit a pull request.

---

## ⭐ If you find this project useful, consider giving it a star!
