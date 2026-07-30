# Hybrid Feature Extraction

This document describes the hybrid acoustic feature representation used in the proposed **Memory-Augmented Adaptive Audio Deepfake Detection** framework.

---

# 📥 Download Precomputed Features

The precomputed hybrid features, XLS-R embeddings, pretrained models, and additional resources are hosted on **Google Drive** because they exceed GitHub's file size limits.

## Google Drive

🔗 **https://drive.google.com/drive/folders/1T8HzCZ8-yL4kgAFyQyiJodvOy5fsJyPg?usp=sharing**

The Google Drive folder contains:

- In-the-Wild extracted features
- ASVspoof 2019 extracted features
- XLS-R embeddings
- Speaker metadata
- Labels
- Pretrained models


---

# Directory Structure

After downloading, organize the files as follows:

```text
datasets/
│
├── in_the_wild/
│   ├── train/
│   ├── validation/
│   └── test/
│
└── asvspoof2019/
    ├── train/
    ├── development/
    └── evaluation/


```

> **Note**
>
> The repository contains the complete feature extraction pipeline. The downloadable files are provided only to reproduce the experimental results without re-extracting features from the original datasets.

---

# Hybrid Feature Overview

| Feature Group | Dimension | Description |
|--------------|----------:|-------------|
| MFCC | 80 | Mel-Frequency Cepstral Coefficients |
| CQCC | 80 | Constant-Q Cepstral Coefficients |
| LFCC | 80 | Linear Frequency Cepstral Coefficients |
| Spectral Features | 14 | Frequency-domain descriptors |
| Prosodic Features | 4 | Pitch and energy statistics |
| Physics-guided Features | 4 | Jitter, Shimmer, HNR, Formant Stability |
| XLS-R Embeddings | 1024 | Self-supervised multilingual speech embeddings |
| **Total** | **1286** | Hybrid Feature Vector |

...
