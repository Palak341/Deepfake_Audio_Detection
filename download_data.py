"""
utils.py
--------
Global configuration, environment/device setup, and small shared helper
functions used across the rest of the pipeline (preprocessing, features,
dataset, model, memory, evaluation, train, inference).

Everything that used to be bare module-level constants in the notebook
("Section 2: Parameters") now lives here so every other module can do:

    from src.utils import config, DEVICE, set_seed
"""

import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Optional third-party dependency flags
# --------------------------------------------------------------------------- #
try:
    import parselmouth  # noqa: F401
    HAS_PARSELMOUTH = True
except Exception as e:
    print(f"[warn] parselmouth unavailable ({e}); prosody/physics features will use librosa fallback.")
    HAS_PARSELMOUTH = False

try:
    from spafe.features.cqcc import cqcc as spafe_cqcc  # noqa: F401
    from spafe.features.lfcc import lfcc as spafe_lfcc  # noqa: F401
    HAS_SPAFE = True
except Exception as e:
    print(f"[warn] spafe unavailable ({e}); CQCC/LFCC will use librosa-based approximations.")
    HAS_SPAFE = False

try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor  # noqa: F401
    HAS_XLSR = True
except Exception as e:
    print(f"[warn] transformers unavailable ({e}); XLS-R branch will be disabled.")
    HAS_XLSR = False

try:
    import faiss  # noqa: F401
    HAS_FAISS = True
except Exception as e:
    print(f"[warn] faiss unavailable ({e}); memory bank will fall back to brute-force numpy search.")
    HAS_FAISS = False


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info():
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config:
    """Central configuration object (mirrors the notebook's parameters cell)."""

    # --- Data locations ---------------------------------------------------
    AUDIO_DIR = os.environ.get(
        "AUDIO_DIR",
        "data/release_in_the_wild",  # local project 'data/' dir by default
    )
    META_CSV = os.environ.get("META_CSV", "data/meta.csv")

    OUTPUT_DIR = "outputs"
    MODELS_DIR = "models"

    MEMORY_DB_PATH = os.path.join(OUTPUT_DIR, "memory_bank.sqlite")
    FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "memory.index")
    EMBED_MODEL_PATH = os.path.join(MODELS_DIR, "rat_embedding_model.pt")
    HEAD_MODEL_PATH = os.path.join(MODELS_DIR, "rat_personalized_head.pt")
    HISTORY_PLOT_PATH = os.path.join(OUTPUT_DIR, "rat_training_history.png")
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # --- Audio ---------------------------------------------------------- #
    SAMPLE_RATE = 16000
    DURATION = 4.0
    N_SAMPLES = int(SAMPLE_RATE * DURATION)
    TRIM_SILENCE = True

    # --- Spectral / feature extraction ----------------------------------- #
    N_FFT = 1024
    HOP_LENGTH = 256
    N_MELS = 64
    N_MFCC = 40
    N_CQCC = 40
    N_LFCC = 40
    FIXED_FRAMES = N_SAMPLES // HOP_LENGTH + 1

    DIM_MFCC = N_MFCC * 2
    DIM_CQCC = N_CQCC * 2
    DIM_LFCC = N_LFCC * 2
    DIM_SPECTRAL = 7 * 2
    DIM_PROSODY = 4
    DIM_PHYSICS = 4
    DIM_XLSR = 1024 if HAS_XLSR else 0

    FEATURE_GROUP_DIMS = {
        "mfcc": DIM_MFCC,
        "cqcc": DIM_CQCC,
        "lfcc": DIM_LFCC,
        "spectral": DIM_SPECTRAL,
        "prosody": DIM_PROSODY,
        "physics": DIM_PHYSICS,
        "xlsr": DIM_XLSR,
    }
    TOTAL_RAW_DIM = sum(FEATURE_GROUP_DIMS.values())

    # --- Embedding model --------------------------------------------------
    EMBED_DIM = 256
    PROJ_HIDDEN = 512

    # --- Labels ------------------------------------------------------------
    LABEL_MAP = {"bona-fide": 0, "spoof": 1}
    LABEL_TO_SUBFOLDER = {"bona-fide": "real", "spoof": "fake"}
    INVERSE_LABEL_MAP = {0: "bona-fide (real)", 1: "spoof (deepfake)"}

    # --- Training ------------------------------------------------------- #
    BATCH_SIZE = 32
    EPOCHS = 12
    LEARNING_RATE = 1e-3
    TRIPLET_MARGIN = 0.3
    LAMBDA_TRIPLET = 0.5
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    RANDOM_SEED = 42
    MAX_SAMPLES = None
    MAX_CLIPS_PER_SPEAKER = 400

    USE_AUGMENTATION = True
    AUGMENT_TARGET_RATIO = 1.0
    AUGMENT_SEED = 123

    # --- Personalization head training ----------------------------------- #
    P_EPOCHS = 15
    P_PATIENCE = 4
    MEMORY_DROPOUT_PROB = 0.35

    # --- Memory bank -------------------------------------------------------
    MEMORY_TOPK = 5
    CONF_THRESH_BOOTSTRAP = 0.99
    CONF_THRESH_UPDATE = 0.98
    SIM_THRESH_UPDATE = 0.80
    MAX_ENTRIES_PER_SPEAKER = 50
    RESET_MEMORY_DB = True
    FEATURE_CACHE_VERSION = "v3"

    ADAPTIVE_TOPK_MIN = 3
    ADAPTIVE_TOPK_MAX = 10

    AGE_DECAY_PER_YEAR = 0.4
    MIN_AGE_WEIGHT = 0.15
    SECONDS_PER_YEAR = 365.25 * 24 * 3600

    PROTOTYPE_TRIGGER_RAW_COUNT = 30
    PROTOTYPE_TARGET_COUNT = 20
    PROTOTYPE_MINIBATCH_THRESHOLD = 1000
    PHYSICS_SIM_THRESH = 0.55


config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Small numeric / feature helpers
# --------------------------------------------------------------------------- #
def mean_std(mat):
    """Concatenate per-coefficient mean and std along the last (time) axis."""
    mat = np.asarray(mat, dtype=np.float32)
    return np.concatenate([mat.mean(axis=-1), mat.std(axis=-1)]).astype(np.float32)


def featurize_row_to_vec(feats, dim_xlsr=None):
    """Flatten a dict of named raw feature vectors into a single 1-D vector."""
    dim_xlsr = config.DIM_XLSR if dim_xlsr is None else dim_xlsr
    return np.concatenate([
        feats["mfcc"], feats["cqcc"], feats["lfcc"],
        feats["spectral"], feats["prosody"], feats["physics"],
        feats["xlsr"] if feats["xlsr"].shape[0] > 0 else np.zeros(dim_xlsr, dtype=np.float32),
    ])


def build_group_slices(feature_group_dims, dim_xlsr):
    """Compute (start, end) slice indices for each named feature group inside
    the flattened raw feature vector produced by `featurize_row_to_vec`."""
    slices = {}
    offset = 0
    for name, d in feature_group_dims.items():
        dd = d if d > 0 else dim_xlsr
        slices[name] = (offset, offset + dd)
        offset += dd
    return slices, offset


# --------------------------------------------------------------------------- #
# Contrastive (triplet) pairing helpers -- used to train the general
# embedding model in train.py
# --------------------------------------------------------------------------- #
def create_positive_pairs(labels, speakers):
    """For each anchor, pick another sample with the same label (same speaker
    preferred) as the positive. Returns index tensor aligned with the batch."""
    idx = torch.arange(len(labels))
    pos_idx = idx.clone()
    labels_np = labels.cpu().numpy()
    speakers_np = np.asarray(speakers)
    for i in range(len(labels)):
        same_speaker = np.where((speakers_np == speakers_np[i]) & (labels_np == labels_np[i].item()))[0]
        same_speaker = same_speaker[same_speaker != i]
        if len(same_speaker) > 0:
            pos_idx[i] = int(np.random.choice(same_speaker))
        else:
            same_label = np.where(labels_np == labels_np[i].item())[0]
            same_label = same_label[same_label != i]
            pos_idx[i] = int(np.random.choice(same_label)) if len(same_label) else i
    return pos_idx


def create_negative_pairs(labels, speakers):
    """Negative = opposite class (spoof for a genuine anchor, or vice versa),
    falling back to a different speaker of the same class if needed."""
    idx = torch.arange(len(labels))
    neg_idx = idx.clone()
    labels_np = labels.cpu().numpy()
    speakers_np = np.asarray(speakers)
    for i in range(len(labels)):
        opp_label = np.where(labels_np != labels_np[i].item())[0]
        if len(opp_label) > 0:
            neg_idx[i] = int(np.random.choice(opp_label))
        else:
            diff_speaker = np.where(speakers_np != speakers_np[i])[0]
            neg_idx[i] = int(np.random.choice(diff_speaker)) if len(diff_speaker) else i
    return neg_idx


def build_loo_memory_batch(idxs, embeddings, labels, speakers, embed_dim, k, drop_prob=0.0):
    """For each sample, retrieve up to k other GENUINE same-speaker embeddings
    (leave-one-out) as its memory context. Candidates are ranked by cosine
    similarity to the current sample's embedding and the top-k most similar
    are kept -- matching how AdaptiveMemoryRetriever selects references at
    inference time, rather than taking whichever k genuine samples happen to
    appear first in dataset order. Returns padded tensor + mask.

    `drop_prob`: with this probability, a sample's memory context is withheld
    entirely (as if it were a brand-new / cold-start speaker), even though
    real same-speaker references were available in this training batch.
    Without this, training only ever showed the personalization head a richly
    populated context, but at deployment time the ConfidenceGate's strict
    thresholds mean most speakers are queried many times before they ever
    accumulate gated memory -- so the head was constantly fed an empty
    context it had never been trained to handle. Randomly dropping context
    during training closes that train/inference mismatch.
    """
    B = len(idxs)
    mem = np.zeros((B, k, embed_dim), dtype=np.float32)
    mask = np.zeros((B, k), dtype=bool)
    for bi, i in enumerate(idxs):
        if drop_prob > 0.0 and np.random.rand() < drop_prob:
            continue
        same_speaker_genuine = np.where(
            (speakers == speakers[i]) & (labels == 0) & (np.arange(len(labels)) != i)
        )[0]
        if len(same_speaker_genuine) > 0:
            query = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-9)
            cand = embeddings[same_speaker_genuine]
            cand_norm = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-9)
            sims = cand_norm @ query
            top_order = np.argsort(-sims)[:k]
            chosen = same_speaker_genuine[top_order]
        else:
            chosen = same_speaker_genuine[:k]
        for j, src_idx in enumerate(chosen):
            mem[bi, j] = embeddings[src_idx]
            mask[bi, j] = True
    return torch.from_numpy(mem), torch.from_numpy(mask)


def print_saved_artifacts(output_dir=None):
    output_dir = output_dir or config.OUTPUT_DIR
    print("Saved artifacts in", os.path.abspath(output_dir), ":")
    for f in sorted(os.listdir(output_dir)):
        print(" -", f)
