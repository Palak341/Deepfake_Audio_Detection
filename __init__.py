"""
dataset.py
----------
Metadata loading, speaker-disjoint train/val/test splitting, hybrid feature
cache construction (with train-only waveform augmentation), and the
PyTorch Dataset/DataLoader wrapping the cached feature vectors.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.features import HybridFeatureExtractor
from src.preprocessing import AudioAugmentor, AudioProcessor
from src.utils import config, featurize_row_to_vec


# --------------------------------------------------------------------------- #
# Metadata loading + speaker-disjoint splitting
# --------------------------------------------------------------------------- #
def load_metadata():
    raw = pd.read_csv(config.META_CSV)
    raw.columns = raw.columns.str.strip()
    for col in ["file", "label"]:
        if col not in raw.columns:
            raise KeyError(f"Expected column {col!r} not found. Got: {raw.columns.tolist()}")

    files = raw["file"].astype(str).str.strip()
    labels = raw["label"].astype(str).str.strip().replace({"real": "bona-fide", "fake": "spoof"})
    df = pd.DataFrame({"file": files, "label": labels})

    if "speaker" in raw.columns:
        df["speaker"] = raw["speaker"].astype(str).str.strip()
    else:
        print("[warn] No 'speaker' column — falling back to row-level split.")
        df["speaker"] = df["file"]

    df = df[df["label"].isin(config.LABEL_MAP)].reset_index(drop=True)
    subfolders = df["label"].map(config.LABEL_TO_SUBFOLDER)
    df["filepath"] = (config.AUDIO_DIR + os.sep + subfolders + os.sep + df["file"])
    df["target"] = df["label"].map(config.LABEL_MAP).astype(int)

    exists = df["filepath"].map(os.path.exists)
    if (~exists).sum() > 0:
        print(f"[warn] {(~exists).sum():,} files missing on disk — dropping them.")
    df = df[exists].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("0 usable audio files found — check AUDIO_DIR / META_CSV.")

    if config.MAX_SAMPLES is not None:
        parts = []
        for _, g in df.groupby("target"):
            n_target = min(len(g), config.MAX_SAMPLES // 2)
            speaker_order = g.groupby("speaker").size().sort_values(ascending=False).index
            selected, count = [], 0
            for spk in speaker_order:
                if count >= n_target:
                    break
                spk_rows = g[g["speaker"] == spk]
                take_n = min(len(spk_rows), config.MAX_CLIPS_PER_SPEAKER, n_target - count)
                spk_rows = spk_rows.sample(n=take_n, random_state=config.RANDOM_SEED)
                selected.append(spk_rows)
                count += take_n
            part = pd.concat(selected).sample(frac=1, random_state=config.RANDOM_SEED)
            parts.append(part.head(n_target))
        df = pd.concat(parts).reset_index(drop=True)

    print(f"Usable clips: {len(df):,} | speakers: {df['speaker'].nunique():,}")
    print(df["label"].value_counts().to_string())
    return df


def make_splits(df):
    """Speaker-disjoint 70/15/15 split (no speaker appears in >1 split).

    Greedily assigns whole speakers (largest clip-count first, an LPT /
    longest-processing-time bin-balancing heuristic) to whichever split is
    currently furthest below its sample-count target. This lands the
    realized split much closer to the intended 70/15/15 BY SAMPLES than a
    naive GroupShuffleSplit (which balances speaker *counts*, not sample
    counts, and drifts badly when speakers have wildly different clip
    counts) while keeping every speaker in exactly one split.
    """
    speaker_counts = df.groupby("speaker").size()
    total = speaker_counts.sum()
    train_frac = 1.0 - config.VAL_SPLIT - config.TEST_SPLIT
    targets = {
        "train": train_frac * total,
        "val": config.VAL_SPLIT * total,
        "test": config.TEST_SPLIT * total,
    }
    current = {"train": 0, "val": 0, "test": 0}
    assignment = {}

    order = speaker_counts.sample(frac=1.0, random_state=config.RANDOM_SEED) \
        .sort_values(ascending=False, kind="stable").index
    for spk in order:
        cnt = int(speaker_counts[spk])
        remaining = {k: targets[k] - current[k] for k in targets}
        dest = max(remaining, key=remaining.get)
        assignment[spk] = dest
        current[dest] += cnt

    df = df.copy()
    df["_split"] = df["speaker"].map(assignment)
    train_df = df[df["_split"] == "train"].drop(columns="_split").reset_index(drop=True)
    val_df = df[df["_split"] == "val"].drop(columns="_split").reset_index(drop=True)
    test_df = df[df["_split"] == "test"].drop(columns="_split").reset_index(drop=True)

    sets = [set(train_df["speaker"]), set(val_df["speaker"]), set(test_df["speaker"])]
    for (n1, s1), (n2, s2) in [(("train", sets[0]), ("val", sets[1])),
                                (("train", sets[0]), ("test", sets[2])),
                                (("val", sets[1]), ("test", sets[2]))]:
        overlap = s1 & s2
        if overlap:
            raise RuntimeError(f"Speaker leakage: {n1} & {n2} share {len(overlap)} speakers!")
    print("Speaker overlap check: 0 shared speakers across train/val/test (verified)")
    realized = {k: v / total for k, v in current.items()}
    print(f"Realized sample-count split -> train={realized['train']:.1%}  "
          f"val={realized['val']:.1%}  test={realized['test']:.1%}  (target 70/15/15)")
    return train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# Hybrid feature cache construction
# --------------------------------------------------------------------------- #
def build_feature_cache(split_df, cache_path, feature_extractor, augment=False):
    """`augment` (only ever True for the train split) oversamples the
    minority (spoof) class with waveform-augmented copies of existing spoof
    clips -- noise/pitch/stretch/gain via AudioAugmentor -- until spoof
    count is within AUGMENT_TARGET_RATIO of bona-fide count. Val/test are
    never augmented, so evaluation numbers stay honest."""
    if os.path.exists(cache_path):
        print(f"[cache hit] {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], data["speakers"]

    X, y, speakers = [], [], []
    n = len(split_df)
    for i, row in split_df.iterrows():
        feats = feature_extractor.combineFeatures(row["filepath"])
        X.append(featurize_row_to_vec(feats))
        y.append(row["target"])
        speakers.append(row["speaker"])
        if (i + 1) % 150 == 0 or (i + 1) == n:
            print(f"  extracted {i + 1}/{n}")

    if augment and config.USE_AUGMENTATION:
        y_arr_tmp = np.array(y)
        n_bona = int((y_arr_tmp == config.LABEL_MAP["bona-fide"]).sum())
        n_spoof = int((y_arr_tmp == config.LABEL_MAP["spoof"]).sum())
        n_needed = max(0, int(n_bona * config.AUGMENT_TARGET_RATIO) - n_spoof)
        print(f"[augment] train spoof={n_spoof} bona-fide={n_bona} "
              f"-> generating {n_needed} augmented spoof samples to reach target ratio "
              f"{config.AUGMENT_TARGET_RATIO}")
        if n_needed > 0:
            spoof_rows = split_df[split_df["target"] == config.LABEL_MAP["spoof"]].reset_index(drop=True)
            aug_rng = np.random.RandomState(config.AUGMENT_SEED)
            for j in range(n_needed):
                src_row = spoof_rows.iloc[aug_rng.randint(len(spoof_rows))]
                wav_tensor = AudioProcessor.process(src_row["filepath"])
                y_orig = wav_tensor.squeeze(0).numpy()
                y_aug = AudioAugmentor.augment(y_orig, aug_rng)
                feats = feature_extractor.combineFeaturesFromArray(y_aug)
                X.append(featurize_row_to_vec(feats))
                y.append(src_row["target"])
                speakers.append(src_row["speaker"])
                if (j + 1) % 150 == 0 or (j + 1) == n_needed:
                    print(f"  augmented {j + 1}/{n_needed}")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    speakers = np.array(speakers)
    np.savez_compressed(cache_path, X=X, y=y, speakers=speakers)
    return X, y, speakers


def build_all_feature_caches(train_df, val_df, test_df, feature_extractor=None):
    """Convenience wrapper building (and/or loading) the train/val/test
    feature caches together."""
    feature_extractor = feature_extractor or HybridFeatureExtractor()
    print("Building feature caches (this runs the full hybrid extractor once per split)...")

    X_train, y_train, spk_train = build_feature_cache(
        train_df, os.path.join(config.OUTPUT_DIR, f"feat_train_{config.FEATURE_CACHE_VERSION}.npz"),
        feature_extractor, augment=True)
    X_val, y_val, spk_val = build_feature_cache(
        val_df, os.path.join(config.OUTPUT_DIR, f"feat_val_{config.FEATURE_CACHE_VERSION}.npz"),
        feature_extractor)
    X_test, y_test, spk_test = build_feature_cache(
        test_df, os.path.join(config.OUTPUT_DIR, f"feat_test_{config.FEATURE_CACHE_VERSION}.npz"),
        feature_extractor)

    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
    return (X_train, y_train, spk_train), (X_val, y_val, spk_val), (X_test, y_test, spk_test)


# --------------------------------------------------------------------------- #
# PyTorch Dataset / DataLoader
# --------------------------------------------------------------------------- #
class TabularRATDataset(Dataset):
    """Wraps a cached (X, y, speakers) triple of hybrid feature vectors."""

    def __init__(self, X, y, speakers):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.speakers = speakers

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.speakers[idx]


def build_dataloaders(train_data, val_data, test_data, batch_size=None):
    batch_size = batch_size or config.BATCH_SIZE
    X_train, y_train, spk_train = train_data
    X_val, y_val, spk_val = val_data
    X_test, y_test, spk_test = test_data

    train_loader = DataLoader(TabularRATDataset(X_train, y_train, spk_train),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabularRATDataset(X_val, y_val, spk_val),
                             batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TabularRATDataset(X_test, y_test, spk_test),
                              batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
