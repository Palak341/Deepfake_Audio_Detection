"""
download_data.py
-----------------
One-time helper to fetch the "In the Wild" audio deepfake dataset from
Kaggle into data/, so you don't have to manually download/unzip a ~5GB
archive.

Usage:
    pip install kagglehub
    python download_data.py

Requires a Kaggle account + API token (kagglehub will prompt you to log
in / paste a token the first time you run this — see
https://github.com/Kaggle/kagglehub#authenticate).

After this finishes, data/ will contain:
    data/release_in_the_wild/real/*.wav
    data/release_in_the_wild/fake/*.wav
    data/meta.csv
"""

import os
import shutil

import kagglehub

DATASET = "abdallamohamed312/in-the-wild-audio-deepfake"
DEST_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    print(f"Downloading '{DATASET}' via kagglehub (this can take a while, ~5GB)...")
    cache_path = kagglehub.dataset_download(DATASET)
    print(f"Downloaded to kagglehub cache: {cache_path}")

    os.makedirs(DEST_DIR, exist_ok=True)

    # kagglehub caches the dataset elsewhere on disk; copy/symlink the bits
    # we need into ./data so AUDIO_DIR / META_CSV defaults in src/utils.py
    # work out of the box.
    for name in os.listdir(cache_path):
        src = os.path.join(cache_path, name)
        dst = os.path.join(DEST_DIR, name)
        if os.path.exists(dst):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    print(f"\nDone. Dataset is ready under: {DEST_DIR}")
    print("Set these before running train.py if your folder names differ:")
    print(f'  export AUDIO_DIR="{DEST_DIR}/release_in_the_wild"')
    print(f'  export META_CSV="{DEST_DIR}/meta.csv"')


if __name__ == "__main__":
    main()
