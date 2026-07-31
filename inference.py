numpy
pandas
matplotlib
scikit-learn
scipy

torch
torchaudio
librosa

# Optional feature-extraction backends (the pipeline degrades gracefully
# to librosa-based fallbacks if any of these are unavailable):
spafe               # CQCC / LFCC
praat-parselmouth   # jitter / shimmer / HNR / formants (prosody + physics)
transformers        # facebook/wav2vec2-xls-r-300m embeddings

# Optional fast nearest-neighbor backend for the speaker memory bank
# (falls back to brute-force numpy cosine similarity if unavailable):
faiss-cpu
