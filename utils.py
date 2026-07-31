"""
features.py
-----------
Hybrid feature extraction: MFCC, LFCC, CQCC, spectral, prosody, physics
(jitter/shimmer/HNR/formant stability), and XLS-R self-supervised embeddings.
"""

import librosa
import numpy as np
import torch
from scipy.fft import dct

from src.preprocessing import AudioProcessor
from src.utils import (
    HAS_PARSELMOUTH, HAS_SPAFE, HAS_XLSR,
    config, mean_std,
)

if HAS_PARSELMOUTH:
    import parselmouth
    from parselmouth.praat import call as pm_call

if HAS_SPAFE:
    from spafe.features.cqcc import cqcc as spafe_cqcc
    from spafe.features.lfcc import lfcc as spafe_lfcc

if HAS_XLSR:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


class HybridFeatureExtractor:
    def __init__(self, device=None):
        from src.utils import DEVICE
        self.device = device or DEVICE
        self._xlsr_processor = None
        self._xlsr_model = None
        if HAS_XLSR:
            try:
                self._xlsr_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/wav2vec2-xls-r-300m")
                self._xlsr_model = Wav2Vec2Model.from_pretrained(
                    "facebook/wav2vec2-xls-r-300m").to(self.device).eval()
                for p in self._xlsr_model.parameters():
                    p.requires_grad_(False)
            except Exception as e:
                print(f"[warn] could not load XLS-R ({e}); disabling XLS-R branch.")
                self._xlsr_model = None

    # --- Individual feature blocks ---------------------------------------- #
    def extractMFCC(self, y, sr=None):
        sr = sr or config.SAMPLE_RATE
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        return mean_std(mfcc)

    def extractCQCC(self, y, sr=None):
        sr = sr or config.SAMPLE_RATE
        if HAS_SPAFE:
            try:
                feat = spafe_cqcc(y, fs=sr, num_ceps=config.N_CQCC)
                return mean_std(feat.T)
            except Exception:
                pass
        cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=config.HOP_LENGTH, n_bins=config.N_CQCC * 2))
        log_cqt = librosa.amplitude_to_db(cqt)
        cqcc_approx = dct(log_cqt, type=2, axis=0, norm="ortho")[:config.N_CQCC]
        return mean_std(cqcc_approx)

    def extractLFCC(self, y, sr=None):
        sr = sr or config.SAMPLE_RATE
        if HAS_SPAFE:
            try:
                feat = spafe_lfcc(y, fs=sr, num_ceps=config.N_LFCC)
                return mean_std(feat.T)
            except Exception:
                pass
        S = np.abs(librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))
        lin_fb = np.linspace(0, S.shape[0] - 1, config.N_LFCC * 2).astype(int)
        log_lin = librosa.amplitude_to_db(S[lin_fb])
        lfcc_approx = dct(log_lin, type=2, axis=0, norm="ortho")[:config.N_LFCC]
        return mean_std(lfcc_approx)

    def extractSpectralFeatures(self, y, sr=None):
        sr = sr or config.SAMPLE_RATE
        hop = config.HOP_LENGTH
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop).mean(axis=0, keepdims=True)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop).mean(axis=0, keepdims=True)
        rms = librosa.feature.rms(y=y, hop_length=hop)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)
        stacked = np.concatenate([centroid, bandwidth, contrast, rolloff, chroma, rms, zcr], axis=0)
        return mean_std(stacked)

    def extractProsodyFeatures(self, y, sr=None):
        """Pitch, energy contour proxy, pause duration, speaking rate."""
        sr = sr or config.SAMPLE_RATE
        if HAS_PARSELMOUTH:
            try:
                snd = parselmouth.Sound(y, sampling_frequency=sr)
                pitch = snd.to_pitch()
                f0 = pitch.selected_array["frequency"]
                f0 = f0[f0 > 0]
                mean_pitch = float(f0.mean()) if len(f0) else 0.0
                std_pitch = float(f0.std()) if len(f0) else 0.0
                intensity = snd.to_intensity()
                values = intensity.values[0]
                pause_ratio = float((values < (values.mean() - values.std())).mean())
                speaking_rate = float(len(f0) / max(len(values), 1))
                return np.array([mean_pitch, std_pitch, speaking_rate, pause_ratio], dtype=np.float32)
            except Exception:
                pass

        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        mean_pitch = float(f0.mean()) if len(f0) else 0.0
        std_pitch = float(f0.std()) if len(f0) else 0.0
        speaking_rate = float(voiced_flag.mean()) if voiced_flag is not None else 0.0
        rms = librosa.feature.rms(y=y)[0]
        pause_ratio = float((rms < rms.mean() * 0.3).mean())
        return np.array([mean_pitch, std_pitch, speaking_rate, pause_ratio], dtype=np.float32)

    def extractPhysicalFeatures(self, y, sr=None):
        """Physics-guided: jitter, shimmer, HNR, formant stability via Praat."""
        sr = sr or config.SAMPLE_RATE
        if HAS_PARSELMOUTH:
            try:
                snd = parselmouth.Sound(y, sampling_frequency=sr)
                point_process = pm_call(snd, "To PointProcess (periodic, cc)", 50, 500)
                jitter = pm_call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                shimmer = pm_call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                harmonicity = pm_call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
                hnr = pm_call(harmonicity, "Get mean", 0, 0)
                formant = snd.to_formant_burg()
                f1_vals = [formant.get_value_at_time(1, t) for t in np.arange(0, snd.duration, 0.05)]
                f1_vals = [v for v in f1_vals if v == v]  # drop NaNs
                formant_stability = float(np.std(f1_vals)) if len(f1_vals) > 1 else 0.0
                jitter = 0.0 if jitter != jitter else float(jitter)
                shimmer = 0.0 if shimmer != shimmer else float(shimmer)
                hnr = 0.0 if hnr != hnr else float(hnr)
                return np.array([jitter, shimmer, hnr, formant_stability], dtype=np.float32)
            except Exception:
                pass
        return np.zeros(4, dtype=np.float32)

    def extractXLSREmbeddings(self, waveform_1d_tensor):
        """facebook/wav2vec2-xls-r-300m -> mean-pooled 1024-D embedding."""
        if self._xlsr_model is None:
            return np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            inputs = self._xlsr_processor(
                waveform_1d_tensor.numpy(), sampling_rate=config.SAMPLE_RATE, return_tensors="pt"
            ).input_values.to(self.device)
            out = self._xlsr_model(inputs).last_hidden_state  # (1, T, 1024)
            pooled = out.mean(dim=1).squeeze(0).cpu().numpy()
        return pooled.astype(np.float32)

    # --- Combined extraction ------------------------------------------------
    def combineFeaturesFromArray(self, y):
        """Same hybrid extraction as combineFeatures, but starting from an
        already-loaded/processed 1-D numpy waveform instead of a file path.
        Lets us featurize in-memory AUGMENTED waveforms without ever writing
        synthetic audio to disk."""
        wav_tensor = torch.from_numpy(y).float()
        feats = {
            "mfcc": self.extractMFCC(y),
            "cqcc": self.extractCQCC(y),
            "lfcc": self.extractLFCC(y),
            "spectral": self.extractSpectralFeatures(y),
            "prosody": self.extractProsodyFeatures(y),
            "physics": self.extractPhysicalFeatures(y),
        }
        if HAS_XLSR and self._xlsr_model is not None:
            feats["xlsr"] = self.extractXLSREmbeddings(wav_tensor)
        else:
            feats["xlsr"] = np.zeros(0, dtype=np.float32)
        return feats

    def combineFeatures(self, file_path):
        """Full hybrid extraction for one file -> dict of named raw vectors."""
        wav_tensor = AudioProcessor.process(file_path)
        y = wav_tensor.squeeze(0).numpy()
        return self.combineFeaturesFromArray(y)
