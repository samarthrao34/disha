"""MFCC feature extraction for the RAVDESS speech baseline."""

from pathlib import Path

import librosa
import numpy as np


SAMPLE_RATE = 16000
N_MFCC = 40


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    if audio.size == 0:
        raise ValueError(f"audio contains no usable signal: {path}")
    return audio.astype(np.float32)


def extract_mfcc_statistics(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if audio is None or audio.size == 0:
        raise ValueError("audio cannot be empty")
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=160,
        n_mels=64,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta.mean(axis=1),
            delta.std(axis=1),
            delta2.mean(axis=1),
            delta2.std(axis=1),
        ]
    )
    return features.astype(np.float32)


def extract_file_features(path: str | Path) -> np.ndarray:
    return extract_mfcc_statistics(load_audio(path))
