import os
from pathlib import Path

import librosa
import numpy as np

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def resolve_audio_path(file_path: str) -> str:
    path = Path(file_path)

    if path.exists():
        if path.is_file():
            return str(path)
        if path.is_dir():
            candidates = sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
            if candidates:
                return str(candidates[0])
            raise FileNotFoundError(f"No supported audio files found in directory: {file_path}")

    if path.suffix.lower() in SUPPORTED_EXTENSIONS:
        alt_path = path.parent / path.stem
        if alt_path.exists() and alt_path.is_dir():
            candidates = sorted([p for p in alt_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
            if candidates:
                return str(candidates[0])

    raise FileNotFoundError(
        f"Audio file not found: {file_path}. Provide a valid .wav/.mp3/.flac/.m4a/.ogg file or an emotion folder."
    )


def load_audio(file_path: str):
    resolved_path = resolve_audio_path(file_path)
    audio, sr = librosa.load(resolved_path, sr=None, mono=True)
    return audio, sr


def extract_feature_vector_from_array(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) == 0:
        raise ValueError("Audio array is empty")

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    pitches, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch = np.mean(pitch_values) if pitch_values.size else 0.0

    energy = np.mean(np.square(audio))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))

    feature_vector = np.concatenate(
        [mfcc_mean, mfcc_std, [pitch, energy, spectral_centroid, spectral_rolloff, zero_crossing_rate]]
    ).astype(np.float32)

    return feature_vector


def extract_feature_vector(file_path: str) -> np.ndarray:
    audio, sr = load_audio(file_path)
    return extract_feature_vector_from_array(audio, sr)


def find_audio_files(data_dir: str):
    data_path = Path(data_dir)
    audio_files = []

    for file_path in data_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            audio_files.append(str(file_path))

    return sorted(audio_files)


def build_dataset(data_dir: str):
    label_names = sorted([name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))])
    if not label_names:
        raise ValueError(f"No emotion folders were found in {data_dir}")

    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    features = []
    labels = []

    for label in label_names:
        emotion_dir = Path(data_dir) / label
        for file_path in sorted(emotion_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                features.append(extract_feature_vector(str(file_path)))
                labels.append(label_to_index[label])

    if not features:
        raise ValueError(f"No audio files were found inside {data_dir}")

    return np.vstack(features), np.array(labels, dtype=np.int32), label_to_index
