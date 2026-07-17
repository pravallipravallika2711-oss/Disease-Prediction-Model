import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio_features import extract_feature_vector_from_array


def test_extract_feature_vector_from_array_returns_expected_shape():
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, endpoint=False))
    features = extract_feature_vector_from_array(audio, 16000)

    assert features.shape[0] == 31
