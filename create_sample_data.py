import os
from pathlib import Path

import numpy as np
import soundfile as sf

EMOTIONS = ["happy", "sad", "angry", "neutral", "fearful"]
SAMPLE_RATE = 22050
DURATION_SECONDS = 1.0
SAMPLES_PER_EMOTION = 3


def generate_tone(emotion: str, index: int, sample_rate: int, duration: int):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if emotion == "happy":
        freq = 440 + 80 * index
        amplitude = 0.6 + 0.1 * index
    elif emotion == "sad":
        freq = 220 - 20 * index
        amplitude = 0.35 + 0.03 * index
    elif emotion == "angry":
        freq = 700 + 110 * index
        amplitude = 0.8 + 0.05 * index
    elif emotion == "neutral":
        freq = 300 + 10 * index
        amplitude = 0.5 + 0.02 * index
    else:
        freq = 180 + 40 * index
        amplitude = 0.4 + 0.05 * index

    signal = amplitude * np.sin(2 * np.pi * freq * t)
    signal += 0.15 * np.sin(2 * np.pi * (freq * 0.5) * t)
    signal = np.clip(signal, -0.95, 0.95)
    return signal


def main():
    root = Path("data")
    root.mkdir(exist_ok=True)

    for emotion in EMOTIONS:
        folder = root / emotion
        folder.mkdir(exist_ok=True)
        for idx in range(SAMPLES_PER_EMOTION):
            audio = generate_tone(emotion, idx, SAMPLE_RATE, DURATION_SECONDS)
            output_path = folder / f"sample_{idx + 1}.wav"
            sf.write(output_path, audio, SAMPLE_RATE)
            print(f"Created {output_path}")


if __name__ == "__main__":
    main()
