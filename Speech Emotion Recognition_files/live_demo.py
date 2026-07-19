import argparse
import sounddevice as sd
import numpy as np

from src.audio_features import extract_feature_vector_from_array
from src.model_utils import load_model, predict_probabilities


def parse_args():
    parser = argparse.ArgumentParser(description="Live microphone emotion demo")
    parser.add_argument("--model-path", default="models/emotion_model.pkl")
    parser.add_argument("--metadata-path", default="models/emotion_labels.json")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def main():
    args = parse_args()
    model, index_to_label = load_model(args.model_path, args.metadata_path)
    print("Recording for 2 seconds...")
    recording = sd.rec(int(args.duration * args.sample_rate), samplerate=args.sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = recording[:, 0]
    features = extract_feature_vector_from_array(audio, args.sample_rate)
    probabilities = predict_probabilities(model, features)
    label_scores = sorted(zip(index_to_label.values(), probabilities), key=lambda item: item[1], reverse=True)

    print("Prediction:")
    for label, score in label_scores:
        print(f"{label}: {score * 100:.1f}%")


if __name__ == "__main__":
    main()
