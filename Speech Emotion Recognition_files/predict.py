import argparse

from src.audio_features import extract_feature_vector
from src.model_utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Predict an emotion from an audio file")
    parser.add_argument("--model-path", required=True, help="Path to the trained model")
    parser.add_argument("--metadata-path", required=True, help="Path to the label mapping metadata")
    parser.add_argument("--audio-path", required=True, help="Path to the audio file to classify")
    return parser.parse_args()


def main():
    args = parse_args()
    model, index_to_label = load_model(args.model_path, args.metadata_path)
    features = extract_feature_vector(args.audio_path)
    prediction_index = model.predict([features])[0]
    print(index_to_label[int(prediction_index)])


if __name__ == "__main__":
    main()
