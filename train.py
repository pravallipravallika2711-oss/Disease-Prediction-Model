import argparse
from pathlib import Path

from src.audio_features import build_dataset
from src.model_utils import evaluate_model, prepare_training_split, save_model, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a speech emotion recognition model")
    parser.add_argument("--data-dir", required=True, help="Folder containing emotion subfolders with audio files")
    parser.add_argument("--model-path", default="models/emotion_model.pkl", help="Where to save the trained model")
    parser.add_argument("--metadata-path", default="models/emotion_labels.json", help="Where to save the label mapping")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    X, y, label_mapping = build_dataset(str(data_dir))

    X_train, X_test, y_train, y_test = prepare_training_split(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

    save_model(model, args.model_path, args.metadata_path, label_mapping)
    print(f"Training complete. Test accuracy: {accuracy:.2f}")
    print(f"Model saved to {args.model_path}")
    print(f"Metadata saved to {args.metadata_path}")


if __name__ == "__main__":
    main()
