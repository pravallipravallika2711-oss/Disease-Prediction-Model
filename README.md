# Emotion Recognition from Speech

This project builds a simple speech emotion recognition pipeline that extracts acoustic features from audio recordings and trains a machine-learning model to classify emotions such as happy, sad, angry, neutral, and fearful.

## Project structure

- `src/audio_features.py` – loads audio files and extracts MFCC, pitch, energy, and spectral features.
- `src/model_utils.py` – trains and saves the classifier.
- `train.py` – trains the model from a folder of labeled audio files.
- `predict.py` – predicts an emotion for a single audio file.

## Dataset layout

Organize your data like this:

```text
data/
  happy/
    sample1.wav
    sample2.wav
  sad/
    sample1.wav
  angry/
    sample1.wav
  neutral/
    sample1.wav
  fearful/
    sample1.wav
```

Each folder name is treated as the emotion label.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --data-dir data --model-path models/emotion_model.pkl --metadata-path models/emotion_labels.json
```

## Prediction

```bash
python predict.py --model-path models/emotion_model.pkl --metadata-path models/emotion_labels.json --audio-path path/to/audio.wav
```

## Live microphone demo

```bash
python live_demo.py --model-path models/emotion_model.pkl --metadata-path models/emotion_labels.json
```

## Streamlit app

```bash
streamlit run app.py
```

## Notes

This starter version uses handcrafted features and a compact neural network classifier. It is a practical baseline that can be extended with CNNs or LSTMs for stronger accuracy on larger datasets such as RAVDESS, TESS, or EMO-DB.
