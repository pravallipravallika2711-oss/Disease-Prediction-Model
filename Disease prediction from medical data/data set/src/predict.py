import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diabetes_model.joblib')


def predict_disease(features):
    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    return int(prediction)


if __name__ == '__main__':
    sample = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 169.5,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50,
    }
    result = predict_disease(sample)
    print(f'Prediction: {result}')
    print('0 = No diabetes, 1 = Diabetes')
