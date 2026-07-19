import streamlit as st
import joblib
import pandas as pd
import os

# Set page config
st.set_page_config(page_title="Disease Prediction", layout="centered")

# Title and description
st.title("🏥 Disease Prediction Model")
st.markdown("### Predict Diabetes Risk Based on Medical Data")
st.write("Enter patient medical information below to predict diabetes risk.")

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'diabetes_model.joblib')
model = joblib.load(MODEL_PATH)

# Create input fields in columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", 0, 17, 0)
    glucose = st.slider("Glucose Level", 44, 199, 100)
    blood_pressure = st.slider("Blood Pressure", 24, 122, 72)
    skin_thickness = st.slider("Skin Thickness", 7, 99, 20)

with col2:
    insulin = st.slider("Insulin Level", 14, 846, 80)
    bmi = st.slider("BMI", 18.2, 67.1, 25.0)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.078, 2.420, 0.5)
    age = st.slider("Age", 21, 81, 35)

# Create prediction button
if st.button("🔍 Predict", use_container_width=True):
    # Prepare input data
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age,
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display result
    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ **HIGH RISK**: Diabetes is likely to occur")
    else:
        st.success("✅ **LOW RISK**: No diabetes prediction")
    
    # Show input summary
    st.markdown("### Patient Data Summary")
    st.dataframe(input_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Model Information:**
- Algorithm: Logistic Regression
- Accuracy: 71%
- Dataset: Diabetes Dataset (768 patients)
- Note: This is a prediction tool and should not replace professional medical advice.
""")
