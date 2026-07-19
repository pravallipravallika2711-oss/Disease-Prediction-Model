# 🩺 Disease Prediction from Medical Data using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based Disease Prediction System developed using Python and Scikit-learn. The system predicts whether a patient is likely to have diabetes based on medical information such as glucose level, blood pressure, BMI, age, insulin level, and other health-related features.

The project demonstrates the complete Machine Learning workflow, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, model comparison, and prediction.

This project was developed as part of an internship to understand how Artificial Intelligence and Machine Learning are applied in healthcare for early disease detection.

---

# 🎯 Objectives

* Predict diseases using structured medical data.
* Perform data preprocessing and feature engineering.
* Train and compare multiple Machine Learning algorithms.
* Evaluate model performance using various metrics.
* Save the best-performing model for future predictions.
* Build a reusable prediction system.

---

# 📂 Project Structure

```text
Disease_Prediction_Model/
│
├── dataset/
│   └── diabetes.csv
│
├── models/
│   ├── diabetes_model.pkl
│   └── scaler.pkl
│
├── notebook/
│   └── disease_prediction.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

This project uses the **Pima Indians Diabetes Dataset**.

### Features

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

### Target

* Outcome

  * 0 → Non-Diabetic
  * 1 → Diabetic

---

# ⚙️ Technologies Used

* Python 3.x
* Jupyter Notebook
* VS Code
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Flask (Optional)

---

# 📚 Machine Learning Algorithms

The following classification algorithms are implemented and compared:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest Classifier

The model with the best performance is selected and saved.

---

# 🔍 Exploratory Data Analysis (EDA)

The project performs several EDA steps including:

* Dataset overview
* Shape of dataset
* Data types
* Missing value analysis
* Statistical summary
* Class distribution
* Correlation heatmap
* Histograms for numerical features

These analyses help understand the dataset before training the models.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

* Loading dataset
* Checking missing values
* Splitting features and target
* Train-Test Split
* Feature Scaling using StandardScaler
* Preparing data for Machine Learning models

---

# 🧠 Model Training

Three classification models are trained:

### 1. Logistic Regression

A simple and efficient baseline classification algorithm.

### 2. Support Vector Machine (SVM)

Suitable for binary classification problems with good accuracy.

### 3. Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy.

---

# 📈 Model Evaluation

Each model is evaluated using:

* Accuracy Score
* Classification Report
* Precision
* Recall
* F1-Score
* Confusion Matrix

The best-performing model is selected based on overall accuracy and classification metrics.

---

# 💾 Model Saving

The trained Random Forest model is saved using Joblib.

Saved files:

```
models/
│
├── diabetes_model.pkl
└── scaler.pkl
```

These files are later used for prediction without retraining the model.

---

# 🚀 How to Run the Project

## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/Disease_Prediction_Model.git
```

---

## Step 2: Move into Project Folder

```bash
cd Disease_Prediction_Model
```

---

## Step 3: Create Virtual Environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

---

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5: Train Model

```bash
python src/train.py
```

---

## Step 6: Run Prediction

```bash
python src/predict.py
```

---

## Step 7 (Optional): Run Flask Application

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

# 📊 Sample Prediction

Example Input

| Feature                    | Value |
| -------------------------- | ----: |
| Pregnancies                |     2 |
| Glucose                    |   148 |
| Blood Pressure             |    72 |
| Skin Thickness             |    35 |
| Insulin                    |     0 |
| BMI                        |  33.6 |
| Diabetes Pedigree Function | 0.627 |
| Age                        |    50 |

Example Output

```
Prediction:
Diabetic
```

---

# 📷 Screenshots

Add screenshots of:

* Dataset Preview
* Correlation Heatmap
* Histograms
* Accuracy Comparison
* Confusion Matrix
* Prediction Output
* Flask Web Application (Optional)

---

# 📈 Future Improvements

* Add XGBoost Classifier
* Hyperparameter Tuning
* Cross Validation
* Feature Selection
* Web-based User Interface
* Streamlit Dashboard
* Deep Learning Models
* Support for Multiple Diseases
* Cloud Deployment
* REST API Integration

---

# 📖 Learning Outcomes

Through this project, I learned:

* Healthcare Machine Learning concepts
* Data preprocessing techniques
* Exploratory Data Analysis
* Feature Scaling
* Classification Algorithms
* Model Evaluation
* Saving and Loading Models
* Predictive Analytics
* Machine Learning Project Structure
* GitHub Project Management

---

# 📌 Requirements

```
Python 3.x
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Flask
Jupyter Notebook
```

Install using:

```bash
pip install -r requirements.txt
```

---

# 🤝 Contributing

Contributions, improvements, and suggestions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

# 📄 License

This project is created for educational and internship purposes.

---

# 👩‍💻 Author

**Pravallika**

Machine Learning Enthusiast

Python Developer

GitHub: https://github.com/your-username

LinkedIn: https://www.linkedin.com/in/your-profile/

---

## ⭐ Support

If you found this project helpful, please consider giving it a **⭐ Star** on GitHub. It helps others discover the project and motivates future improvements.
