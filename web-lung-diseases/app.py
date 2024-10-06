from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load and prepare the model once when the application starts
def load_model():
    # Load dataset 
    dataframe = pd.read_csv('dataset_sudahnormalisasi.csv')

    # Prepare features and labels
    data = dataframe[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                      'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
                      'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
                      'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER']]

    # Label encoding for categorical features
    le = LabelEncoder()
    data['GENDER'] = le.fit_transform(data['GENDER'])
    data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])

    # Normalize features
    scaler = MinMaxScaler()
    features = data.drop('LUNG_CANCER', axis=1)
    labels = data['LUNG_CANCER']
    scaled_features = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=0)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)

    return model, le, scaler

model, label_encoder, scaler = load_model()

@app.route('/landing_page')
def landing_page():
    return render_template('index.html')

# Fungsi untuk klasifikasi berdasarkan input pengguna
def classify(input_data):
    # Load dataset
    try:
        dataframe = pd.read_csv('dataset_sudahnormalisasi.csv')
    except FileNotFoundError:
        return "Dataset tidak ditemukan", None

    # Prepare features and labels
    data = dataframe[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                      'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
                      'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
                      'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER']]

    # Label encoding for categorical features
    le = LabelEncoder()
    data['GENDER'] = le.fit_transform(data['GENDER'])
    data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])

    # Normalize features
    scaler = MinMaxScaler()
    features = data.drop('LUNG_CANCER', axis=1)
    labels = data['LUNG_CANCER']
    scaled_features = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=0)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)

    # Normalize the user input
    input_df = pd.DataFrame([input_data], columns=features.columns)

    # Re-fit LabelEncoder with all possible values
    all_genders = ['M', 'F']  # Add all possible values 
    le.fit(all_genders)
    input_df['GENDER'] = le.transform(input_df['GENDER'])  # Encode user input
    scaled_input = scaler.transform(input_df)

    # Predict based on user input
    prediction = model.predict(scaled_input)

    # Use the model to predict the test data for confusion matrix and classification report
    y_pred = model.predict(X_test)

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred)

    # Generate Confusion Matrix Plot
    plt.figure(figsize=(6, 4))
    plt.matshow(cm, cmap='Blues', alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j, y=i, s=cm[i, j], va='center', ha='center')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return classification, plot_url, prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify_page():
    if request.method == 'POST':
        # Collect form data
        form_data = {
            'GENDER': request.form['GENDER'],
            'AGE': float(request.form['AGE']),
            'SMOKING': int(request.form['SMOKING']),
            'YELLOW_FINGERS': int(request.form['YELLOW_FINGERS']),
            'ANXIETY': int(request.form['ANXIETY']),
            'PEER_PRESSURE': int(request.form['PEER_PRESSURE']),
            'CHRONIC_DISEASE': int(request.form['CHRONIC_DISEASE']),
            'FATIGUE': int(request.form['FATIGUE']),
            'ALLERGY': int(request.form['ALLERGY']),
            'WHEEZING': int(request.form['WHEEZING']),
            'ALCOHOL_CONSUMING': int(request.form['ALCOHOL_CONSUMING']),
            'COUGHING': int(request.form['COUGHING']),
            'SHORTNESS_OF_BREATH': int(request.form['SHORTNESS_OF_BREATH']),
            'SWALLOWING_DIFFICULTY': int(request.form['SWALLOWING_DIFFICULTY']),
            'CHEST_PAIN': int(request.form['CHEST_PAIN'])
        }

        # Classify user input
        classification_report, plot_url, prediction = classify(form_data)
        
        result = "Lung Cancer Detected" if prediction == 1 else "No Lung Cancer Detected"
        
        return render_template('classify.html', classification=classification_report, plot_url=plot_url, result=result)
    return render_template('classify.html', classification=None)

if __name__ == '__main__':
    app.run(debug=True)