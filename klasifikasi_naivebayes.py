import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib

matplotlib.use('TkAgg')

pd.options.mode.chained_assignment = None

# Membaca data
dataframe = pd.read_csv('datasetbaru_sudahnormalisasi.csv')

# Seleksi kolom yang akan digunakan
data = dataframe[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER']]

# Tampilkan data awal
print("data awal".center(75, "="))
print(data)
print("=".center(75, "="))

# Pengecekan missing value
print("Pengecekan missing value".center(75, "="))
print(data.isnull().sum())
print("=".center(75, "="))

# Mendeteksi outlier menggunakan z-score
print('Hasil Outlier'.center(75, "="))
def detect_outlier(data, threshold=3):
    outliers = []
    mean = np.mean(data)
    std = np.std(data)
    
    for yy in data:
        z_score = (yy - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(yy)
    
    return outliers

outliers = {}
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    if col != 'LUNG_CANCER':
        outliers[col] = detect_outlier(data[col])

# Menampilkan hasil outlier
for col, outlier_values in outliers.items():
    if outlier_values:
        print(f"Outlier pada kolom {col}: {outlier_values}")
    else:
        print(f"Tidak ada outlier pada kolom {col}")

print("=".center(75, "=")) 

# Menghapus outlier dari data
def remove_outliers(data, outliers):
    for col, outlier_values in outliers.items():
        if outlier_values:
            data = data[~data[col].isin(outlier_values)]
    return data

data_cleaned = remove_outliers(data, outliers)

# encoding kategori variabel
le = LabelEncoder()
data_cleaned['GENDER'] = le.fit_transform(data_cleaned['GENDER'])
data_cleaned['LUNG_CANCER'] = le.fit_transform(data_cleaned['LUNG_CANCER'])

# Normalisasi data menggunakan metode minmax normalization
min_max_scaler = MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_cleaned[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']])
normalized = pd.DataFrame(np_scaled, columns=['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'])
normalized['LUNG_CANCER'] = data_cleaned['LUNG_CANCER'].values

# Grouping variabel
X = normalized[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']].values
y = normalized['LUNG_CANCER'].values

# Pembagian training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Pemodelan Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)

# Perhitungan confusion matrix
cm = confusion_matrix(y_test, Y_pred)
print('CLASSIFICATION REPORT NAIVE BAYES'.center(75, '='))
print(classification_report(y_test, Y_pred, zero_division=1))

# Akurasi, Precision, Sensitivity, Specificity
accuracy = accuracy_score(y_test, Y_pred)
precision = precision_score(y_test, Y_pred, zero_division=1)
TN = cm[1][1]
FN = cm[1][0]
TP = cm[0][0]
FP = cm[0][1]
sens = TN / (TN + FP) * 100 if (TN + FP) != 0 else 0
spec = TP / (TP + FN) * 100 if (TP + FN) != 0 else 0

print(f'Akurasi: {accuracy * 100:.2f}%')
print(f'Sensitivity: {sens:.2f}%')
print(f'Specificity: {spec:.2f}%')
print(f'Precision: {precision * 100:.2f}%')

# Menampilkan Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

