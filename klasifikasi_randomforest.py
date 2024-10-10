import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

pd.options.mode.chained_assignment = None

# Membaca data
dataframe = pd.read_csv('datasetbaru_sudahnormalisasi.csv')
# Seleksi kolom yang akan digunakan
data = dataframe[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER']]

# Tampilkan data awal
print("Data Awal".center(75, "="))
print(data)
print("=".center(75, "="))

# Pengecekan missing value
print("Pengecekan Missing Value".center(75, "="))
print(data.isnull().sum())
print("=".center(75, "="))

# Encoding kategori variabel
le = LabelEncoder()
data['GENDER'] = le.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])

# Normalisasi data menggunakan metode MinMax Normalization
min_max_scaler = MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']])
normalized = pd.DataFrame(np_scaled, columns=['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'])
normalized['LUNG_CANCER'] = data['LUNG_CANCER'].values

# Grouping variabel
X = normalized[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']].values
y = normalized['LUNG_CANCER'].values

# Pembagian training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Mencari parameter terbaik untuk Random Forest
param_grid = {'n_estimators': [100, 200, 300, 400, 500]}  # Coba 100 hingga 500 pohon
rf_classifier = RandomForestClassifier(random_state=0)

# Melakukan pencarian grid
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Menampilkan hasil pencarian
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Score: {grid_search.best_score_}')

# Menggunakan model terbaik untuk prediksi
best_rf_classifier = grid_search.best_estimator_
Y_pred = best_rf_classifier.predict(X_test)

# Perhitungan confusion matrix
cm = confusion_matrix(y_test, Y_pred)
print('CLASSIFICATION REPORT RANDOM FOREST'.center(75, '='))
print(classification_report(y_test, Y_pred, zero_division=1))

# Akurasi, Precision, Sensitivity, Specificity
accuracy = accuracy_score(y_test, Y_pred)
precision = precision_score(y_test, Y_pred, zero_division=1)
f1 = f1_score(y_test, Y_pred, zero_division=1)
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
print(f'F1-Score: {f1 * 100:.2f}%')

# Menampilkan Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()
