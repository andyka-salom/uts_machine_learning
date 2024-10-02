import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

fileLocation = '../mlprak_uts/dataset_sudahnormalisasi.csv'
df = pd.read_csv(fileLocation)
df.columns = ['GENDER','AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
              'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
              'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
              'SWALLOWING_DIFFICULTY', 'CHEST_PAIN','LUNG_CANCER']

print("\n df yang telah dibaca:")
print(df)
print("============================================================")

print("\n Deteksi df yang missing Value:")
missvalue = df.isna().sum()
print(missvalue)
print("============================================================")

# encoding kategori variabel
le = LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])

# grouping df antara df training dan df target
X = df[['GENDER','AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
        'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
        'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
        'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']] 
y = df['LUNG_CANCER'] 

# Split df training dan df testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Membuat decision tree
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# Membuat prediksi dari df x
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print("============================================================\n")

# Laporan klasifikasi dan Confusion Matrix
class_report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='pink')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# cm
conf_matrix_df = pd.DataFrame(conf_matrix, index=['True Class 0', 'True Class 1'], columns=['Pred Class 0', 'Pred Class 1'])
print("\nConfusion Matrix:")
print(conf_matrix_df)
print("============================================================\n")

# Ekstrak precision, recall, dan F1 Score
precision_class_1 = class_report['1']['precision']
precision_class_0 = class_report['0']['precision']
recall_class_1 = class_report['1']['recall']
recall_class_0 = class_report['0']['recall']
f1_class_1 = class_report['1']['f1-score']
f1_class_0 = class_report['0']['f1-score']

# Membuat tabel ringkasan dari hasil prediksi
summary_table = pd.DataFrame({
    '': ['Prediksi Class 1', 'Prediksi Class 0', 'Class Recall'],
    'True Class 1': [conf_matrix[1, 1], conf_matrix[1, 0], f"{recall_class_1:.2%}"],
    'True Class 0': [conf_matrix[0, 1], conf_matrix[0, 0], f"{recall_class_0:.2%}"],
    'Class Precision': [f"{precision_class_1:.2%}", f"{precision_class_0:.2%}", ""],
    'F1 Score': [f"{f1_class_1:.2%}", f"{f1_class_0:.2%}", ""]
})

print("\nSummary Table:")
print(tabulate(summary_table, headers='keys', tablefmt='grid', showindex=False))
print("============================================================")

# Plot Decision Tree
plt.figure(figsize=(20, 15))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Class 0', 'Class 1'], fontsize=10)
plt.show()