import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

fileLocation = '../mlprak_uts/dataset.csv'
df = pd.read_csv(fileLocation)

print("\n Data yang telah dibaca:")
print(df)
print("============================================================")

# mendeteksi missvalue
print("\n Deteksi data yang missing Value:")
missvalue = df.isna().sum()
print(missvalue)
print("============================================================")

print("\n Baris dengan missing values:")
print(df[df.isna().any(axis=1)])
print("============================================================")

# outlier pada kolom "AGE"
def detect_outlier(AGE):  
    outliers = []  
    threshold = 3  
    mean_AGE = np.mean(AGE)
    std_AGE = np.std(AGE)  

    for y in AGE:  
        z_score = (y - mean_AGE) / std_AGE  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['AGE'].values)  
print("\n Outlier data points in 'AGE':", outlier_datapoints)

# outlier pada kolom "SMOKING"
def detect_outlier(SMOKING):  
    outliers = []  
    threshold = 3  
    mean_SMOKING = np.mean(SMOKING)
    std_SMOKING = np.std(SMOKING)  

    for y in SMOKING:  
        z_score = (y - mean_SMOKING) / std_SMOKING  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['SMOKING'].values)  
print("\n Outlier data points in 'SMOKING':", outlier_datapoints)

# outlier pada kolom "YELLOW_FINGERS"
def detect_outlier(YELLOW_FINGERS):  
    outliers = []  
    threshold = 3  
    mean_YELLOW_FINGERS = np.mean(YELLOW_FINGERS)
    std_YELLOW_FINGERS = np.std(YELLOW_FINGERS)  

    for y in YELLOW_FINGERS:  
        z_score = (y - mean_YELLOW_FINGERS) / std_YELLOW_FINGERS  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['YELLOW_FINGERS'].values)  
print("\n Outlier data points in 'YELLOW_FINGERS':", outlier_datapoints)

# outlier pada kolom "ANXIETY"
def detect_outlier(ANXIETY):  
    outliers = []  
    threshold = 3  
    mean_ANXIETY = np.mean(ANXIETY)
    std_ANXIETY = np.std(ANXIETY)  

    for y in ANXIETY:  
        z_score = (y - mean_ANXIETY) / std_ANXIETY  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['ANXIETY'].values)  
print("\n Outlier data points in 'ANXIETY':", outlier_datapoints)

# outlier pada kolom "PEER_PRESSURE"
def detect_outlier(PEER_PRESSURE):  
    outliers = []  
    threshold = 3  
    mean_PEER_PRESSURE = np.mean(PEER_PRESSURE)
    std_PEER_PRESSURE = np.std(PEER_PRESSURE)  

    for y in PEER_PRESSURE:  
        z_score = (y - mean_PEER_PRESSURE) / std_PEER_PRESSURE  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['PEER_PRESSURE'].values)  
print("\n Outlier data points in 'PEER_PRESSURE':", outlier_datapoints)

# outlier pada kolom "CHRONIC_DISEASE"
def detect_outlier(CHRONIC_DISEASE):  
    outliers = []  
    threshold = 3  
    mean_CHRONIC_DISEASE = np.mean(CHRONIC_DISEASE)
    std_CHRONIC_DISEASE = np.std(CHRONIC_DISEASE)  

    for y in CHRONIC_DISEASE:  
        z_score = (y - mean_CHRONIC_DISEASE) / std_CHRONIC_DISEASE  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['CHRONIC_DISEASE'].values)  
print("\n Outlier data points in 'CHRONIC_DISEASE':", outlier_datapoints)

# outlier pada kolom "FATIGUE"
def detect_outlier(FATIGUE):  
    outliers = []  
    threshold = 3  
    mean_FATIGUE = np.mean(FATIGUE)
    std_FATIGUE = np.std(FATIGUE)  

    for y in FATIGUE:  
        z_score = (y - mean_FATIGUE) / std_FATIGUE  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['FATIGUE'].values)  
print("\n Outlier data points in 'FATIGUE':", outlier_datapoints)

# outlier pada kolom "ALLERGY"
def detect_outlier(ALLERGY):  
    outliers = []  
    threshold = 3  
    mean_ALLERGY = np.mean(ALLERGY)
    std_ALLERGY = np.std(ALLERGY)  

    for y in ALLERGY:  
        z_score = (y - mean_ALLERGY) / std_ALLERGY  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['ALLERGY'].values)  
print("\n Outlier data points in 'ALLERGY':", outlier_datapoints)

# outlier pada kolom "WHEEZING"
def detect_outlier(WHEEZING):  
    outliers = []  
    threshold = 3  
    mean_WHEEZING = np.mean(WHEEZING)
    std_WHEEZING = np.std(WHEEZING)  

    for y in WHEEZING:  
        z_score = (y - mean_WHEEZING) / std_WHEEZING  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['WHEEZING'].values)  
print("\n Outlier data points in 'WHEEZING':", outlier_datapoints)

# outlier pada kolom "ALCOHOL_CONSUMING"
def detect_outlier(ALCOHOL_CONSUMING):  
    outliers = []  
    threshold = 3  
    mean_ALCOHOL_CONSUMING = np.mean(ALCOHOL_CONSUMING)
    std_ALCOHOL_CONSUMING = np.std(ALCOHOL_CONSUMING)  

    for y in ALCOHOL_CONSUMING:  
        z_score = (y - mean_ALCOHOL_CONSUMING) / std_ALCOHOL_CONSUMING  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['ALCOHOL_CONSUMING'].values)  
print("\n Outlier data points in 'ALCOHOL_CONSUMING':", outlier_datapoints)

# outlier pada kolom "COUGHING"
def detect_outlier(COUGHING):  
    outliers = []  
    threshold = 3  
    mean_COUGHING = np.mean(COUGHING)
    std_COUGHING = np.std(COUGHING)  

    for y in COUGHING:  
        z_score = (y - mean_COUGHING) / std_COUGHING  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['COUGHING'].values)  
print("\n Outlier data points in 'COUGHING':", outlier_datapoints)

# outlier pada kolom "SHORTNESS_OF_BREATH"
def detect_outlier(SHORTNESS_OF_BREATH):  
    outliers = []  
    threshold = 3  
    mean_SHORTNESS_OF_BREATH = np.mean(SHORTNESS_OF_BREATH)
    std_SHORTNESS_OF_BREATH = np.std(SHORTNESS_OF_BREATH)  

    for y in SHORTNESS_OF_BREATH:  
        z_score = (y - mean_SHORTNESS_OF_BREATH) / std_SHORTNESS_OF_BREATH  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['SHORTNESS_OF_BREATH'].values)  
print("\n Outlier data points in 'SHORTNESS_OF_BREATH':", outlier_datapoints)

# outlier pada kolom "SWALLOWING_DIFFICULTY"
def detect_outlier(SWALLOWING_DIFFICULTY):  
    outliers = []  
    threshold = 3  
    mean_SWALLOWING_DIFFICULTY = np.mean(SWALLOWING_DIFFICULTY)
    std_SWALLOWING_DIFFICULTY = np.std(SWALLOWING_DIFFICULTY)  

    for y in SWALLOWING_DIFFICULTY:  
        z_score = (y - mean_SWALLOWING_DIFFICULTY) / std_SWALLOWING_DIFFICULTY  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['SWALLOWING_DIFFICULTY'].values)  
print("\n Outlier data points in 'SWALLOWING_DIFFICULTY':", outlier_datapoints)

# outlier pada kolom "CHEST_PAIN"
def detect_outlier(CHEST_PAIN):  
    outliers = []  
    threshold = 3  
    mean_CHEST_PAIN = np.mean(CHEST_PAIN)
    std_CHEST_PAIN = np.std(CHEST_PAIN)  

    for y in CHEST_PAIN:  
        z_score = (y - mean_CHEST_PAIN) / std_CHEST_PAIN  
        if np.abs(z_score) > threshold:  
            outliers.append(y)  
    return outliers  
outlier_datapoints = detect_outlier(df['CHEST_PAIN'].values)  
print("\n Outlier data points in 'CHEST_PAIN':", outlier_datapoints)

# inisialisasi df_cleaned dan membuat variabel untuk mengecek (columns_check)
df_cleaned = df.copy()
columns_to_check = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                    'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
                    'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
                    'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

# loop pada setiap kolom dan hapus outlier
for column in columns_to_check:  
    outliers = detect_outlier(df_cleaned[column].values)  
    
    # Buat mask untuk baris yang bukan outlier  
    mask = ~df_cleaned[column].isin(outliers)  
    df_cleaned = df_cleaned[mask]  

print("\nData setelah menghapus outliers:")  
print(df_cleaned)
print("============================================================")

print("\n Missing values setelah semua replace, drop, dan outlier removal:")
print(df_cleaned.isna().sum())
print("============================================================")

# menginisialisasi scaler
scaler = MinMaxScaler()

# kolom numerik yang dinormalisasi
numeric_columns = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                    'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
                    'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 
                    'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

# normalisasi kolom numerik
df_cleaned[numeric_columns] = scaler.fit_transform(df_cleaned[numeric_columns])

print("\nData setelah normalisasi Min-Max:")
print(df_cleaned)
print("============================================================")

# menyimpan data yang sudah dinormalisasi ke file CSV baru
output_file_normalized = '../mlprak_uts/dataset_sudahnormalisasi.csv'
df_cleaned.to_csv(output_file_normalized, index=False)
print(f"\nData yang telah dinormalisasi disimpan ke {output_file_normalized}")