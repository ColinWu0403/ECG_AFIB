import pandas as pd
import numpy as np
import os
import neurokit2 as nk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define directories
normal_dir = '../data/revlis_data/AF_TEST/csv/Normal_csv'
af_dir = '../data/revlis_data/AF_TEST/csv/AF_Arr_csv'

# Initialize lists to hold the features and labels
ecg_data = []
af_labels = []

# Function to extract features from ECG file
def extract_features(file_path):
    ecg_df = pd.read_csv(file_path)
    ecg_signal = ecg_df['Lead2'].values
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=1000)
    ecg_processed, info = nk.ecg_process(ecg_cleaned, sampling_rate=1000)
    hrv_metrics = nk.hrv_time(ecg_processed, sampling_rate=1000)

    sdnn = hrv_metrics['HRV_SDNN'].values[0]
    rmssd = hrv_metrics['HRV_RMSSD'].values[0]
    meanNN = hrv_metrics['HRV_MeanNN'].values[0]

    mean_val = np.mean(ecg_cleaned)
    std_val = np.std(ecg_cleaned)
    max_val = np.max(ecg_cleaned)
    min_val = np.min(ecg_cleaned)

    return [mean_val, std_val, max_val, min_val, sdnn, rmssd, meanNN]

# Process Normal files
for filename in os.listdir(normal_dir):
    file_path = os.path.join(normal_dir, filename)
    if os.path.isfile(file_path):
        features = extract_features(file_path)
        ecg_data.append(features)
        af_labels.append(0)  # Label 0 for Normal

# Process AF files
for filename in os.listdir(af_dir):
    file_path = os.path.join(af_dir, filename)
    if os.path.isfile(file_path):
        features = extract_features(file_path)
        ecg_data.append(features)
        af_labels.append(1)  # Label 1 for AF

# Convert the lists to NumPy arrays
x = np.array(ecg_data)
y = np.array(af_labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Predict on the test set
y_pred = clf.predict(x_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
