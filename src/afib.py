import pandas as pd
import numpy as np
import os
import joblib
import neurokit2 as nk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def process_data():
    # Load the profiles file to get the filenames and AFib status
    profiles = pd.read_csv('../data/revlis_data/profiles.csv')

    # Iterate through each entry in profiles
    for _, row in profiles.iterrows():
        filename = row['filename']
        filename = str(filename + ".csv")
        af_similarity = row['AF_Similarity']

        # Construct the full path to the ECG file
        file_path = os.path.join('../data/revlis_data/csv', filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            # print(f"File {file_path} does not exist. Skipping.")
            continue

        # Read the ECG file
        ecg_df = pd.read_csv(file_path)

        # Extract the ECG signal from Lead2
        ecg_signal = ecg_df['Lead2'].values

        # Clean the ECG signal using neurokit2
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=1000)

        # Process the cleaned ECG signal using neurokit2
        ecg_processed, info = nk.ecg_process(ecg_cleaned, sampling_rate=1000)
        hrv_metrics = nk.hrv_time(ecg_processed, sampling_rate=1000)

        # Extract HRV features
        sdnn = hrv_metrics['HRV_SDNN'].values[0]
        rmssd = hrv_metrics['HRV_RMSSD'].values[0]
        meanNN = hrv_metrics['HRV_MeanNN'].values[0]

        # Example: Calculate some simple features from the ECG signal
        mean_val = np.mean(ecg_cleaned)
        std_val = np.std(ecg_cleaned)
        max_val = np.max(ecg_cleaned)
        min_val = np.min(ecg_cleaned)

        # Append the features and the label to the lists
        ecg_data.append([mean_val, std_val, max_val, min_val, sdnn, rmssd, meanNN])
        
        af_threshold = 0.9  # Threshold for AFib similarity set to 90 %
        
        af_labels.append(1 if af_similarity > af_threshold else 0)
        # print([mean_val, std_val, max_val, min_val, sdnn, rmssd, meanNN])

def train_model():
    # Convert the lists to NumPy arrays
    x = np.array(ecg_data)
    y = np.array(af_labels)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)
    return clf, x_train, x_test, y_test, y_train, y_test

def save_model(clf):
    # Save the trained model
    model_save_path = '../papers/random_forest_model.pkl'
    joblib.dump(clf, model_save_path)

def main():
    process_data()
    clf, x_train, x_test, y_test, y_train, y_test = train_model()
    # Predict on the test set
    y_pred = clf.predict(x_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Initialize lists to hold the features and labels
    ecg_data = []
    af_labels = []
    
    main()