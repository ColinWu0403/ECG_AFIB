import pandas as pd
import numpy as np
import os
import joblib
import neurokit2 as nk
from sklearn.metrics import classification_report, accuracy_score

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

def process_data(normal_dir, af_dir):
    # Process Normal files
    for filename in os.listdir(normal_dir):
        file_path = os.path.join(normal_dir, filename)
        if os.path.isfile(file_path):
            features = extract_features(file_path)
            ecg_data.append(features)
            af_labels.append(0)  # Label 0 for Normal
            file_names.append(filename)

    # Process AF files
    for filename in os.listdir(af_dir):
        file_path = os.path.join(af_dir, filename)
        if os.path.isfile(file_path):
            features = extract_features(file_path)
            ecg_data.append(features)
            af_labels.append(1)  # Label 1 for AF
            file_names.append(filename)

def evaluate_model():
    # Convert the lists to NumPy arrays
    x_test = np.array(ecg_data)
    y_test = np.array(af_labels)

    # Predict on the test set
    y_pred = clf.predict(x_test)

    # Evaluate the model
    print("Accuracy on new test data:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred, y_test

def main():
    process_data(normal_dir, af_dir)
    y_pred, y_test = evaluate_model()
    
    # Prepare a DataFrame with predictions vs actual labels
    results_df = pd.DataFrame({
        'File Name': file_names,
        'Predicted Label': y_pred,
        'Actual Label': y_test
    })

    results_df['Correct'] = results_df['Predicted Label'] == results_df['Actual Label']

    results_df.to_csv('../data/revlis_data/af_prediction_results.csv', index=False)

if __name__ == "__main__":
    # Define directories
    normal_dir = '../data/revlis_data/AF_TEST/csv/Normal_csv'
    af_dir = '../data/revlis_data/AF_TEST/csv/AF_Arr_csv'

    model_save_path = '../papers/random_forest_model.pkl'
    clf = joblib.load(model_save_path)

    # Initialize lists to hold the features and labels
    ecg_data = []
    af_labels = []
    file_names = []
    predictions = []
    
    main()