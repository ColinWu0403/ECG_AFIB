import pandas as pd
import numpy as np
import os
import joblib
import neurokit2 as nk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
    global ecg_data, af_labels, file_names

    ecg_data = []
    af_labels = []
    file_names = []
    
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
    
    # Convert to NumPy arrays
    x_data = np.array(ecg_data)
    y_labels = np.array(af_labels)

    return x_data, y_labels

def evaluate_model(mode):    
    x_test, y_test = process_data(normal_dir, af_dir)
    
    x_test_cnn = np.expand_dims(x_test, axis=2) 
    y_test_cnn = tf.keras.utils.to_categorical(y_test, num_classes=2)

    # Check shapes before evaluation
    print(f"x_test_cnn shape: {x_test_cnn.shape}")
    print(f"y_test_cnn shape: {y_test_cnn.shape}")

    if mode == '1':
        print("Evaluating Random Forest model")
        # Predict on the test set
        y_pred = clf.predict(x_test)

        # Evaluate the model
        print("Accuracy on new test data:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    elif mode == '2':
        print("Evaluating CNN model")
        
        try:
            # Evaluate the model
            loss, accuracy = clf.evaluate(x_test_cnn, y_test_cnn)
            print(f"Test Loss: {loss}")
            print(f"Test Accuracy: {accuracy}")
            
            # Predict on the test set
            y_pred_prob = clf.predict(x_test_cnn)
            y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class indices
            print(classification_report(y_test, y_pred))

        except Exception as e:
            print(f"Error during evaluation: {e}")

    return y_pred, y_test

def main():
    # process_data(normal_dir, af_dir)
    y_pred, y_test = evaluate_model(choice)
    
    # Prepare a DataFrame with predictions vs actual labels
    results_df = pd.DataFrame({
        'File Name': file_names,
        'Predicted Label': y_pred,
        'Actual Label': y_test
    })

    results_df['Correct'] = results_df['Predicted Label'] == results_df['Actual Label']

    if choice == '1':
        results_df.to_csv('../data/revlis_data/af_prediction_results_random_forest.csv', index=False)
    elif choice == '2':
        results_df.to_csv('../data/revlis_data/af_prediction_results_cnn.csv', index=False)

if __name__ == "__main__":
    # Define directories
    normal_dir = '../data/revlis_data/AF_TEST/csv/Normal_csv'
    af_dir = '../data/revlis_data/AF_TEST/csv/AF_Arr_csv'

    choice = input("Choose type of model:\n1) Random Forest\n2) CNN: ")
    if choice == '1':
        model_save_path = '../papers/random_forest_model.pkl'
        clf = joblib.load(model_save_path)
    elif choice == '2':
        model_save_path = '../papers/cnn_model.keras'
        clf = tf.keras.models.load_model(model_save_path)
        clf.summary()
        print(clf.input_shape)  # This will show the expected input shape of the model

    # Initialize lists to hold the features and labels
    ecg_data = []
    af_labels = []
    file_names = []
    predictions = []
    
    main()