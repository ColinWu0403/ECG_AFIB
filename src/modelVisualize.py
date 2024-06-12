import wfdb
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import tensorflow as tf


# Load the model
def load_model(model_path):
    model = joblib.load(model_path)
    return model


# Preprocess the data for prediction
def preprocess_data(df):
    # Check if the DataFrame is empty after filtering
    if df.empty:
        print(df)
        raise ValueError("No data left after filtering. Adjust the filter criteria.")

    # Normalize the data
    features = ['hrv_sdnn', 'hrv_rmssd', "hrv_mean", 'cv', "heart_rate_std", "heart_rate_mean", "sd1", "sd2"]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Add sampling_rate column
    df['sampling_rate'] = df['sampling_rate'].astype(int)

    return df, df[features], df[features].join(df['sampling_rate'])


# Load the data for a specific patient
def load_data(file_path, specified_record):
    df = pd.read_csv(file_path)
    df['record_name'] = df['record_name'].astype(str)
    df = df[df['record_name'] == str(specified_record)]
    actual_afib_annotations = df['num_AFIB_annotations'].values  # Extract AFib annotations
    return df, actual_afib_annotations


# Plot heart rate mean with predictions
def plot_heart_rate_with_predictions(df, predictions):
    plt.figure(figsize=(15, 6))

    time = df['start_time'].values  # Assuming 'start_time' is in seconds
    heart_rate_mean = df['heart_rate_mean'].values

    for i in range(len(predictions) - 1):
        start_time = time[i]
        end_time = time[i + 1]
        pred = predictions[i]

        if pred == 1:
            color = 'red'  # AFib
            line_width = 3  # thicker line for AFib
        else:
            color = 'green'  # Normal
            line_width = 1

        plt.plot([start_time, end_time], [heart_rate_mean[i], heart_rate_mean[i + 1]], color=color,
                 linewidth=line_width)

    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate Mean')
    plt.title('Heart Rate Mean with AFib Predictions ' + record_name)
    plt.show()


# Plot hrv_sdnn with predictions
def plot_hrv_sdnn_with_predictions(df, predictions):
    plt.figure(figsize=(15, 6))

    time = df['start_time'].values  # Assuming 'start_time' is in seconds
    hrv_sdnn = df['hrv_sdnn'].values

    for i in range(len(predictions) - 1):
        start_time = time[i]
        end_time = time[i + 1]
        pred = predictions[i]

        if pred == 1:
            color = 'red'  # AFib
            line_width = 3  # thicker line for AFib
        else:
            color = 'green'  # Normal
            line_width = 1

        plt.plot([start_time, end_time], [hrv_sdnn[i], hrv_sdnn[i + 1]], color=color, linewidth=line_width)

    plt.xlabel('Time (s)')
    plt.ylabel('HRV SDNN')
    plt.title('HRV SDNN with AFib Predictions ' + record_name)
    plt.show()


# Plot cv with predictions
def plot_cv_with_predictions(df, predictions):
    plt.figure(figsize=(15, 6))

    time = df['start_time'].values  # Assuming 'start_time' is in seconds
    cv = df['cv'].values

    for i in range(len(predictions) - 1):
        start_time = time[i]
        end_time = time[i + 1]
        pred = predictions[i]

        if pred == 1:
            color = 'red'  # AFib
            line_width = 3  # thicker line for AFib
        else:
            color = 'green'  # Normal
            line_width = 1

        plt.plot([start_time, end_time], [cv[i], cv[i + 1]], color=color, linewidth=line_width)

    plt.xlabel('Time (s)')
    plt.ylabel('CV')
    plt.title('CV with AFib Predictions ' + record_name)
    plt.show()


# Plot hrv_rmssd with predictions
def plot_hrv_rmssd_with_predictions(df, predictions):
    plt.figure(figsize=(15, 6))

    time = df['start_time'].values  # Assuming 'start_time' is in seconds
    hrv_rmssd = df['hrv_rmssd'].values

    for i in range(len(predictions) - 1):
        start_time = time[i]
        end_time = time[i + 1]
        pred = predictions[i]

        if pred == 1:
            color = 'red'  # AFib
            line_width = 3  # thicker line for AFib
        else:
            color = 'green'  # Normal
            line_width = 1

        plt.plot([start_time, end_time], [hrv_rmssd[i], hrv_rmssd[i + 1]], color=color, linewidth=line_width)

    plt.xlabel('Time (s)')
    plt.ylabel('RMSSD')
    plt.title('RMSSD with AFib Predictions ' + record_name)
    plt.show()


# Plot ECG signal with markers for Afib predictions
def plot_ecg_with_predictions(ecg_signal, predictions, actual_afib, sampling_rate, start_time, end_time):
    print("Predictions: " + str(np.count_nonzero(predictions)))
    print(predictions)
    print("Actual AFIB: " + str(np.count_nonzero(actual_afib)))
    print(actual_afib)
    plt.figure(figsize=(15, 6))

    # Calculate the indices corresponding to the start and end times
    start_index = int(start_time * sampling_rate)
    end_index = min(int(end_time * sampling_rate), len(ecg_signal))  # Ensure end_index does not exceed signal length

    # Extract the ECG signal and its corresponding time array for the specified interval
    ecg_interval = ecg_signal[start_index:end_index]
    time_interval = np.arange(start_time, start_time + len(ecg_interval) / sampling_rate,
                              1 / sampling_rate) / 60  # Convert to minutes

    # Plot ECG signal
    plt.plot(time_interval, ecg_interval, color='black')

    # Add markers for Afib predictions
    interval_length = 10  # 10-second intervals
    start_prediction_index = int(start_time / interval_length)
    end_prediction_index = int(end_time / interval_length)

    # Extract the relevant predictions for the interval
    relevant_predictions = predictions[start_prediction_index:end_prediction_index]
    relevant_actual_afib = actual_afib[start_prediction_index:end_prediction_index]

    for i, (pred, actual) in enumerate(zip(relevant_predictions, relevant_actual_afib)):
        interval_start = start_time + i * interval_length  # Interval start in seconds
        interval_end = start_time + (i + 1) * interval_length  # Interval end in seconds

        if interval_start >= end_time:  # Stop if interval exceeds signal length
            break

        interval_start_min = interval_start / 60  # Convert interval start to minutes
        interval_end_min = interval_end / 60  # Convert interval end to minutes

        if pred == 1:
            if actual == 1:
                plt.axvspan(interval_start_min, interval_end_min, color='red', alpha=1)  # Mark as red if actual and predict
            else:
                plt.axvspan(interval_start_min, interval_end_min, color='yellow', alpha=0.5)  # Mark as yellow only predict
        else:
            if actual == 1:
                plt.axvspan(interval_start_min, interval_end_min, color='blue', alpha=0.75)  # Mark as blue if actual
            else:
                plt.axvspan(interval_start_min, interval_end_min, color='green', alpha=0.05)  # Mark as green if normal

    # Convert start and end times to hours
    start_time_hours = start_time / 3600
    end_time_hours = end_time / 3600

    main_title = f"ECG Signal with Afib Predictions: {record_name}"
    subtitle1 = f"Start Time: {start_time / 60:.2f} minutes ({start_time_hours:.2f} hours)"
    subtitle2 = f"End Time: {end_time / 60:.2f} minutes ({end_time_hours:.2f} hours)"

    # Include start time in the title (converted to minutes)
    plt.xlabel('Time (minutes)')
    plt.ylabel('ECG Signal')
    plt.title(f'{main_title}\n{subtitle1}\n{subtitle2}')
    plt.show()


# Function to load the model conditionally
def load_model_type(model_path, model_type):
    if model_type == "1":
        model = load_model(model_path)
    elif model_type in ["2", "3", "6"]:
        model = tf.keras.models.load_model(model_path)
    elif model_type == "5":
        model = xgb.Booster()
        model.load_model(model_path)
    else:
        raise ValueError("Unsupported model type.")
    return model


# Function to predict conditionally based on the model type
def predict(model, model_type, features):
    if model_type == "1":
        return model.predict(features)
    elif model_type in ["2", "3"]:
        features = np.array(features).reshape((features.shape[0], 1, features.shape[1]))  # LSTM and CNN
        predictions = model.predict(features)
        return np.argmax(predictions, axis=1)
    elif model_type == "5":
        d_matrix = xgb.DMatrix(features)
        return model.predict(d_matrix)
    elif model_type == "6":
        features = np.array(features).reshape((features.shape[0], features.shape[1], 1))  # ResNet
        predictions = model.predict(features)
        return np.argmax(predictions, axis=1)
    else:
        raise ValueError("Unsupported model type.")


record_name = "4043"


# Main function to run the predictions and plot the results
def main():
    print("Models:")
    print("1. Random Forest Classifier")
    print("2. LSTM")
    print("3. CNN")
    print("4. SVM")
    print("5. Gradient Boosting (XGBoost)")
    print("6. ResNet")
    print("0. Exit")

    model_type = str(input("Enter the type of model: "))
    if model_type == "0":
        return
    elif model_type == "1":
        model_path = '../models/random_forest_model.pkl'
    elif model_type == "2":
        model_path = '../models/LSTM_model.keras'
    elif model_type == "3":
        model_path = '../models/CNN_model.keras'
    elif model_type == "4":
        model_path = '../models/SVM_model.pkl'
    elif model_type == "5":
        model_path = '../models/XGBoost_model.model'
    elif model_type == "6":
        model_path = '../models/resnet_model.keras'
    else:
        print("Error: model does not exist.")
        return

    # Load your trained model
    model = load_model_type(model_path, model_type)

    while True:
        print("Choose the type of plot:")
        print("1. Heart Rate Mean")
        print("2. HRV SDNN")
        print("3. HRV RMSSD")
        print("4. CV (Coefficient of Variation)")
        print("5. ECG Signal Visualization")
        print("0. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            plot_function = plot_heart_rate_with_predictions
        elif choice == "2":
            plot_function = plot_hrv_sdnn_with_predictions
        elif choice == "3":
            plot_function = plot_hrv_rmssd_with_predictions
        elif choice == "4":
            plot_function = plot_cv_with_predictions
        elif choice == "5":
            plot_function = plot_ecg_with_predictions
        elif choice == "0":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please choose again.")
            continue

        # Load the ECG data
        df, actual_afib_annotations = load_data('../data/afdb_data.csv', record_name)

        # Preprocess the data
        df, features, features_sample = preprocess_data(df)

        # Predict AFib in each interval
        predictions = predict(model, model_type, features)

        if choice != "5":
            # Plot based on the selected type
            plot_function(df, predictions)
        else:
            record_path = str("../afdb/0" + record_name)
            # Load the ECG data
            signals, fields = wfdb.rdsamp(record_path)  # Assuming you have the PhysioNet database downloaded
            ecg_signal = signals[:, 0]  # Extract the first channel (ECG signal)
            sampling_rate = int(features_sample["sampling_rate"].iloc[0])  # Sampling rate from the features DataFrame

            # Define the time interval for plotting (30 minutes)
            interval_length_minutes = 30
            interval_length_seconds = interval_length_minutes * 60
            total_duration_seconds = len(ecg_signal) / sampling_rate
            num_intervals = int(total_duration_seconds / interval_length_seconds)

            for i in range(num_intervals):
                start_time = i * interval_length_seconds
                end_time = (i + 1) * interval_length_seconds

                # Plot ECG with Afib predictions for each 30-minute interval
                plot_ecg_with_predictions(ecg_signal, predictions, actual_afib_annotations, sampling_rate, start_time, end_time)


if __name__ == "__main__":
    main()
