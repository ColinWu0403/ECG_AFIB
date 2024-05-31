import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Load the model
def load_model():
    model = joblib.load('../models/random_forest_model.pkl')
    return model

# Preprocess the data for prediction
def preprocess_data(df):
    # Check if the DataFrame is empty after filtering
    if df.empty:
        print(df)
        raise ValueError("No data left after filtering. Adjust the filter criteria.")

    # Normalize the data
    features = ['hrv_sdnn', 'hrv_rmssd', "hrv_mean", 'cv', "num_N_annotations"]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, features

# Load the data for a specific patient
def load_data(file_path, specified_record):
    df = pd.read_csv(file_path)
    df['record_name'] = df['record_name'].astype(str)
    df = df[df['record_name'] == str(specified_record)]
    return df

record_name = "4043"

# Plot heart rate mean with predictions
def plot_heart_rate_with_predictions(df, predictions, interval=10):
    plt.figure(figsize=(15, 6))

    time = df['start_time'].values  # Assuming 'start_time' is in seconds
    heart_rate_mean = df['heart_rate_mean'].values

    for i in range(len(predictions) - 1):
        start_time = time[i]
        end_time = time[i + 1]
        pred = predictions[i]

        if pred == 1:
            color = 'red'  # AFib
        else:
            color = 'green'  # Normal

        plt.plot([start_time, end_time], [heart_rate_mean[i], heart_rate_mean[i + 1]], color=color, linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate Mean')
    plt.title('Heart Rate Mean with AFib Predictions ' + record_name)
    plt.show()

# Plot hrv_sdnn with predictions
def plot_hrv_sdnn_with_predictions(df, predictions, interval=10):
    plt.figure(figsize=(15, 6))

    time = df['start_time'].values  # Assuming 'start_time' is in seconds
    hrv_sdnn = df['hrv_sdnn'].values

    for i in range(len(predictions) - 1):
        start_time = time[i]
        end_time = time[i + 1]
        pred = predictions[i]

        if pred == 1:
            color = 'red'  # AFib
        else:
            color = 'green'  # Normal

        plt.plot([start_time, end_time], [hrv_sdnn[i], hrv_sdnn[i + 1]], color=color, linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('HRV SDNN')
    plt.title('HRV SDNN with AFib Predictions ' + record_name)
    plt.show()

# Main function to run the predictions and plot the results
def main():
    while True:
        print("Choose the type of plot:")
        print("1. Heart Rate Mean")
        print("2. HRV SDNN")
        print("0. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            plot_type = 'heart_rate_mean'
            plot_function = plot_heart_rate_with_predictions
        elif choice == "2":
            plot_type = 'hrv_sdnn'
            plot_function = plot_hrv_sdnn_with_predictions
        elif choice == "0":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please choose again.")
            continue

        # Load your trained model
        model = load_model()

        # Load the ECG data
        df = load_data('../data/afdb_data.csv', record_name)  # Specify the path to the data CSV file and record name
        interval = 10  # Interval in seconds

        # Preprocess the data
        df, features = preprocess_data(df)

        # Predict AFib in each interval
        predictions = model.predict(df[features])

        # Plot based on the selected type
        plot_function(df, predictions, interval)

if __name__ == "__main__":
    main()
