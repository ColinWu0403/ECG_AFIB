import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
            line_witdh = 3  # thicker line for AFib
        else:
            color = 'green'  # Normal
            line_witdh = 1

        plt.plot([start_time, end_time], [heart_rate_mean[i], heart_rate_mean[i + 1]], color=color, linewidth=line_witdh)

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
            linewidth = 3  # thicker line for AFib
        else:
            color = 'green'  # Normal
            linewidth = 1

        plt.plot([start_time, end_time], [hrv_sdnn[i], hrv_sdnn[i + 1]], color=color, linewidth=linewidth)

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
            linewidth = 3  # thicker line for AFib
        else:
            color = 'green'  # Normal
            linewidth = 1

        plt.plot([start_time, end_time], [cv[i], cv[i + 1]], color=color, linewidth=linewidth)

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
            linewidth = 3  # thicker line for AFib
        else:
            color = 'green'  # Normal
            linewidth = 1

        plt.plot([start_time, end_time], [hrv_rmssd[i], hrv_rmssd[i + 1]], color=color, linewidth=linewidth)

    plt.xlabel('Time (s)')
    plt.ylabel('RMSSD')
    plt.title('RMSSD with AFib Predictions ' + record_name)
    plt.show()


# Main function to run the predictions and plot the results
def main():
    print("Models:")
    print("1. Random Forest Classifier")

    model_type = str(input("Enter the type of model: "))
    if model_type == "1":
        model_path = '../models/random_forest_model.pkl'
    else:
        print("Error: model does not exist.")
        return

    while True:
        print("Choose the type of plot:")
        print("1. Heart Rate Mean")
        print("2. HRV SDNN")
        print("3. HRV RMSSD")
        print("4. CV (Coefficient of Variation)")
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
        elif choice == "0":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please choose again.")
            continue

        # Load your trained model
        model = load_model(model_path)

        # Load the ECG data
        df = load_data('../data/afdb_data.csv', record_name)

        # Preprocess the data
        df, features = preprocess_data(df)

        # Predict AFib in each interval
        predictions = model.predict(df[features])

        # Plot based on the selected type
        plot_function(df, predictions)


if __name__ == "__main__":
    main()
