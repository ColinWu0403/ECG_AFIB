import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_poincare(csv_file, filename):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Extract sd1 and sd2 values
    sd1 = data['sd1']
    sd2 = data['sd2']

    # Create the x and y values for the plot using sd1 and sd2
    x = sd1[:-1]
    y = sd2[1:]

    # Calculate the mean and standard deviations for plotting purposes
    mean_sd1 = np.mean(sd1)
    mean_sd2 = np.mean(sd2)

    # Plot the Poincaré plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c='blue', alpha=0.5, label='SD1 vs SD2')
    plt.xlabel('SD1')
    plt.ylabel('SD2')
    plt.title('Poincaré Plot: ' + filename)

    # Plot the lines for mean SD1 and SD2
    plt.axline((mean_sd1, mean_sd1), slope=1, color='red', linestyle='--', label='Line of Identity')
    plt.axline((mean_sd1, mean_sd1), slope=-1, color='green', linestyle='--', label='Perpendicular Line')

    # Annotate the plot with SD1 and SD2 mean values
    plt.text(mean_sd1, mean_sd2, f'Mean SD1: {mean_sd1:.2f}\nMean SD2: {mean_sd2:.2f}', fontsize=12, color='black')

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cv(csv_file_path, filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Check if the 'cv' column exists
    if 'cv' not in df.columns:
        print("The 'cv' column does not exist in the CSV file.")
        return

    # Extract the 'cv' and 'start_time' column values
    cv_values = df['cv']
    start_time = df['start_time']

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(start_time, cv_values, marker='o', linestyle='-')
    plt.title('CV Value Over Time: ' + filename)
    plt.xlabel('Time (minutes)')
    plt.ylabel('CV')
    plt.grid(True)
    plt.show()


def plot_heart_rate_mean_vs_std(csv_file_path, filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Check if the 'heart_rate_mean' and 'heart_rate_std' columns exist
    if 'heart_rate_mean' not in df.columns or 'heart_rate_std' not in df.columns:
        print("The 'heart_rate_mean' or 'heart_rate_std' column does not exist in the CSV file.")
        return

    # Extract the 'heart_rate_mean' and 'heart_rate_std' column values
    heart_rate_mean = df['heart_rate_mean']
    heart_rate_std = df['heart_rate_std']

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(heart_rate_mean, heart_rate_std, marker='o')
    plt.title('Heart Rate Mean vs. Heart Rate Std: ' + filename)
    plt.xlabel('Heart Rate Mean')
    plt.ylabel('Heart Rate Std')
    plt.grid(True)
    plt.show()


def plot_heart_rate_mean(csv_file_path, filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Check if the 'heart_rate_mean' column exists
    if 'heart_rate_mean' not in df.columns:
        print("The 'heart_rate_mean' column does not exist in the CSV file.")
        return

    # Extract the 'heart_rate_mean' and 'start_time' column values
    heart_rate_mean = df['heart_rate_mean']
    start_time = df['start_time']

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(start_time, heart_rate_mean, marker='o', linestyle='-')
    plt.title('Heart Rate Mean Over Time: ' + filename)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Heart Rate Mean')
    plt.grid(True)
    plt.show()


def plot_heart_rate_std(csv_file_path, filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Check if the 'heart_rate_std' column exists
    if 'heart_rate_std' not in df.columns:
        print("The 'heart_rate_std' column does not exist in the CSV file.")
        return

    # Extract the 'heart_rate_std' and 'start_time' column values
    heart_rate_std = df['heart_rate_std']
    start_time = df['start_time']

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(start_time, heart_rate_std, marker='o', linestyle='-')
    plt.title('Heart Rate Std Over Time: ' + filename)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Heart Rate Std')
    plt.grid(True)
    plt.show()


def plot_sdnn(csv_file, filename):
    # Load the data
    data = pd.read_csv(csv_file)

    # Plot SDNN
    plt.figure(figsize=(10, 6))
    plt.plot(data['start_time'], data['hrv_sdnn'], label='HRV SDNN', color='green')
    plt.xlabel('Time (minutes)')
    plt.ylabel('SDNN (ms)')
    plt.title('HRV SDNN Over Time: ' + filename)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_rmssd(csv_file, filename):
    # Load the data
    data = pd.read_csv(csv_file)

    # Plot RMSSD
    plt.figure(figsize=(10, 6))
    plt.plot(data['start_time'], data['hrv_rmssd'], label='HRV RMSSD', color='blue')
    plt.xlabel('Time (minutes)')
    plt.ylabel('RMSSD (ms)')
    plt.title('HRV RMSSD Over Time: ' + filename)
    plt.legend()
    plt.grid(True)
    plt.show()


def get_file_name(path):
    if path.endswith("/"):
        path = path[:-1]
    last_part = path.split("/")[-1]
    if last_part.endswith(".csv"):
        last_part = last_part[:-4]
    return last_part


def main():
    csv_file = '../data/30_sec_intervals/04048_features.csv'  # Replace with the path to your CSV file

    print("Select the type of plot:")
    print("1. Heart Rate Mean")
    print("2. Heart Rate Std")
    print("3. Heart Rate Mean vs Heart Rate Std")
    print("4. Coefficient of Variation (CV)")
    print("5. Poincare Plot")
    print("6. SDNN")
    print("7. RMSSD")
    print("0. Exit")

    while True:
        choice = input("Enter the number of your choice: ")

        if choice == "0":
            exit(0)
        elif choice == '1':
            plot_heart_rate_mean(csv_file, get_file_name(csv_file))
        elif choice == '2':
            plot_heart_rate_std(csv_file, get_file_name(csv_file))
        elif choice == '3':
            plot_heart_rate_mean_vs_std(csv_file, get_file_name(csv_file))
        elif choice == '4':
            plot_cv(csv_file, get_file_name(csv_file))
        elif choice == '5':
            plot_poincare(csv_file, get_file_name(csv_file))
        elif choice == '6':
            plot_sdnn(csv_file, get_file_name(csv_file))
        elif choice == '7':
            plot_rmssd(csv_file, get_file_name(csv_file))
        else:
            print("Invalid choice. Please enter a number between 0 to 7.")


if __name__ == "__main__":
    main()
