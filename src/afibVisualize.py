import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from PIL import Image

# Replace 'your_file.csv' with the path to your CSV file
record_name = "20240201112855_034635"
csv_file = f'../data/revlis_data/csv/{record_name}.csv'

# Read the CSV file
data = pd.read_csv(csv_file)

# Check if 'Lead2' column exists
if 'Lead2' not in data.columns:
    raise ValueError("The column 'Lead2' is not in the CSV file.")

# Extract the 'Lead2' column
lead2_signal = data['Lead2']
lead2_signal = lead2_signal.to_numpy()

# Calculate time in seconds
sampling_rate = 1000  # Sampling rate in Hz
time = np.arange(len(lead2_signal)) / sampling_rate

# Remove baseline wandering using neurokit2
cleaned_signal = nk.ecg_clean(lead2_signal, sampling_rate=sampling_rate)

# Plot the ECG signal with a larger figure size
plt.figure(figsize=(48, 6), dpi=300)

# Plot the original signal
plt.subplot(2, 1, 1)
plt.plot(time, lead2_signal, color='blue', linewidth=0.5)
plt.title(f'Original ECG Signal: {record_name}')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot the cleaned signal
plt.subplot(2, 1, 2)
plt.plot(time, cleaned_signal, color='green', linewidth=0.5)
plt.title(f'Cleaned ECG Signal: {record_name}')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)


plt.tight_layout()
plt.savefig('../reports/ecg_signal.png')
plt.close()

# Display the saved image
img = Image.open('../reports/ecg_signal.png')
img.show()
