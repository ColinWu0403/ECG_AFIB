import neurokit2 as nk
import matplotlib.pyplot as plt

# Download afdb
ecg_signal = nk.data(dataset="ecg_3000hz")

# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=3000)

# Delineate
_, waves = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, method="dwt", show=True, show_type='all')

# Process the ECG signal
ecg_signals, info = nk.ecg_process(ecg_signal, sampling_rate=3000)

# Plot the ECG signal, R-peaks, and delineated waves
nk.ecg_plot(ecg_signals, info)
plt.title("ECG Signal with R-peaks and Delineated Waves")

# Show the plot
plt.show()