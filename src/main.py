import neurokit2 as nk
import matplotlib.pyplot as plt
#
# # Simulate an ECG signal
# ecg_signal = nk.ecg_simulate(duration=10, noise=0.01, sampling_rate=1000, method="ecgsyn")
#
# # Preprocess the signal
# ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=1000)
#
# # Find R-peaks
# r_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=1000)
#
# # Extract features
# ecg_rate = nk.ecg_rate(r_peaks, sampling_rate=1000)
# hrv = nk.hrv_time(r_peaks, sampling_rate=1000)
#
# # Process the cleaned ECG signal
# ecg_signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=1000)
#
# # Visualize the ECG signal and R-peaks
# plt.figure(figsize=(12, 6))
# nk.ecg_plot(ecg_signals)
# # plt.show()
#
# # Print extracted features
# # print("Heart Rate (bpm):", ecg_rate)
# # print("HRV Time-Domain Features:", hrv)
# print("Heart Rate (bpm):", info["Heart_Rate"])
# print("HRV Time-Domain Features:", nk.hrv_time(info["R_Peaks"], sampling_rate=1000))

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