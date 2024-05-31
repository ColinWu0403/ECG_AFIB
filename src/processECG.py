import wfdb
import neurokit2 as nk
import pandas as pd
import numpy as np
import os

total_N_annotations = 0
total_AFIB_annotations = 0


def process_ecg_interval(record_path, record_name, start_sample, end_sample, interval_index):
    global total_N_annotations
    global total_AFIB_annotations

    # Read the header of the record to get metadata
    record_header = wfdb.rdheader(record_path)
    total_samples = record_header.sig_len

    # Ensure that end_sample does not exceed the total length of the signal
    end_sample = min(end_sample, total_samples)

    if end_sample - start_sample < 10:  # Ensure the interval has at least 10 samples
        print(f"Error processing interval {interval_index}: The data length is too small to be segmented.")
        return None

    record_segment = wfdb.rdrecord(record_path, sampfrom=start_sample, sampto=end_sample)
    ecg_signal = record_segment.p_signal[:, 0]  # Assuming the first channel is ECG
    sampling_rate = record_segment.fs

    # Read the annotations (if they exist)
    try:
        annotations = wfdb.rdann(record_path, 'atr', sampfrom=start_sample, sampto=end_sample)
        print("atr annotations")
        print(annotations)
        print("sample" + str(annotations.sample))
        print("symbol" + str(annotations.symbol) + str(annotations.subtype))
        print("aux_note" + str(annotations.aux_note))
        print("")
        # print("custom label: " + str(annotations.custom_labels))
        # print("description: " + str(annotations.description))
    except FileNotFoundError:
        annotations = None

    try:
        qrs_annotations = wfdb.rdann(record_path, 'qrs', sampfrom=start_sample, sampto=end_sample)
    except FileNotFoundError:
        qrs_annotations = None

    # Process the ECG signal
    try:
        ecg_signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    except Exception as e:
        print(f"Error processing interval {interval_index}: {e}")
        return None

    # Extract relevant features
    heart_rate = ecg_signals["ECG_Rate"]
    signal_quality = ecg_signals["ECG_Quality"]
    avg_quality = np.mean(signal_quality)
    r_peaks = info["ECG_R_Peaks"]
    hrv_time = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)

    # Calculate intervals using helper functions
    pr_interval_mean, pr_interval_std = calculate_pr_interval(info, sampling_rate)
    qrs_duration_mean, qrs_duration_std = calculate_qrs_duration(info, sampling_rate)
    qt_interval_mean, qt_interval_std = calculate_qt_interval(info, sampling_rate)

    # Calculate Coefficient of Variation (CV)
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # Convert to milliseconds
    cv = hrv_time["HRV_SDNN"].iloc[0] / hrv_time["HRV_MeanNN"].iloc[0]

    # Poincaré Plot Analysis
    sd1, sd2 = calculate_poincare(rr_intervals)

    # Prepare a dictionary of extracted features
    features = {
        "record_name": record_name,
        "start_time": interval_index / 6,  # in minutes
        "sampling_rate": sampling_rate,
        "heart_rate_mean": heart_rate.mean(),
        "heart_rate_std": heart_rate.std(),
        "signal_quality": avg_quality,  # see explanation below
        "pr_interval_mean": pr_interval_mean,  # all values for intervals are in milliseconds
        "pr_interval_std": pr_interval_std,
        "qrs_duration_mean": qrs_duration_mean,
        "qrs_duration_std": qrs_duration_std,
        "qt_interval_mean": qt_interval_mean,
        "qt_interval_std": qt_interval_std,
        "hrv_rmssd": hrv_time["HRV_RMSSD"].iloc[0],
        # Square root of the mean of the squared successive differences between adjacent RR intervals.
        "hrv_mean": hrv_time["HRV_MeanNN"].iloc[0],  # The mean of the RR intervals in milliseconds
        "hrv_sdnn": hrv_time["HRV_SDNN"].iloc[0],  # The standard deviation of the RR intervals in milliseconds
        "cv": cv,
        # Coefficient of Variation (CV) the ratio of the standard deviation of the RR intervals to the mean RR interval.
        "sd1": sd1,  # Coordinates for scatter plots where each RR interval is plotted against the previous RR interval.
        "sd2": sd2,
        # Add additional features to the features dictionary

        # signal_quality:
        # it is a value from 0 to 1:
        # 1 corresponds to heartbeats that are the closest to the average sample
        # 0 corresponds to the most distant heartbeat from that average sample.

        # Useful Resources:
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/
        # https://ecg.utah.edu/lesson/3
        # https://emedicine.medscape.com/article/2172196-overview?form=fpf
        # https://www.researchgate.net/publication/258514321_The_Usefulness_of_the_Coefficient_of_Variation_of_Electrocardiographic_RR_Interval_as_an_Index_of_Cardiovascular_Function_and_its_Correlation_with_Age_and_Stroke
    }

    # deal with ptb database data
    ptb_afib = False
    contains_lr = 'lr' in features["record_name"]
    contains_hr = 'hr' in features["record_name"]
    if contains_lr or contains_hr:
        ptb_afib = True

    # Add annotation-related features
    if annotations is not None:
        features["num_annotations"] = len(annotations.sample)

        # Count annotations and update total counts
        num_N_annotations = 0
        num_AFIB_annotations = 0
        aux_notes = annotations.aux_note
        if aux_notes:
            for note in aux_notes:
                if note == '(N':
                    num_N_annotations += 1
                    total_N_annotations += 1
                elif note == '(AFIB':
                    num_AFIB_annotations += 1
                    total_AFIB_annotations += 1

        features["num_N_annotations"] = num_N_annotations
        features["num_AFIB_annotations"] = num_AFIB_annotations
        features["total_N_annotations"] = total_N_annotations
        features["total_AFIB_annotations"] = total_AFIB_annotations
    else:
        if ptb_afib:
            features["num_annotations"] = 1
            features["num_AFIB_annotations"] = 1
            features["total_AFIB_annotations"] = 1
        else:
            features["num_annotations"] = 0
            features["num_AFIB_annotations"] = 0
            features["total_AFIB_annotations"] = 0

        features["num_N_annotations"] = 0
        features["total_N_annotations"] = 0

    if qrs_annotations is not None:
        features["num_qrs_annotations"] = len(qrs_annotations.sample)
    else:
        features["num_qrs_annotations"] = 0

    return features


def process_ecg_record(record_path, record_name):
    # Read the header of the record to get metadata
    record = wfdb.rdheader(record_path)
    sampling_rate = record.fs
    total_samples = record.sig_len

    # Split the signal into 10 second intervals
    ten_sec_intervals = sampling_rate * 10

    # Calculate the number of intervals needed to cover the entire signal
    num_intervals = total_samples // ten_sec_intervals
    if total_samples % ten_sec_intervals != 0:
        num_intervals += 1  # Add one more interval for the remaining samples
    all_features = []

    for i in range(num_intervals):
        start_sample = i * ten_sec_intervals
        end_sample = start_sample + ten_sec_intervals
        try:
            features = process_ecg_interval(record_path, record_name, start_sample, end_sample, i)
            if features is not None:
                all_features.append(features)
        except Exception as e:
            print(f"Error processing interval {i}: {e}")

    return all_features


def count_annotations(file_path, target_symbols):
    count_dict = {symbol: 0 for symbol in target_symbols}

    with open(file_path, 'rb') as f:
        content = f.read()

    content_str = content.decode('latin1')  # Decode binary content to string
    for symbol in target_symbols:
        count_dict[symbol] = content_str.count(symbol)

    return count_dict


def calculate_pr_interval(info, sampling_rate):
    p_peaks = info["ECG_P_Peaks"]
    r_peaks = info["ECG_R_Peaks"]
    pr_intervals = []
    for p_peak, r_peak in zip(p_peaks, r_peaks):
        if not np.isnan(p_peak) and not np.isnan(r_peak):
            pr_interval = ((r_peak - p_peak) / sampling_rate) * 1000
            pr_intervals.append(pr_interval)
    pr_interval_mean = np.mean(pr_intervals) if pr_intervals else 0
    pr_interval_std = np.std(pr_intervals) if pr_intervals else 0
    return pr_interval_mean, pr_interval_std


def calculate_qrs_duration(info, sampling_rate):
    q_peaks = info["ECG_Q_Peaks"]
    r_onsets = info["ECG_R_Onsets"]
    s_peaks = info["ECG_S_Peaks"]
    qrs_durations = []
    for i in range(min(len(q_peaks), len(s_peaks))):
        if not np.isnan(q_peaks[i]) and not np.isnan(r_onsets[i]) and not np.isnan(s_peaks[i]):
            nearest_q_onset = min(q_peaks[i], r_onsets[i])
            qrs_duration = ((s_peaks[i] - nearest_q_onset) / sampling_rate) * 1000
            qrs_durations.append(qrs_duration)
    qrs_duration_mean = np.mean(qrs_durations) if qrs_durations else 0
    qrs_duration_std = np.std(qrs_durations) if qrs_durations else 0
    return qrs_duration_mean, qrs_duration_std


def calculate_qt_interval(info, sampling_rate):
    q_peaks = info["ECG_Q_Peaks"]
    t_offsets = info["ECG_T_Offsets"]
    qt_intervals = []
    for i in range(min(len(q_peaks), len(t_offsets))):
        if not np.isnan(q_peaks[i]) and not np.isnan(t_offsets[i]):
            nearest_q_peak = min(q_peaks[i], t_offsets[i])
            qt_interval = ((t_offsets[i] - nearest_q_peak) / sampling_rate) * 1000
            qt_intervals.append(qt_interval)
    qt_interval_mean = np.mean(qt_intervals) if qt_intervals else 0
    qt_interval_std = np.std(qt_intervals) if qt_intervals else 0
    return qt_interval_mean, qt_interval_std


def calculate_poincare(rr_intervals):
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    sd1 = np.std(np.subtract(rr_n1, rr_n) / np.sqrt(2))
    sd2 = np.std(np.add(rr_n1, rr_n) / np.sqrt(2))
    return sd1, sd2


def load_and_combine_data(data_dir):
    # Initialize an empty list to hold DataFrames
    data_frames = []

    # Iterate over all files in the data directory
    for file_name in os.listdir(data_dir):
        if file_name == "2023_dataframe.csv":  # ignore this file
            continue
        elif file_name == "afdb_data.csv":
            continue
        elif file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            data_frames.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df


def add_has_afib_column(df):
    # Create the has_AFIB column based on num_AFIB_annotations
    df['has_AFIB'] = (df['num_AFIB_annotations'] > 0).astype(int)
    return df


def save_combined_data(df, output_file):
    # Save the combined DataFrame to a new CSV file
    df.to_csv(output_file, index=False)


def extract_file_paths_and_names(csv_file, f_type):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Initialize lists to hold full paths and file names
    full_paths = []
    file_names = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        if f_type == 'hr':
            full_path = row['filename_hr']
        elif f_type == 'lr':
            full_path = row['filename_hr']
        else:
            full_path = ""

        file_name = full_path.split('/')[-1]

        full_paths.append(full_path)
        file_names.append(file_name)

    return full_paths, file_names


def process_all_ecg_records(csv_file, ptb_dir, output_file, f_type):
    records, file_names = extract_file_paths_and_names(csv_file, f_type)

    print(records)
    print(file_names)

    all_features = []
    i = 0
    for record_name in records:
        record_path = os.path.join(ptb_dir, record_name)
        features = process_ecg_record(record_path, file_names[i])
        if features:
            all_features.extend(features)
        i = i + 1

    df = pd.DataFrame(all_features)
    df.to_csv(output_file, index=False)
    print(f"Saved all features to {output_file}")


def main():
    print("1. process ECG signals from afib/ and export to .csv files")
    print("2. process ECG signals from ptb/ and export to .csv files")
    print("3. combine all data")
    choice = input("Enter the number of your choice: ")

    if choice == "1":
        # Define the directory containing the ECG records
        ptb_dir = "../afdb"

        records = ["04043"]  # add records here

        for record_name in records:
            record_path = os.path.join(ptb_dir, record_name)
            features = process_ecg_record(record_path, record_name)

            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(features)
            file_name = "../data/10_sec_intervals/" + record_name + "_features.csv"

            # Save the DataFrame to a CSV file
            df.to_csv(file_name, index=False)
    elif choice == "2":
        ptb_dir = "../ptb"
        csv_file = "../ptb/ptbxl_afib.csv"
        output_file = "../data/10_sec_intervals/hr_features.csv"

        if not os.path.exists("../data/10_sec_intervals"):
            os.makedirs("../data/10_sec_intervals")

        process_all_ecg_records(csv_file, ptb_dir, output_file, "hr")
    elif choice == "3":
        data_dir = "../data/10_sec_intervals"
        output_file = "../data/afdb_data.csv"

        combined_df = load_and_combine_data(data_dir)

        combined_df = add_has_afib_column(combined_df)

        save_combined_data(combined_df, output_file)
    else:
        return


if __name__ == "__main__":
    main()
