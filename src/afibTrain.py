import pandas as pd
import numpy as np
import os
import joblib
import neurokit2 as nk
from keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def process_data():
    # Load the profiles file to get the filenames and AFib status
    profiles = pd.read_csv('../data/revlis_data/profiles.csv')

    # Iterate through each entry in profiles
    for _, row in profiles.iterrows():
        filename = row['filename']
        filename = str(filename + ".csv")
        af_similarity = row['AF_Similarity']

        # Construct the full path to the ECG file
        file_path = os.path.join('../data/revlis_data/csv', filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            # print(f"File {file_path} does not exist. Skipping.")
            continue

        # Read the ECG file
        ecg_df = pd.read_csv(file_path)

        # Extract the ECG signal from Lead2
        ecg_signal = ecg_df['Lead2'].values

        # Clean the ECG signal using neurokit2
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=1000)

        # Process the cleaned ECG signal using neurokit2
        ecg_processed, info = nk.ecg_process(ecg_cleaned, sampling_rate=1000)
        hrv_metrics = nk.hrv_time(ecg_processed, sampling_rate=1000)

        # Extract HRV features
        sdnn = hrv_metrics['HRV_SDNN'].values[0]
        rmssd = hrv_metrics['HRV_RMSSD'].values[0]
        meanNN = hrv_metrics['HRV_MeanNN'].values[0]

        # Example: Calculate some simple features from the ECG signal
        mean_val = np.mean(ecg_cleaned)
        std_val = np.std(ecg_cleaned)
        max_val = np.max(ecg_cleaned)
        min_val = np.min(ecg_cleaned)

        # Append the features and the label to the lists
        ecg_data.append([mean_val, std_val, max_val, min_val, sdnn, rmssd, meanNN])
        
        af_threshold = 0.9  # Threshold for AFib similarity set to 90 %
        
        af_labels.append(1 if af_similarity > af_threshold else 0)
        # print([mean_val, std_val, max_val, min_val, sdnn, rmssd, meanNN])

# Function to build the CNN model
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=123, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.4818354119145667))

    model.add(Conv1D(filters=126, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.2843527405365337))

    model.add(Conv1D(filters=50, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.4432529199866734))

    model.add(Flatten())
    model.add(Dense(454, activation='relu'))
    model.add(Dropout(0.4377679671880249))
    
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(mode):
    # Convert the lists to NumPy arrays
    x = np.array(ecg_data)
    y = np.array(af_labels)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print("Training the model...")
    if mode == 1:
        # Train a Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        return model, x_train, x_test, y_test, y_train, y_test
    elif mode == 2:

        # Prepare the data for CNN
        x_train_cnn = np.expand_dims(x_train, axis=2)
        x_test_cnn = np.expand_dims(x_test, axis=2)
        
        print(x_train_cnn.shape)
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
        
        # Build CNN model
        input_shape = (x_train_cnn.shape[1], 1)
        model = build_cnn_model(input_shape)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        # Train the model
        model.fit(x_train_cnn, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        
        return model, x_train_cnn, x_test_cnn, y_test, y_train, y_test
    else:
        print("Invalid Mode")
        return None, None, None, None, None

def save_model(model, type):
    if type == 1:
        # Save the trained model
        model_save_path = '../papers/random_forest_model.pkl'
        joblib.dump(model, model_save_path)
    else:
        model_save_path = '../papers/cnn_model.keras'
        model.save(model_save_path)

def main():
    process_data()
    
    print("1)Random Forest\n2)CNN")
    choice = int(input("Model Type: "))
    # if !(choice == 1 or choice == 2):
    #     print("Invalid choice")
    #     return
    
    model, x_train, x_test, y_test, y_train, y_test = train_model(choice)

    if model is None:
        print("Model training failed.")
        return
    
    if choice == 1:
        print("Random Forest Model")

        save_model(model, choice)
        
        # Predict on the test set
        y_pred = model.predict(x_test)
        
        # Evaluate the model
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    else:
        print("CNN Model")

        save_model(model, choice)

        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Test Accuracy: {accuracy}")
        
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)  # Convert one-hot encoded predictions to class indices
        y_test_indices = np.argmax(y_test, axis=1)  # Convert one-hot encoded true labels to class indices
        print("Classification Report:")
        print(classification_report(y_test_indices, y_pred))
        
if __name__ == "__main__":
    # Initialize lists to hold the features and labels
    ecg_data = []
    af_labels = []

    main()