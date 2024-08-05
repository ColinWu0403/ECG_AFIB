import pandas as pd
import numpy as np
import os
import joblib
import neurokit2 as nk
from keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import optuna
from optuna.integration import TFKerasPruningCallback

def process_data():
    profiles = pd.read_csv('../data/revlis_data/profiles.csv')
    for _, row in profiles.iterrows():
        filename = row['filename']
        filename = str(filename + ".csv")
        af_similarity = row['AF_Similarity']
        file_path = os.path.join('../data/revlis_data/csv', filename)
        if not os.path.exists(file_path):
            continue
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
        ecg_data.append([mean_val, std_val, max_val, min_val, sdnn, rmssd, meanNN])
        af_threshold = 0.9
        af_labels.append(1 if af_similarity > af_threshold else 0)

def build_cnn_model(input_shape, params):
    model = Sequential()
    print(f"Building model with input shape: {input_shape}")

    # First Conv1D Layer
    model.add(Conv1D(params['filters1'], kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(params['dropout1']))
    print(f"After first layer: {model.output_shape}")

    # Second Conv1D Layer
    model.add(Conv1D(params['filters2'], kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(params['dropout2']))
    print(f"After second layer: {model.output_shape}")

    # Third Conv1D Layer
    model.add(Conv1D(params['filters3'], kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(params['dropout3']))
    print(f"After third layer: {model.output_shape}")

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(params['dense_units'], activation='relu'))
    model.add(Dropout(params['dense_dropout']))

    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(mode):
    x = np.array(ecg_data)
    y = np.array(af_labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    if mode == 1:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        return model, x_train, x_test, y_train, y_test
    elif mode == 2:
        x_train_cnn = np.expand_dims(x_train, axis=2)
        x_test_cnn = np.expand_dims(x_test, axis=2)
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
        input_shape = (x_train_cnn.shape[1], 1)
        model = build_cnn_model(input_shape, best_params)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model.fit(x_train_cnn, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        return model, x_train_cnn, x_test_cnn, y_train, y_test
    else:
        print("Invalid Mode")
        return None, None, None, None, None

def save_model(model, type):
    if type == 1:
        model_save_path = '../papers/random_forest_model.pkl'
        joblib.dump(model, model_save_path)
    else:
        model_save_path = '../papers/cnn_model.keras'
        model.save(model_save_path)

def evaluate_model(model, x_test, y_test):
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    class_report = classification_report(y_true, y_pred, output_dict=True)

    print("Accuracy:", accuracy)
    print("ROC AUC Score:", roc_auc)
    print("Classification Report:", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", conf_matrix)

def objective(trial):
    x = np.array(ecg_data)
    y = np.array(af_labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train_cnn = np.expand_dims(x_train, axis=2)
    x_test_cnn = np.expand_dims(x_test, axis=2)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    input_shape = (x_train_cnn.shape[1], 1)

    params = {
        'filters1': trial.suggest_int('filters1', 32, 128),
        'dropout1': trial.suggest_uniform('dropout1', 0.1, 0.5),
        'filters2': trial.suggest_int('filters2', 32, 128),
        'dropout2': trial.suggest_uniform('dropout2', 0.1, 0.5),
        'filters3': trial.suggest_int('filters3', 32, 128),
        'dropout3': trial.suggest_uniform('dropout3', 0.1, 0.5),
        'dense_units': trial.suggest_int('dense_units', 128, 512),
        'dense_dropout': trial.suggest_uniform('dense_dropout', 0.1, 0.5)
    }

    model = build_cnn_model(input_shape, params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[TFKerasPruningCallback(trial, 'val_loss'), early_stopping], verbose=0)

    loss, accuracy = model.evaluate(x_test_cnn, y_test, verbose=0)
    return accuracy

def main():
    process_data()
    print("1)Random Forest\n2)CNN")
    choice = int(input("Model Type: "))
    if choice != 1 and choice != 2:
        print("Invalid choice")
        return

    if choice == 2:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        print(f"Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print(f"  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        global best_params
        best_params = trial.params

    model, x_train, x_test, y_train, y_test = train_model(choice)

    if model is None:
        print("Model training failed.")
        return

    if choice == 1:
        print("Random Forest Model")
        save_model(model, choice)
        y_pred = model.predict(x_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    else:
        print("CNN Model")
        save_model(model, choice)
        evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    ecg_data = []
    af_labels = []
    best_params = None
    main()
