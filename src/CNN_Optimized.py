import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from keras.src.callbacks import EarlyStopping
from keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.utils import to_categorical
import optuna
from optuna.integration import TFKerasPruningCallback


# Load and prepare data functions remain unchanged
def load_data(file_path):
    return pd.read_csv(file_path)


def prepare_data(df):
    df = df[df['hrv_sdnn'] <= 500]
    df = df[df['hrv_rmssd'] <= 500]
    df = df[df['cv'] <= 0.5]
    df = df[df['signal_quality'] >= 0.5]

    features = ['hrv_sdnn', 'hrv_rmssd', "hrv_mean", 'cv']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    x = df[features]
    y = df['has_AFIB']

    smote = SMOTE(random_state=42)
    x_res, y_res = smote.fit_resample(x, y)

    x_res = x_res.values.reshape((x_res.shape[0], 1, x_res.shape[1]))
    y_res = to_categorical(y_res)

    return train_test_split(x_res, y_res, test_size=0.2, random_state=42)


# Define the model-building function with hyperparameters from Optuna
def build_cnn_model(input_shape, params):
    model = Sequential()
    model.add(
        Conv1D(params['filters1'], kernel_size=params['kernel_size1'], activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=params['pool_size1']))
    model.add(Dropout(params['dropout1']))

    model.add(Conv1D(params['filters2'], kernel_size=params['kernel_size2'], activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=params['pool_size2']))
    model.add(Dropout(params['dropout2']))

    model.add(Conv1D(params['filters3'], kernel_size=params['kernel_size3'], activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=params['pool_size3']))
    model.add(Dropout(params['dropout3']))

    model.add(Flatten())
    model.add(Dense(params['dense_units'], activation='relu'))
    model.add(Dropout(params['dense_dropout']))

    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(model, x_test, y_test):
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    class_report = classification_report(y_true, y_pred, output_dict=True)

    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df['labels'] = class_report_df.index
    cols = class_report_df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    class_report_df = class_report_df[cols]

    create_classification_report_image(class_report_df)
    create_pdf(accuracy, roc_auc, conf_matrix)
    delete_images()


def create_classification_report_image(class_report_df):
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    cell_text = class_report_df.values
    table = plt.table(cellText=cell_text,
                      colLabels=class_report_df.columns,
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.savefig("classification_report.png", bbox_inches='tight')
    plt.close()


def create_pdf(accuracy, roc_auc, conf_matrix):
    pdf_filename = "../reports/model_evaluation_CNN.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    c.drawImage("classification_report.png", 55, 250, width=500, preserveAspectRatio=True, mask='auto')
    c.drawString(270, height - 50, "Accuracy")
    c.drawString(242, height - 70, f"{accuracy}")
    c.drawString(255, height - 100, "ROC AUC Score")
    c.drawString(242, height - 120, f"{roc_auc}")
    c.drawString(245, height - 150, "Classification Report")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.close()

    c.drawImage("confusion_matrix.png", 65, 0, width=500, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()


def delete_images():
    os.remove("classification_report.png")
    os.remove("confusion_matrix.png")


def objective(trial):
    df = load_data(filename)
    x_train, x_test, y_train, y_test = prepare_data(df)
    input_shape = (x_train.shape[1], x_train.shape[2])

    params = {
        'filters1': trial.suggest_int('filters1', 32, 256),
        'kernel_size1': trial.suggest_int('kernel_size1', 1, 3),
        'pool_size1': trial.suggest_int('pool_size1', 1, 2),
        'dropout1': trial.suggest_uniform('dropout1', 0.1, 0.5),
        'filters2': trial.suggest_int('filters2', 32, 256),
        'kernel_size2': trial.suggest_int('kernel_size2', 1, 3),
        'pool_size2': trial.suggest_int('pool_size2', 1, 2),
        'dropout2': trial.suggest_uniform('dropout2', 0.1, 0.5),
        'filters3': trial.suggest_int('filters3', 32, 256),
        'kernel_size3': trial.suggest_int('kernel_size3', 1, 3),
        'pool_size3': trial.suggest_int('pool_size3', 1, 2),
        'dropout3': trial.suggest_uniform('dropout3', 0.1, 0.5),
        'dense_units': trial.suggest_int('dense_units', 128, 1024),
        'dense_dropout': trial.suggest_uniform('dense_dropout', 0.1, 0.5)
    }

    model = build_cnn_model(input_shape, params)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
              callbacks=[TFKerasPruningCallback(trial, 'val_loss'), early_stopping], verbose=0)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy


def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print(f"Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train and evaluate the best model
    df = load_data(filename)
    x_train, x_test, y_train, y_test = prepare_data(df)
    input_shape = (x_train.shape[1], x_train.shape[2])

    model = build_cnn_model(input_shape, trial.params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    filename = '../data/afdb_data.csv'

    main()
