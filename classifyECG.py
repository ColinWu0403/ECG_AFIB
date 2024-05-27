import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def load_data(file_path):
    # Load the data
    return pd.read_csv(file_path)


def prepare_data(df):
    # Filter out rows where SDNN > 500 ms
    df = df[df['hrv_sdnn'] <= 500]

    # Filter out rows where RMSSD > 500 ms
    df = df[df['hrv_rmssd'] <= 500]

    # Filter out rows where cv > 0.5 (50 % variability)
    df = df[df['cv'] <= 0.5]

    # Normalize the data
    features = ['hrv_sdnn', 'hrv_rmssd', 'cv']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Prepare the data
    X = df[features]  # Features: SDNN, RMSSD, and cv
    y = df['has_AFIB']  # Target: whether the patient has AFib

    # Address class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    x_res, y_res = smote.fit_resample(X, y)

    return train_test_split(x_res, y_res, test_size=0.2, random_state=42)


def train_model(x_train, y_train):
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Train the model using Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    # Make predictions
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Convert classification report to DataFrame
    class_report_df = pd.DataFrame(class_report).transpose()

    # Add labels column
    class_report_df['labels'] = class_report_df.index

    # Rearrange columns so "labels" comes first
    cols = class_report_df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    class_report_df = class_report_df[cols]

    # Create classification report table image
    create_classification_report_image(class_report_df)

    # Create PDF report
    create_pdf(accuracy, roc_auc, conf_matrix)
    delete_images()


def create_classification_report_image(class_report_df):
    plt.figure(figsize=(10, 6))
    plt.axis('off')  # Hide axis
    cell_text = class_report_df.map(lambda x: f"{x:.8f}" if isinstance(x, float) else str(x))
    table = plt.table(cellText=cell_text.values,
                      colLabels=class_report_df.columns,
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.savefig("classification_report.png", bbox_inches='tight')
    plt.close()


def create_pdf(accuracy, roc_auc, conf_matrix):
    pdf_filename = "reports/model_evaluation_Random_Forest_30_seconds.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    # Add classification report image
    c.drawImage("classification_report.png", 55, 330, width=500, preserveAspectRatio=True, mask='auto')

    # Add text labels
    c.drawString(270, height - 50, f"Accuracy")
    c.drawString(242, height - 70, f"{accuracy}")
    c.drawString(255, height - 100, f"ROC AUC Score")
    c.drawString(242, height - 120, f"{roc_auc}")

    c.drawString(245, height - 150, "Classification Report")

    # Add confusion matrix image
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


filename = 'data/afdb_data.csv'


def main():
    # Load the data
    df = load_data(filename)

    # Prepare the data
    x_train, x_test, y_train, y_test = prepare_data(df)

    # Train the model
    model = train_model(x_train, y_train)

    # Evaluate the model
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
