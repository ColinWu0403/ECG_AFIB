import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import seaborn as sns

def load_data(file_path):
    # Load the data
    return pd.read_csv(file_path)

def prepare_data(df):
    # Normalize the data
    features = ['hrv_sdnn', 'hrv_rmssd', 'cv']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Prepare the data
    X = df[features]  # Features: SDNN, RMSSD, and cv
    y = df['has_AFIB']  # Target: whether the patient has AFib

    # Address class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("Support: " + str(len(X_res)) + " " + str(len(y_res)))
    return train_test_split(X_res, y_res, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Train the model using Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("Accuracy:", accuracy)
    print("ROC AUC Score:", roc_auc)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: ' + filename)
    plt.show()

    return accuracy, conf_matrix, class_report, roc_auc

def plot_decision_boundary(X, y, model):
    def plot(ax):
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3)
        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.Paired)
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)
        ax.set_xlabel('SDNN')
        ax.set_ylabel('RMSSD')

    fig, ax = plt.subplots(figsize=(10, 6))
    plot(ax)
    plt.title('Decision Boundary')
    plt.show()

filename = 'data/afdb_data.csv'

def main():
    # Load the data
    df = load_data(filename)

    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Plot the decision boundary
    # plot_decision_boundxary(X_test, y_test, model)

if __name__ == "__main__":
    main()
