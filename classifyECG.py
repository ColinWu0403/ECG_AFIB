import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

def load_data(file_path):
    # Load the data
    return pd.read_csv(file_path)

def create_afib_target(df, threshold=0):
    # Create a binary target column based on num_AFIB_annotations
    df['afib'] = (df['num_AFIB_annotations'] > threshold).astype(int)
    return df

def prepare_data(df):
    # Prepare the data
    X = df[['hrv_sdnn', 'hrv_rmssd']]  # Features: SDNN and RMSSD
    y = df['afib']  # Target: whether the patient has AFib
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    # Train the model using Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, conf_matrix, class_report

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

def main():
    # Load the data
    df = load_data('data/08219_features.csv')  # Replace 'your_data.csv' with your actual file name

    # Create the AFib target column
    df = create_afib_target(df)

    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Plot the decision boundary
    plot_decision_boundary(X_test, y_test, model)

if __name__ == "__main__":
    main()
