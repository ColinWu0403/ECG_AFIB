import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler

def calculate_error(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

def plot_data(X, y, df):
    # Plotting each feature against the target variable
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(['old', 'sex', 'height', 'diabetes']):
        if col in ['sex', 'diabetes']:
            # For categorical features, use original non-dummy encoded values for the plot
            X_plot = df[col]
        else:
            X_plot = X[col]

        # Calculate linear regression
        slope, intercept, r, p, std_err = stats.linregress(X_plot, y)
        regression_line = slope * X_plot + intercept

        # print(col + ": " + str(r))

        axes[i].scatter(X_plot, y, alpha=0.5, label='Actual')
        axes[i].plot(X_plot, regression_line, color='red', label='Linear Regression Line')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Weight')
        axes[i].legend()

        plt.tight_layout()
        plt.show()

def user_input_prediction(model, scaler):
    # Infinite loop for user input
    while True:
        try:
            # Accept user input for height
            height_input = input("Enter the height in cm (or type 'exit' to quit): ")
            if height_input == 'exit':
                break
            height = float(height_input)

            # Prepare the input afdb for prediction
            input_data = [[30, 1, height, 0]]  # Example values for age, sex, height, diabetes
            input_data_scaled = scaler.transform(input_data)

            # Predict the weight
            predicted_weight = model.predict(input_data_scaled)[0]
            print("Predicted weight:", predicted_weight)
        except ValueError:
            print("Please enter a valid number for height.")

def main():
    df = pd.read_csv('../data/2023_dataframe.csv')

    # Define features and target
    X = df[['old', 'sex', 'height', 'diabetes']]
    y = df['weight']

    # Encode categorical variables (assuming 'sex' is binary and 'diabetes' has multiple levels)
    X.loc[:, 'sex'] = X['sex'].astype('category')
    X.loc[:, 'diabetes'] = X['diabetes'].astype('category')
    X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables

    # Split the data into training and testing sets, 80:20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    while True:
        print("\nChoose a function:")
        print("1. Train model and predict")
        print("2. Calculate error")
        print("3. Plot data")
        print("4. User input for predicted value")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            model, scaler = train_model(X_train, y_train)
            user_input_prediction(model, scaler)
        elif choice == '2':
            model, scaler = train_model(X_train, y_train)
            calculate_error(model, scaler, X_test, y_test)
        elif choice == '3':
            plot_data(X, y, df)
        elif choice == '4':
            model, scaler = train_model(X_train, y_train)
            user_input_prediction(model, scaler)
        elif choice == '5':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()