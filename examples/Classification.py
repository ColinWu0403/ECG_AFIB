import pandas as pd
import matplotlib.pyplot as plt

def calculate_bmi(df):
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    return df

# Define BMI categories
def classify_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

def plot_bmi_classification(df, color_map):
    plt.figure(figsize=(8, 6))
    for bmi_category, color in color_map.items():
        plt.scatter(df[df['bmi_category'] == bmi_category]['height'],
                    df[df['bmi_category'] == bmi_category]['weight'],
                    color=color,
                    label=bmi_category)
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.title('BMI Classification')
    plt.legend()
    plt.show()

# Define colors for each BMI category
color_map = {
    'Underweight': 'blue',
    'Normal weight': 'green',
    'Overweight': 'orange',
    'Obese': 'red'
}

def main():
    # Define the file path and color map
    filepath = '../data/2023_dataframe.csv'

    # Load the data
    df = pd.read_csv(filepath)

    # Calculate BMI
    df = calculate_bmi(df)

    # Classify individuals based on BMI
    df['bmi_category'] = df['bmi'].apply(classify_bmi)

    # Print the first few rows to verify
    print(df[['old', 'height', 'weight', 'bmi', 'bmi_category']].head(10))

    # Plot the BMI classification
    plot_bmi_classification(df, color_map)

if __name__ == "__main__":
    main()
