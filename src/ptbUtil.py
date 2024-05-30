import pandas as pd


def find_afib_records():
    # Load the CSV file
    input_file = '../ptb/ptbxl_database.csv'
    df = pd.read_csv(input_file)

    # Filter the rows where the "report" column contains "atrial fibrillation"
    filtered_df = df[df['report'].str.contains('atrial fibrillation', case=False, na=False)]

    # Export the filtered rows to a new CSV file
    output_file = '../ptb/ptbxl_afib.csv'
    filtered_df.to_csv(output_file, index=False)

    print(f"Filtered data has been saved to {output_file}")

def find_afib():
    sds = pd.read_csv('../ptb/ptbxl_afib.csv')