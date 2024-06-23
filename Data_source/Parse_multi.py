import pandas as pd

# Define all possible text categories
columns = [
    "LiFSI", "LiDFOB", "LiBF4", "DME", "DEE", "DMP", "DEGDME", "TTE", "BTFE", "BDE", "HFE", "TFEPE", "B", "TFMOB", "LiPO2F2", "LiNO3", "FEC", "VC"
]

# Read the CSV file
df = pd.read_csv('Data_Ori_Full.csv')

# Create a new DataFrame to store the transformed data
new_df = pd.DataFrame(columns=columns)

# Iterate through columns first, then rows
for column in df.columns:
    for _, row in df.iterrows():
        cell = row[column]
        if pd.isna(cell):
            continue  # Skip empty values

        # Split data into a list
        data_list = cell.split(',')

        # Assume the first half is text, the second half is numeric
        split_index = len(data_list) // 2
        texts = data_list[:split_index]
        values = data_list[split_index:]

        # Create a temporary dictionary to store the row data
        data = {}
        for text, value in zip(texts, values):
            try:
                data[text] = float(value)
            except ValueError:
                print(f"Cannot convert value to float: {value}")
                continue

        # Initialize all columns to NaN in the new row
        new_row = pd.Series(index=columns, dtype=float)

        # Fill the corresponding numbers into the new row
        for text, value in data.items():
            if text in new_row.index:
                new_row[text] = value

        # Add the new row as a DataFrame to the new DataFrame
        if not new_row.dropna().empty:  # Ensure the new row is not all NaN
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

# Print the new DataFrame and save it to a new CSV file
print(new_df)
new_df.to_csv('Origin_data.csv', index=False)