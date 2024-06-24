import pandas as pd

# Read the two csv files
df1 = pd.read_csv("Sample_data.csv")
df2 = pd.read_csv("Electrolyte formulation_Parse.csv")

# Specify the column name to filter
column_to_filter = "Sample"  # Replace "Sample" with the actual column name

# Get the values list of the specified column in csv1
filter_values = df1[column_to_filter].tolist()

# Filter the rows in csv2 that contain the specified values
filtered_df = df2[df2[column_to_filter].isin(filter_values)]

# Save the filtered data to a new csv file
filtered_df.to_csv("filtered_data.csv", index=False)

print("Filtering completed! The results have been saved to Filtered_data.csv")