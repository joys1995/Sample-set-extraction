import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("Origin_data.csv")

# Select feature columns 
feature_cols = data.columns[1:15]

# Standardize the features
scaler = StandardScaler()
data[feature_cols] = scaler.fit_transform(data[feature_cols])

# Save the standardized data
data.to_csv("STD_data.csv", index=False)