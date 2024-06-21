import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv('STD_data.csv')

# Extract feature columns
features = data.iloc[:, 1:]

# Create PCA object
pca = PCA()

# Perform PCA on feature columns
pca_result = pca.fit_transform(features)

# Calculate cumulative explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# --- Plot the cumulative explained variance ratio ---
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio Plot')

# Save the plot
plt.savefig('Cumulative_variance_ratio_plot.png')