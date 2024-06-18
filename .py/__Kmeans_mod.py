import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('__STD_data.csv')

# Extract features (excluding the sample ID column)
features = data.iloc[:, 1:].values

# Set clustering model parameters
num_clusters = 400

# Create and fit the K-means clustering model
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
kmeans.fit(features)  # Directly fit to the features without scaling

# Get cluster labels for each sample
labels = kmeans.labels_

# Create a DataFrame to store clustering results
clustered_data = pd.DataFrame({'Sample': data.iloc[:, 0], 'Cluster': labels})

# Find the representative sample in each cluster (closest to the cluster center)
representative_samples = []
for cluster in range(num_clusters):
    cluster_points = features[labels == cluster]
    cluster_center = kmeans.cluster_centers_[cluster]
    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
    closest_index = np.argmin(distances)
    representative_sample_index = clustered_data[clustered_data['Cluster'] == cluster].iloc[closest_index].name
    representative_samples.append(data.iloc[representative_sample_index])

# Create a DataFrame of representative samples
representative_samples_df = pd.DataFrame(representative_samples)

# Sort by sample ID
representative_samples_sorted = representative_samples_df.sort_values(by='Sample')

# Save the representative samples to a new CSV file
representative_samples_sorted.to_csv('Kmeans_STD_mod_400.csv', index=False)