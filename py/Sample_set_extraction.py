import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import resample

# Read dataset
data = pd.read_csv('STD_data.csv')
sample_id_column = 'Sample'
features = data.iloc[:, 1:]

# PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)
data['PC1'] = reduced_features[:, 0]
data['PC2'] = reduced_features[:, 1]

# Stratified sampling to obtain a medium sample
medium_sample_size = 1000  # Adjustable parameter
for i in range(2):
    data[f'PC{i+1}_quantile'] = pd.qcut(data[f'PC{i+1}'], 4, labels=False, duplicates='drop')

stratified_sample = data.groupby(['PC1_quantile', 'PC2_quantile'], group_keys=False).apply(
    lambda x: resample(x, n_samples=min(medium_sample_size, len(x)), random_state=42),
    include_groups=False
)

# KMeans clustering with n_clusters set to the target small sample size
final_sample_size = 250  # Desired final sample size
kmeans = KMeans(n_clusters=final_sample_size, init='k-means++', random_state=42)
kmeans.fit(stratified_sample[['PC1', 'PC2']])

# Assign cluster labels to each sample
stratified_sample['cluster_label'] = kmeans.labels_

# Select the sample closest to the cluster center as representative
final_sample = stratified_sample.groupby('cluster_label', group_keys=False).apply(
    lambda x: x.iloc[((x[['PC1', 'PC2']] - kmeans.cluster_centers_[x['cluster_label'].iloc[0], :])**2).sum(axis=1).idxmin()]
    if len(x) < 1 else x.iloc[0],  # Select the first row if there's only one sample
    include_groups=False  # Exclude grouping columns
)

# Organize output
final_sample = final_sample.sort_values(by=sample_id_column)
output_columns = [sample_id_column] + list(features.columns)
final_sample = final_sample[output_columns]

# Save results
final_sample.to_csv('Sample_data.csv', index=False)