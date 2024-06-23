import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load all datasets
df_full = pd.read_csv('STD_data.csv')
df_sample1 = pd.read_csv('Sample_data.csv')

# Remove sample ID columns
df_full = df_full.iloc[:, 1:]
df_sample1 = df_sample1.iloc[:, 1:]

# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
df_full_pca = pca.fit_transform(df_full)
df_sample1_pca = pca.transform(df_sample1)

# Create plot
fig, axes = plt.subplots(figsize=(12, 9))

# Define list of sample datasets and their names
sample_datasets = [df_sample1_pca]
sample_names = ['Sample_set']

# Access data and name for plotting
sample_data = sample_datasets[0]
sample_name = sample_names[0]

# Create scatter plot
axes.scatter(df_full_pca[:, 0], df_full_pca[:, 1], color='blue', alpha=0.2, label='Full Dataset')
axes.scatter(sample_data[:, 0], sample_data[:, 1], color='red', alpha=0.5, label=sample_name)
axes.set_xlabel('Principal Component 1')
axes.set_ylabel('Principal Component 2')
axes.set_title(f'{sample_name} vs Full Dataset')
axes.legend()

# Save the figure
plt.tight_layout()
plt.savefig('PCA_comparison.png', dpi=300)