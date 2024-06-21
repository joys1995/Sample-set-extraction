import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load full dataset and sample dataset
df_full = pd.read_csv('STD_data.csv')
df_sample = pd.read_csv('Sample_data.csv')

# Exclude sample ID column (assuming it is the first column)
df_full = df_full.iloc[:, 1:]
df_sample = df_sample.iloc[:, 1:]

# Descriptive statistics
print("Full Dataset Description:")
print(df_full.describe())
print("\
Sample Dataset Description:")
print(df_sample.describe())

# Function to save plots
def save_plot(fig, filename):
    fig.savefig(filename, dpi=300)  # Increase image resolution
    plt.show()

# --- Combine all plots into one figure ---
num_cols = len(df_full.columns)
num_rows = 4  # Each row represents a histogram, QQ plot, density plot, and box plot

fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3))

for i, col in enumerate(df_full.columns):
    # Histogram
    sns.histplot(df_full[col], kde=True, color='blue', label='Full Dataset', stat='density', ax=axes[0, i])
    sns.histplot(df_sample[col], kde=True, color='red', label='Sample Dataset', stat='density', ax=axes[0, i])
    axes[0, i].set_title(f'Distribution of {col}')
    axes[0, i].legend()

    # QQ plot
    stats.probplot(df_full[col], dist="norm", plot=axes[1, i])
    axes[1, i].set_title(f'QQ Plot of Full Dataset - {col}')
    stats.probplot(df_sample[col], dist="norm", plot=axes[2, i])
    axes[2, i].set_title(f'QQ Plot of Sample Dataset - {col}')

    # Density plot
    sns.kdeplot(df_full[col], color='blue', label='Full Dataset', ax=axes[3, i])
    sns.kdeplot(df_sample[col], color='red', label='Sample Dataset', ax=axes[3, i])
    axes[3, i].set_title(f'Density Plot of {col}')
    axes[3, i].legend()

# Adjust layout to prevent overlapping of plots
plt.tight_layout()

# Save the final figure
save_plot(fig, 'Description_comparison.png')