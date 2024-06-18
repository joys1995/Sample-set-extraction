import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 读取大样本和样本数据
df_full = pd.read_csv('__STD_data.csv')
df_sample = pd.read_csv('Kmeans_STD_mod_200_layer_mod_1.csv')

# 排除样本编号列（假设样本编号在第一列）
df_full = df_full.iloc[:, 1:]
df_sample = df_sample.iloc[:, 1:]

# 描述性统计
print("Full Dataset Description:")
print(df_full.describe())
print("\
Sample Dataset Description:")
print(df_sample.describe())

# 创建一个函数来保存图表
def save_plot(fig, filename):
    fig.savefig(filename, dpi=300)  # 提高图片分辨率
    plt.show()

# --- 将所有图表整合到一张图 ---
num_cols = len(df_full.columns)
num_rows = 4  # 直方图、QQ图、密度图、箱线图各占一行

fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3))

for i, col in enumerate(df_full.columns):
    # 直方图
    sns.histplot(df_full[col], kde=True, color='blue', label='Full Dataset', stat='density', ax=axes[0, i])
    sns.histplot(df_sample[col], kde=True, color='red', label='Sample Dataset', stat='density', ax=axes[0, i])
    axes[0, i].set_title(f'Distribution of {col}')
    axes[0, i].legend()

    # QQ图
    stats.probplot(df_full[col], dist="norm", plot=axes[1, i])
    axes[1, i].set_title(f'QQ Plot of Full Dataset - {col}')
    stats.probplot(df_sample[col], dist="norm", plot=axes[2, i])
    axes[2, i].set_title(f'QQ Plot of Sample Dataset - {col}')

    # 密度图
    sns.kdeplot(df_full[col], color='blue', label='Full Dataset', ax=axes[3, i])
    sns.kdeplot(df_sample[col], color='red', label='Sample Dataset', ax=axes[3, i])
    axes[3, i].set_title(f'Density Plot of {col}')
    axes[3, i].legend()

# 调整布局，避免图表重叠
plt.tight_layout()

# 保存最终的图表
save_plot(fig, 'Kmeans_STD_200_mod_layer_mod_1_all_comparisons.png')