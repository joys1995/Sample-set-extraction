import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 读取所有数据集
df_full = pd.read_csv('__STD_data.csv')
df_sample1 = pd.read_csv('Kmeans_STD_400.csv')
df_sample2 = pd.read_csv('Kmeans_STD_mod_400.csv')
df_sample3 = pd.read_csv('Kmeans_STD_200.csv')
df_sample4 = pd.read_csv('Kmeans_STD_mod_200.csv')
df_sample5 = pd.read_csv('Kmeans_STD_mod_200_layer_mod.csv')
df_sample6 = pd.read_csv('Kmeans_STD_mod_200_layer_mod_1.csv')  # 添加新的样本集

# 去除样本编号列
df_full = df_full.iloc[:, 1:]
df_sample1 = df_sample1.iloc[:, 1:]
df_sample2 = df_sample2.iloc[:, 1:]
df_sample3 = df_sample3.iloc[:, 1:]
df_sample4 = df_sample4.iloc[:, 1:]
df_sample5 = df_sample5.iloc[:, 1:]
df_sample6 = df_sample6.iloc[:, 1:]  # 添加新的样本集

# 标准化数据
scaler = StandardScaler()
df_full_scaled = scaler.fit_transform(df_full)
df_sample1_scaled = scaler.transform(df_sample1)
df_sample2_scaled = scaler.transform(df_sample2)
df_sample3_scaled = scaler.transform(df_sample3)
df_sample4_scaled = scaler.transform(df_sample4)
df_sample5_scaled = scaler.transform(df_sample5)
df_sample6_scaled = scaler.transform(df_sample6)  # 添加新的样本集

# 降维到二维空间
pca = PCA(n_components=2)
df_full_pca = pca.fit_transform(df_full_scaled)
df_sample1_pca = pca.transform(df_sample1_scaled)
df_sample2_pca = pca.transform(df_sample2_scaled)
df_sample3_pca = pca.transform(df_sample3_scaled)
df_sample4_pca = pca.transform(df_sample4_scaled)
df_sample5_pca = pca.transform(df_sample5_scaled)
df_sample6_pca = pca.transform(df_sample6_scaled)  # 添加新的样本集

# 创建画布
fig, axes = plt.subplots(3, 2, figsize=(10, 15))

# 定义样本数据集列表
sample_datasets = [df_sample1_pca, df_sample2_pca, df_sample3_pca, df_sample4_pca, df_sample5_pca, df_sample6_pca]  # 添加新的样本集
sample_names = ['Kmeans_STD_400', 'Kmeans_STD_mod_400', 'Kmeans_STD_200', 'Kmeans_STD_mod_200', 'Kmeans_STD_mod_200_layer_mod', 'Kmeans_STD_mod_200_layer_mod_1']  # 添加新的样本集

# 绘制对比分布
for i in range(6):
    row = i // 2
    col = i % 2
    axes[row, col].scatter(df_full_pca[:, 0], df_full_pca[:, 1], color='blue', alpha=0.2, label='Full Dataset')
    axes[row, col].scatter(sample_datasets[i][:, 0], sample_datasets[i][:, 1], color='red', alpha=0.5, label=sample_names[i])
    axes[row, col].set_xlabel('Principal Component 1')
    axes[row, col].set_ylabel('Principal Component 2')
    axes[row, col].set_title(f'{sample_names[i]} vs Full Dataset')
    axes[row, col].legend()

# 保存图像
plt.tight_layout()
plt.savefig('STD_sample_comparison_1.png', dpi=300)