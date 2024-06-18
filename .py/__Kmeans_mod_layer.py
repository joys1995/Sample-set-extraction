import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# 加载CSV文件中的数据
data = pd.read_csv('__STD_data.csv')

# 提取特征（排除样本ID列）
features = data.iloc[:, 1:].values

# 设置聚类模型参数
num_clusters = 40

# 创建并拟合K-means聚类模型 (using original features)
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
kmeans.fit(features)  # Fit to non-standardized features

# 获取每个样本的簇标签
labels = kmeans.labels_

# 创建一个DataFrame存储聚类结果
clustered_data = pd.DataFrame({'Sample': data.iloc[:, 0], 'Cluster': labels})

# 将原始特征添加到clustered_data数据框中
for i in range(features.shape[1]):
    clustered_data[f'Feature_{i}'] = features[:, i]  # Use original features

# 用于使用分位数分层采样找到代表性样本的函数
def find_representative_sample(cluster_points, cluster_center, cluster_indices):
    # 计算到簇中心的距离
    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
    
    # 将距离添加到DataFrame
    df = pd.DataFrame({'Index': cluster_indices, 'Distance': distances})
    
    # 按分位数分层
    quantiles = pd.qcut(df['Distance'], q=5, labels=False)
    df['Quantile'] = quantiles
    
    # 从每个分位数中选择一个样本（每个分位数中距离最小的样本）
    representative_samples = df.groupby('Quantile').apply(lambda x: x.loc[x['Distance'].idxmin()], include_groups=False)
    
    return representative_samples['Index'].values

# 在每个簇中找到代表性样本
representative_samples = []
for cluster in range(num_clusters):
    cluster_points = features[labels == cluster]  # Use original features here
    cluster_center = kmeans.cluster_centers_[cluster]
    cluster_indices = clustered_data[clustered_data['Cluster'] == cluster].index
    
    representative_indices = find_representative_sample(cluster_points, cluster_center, cluster_indices)
    for index in representative_indices:
        representative_samples.append(data.loc[index])

# 创建代表性样本的DataFrame
representative_samples_df = pd.DataFrame(representative_samples)

# 按样本ID排序
representative_samples_sorted = representative_samples_df.sort_values(by='Sample')

# 将代表性样本保存到新的CSV文件中
representative_samples_sorted.to_csv('Kmeans_STD_mod_200_layer.csv', index=False) 