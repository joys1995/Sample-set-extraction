import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import resample

# 读取数据集
data = pd.read_csv('__STD_data.csv')

sample_id_column = 'Sample'
features = data.iloc[:, 1:]

# PCA降维
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)
data['PC1'] = reduced_features[:, 0]
data['PC2'] = reduced_features[:, 1]

# 分层抽样，得到中型样本
medium_sample_size = 1000  # 可调整参数
for i in range(2):
    data[f'PC{i+1}_quantile'] = pd.qcut(data[f'PC{i+1}'], 4, labels=False, duplicates='drop') 

stratified_sample = data.groupby(['PC1_quantile', 'PC2_quantile'], group_keys=False).apply(
    lambda x: resample(x, n_samples=min(medium_sample_size, len(x)), random_state=42),
    include_groups=False
)

# KMeans聚类，n_clusters设置为目标小样本数量
final_sample_size = 200 # 设置为你想要的最终样本数量
kmeans = KMeans(n_clusters=final_sample_size, init='k-means++', random_state=42)
kmeans.fit(stratified_sample[['PC1', 'PC2']])

# 为每个样本添加所属簇标签
stratified_sample['cluster_label'] = kmeans.labels_

# 从每个簇中选择距离中心点最近的样本作为代表
final_sample = stratified_sample.groupby('cluster_label', group_keys=False).apply(
    lambda x: x.iloc[((x[['PC1', 'PC2']] - kmeans.cluster_centers_[x['cluster_label'].iloc[0], :])**2).sum(axis=1).idxmin()]  
    if len(x) < 1 else x.iloc[0],  # 如果只有一个样本，直接选择第一行
    include_groups=False  # 排除分组列
)

# 整理输出结果
final_sample = final_sample.sort_values(by=sample_id_column)
output_columns = [sample_id_column] + list(features.columns)
final_sample = final_sample[output_columns]

# 保存结果
final_sample.to_csv('Kmeans_STD_mod_200_layer_mod.csv', index=False)