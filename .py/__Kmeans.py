import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 从CSV文件中加载数据
data = pd.read_csv('__STD_data.csv')

# 提取特征数据（排除样本编号列）
features = data.iloc[:, 1:].values

# 设置聚类模型参数
num_clusters = 200  # 聚类的数量

# 创建K-means聚类模型
kmeans = KMeans(n_clusters=num_clusters)

# 对特征数据进行聚类
kmeans.fit(features)

# 获取每个样本的聚类标签
labels = kmeans.labels_

# 创建一个新的DataFrame来保存聚类结果
clustered_data = pd.DataFrame({'Sample': data.iloc[:, 0], 'Cluster': labels})

# 提取每个聚类的代表子集
representative_samples = clustered_data.groupby('Cluster').first().reset_index()

# 按样本编号排序
representative_samples_sorted = representative_samples.sort_values(by='Sample')

# 将代表子集保存为新的CSV文件（包含完整样本信息）
representative_samples_full = data[data.iloc[:, 0].isin(representative_samples_sorted['Sample'])]
representative_samples_full_sorted = representative_samples_full.sort_values(by=data.columns[0])
representative_samples_full_sorted.to_csv('Kmeans_STD_200.csv', index=False)

"""
# 可视化聚类结果并保存为图片
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'gray']  # 每个簇的颜色

# 仅绘制代表子集
for cluster in range(num_clusters):
    rep_sample = representative_samples_sorted[representative_samples_sorted['Cluster'] == cluster]
    sample_index = rep_sample.index[0]
    plt.scatter(data.iloc[sample_index, 1], data.iloc[sample_index, 2], c=colors[cluster], label='Cluster {}'.format(cluster))

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Result (Representative Samples)')
plt.legend()
plt.savefig('clustering_result.png')

print("聚类结果已保存为 'clustering_result.png'")
"""