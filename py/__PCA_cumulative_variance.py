import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 读取数据集
data = pd.read_csv('__STD_data.csv')

# 提取特征因子列
features = data.iloc[:, 1:]

# 创建PCA对象
pca = PCA()

# 对特征因子列进行PCA分析
pca_result = pca.fit_transform(features)

# 计算累计解释方差比例
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

#  ---  绘制散点图  ---
plt.scatter(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio Plot')

# 保存累计解释方差比例曲线图片
plt.savefig('STD_cumulative_variance_ratio_scatter.png')