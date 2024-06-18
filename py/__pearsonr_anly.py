import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('Book2.csv')

# 删除样本编号列（假设第一列为样本编号）
df = df.drop(columns=['Sample'])

# 计算皮尔逊相关系数矩阵
corr_matrix = df.corr(method='pearson')

# 设置绘图风格
sns.set(style='white')

# 创建一个绘图对象
plt.figure(figsize=(10, 8))

# 绘制热图，添加 annot_kws 以自定义注释的样式
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot_kws={"size": 10, "color": "black", "weight": "bold"})

# 设置标题
plt.title('Pearson Correlation Coefficient Heatmap')

# 保存热图到文件
plt.savefig('_correlation_heatmap.png')