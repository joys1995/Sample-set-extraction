import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv("__Book_Full.csv")

# 选择特征因子列
feature_cols = data.columns[1:11]

# 标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[feature_cols])

# 更新数据
data[feature_cols] = scaled_data

# 保存为.csv文件
data.to_csv("STD_data.csv", index=False)