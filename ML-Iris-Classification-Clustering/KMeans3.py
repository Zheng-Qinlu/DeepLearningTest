import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_iris

# 载入鸢尾花数据集
iris = load_iris()
X = iris.data # 使用所有特征
y = iris.target

# 初始化K均值聚类模型，设置聚类数为2
kmeans = KMeans(n_clusters=2, random_state=42)

# 训练聚类模型
y_pred = kmeans.fit_predict(X)

# 计算轮廓系数
silhouette_avg = silhouette_score(X, y_pred)

# 计算准确率
acc = accuracy_score(y, y_pred)

# 计算F1分数、精确率和召回率
f1_macro = f1_score(y, y_pred, average='macro')
precision_macro = precision_score(y, y_pred, average='macro')
recall_macro = recall_score(y, y_pred, average='macro')

# 打印结果
print(f"轮廓系数：{silhouette_avg}")
print(f"正确率：{acc}")
print(f"F1分数：{f1_macro}")
print(f"精确率：{precision_macro}")
print(f"召回率：{recall_macro}")

