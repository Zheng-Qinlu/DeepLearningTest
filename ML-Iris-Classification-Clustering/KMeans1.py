import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_iris


def evaluate_clustering(X, y, n_clusters):
    # 初始化K均值聚类模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

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

    return y_pred, silhouette_avg, acc, f1_macro, precision_macro, recall_macro, kmeans


def visualize_clusters(X, y_pred, cluster_centers, class_names):
    # 可视化聚类结果及聚类中心
    plt.figure()
    for i in range(len(class_names)):
        plt.scatter(X[y_pred == i][:, 0], X[y_pred == i][:, 1], label=f'{class_names[i]}')
        plt.text(cluster_centers[i, 0], cluster_centers[i, 1], f'{class_names[i]}', fontsize=12, color='red')

    plt.xlabel('花萼长度')
    plt.ylabel('花萼宽度')
    plt.title('鸢尾花K均值聚类结果')
    plt.legend()
    plt.show()


# 载入鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]  # 使用花萼长度和花萼宽度作为特征
y = iris.target

# 类别名称
class_names = iris.target_names

n_clusters = 3

# 评估聚类结果
y_pred, silhouette_avg, acc, f1_macro, precision_macro, recall_macro, kmeans = evaluate_clustering(X, y, n_clusters)

# 打印结果
print(f"轮廓系数：{silhouette_avg}")
print(f"正确率：{acc}")
print(f"F1分数：{f1_macro}")
print(f"精确率：{precision_macro}")
print(f"召回率：{recall_macro}")

# 可视化聚类结果
visualize_clusters(X, y_pred, kmeans.cluster_centers_, class_names)
