import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_iris

# 载入鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]  # 使用花萼长度和花萼宽度作为特征
y = iris.target

# 划分数据集为训练集和测试集（6:4比例）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 初始化SVM模型
svm_classifier = SVC(C=0.5,decision_function_shape='ovr')

# 训练模型
svm_classifier.fit(X_train, y_train)

# 预测训练集和测试集
y_train_pred = svm_classifier.predict(X_train)
y_test_pred = svm_classifier.predict(X_test)

# 计算准确率
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

# 计算F1分数、精确率和召回率
f1_macro = f1_score(y_test, y_test_pred, average='macro')
precision_macro = precision_score(y_test, y_test_pred, average='macro')
recall_macro = recall_score(y_test, y_test_pred, average='macro')

# 打印结果
print(f"训练集准确率：{acc_train}")
print(f"测试集准确率：{acc_test}")
print(f"测试集F1分数：{f1_macro}")
print(f"测试集精确率：{precision_macro}")
print(f"测试集召回率：{recall_macro}")

# 可视化决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, marker='o', s=40, edgecolor='k')
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.title('鸢尾花分类结果 (SVM)')
plt.show()
