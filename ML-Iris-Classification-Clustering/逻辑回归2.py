import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_iris

# 载入鸢尾花数据集
iris = load_iris()
X = iris.data  # 使用花萼长度和花萼宽度作为特征
y = iris.target

# 划分数据集为训练集和测试集（6:4比例）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 初始化逻辑回归模型
logistic_regression = LogisticRegression(C=1.0, multi_class='ovr', solver='liblinear', tol=0.0001, penalty='l2')

# 训练模型
logistic_regression.fit(X_train, y_train)

# 预测训练集和测试集
y_train_pred = logistic_regression.predict(X_train)
y_test_pred = logistic_regression.predict(X_test)

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
