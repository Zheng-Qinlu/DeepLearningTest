import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_iris

# 载入鸢尾花数据集
iris = load_iris()
X = iris.data  # 使用花萼长度和花萼宽度作为特征
y = iris.target

# 划分数据集为训练集和测试集（6:4比例）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 定义要搜索的参数值，包括penalty
param_grid = {'tol': [1e-4, 1e-3, 1e-2, 1e-1],
              'C': [0.01, 0.1, 0.5, 0.8, 1.0, 5.0],
              'penalty': ['l1', 'l2']}

# 初始化LinearSVC模型
svm_classifier = LinearSVC()

# 使用GridSearchCV进行参数搜索
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 获取最佳参数和最佳交叉验证分数
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# 使用最佳参数来训练模型
best_svm_classifier = LinearSVC(C=best_params['C'], tol=best_params['tol'], penalty=best_params['penalty'], multi_class='ovr')
best_svm_classifier.fit(X_train, y_train)

# 预测测试集
y_test_pred = best_svm_classifier.predict(X_test)

# 计算准确率
acc_test = accuracy_score(y_test, y_test_pred)

# 计算F1分数、精确率和召回率
f1_macro = f1_score(y_test, y_test_pred, average='macro')
precision_macro = precision_score(y_test, y_test_pred, average='macro')
recall_macro = recall_score(y_test, y_test_pred, average='macro')

# 打印结果
print(f"最佳参数: {best_params}")
print(f"最佳交叉验证分数: {best_score}")
print(f"测试集准确率：{acc_test}")
print(f"测试集F1分数：{f1_macro}")
print(f"测试集精确率：{precision_macro}")
print(f"测试集召回率：{recall_macro}")
