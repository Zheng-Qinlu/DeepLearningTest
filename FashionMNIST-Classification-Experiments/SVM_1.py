import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import mnist
import time

# 导入MNIST数据集
mnist = datasets.fetch_openml('mnist_784')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14, random_state=42)

# 创建SVM模型
start_time = time.time()
svm_model = SVC(C=5.0, gamma=0.05, kernel='rbf', max_iter=100)

# 训练SVM模型
svm_model.fit(X_train, y_train)
end_time = time.time()

# 预测测试数据
y_pred = svm_model.predict(X_test)

# 输出训练时间
training_time = end_time - start_time
print(f"训练时间: {training_time:.2f} 秒")

# 计算训练集和测试集正确率
train_accuracy = accuracy_score(y_train, svm_model.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)
print(f"训练集准确率: {train_accuracy:.2%}")
print(f"测试集准确率: {test_accuracy:.2%}")

# 输出每个类别的正确率
class_report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)], output_dict=True)
print("每个类别的正确率:")
for i in range(10):
    class_name = str(i)
    precision = class_report[class_name]["precision"]
    print(f"类别 {class_name}: {precision:.2%}")

# 生成混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("混淆矩阵")
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, [str(i) for i in range(10)], rotation=45)
plt.yticks(tick_marks, [str(i) for i in range(10)])
plt.ylabel('实际标签')
plt.xlabel('预测标签')

for i in range(10):
    for j in range(10):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

plt.show()
