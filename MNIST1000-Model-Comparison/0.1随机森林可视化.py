import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 超参数
meta_batch_size = 5
input_size = 28 * 28  # FashionMNIST图像大小
output_size = 10  # FashionMNIST类别数
num_epochs = 5

# 数据预处理
transform = transforms.Compose([transforms.ToTensor()])
# 使用FashionMNIST替换MNIST
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 选择小样本
small_train_indices = [i for i in range(1000)]  # 选择前1000个样本
small_train_dataset = torch.utils.data.Subset(train_dataset, small_train_indices)

# 将小样本数据集分为训练集和验证集
train_indices, val_indices = train_test_split(small_train_indices, test_size=0.2, random_state=42)

train_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, train_indices),
                                               batch_size=meta_batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, val_indices),
                                             batch_size=meta_batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 准备训练数据
X_train, y_train = [], []
for data, labels in train_dataloader:
    data = data.view(meta_batch_size, -1, input_size).numpy()
    labels = labels.numpy()
    X_train.append(data)
    y_train.append(labels)

X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# 准备验证数据
X_val, y_val = [], []
for data, labels in val_dataloader:
    data = data.view(meta_batch_size, -1, input_size).numpy()
    labels = labels.numpy()
    X_val.append(data)
    y_val.append(labels)

X_val = np.concatenate(X_val, axis=0)
y_val = np.concatenate(y_val, axis=0)

# 准备测试数据
X_test, y_test = [], []
for data, labels in test_dataloader:
    data = data.view(1, -1, input_size).numpy()
    labels = labels.numpy()
    X_test.append(data)
    y_test.append(labels)

X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

# 调整数据形状以适应RandomForestClassifier
X_train = X_train.reshape((X_train.shape[0], -1))
X_val = X_val.reshape((X_val.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# 创建并训练随机森林模型
num_trees_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
val_accuracies = []

for num_trees in num_trees_list:
    rf_model = RandomForestClassifier(n_estimators=num_trees, random_state=42)
    rf_model.fit(X_train, y_train)

    # 在验证集上评估模型
    y_val_pred = rf_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_accuracies.append(val_accuracy)

    print(f'Trees: {num_trees}, Validation Accuracy: {val_accuracy * 100:.2f}%')

# 绘制树的数量与准确率的关系图
plt.plot(num_trees_list, val_accuracies, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs Number of Trees')
plt.show()

# 在测试集上评估模型（使用最佳树的数量）
best_num_trees = num_trees_list[np.argmax(val_accuracies)]
rf_model = RandomForestClassifier(n_estimators=best_num_trees, random_state=42)
rf_model.fit(X_train, y_train)

y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Best Number of Trees: {best_num_trees}, Test Accuracy: {test_accuracy * 100:.2f}%')
