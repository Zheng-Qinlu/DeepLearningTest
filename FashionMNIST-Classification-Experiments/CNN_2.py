import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report

# 数据加载与预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

# 划分训练、验证、测试集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 构建卷积神经网络模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc1_dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc1_dropout(x)  # 在全连接层添加Dropout
        x = self.fc2(x)
        return x

model = ConvNet()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练和验证
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

start_time = time.time()

for epoch in range(1, 51):
    train_loss = 0.0
    train_correct = 0
    val_loss = 0.0
    val_correct = 0

    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == target).sum().item()

    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == target).sum().item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_accuracies.append(train_correct / train_size)
    val_accuracies.append(val_correct / val_size)

    if epoch == 50:
        final_train_loss = train_losses[-1]
        final_train_accuracy = train_accuracies[-1]
        final_val_loss = val_losses[-1]
        final_val_accuracy = val_accuracies[-1]

end_time = time.time()
training_time_seconds = end_time - start_time

# 测试正确率
test_correct = 0
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == target).sum().item()
        all_predictions.extend(predicted.tolist())
        all_targets.extend(target.tolist())

test_accuracy = test_correct / len(test_dataset)

# 输出模型参数量
total_params = sum(p.numel() for p in model.parameters())

# 输出每个类别的正确率
class_correct = list(0. for _ in range(10))
class_total = list(0. for _ in range(10))

for data, target in test_loader:
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == target).squeeze()
    for i in range(len(target)):
        label = target[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(10)]

# 输出混淆矩阵
confusion = confusion_matrix(all_targets, all_predictions)
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 输出训练和验证损失图
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), train_losses, label='Train Loss')
plt.plot(range(1, 51), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 输出训练和验证正确率图
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), train_accuracies, label='Train Accuracy')
plt.plot(range(1, 51), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 输出结果
print(f"模型参数量: {total_params}")
print(f"最后一个Epoch的训练损失值: {final_train_loss}")
print(f"最后一个Epoch的训练正确率: {final_train_accuracy}")
print(f"最后一个Epoch的验证损失值: {final_val_loss}")
print(f"最后一个Epoch的验证正确率: {final_val_accuracy}")
print(f"测试正确率: {test_accuracy}")
print(f"训练时间: {training_time_seconds} 秒")
print("每个类别的正确率:")
for i, label in enumerate(class_labels):
    print(f"{label}: {class_accuracy[i]:.2f}%")
print("混淆矩阵:")
print(confusion)
print("分类报告:")
print(classification_report(all_targets, all_predictions, target_names=class_labels))
