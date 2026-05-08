import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time  # 引入时间库

# 数据加载与预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# 构建神经网络模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NeuralNet()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
# 使用SGD with Momentum，设置学习率和动量参数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 你可以根据需要调整学习率和动量值

# 训练和验证
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

start_time = time.time()  # 记录训练开始时间

for epoch in range(1, 51):  # 从1开始，以便更容易输出第4个Epoch的结果
    train_loss = 0.0
    train_correct = 0
    val_loss = 0.0
    val_correct = 0

    # 训练集上的训练
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

    # 验证集上的验证
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

    if epoch == 4:  # 输出第4个Epoch的结果
        print(f"第4个Epoch的训练损失值: {train_losses[3]}")
        print(f"第4个Epoch的训练正确率: {train_accuracies[3]}")
        print(f"第4个Epoch的验证损失值: {val_losses[3]}")
        print(f"第4个Epoch的验证正确率: {val_accuracies[3]}")

end_time = time.time()  # 训练结束时间
training_time_seconds = end_time - start_time  # 计算训练时间（秒）
print(f"训练时间: {training_time_seconds} 秒")

# 绘制训练和验证损失图
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), train_losses, label='Train Loss')
plt.plot(range(1, 51), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练和验证正确率图
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), train_accuracies, label='Train Accuracy')
plt.plot(range(1, 51), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 输出最后一个Epoch的训练和验证结果
final_train_loss = train_losses[-1]
final_train_accuracy = train_accuracies[-1]
final_val_loss = val_losses[-1]
final_val_accuracy = val_accuracies[-1]
print(f"最后一个Epoch的训练损失值: {final_train_loss}")
print(f"最后一个Epoch的训练正确率: {final_train_accuracy}")
print(f"最后一个Epoch的验证损失值: {final_val_loss}")
print(f"最后一个Epoch的验证正确率: {final_val_accuracy}")

# 训练完成后在测试集上进行测试
test_correct = 0
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == target).sum().item()

test_accuracy = test_correct / len(test_dataset)
print(f"测试正确率: {test_accuracy}")

# 输出模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params}")
