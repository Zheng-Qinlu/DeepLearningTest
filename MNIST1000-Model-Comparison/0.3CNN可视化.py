import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 超参数
batch_size = 64
input_channels = 1
output_size = 10
learning_rate = 0.001
num_epochs = 5

# 数据预处理
transform = transforms.Compose([transforms.ToTensor()])
# 使用FashionMNIST替换MNIST
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 选择小样本
small_train_indices = [i for i in range(1000)]
small_train_dataset = Subset(train_dataset, small_train_indices)

train_dataloader = DataLoader(small_train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(SimpleCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(32 * 7 * 7, output_size)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 创建CNN模型
simple_cnn_model = SimpleCNN(input_channels, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(simple_cnn_model.parameters(), lr=learning_rate)

# 添加学习率调度器
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

# 存储每个epoch的损失值和学习率
losses = []
learning_rates = []

# 训练模型
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for data, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = simple_cnn_model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 在每个 epoch 结束时进行学习率调度
    scheduler.step()

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    losses.append(avg_epoch_loss)
    learning_rates.append(scheduler.get_last_lr()[0])

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss} , Learning Rate: {scheduler.get_last_lr()[0]}')

# 绘制损失曲线
plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.show()

# 绘制学习率变化曲线
plt.plot(range(1, num_epochs + 1), learning_rates, marker='o', linestyle='-', color='r')
plt.title('Learning Rate Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()

# 在测试集上评估模型
simple_cnn_model.eval()

total_correct = 0
total_samples = 0

with torch.no_grad():
    for data, labels in test_dataloader:
        test_outputs = simple_cnn_model(data)
        _, predicted = torch.max(test_outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print(f'Test Accuracy: {accuracy * 100:.2f}%')
