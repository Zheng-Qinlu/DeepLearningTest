import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 超参数
batch_size = 64
input_channels = 1
output_size = 10
learning_rate = 0.001
num_epochs = 5

# 数据预处理
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

small_train_indices = [i for i in range(1000)]
small_train_dataset = Subset(train_dataset, small_train_indices)

train_dataloader = DataLoader(small_train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# LeNet-5 模型
class LeNet5(nn.Module):
    def __init__(self, input_channels, output_size):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, output_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 创建LeNet-5模型
lenet5_model = LeNet5(input_channels, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet5_model.parameters(), lr=learning_rate)

# 添加学习率调度器
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

# 训练模型
for epoch in range(num_epochs):
    for data, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = lenet5_model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 在每个 epoch 结束时进行学习率调度
    scheduler.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()} , Learning Rate: {scheduler.get_last_lr()[0]}')

# 在测试集上评估模型
lenet5_model.eval()

total_correct = 0
total_samples = 0

with torch.no_grad():
    for data, labels in test_dataloader:
        test_outputs = lenet5_model(data)
        _, predicted = torch.max(test_outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print(f'Test Accuracy: {accuracy * 100:.2f}%')
