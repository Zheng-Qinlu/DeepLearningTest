import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 超参数
meta_batch_size = 5
input_size = 28 * 28  # FashionMNIST图像大小
hidden_size = 20
output_size = 10  # FashionMNIST类别数
learning_rate = 0.001
num_epochs = 5

# 数据预处理
transform = transforms.Compose([transforms.ToTensor()])
# 使用FashionMNIST替换MNIST
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 选择小样本
small_train_indices = [i for i in range(1000)]  # 选择前1000个样本
small_train_dataset = Subset(train_dataset, small_train_indices)

train_dataloader = DataLoader(small_train_dataset, batch_size=meta_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 为了在测试时逐个样本进行

# 修改后的元学习器模型，添加批量归一化
class MetaLearner(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MetaLearner, self).__init__()

        # 共享的LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 批量归一化层
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # 初始化LSTM权重
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                # 设置遗忘门的偏置为较大的值
                param.data.fill_(1.0)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # 对LSTM的最后一维进行批量归一化
        lstm_out = self.batch_norm(lstm_out.transpose(2, 1)).transpose(2, 1)

        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

# 创建元学习器模型
meta_model = MetaLearner(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(meta_model.parameters(), lr=learning_rate)

# 添加学习率调度器
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)  # 可以根据需要调整 step_size 和 gamma 参数

# 记录训练过程中的损失和学习率
train_losses = []
learning_rates = []

# 训练模型
for epoch in range(num_epochs):
    epoch_losses = []

    for data, labels in train_dataloader:
        data = data.view(meta_batch_size, -1, input_size)  # 调整数据形状以适应LSTM输入
        optimizer.zero_grad()
        outputs = meta_model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    # 在每个 epoch 结束时进行学习率调度
    scheduler.step()

    # 记录训练过程中的损失和学习率
    train_losses.append(np.mean(epoch_losses))
    learning_rates.append(scheduler.get_last_lr()[0])

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_losses[-1]} , Learning Rate: {learning_rates[-1]}')

# 在测试集上评估模型
meta_model.eval()

# 计算测试准确率
total_correct = 0
total_samples = 0

with torch.no_grad():
    for data, labels in test_dataloader:
        data = data.view(1, -1, input_size)  # 调整数据形状以适应LSTM输入
        test_outputs = meta_model(data)
        _, predicted = torch.max(test_outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 可视化损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# 可视化学习率变化曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), learning_rates, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Over Epochs')
plt.legend()
plt.show()
