import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 超参数
meta_batch_size = 5
input_size = 28 * 28  # MNIST图像大小
hidden_size = 20
output_size = 10  # MNIST类别数
learning_rate = 0.001
num_epochs = 5

# 数据预处理
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=meta_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 为了在测试时逐个样本进行

# 修改后的元学习器模型，添加参数共享
class MetaLearner(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MetaLearner, self).__init__()

        # 共享的LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

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
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

# 创建元学习器模型
meta_model = MetaLearner(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(meta_model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for data, labels in train_dataloader:
        data = data.view(meta_batch_size, -1, input_size)  # 调整数据形状以适应LSTM输入
        optimizer.zero_grad()
        outputs = meta_model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 在测试集上评估模型
meta_model.eval()

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
