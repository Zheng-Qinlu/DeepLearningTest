"""
基于 ViT 的 CIFAR10 图像分类
实验二：Vision Transformer 图像分类实现
针对 M1 芯片优化版本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import os
import math
from typing import List
import random

# 可选：绘图（若环境未安装 matplotlib，则自动降级为写入 CSV）
try:
    import matplotlib.pyplot as plt  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# 路径设置：将模型固定保存在当前脚本所在的 test2 文件夹中
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'vit_cifar10_best.pth')
PLOT_PATH = os.path.join(SCRIPT_DIR, 'training_curves.png')
CSV_PATH = os.path.join(SCRIPT_DIR, 'training_metrics.csv')


# ==================== 1. 数据预处理和加载 ====================

# 训练集数据增强（保持 CIFAR10 原生 32x32 尺寸）
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
])

# 验证/测试集数据处理（32x32 标准化）
trans_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

# 下载并加载 CIFAR10 数据集
print("加载 CIFAR10 数据集...")
trainset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=True, download=True, transform=trans_train)
    
testset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=False, download=False, transform=trans_valid)

# Batch size和num_workers配置
# M1芯片: batch_size=64, num_workers=2
# CUDA GPU (8GB): batch_size=128, num_workers=4
# CUDA GPU (16GB+): batch_size=256, num_workers=8
batch_size = 64  # 可根据GPU显存调整
num_workers = 2  # 可根据CPU核心数调整

trainloader = DataLoader(trainset, batch_size=batch_size, 
                         shuffle=True, num_workers=num_workers)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

# CIFAR10 类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"训练集样本数: {len(trainset)}")
print(f"测试集样本数: {len(testset)}")


# ==================== 2. 模型构建 ====================

class Attention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        
        return self.to_out(out)


class FeedForward(nn.Module):
    """前向MLP网络"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    """Transformer编码器"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., drop_path_rate: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        # 为每一层分配不同的 drop_path 比例（线性递增）
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist() if drop_path_rate > 0 else [0.0] * depth
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                DropPath(dpr[i]) if dpr[i] > 0 else nn.Identity()
            ]))

    def forward(self, x):
        for attn, ff, drop in self.layers:
            x = x + drop(attn(x))  # 残差连接 + DropPath
            x = x + drop(ff(x))    # 残差连接 + DropPath
        return self.norm(x)


class ViT(nn.Module):
    """Vision Transformer主模型"""
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth,
                 heads, mlp_dim, pool='cls', channels=3, dim_head=64, 
                 dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Patch Embedding
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, drop_path_rate=0.1)

        self.pool = pool
        self.to_latent = nn.Identity()

        # MLP分类头（加入 LayerNorm 提升稳定性）
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Patch embedding
        x = self.to_patch_embedding(img)
        b, c, n = x.shape
        x = x.transpose(1, 2)  # (b, n, c)

        # 添加cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置编码
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Transformer编码
        x = self.transformer(x)

        # 池化
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# ==================== 3. 训练函数 ====================

class DropPath(nn.Module):
    """Stochastic Depth / DropPath.
    在训练时按比例随机丢弃残差分支；推理时为恒等映射。
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # x: (B, N, C)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0/1 mask
        return x.div(keep_prob) * random_tensor

# ==================== Mixup / CutMix 与 EMA ====================

def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """支持软标签的交叉熵（用于 mixup / cutmix）。target_probs 为概率分布。"""
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()

def sample_beta_distribution(alpha: float, device):
    return torch.distributions.Beta(alpha, alpha).sample().to(device)

def apply_mixup_cutmix(inputs: torch.Tensor, targets: torch.Tensor, device, p: float = 0.5, alpha: float = 0.2):
    """随机选择使用 Mixup 或 CutMix；返回混合后的 inputs, targets_onehot, 标记是否混合。
    若未混合则返回原始数据和 one-hot 标签。
    """
    batch_size = inputs.size(0)
    num_classes = 10  # CIFAR10 固定
    targets_onehot = torch.zeros(batch_size, num_classes, device=device)
    targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

    if random.random() > p:
        return inputs, targets_onehot, False

    use_cutmix = random.random() < 0.5
    lam = float(sample_beta_distribution(alpha, device))
    rand_index = torch.randperm(batch_size, device=device)
    targets_shuffled = targets_onehot[rand_index]

    if use_cutmix:
        # CutMix: 随机框
        H, W = inputs.size(2), inputs.size(3)
        cut_rat = math.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        # 随机中心
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        x1 = int(max(cx - cut_w // 2, 0))
        y1 = int(max(cy - cut_h // 2, 0))
        x2 = int(min(cx + cut_w // 2, W))
        y2 = int(min(cy + cut_h // 2, H))
        inputs_mix = inputs.clone()
        inputs_mix[:, :, y1:y2, x1:x2] = inputs[rand_index, :, y1:y2, x1:x2]
        # 重新计算 lam（实际区域占比）
        area = (x2 - x1) * (y2 - y1)
        lam = 1. - area / (H * W)
        targets_mixed = targets_onehot * lam + targets_shuffled * (1 - lam)
        return inputs_mix, targets_mixed, True
    else:
        # Mixup: 全图线性混合
        inputs_mix = inputs * lam + inputs[rand_index] * (1 - lam)
        targets_mixed = targets_onehot * lam + targets_shuffled * (1 - lam)
        return inputs_mix, targets_mixed, True

# EMA 支持
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = self._clone(model)
        self.model.eval()

    def _clone(self, model):
        ema_model = type(model)(
            image_size=32, patch_size=4, num_classes=10, dim=256, depth=6,
            heads=4, mlp_dim=1024, dropout=0.0, emb_dropout=0.0
        )
        ema_model.load_state_dict(model.state_dict())
        return ema_model.to(next(model.parameters()).device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if k in msd:
                v.copy_(v * d + msd[k] * (1 - d))

_EMA_INSTANCE: ModelEMA | None = None

def init_ema(model: nn.Module, decay: float = 0.999):
    global _EMA_INSTANCE
    _EMA_INSTANCE = ModelEMA(model, decay=decay)

def update_ema(model: nn.Module):
    if _EMA_INSTANCE is not None:
        _EMA_INSTANCE.update(model)

def get_ema_net(model: nn.Module):
    return _EMA_INSTANCE.model if _EMA_INSTANCE is not None else model

class EarlyStopping:
    """基于监控指标的早停：当指标在 patience 个 epoch 内未超过 best+min_delta 时触发停止。
    这里默认监控的是 test_acc（越大越好）。
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, current: float) -> bool:
        if self.best is None or current > self.best + self.min_delta:
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop

def train(epoch, net, trainloader, criterion, optimizer, device):
    """训练一个epoch"""
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # 可选 Mixup/CutMix
        mixed_inputs, mixed_targets, mixing_used = apply_mixup_cutmix(inputs, targets, device)

        outputs = net(mixed_inputs)
        # 使用软标签交叉熵以支持 mixup/cutmix 的 one-hot 目标
        loss = soft_cross_entropy(outputs, mixed_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        update_ema(net)  # EMA 更新

        train_loss += loss.item()
        # 计算准确率（对 mixup/cutmix 用原始 targets 评估）
        with torch.no_grad():
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch} 完成 | 用时: {epoch_time:.2f}s | '
          f'平均Loss: {train_loss/len(trainloader):.3f} | '
          f'训练准确率: {100.*correct/total:.3f}%')
    
    return train_loss / len(trainloader)


# ==================== 4. 测试函数 ====================

def test(epoch, net, testloader, criterion, device):
    """测试模型"""
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # 使用 EMA 权重评估（若存在）
    ema_to_eval = get_ema_net(net)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = ema_to_eval(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # 计算准确率
    acc = 100. * correct / total
    print(f'\n测试结果 - Epoch {epoch}:')
    print(f'平均Loss: {test_loss/len(testloader):.3f}')
    print(f'测试准确率: {acc:.3f}% ({correct}/{total})')
    
    # 保存最佳模型
    if acc > best_acc:
        print(f'保存模型... (准确率从 {best_acc:.3f}% 提升到 {acc:.3f}%)')
        state = {
            'net': ema_to_eval.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, CHECKPOINT_PATH)
        best_acc = acc
    
    return test_loss / len(testloader), acc


# ==================== 5. 主函数 ====================

def maybe_load_checkpoint(net, device):
    """如果存在已保存模型，询问用户是否加载。返回 True 如果已加载。"""
    if os.path.isfile(CHECKPOINT_PATH):
        print(f"检测到已存在的模型文件: {CHECKPOINT_PATH}")
        try:
            choice = input("是否加载该模型继续测试而不重新训练? [y/N]: ").strip().lower()
        except Exception:
            choice = 'y'  # 非交互环境默认加载
        if choice == 'y':
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            net.load_state_dict(ckpt['net'])
            acc_val = ckpt.get('acc', None)
            epoch_val = ckpt.get('epoch', '?')
            try:
                acc_str = f"{float(acc_val):.3f}%" if acc_val is not None else "?"
            except Exception:
                acc_str = "?"
            print(f"已加载模型 (保存于 epoch {epoch_val} | acc={acc_str})")
            # 设置全局最佳，避免重复保存
            global best_acc
            try:
                best_acc = float(acc_val) if acc_val is not None else 0.0
            except Exception:
                best_acc = 0.0
            return True
        else:
            print("选择重新训练，将覆盖已有模型。")
    return False

def plot_and_save_curves(history, plot_path=PLOT_PATH, csv_path=CSV_PATH):
    """绘制并保存训练曲线，history 为 dict: epoch, train_loss, test_loss, test_acc, lr"""
    import csv
    # 写入 CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_loss', 'test_acc', 'lr'])
        for i in range(len(history['epoch'])):
            writer.writerow([
                history['epoch'][i],
                f"{history['train_loss'][i]:.6f}",
                f"{history['test_loss'][i]:.6f}",
                f"{history['test_acc'][i]:.4f}",
                f"{history['lr'][i]:.8f}",
            ])
    print(f"训练指标已保存 CSV: {csv_path}")

    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib 不可用，跳过曲线绘制。可后续安装 matplotlib 再绘图。")
        return
    plt.figure(figsize=(10,5))
    epochs = history['epoch']
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curves'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, history['test_acc'], label='Test Acc', color='orange')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy Curve'); plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"训练曲线已保存: {plot_path}")

if __name__ == '__main__':
    # 设置设备 (支持CUDA/MPS/CPU)
    # 可以通过环境变量 CUDA_VISIBLE_DEVICES 指定GPU,例如: export CUDA_VISIBLE_DEVICES=0
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.current_device()}")
        
        # CUDA 性能优化
        torch.backends.cudnn.benchmark = True  # 自动寻找最优算法
        torch.backends.cudnn.deterministic = False  # 提高性能
        
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple M1 GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    # 创建ViT模型 (针对 CIFAR10 调整参数)
    print("\n创建 ViT 模型...")
    net = ViT(
        image_size=32,
        patch_size=4,       # 4x4 的 patch -> 8x8=64 个 patch
        num_classes=10,
        dim=256,
        depth=6,            # 6层Transformer编码器
        heads=4,            # 4个注意力头
        mlp_dim=1024,       # MLP隐藏层维度
        dropout=0.0,
        emb_dropout=0.0
    )
    
    net = net.to(device)
    # 初始化 EMA
    init_ema(net, decay=0.9995)
    
    # 统计参数量
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.05)

    # 训练配置
    num_epochs = 50  # 若需要更长训练可提升到 150
    warmup_epochs = 10

    # Warmup + Cosine 学习率调度
    def lr_lambda(epoch_idx: int):
        # epoch_idx 从 0 开始，代表已完成的 epoch 数
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        # 余弦部分范围 [0, 1]
        denom = max(1, (num_epochs - warmup_epochs - 1))
        progress = float(epoch_idx - warmup_epochs) / float(denom)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    best_acc = 0

    already_loaded = maybe_load_checkpoint(net, device)
    # 同步 EMA 到当前网络权重（无论是否加载过 checkpoint）
    init_ema(net, decay=0.9995)

    if already_loaded:
        # 直接评估一次
        test_loss, test_acc = test(0, net, testloader, criterion, device)
        print(f"直接使用已保存模型完成测试: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
    else:
        history = { 'epoch': [], 'train_loss': [], 'test_loss': [], 'test_acc': [], 'lr': [] }
        early_stopper = EarlyStopping(patience=5, min_delta=0.001)
        print(f"\n开始训练 {num_epochs} 个 epochs...")
        print("=" * 70)
        for epoch in range(1, num_epochs + 1):
            train_loss = train(epoch, net, trainloader, criterion, optimizer, device)
            test_loss, test_acc = test(epoch, net, testloader, criterion, device)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.6f}")
            print("=" * 70)
            # 日志记录
            with open('training_log.txt', 'a') as f:
                f.write(f'Epoch {epoch}: Train Loss={train_loss:.4f}, '
                        f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%, '
                        f'LR={current_lr:.6f}\n')
            # 收集历史
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['lr'].append(current_lr)
            # 早停检测（基于测试准确率）
            if early_stopper.step(test_acc):
                print(f"早停触发：连续 {early_stopper.patience} 个 epoch 未超过最佳 {early_stopper.best:.3f}%+{early_stopper.min_delta} 的提升。提前结束训练。")
                break
        print(f"\n训练完成!")
        print(f"最佳测试准确率: {best_acc:.3f}%")
        print(f"模型已保存至: {CHECKPOINT_PATH}")
        # 绘制与保存曲线
        plot_and_save_curves(history)
