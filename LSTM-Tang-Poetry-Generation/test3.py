"""
实验3: 自动写诗
基于LSTM的唐诗生成模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random

# ==================== 配置参数 ====================
class Config:
    data_path = '/home/zql/test3/tang.npz'  # 数据集路径

    # 模型参数（提高容量）
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.3

    # 训练参数（扩大计算规模）
    batch_size = 256
    grad_accum_steps = 2
    num_epochs = 15
    lr = 0.0015

    # 生成参数（已用于改进生成质量与结构约束）
    max_gen_len = 125
    temperature = 0.8
    top_k = 50
    top_p = 0.9
    # 额外禁止的特殊token，避免出现 </s> 等
    ban_tokens = ['<START>', '<EOP>', '</s>', '<PAD>', '<pad>', '<UNK>', '<unk>']
    # 结构化生成控制：
    # - target_line_len 对“藏头诗”仍然生效；
    # - 普通结构化生成会自动使用起始句的字数作为每句长度；
    # - target_num_lines 为 None 时启用“随机句数（偏向4/8）”策略。
    target_num_lines = None   # None 表示自动随机选择；设置为正整数则固定
    target_line_len = 7       # 藏头诗每句字数（默认7）；普通生成忽略此项

    # 模型保存/加载（固定保存到脚本所在的 test3 目录）
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, 'poetry_model.pth')
    use_pretrained = True

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== 数据准备 ====================
def prepare_data(data_path):
    """
    加载唐诗数据集
    返回: dataloader, ix2word, word2ix
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件 {data_path} 不存在！")

    datas = np.load(data_path, allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()

    print(f"数据集加载成功！")
    print(f"诗词数量: {len(data)}")
    print(f"词汇表大小: {len(word2ix)}")

    data = torch.from_numpy(data).long()

    dataloader = DataLoader(
        data,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=2,  # Linux上可提升IO
        drop_last=True
    )

    return dataloader, ix2word, word2ix


# ==================== 模型构建 ====================
class PoetryModel(nn.Module):
    """
    诗歌生成模型：多层LSTM + Dropout + 线性输出
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size()
        embeds = self.embeddings(input)

        if hidden is None:
            h_0 = input.new_zeros((self.num_layers, batch_size, self.hidden_dim), dtype=torch.float32)
            c_0 = input.new_zeros((self.num_layers, batch_size, self.hidden_dim), dtype=torch.float32)
        else:
            h_0, c_0 = hidden

        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.dropout(output)
        output = self.linear(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden


# ==================== 模型训练 ====================
def train(dataloader, word2ix, ix2word):
    """
    训练模型：每个 epoch 打印摘要信息，使用梯度累积与LR调度
    """
    vocab_size = len(word2ix)

    model = PoetryModel(
        vocab_size=vocab_size,
        embedding_dim=Config.embedding_dim,
        hidden_dim=Config.hidden_dim,
        num_layers=Config.num_layers,
        dropout=Config.dropout,
    ).to(Config.device)

    # 预训练加载
    start_epoch = 0
    if Config.use_pretrained and os.path.exists(Config.model_path):
        try:
            checkpoint = torch.load(Config.model_path, map_location=Config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"✓ 加载预训练模型成功，从第 {start_epoch + 1} 轮继续训练")
        except Exception as e:
            print(f"✗ 加载预训练模型失败: {e}")
            print("从头开始训练...")

    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # 加载优化器状态
    if Config.use_pretrained and os.path.exists(Config.model_path):
        try:
            checkpoint = torch.load(Config.model_path, map_location=Config.device)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            pass

    print(f"\n开始训练...")
    print(f"设备: {Config.device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练配置: batch_size={Config.batch_size}, grad_accum_steps={Config.grad_accum_steps}, epochs={Config.num_epochs}, lr={Config.lr}")
    print(f"总批次数: {len(dataloader)} 批次/轮")
    print("=" * 60)

    import time
    train_start_time = time.time()

    for epoch in range(start_epoch, start_epoch + Config.num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        step_in_epoch = 0
        optimizer.zero_grad(set_to_none=True)
        epoch_start_time = time.time()

        print(f"\nEpoch [{epoch+1}/{start_epoch + Config.num_epochs}] 开始")

        for batch_idx, data in enumerate(dataloader):
            data = data.to(Config.device)

            input_data = data[:, :-1]
            target = data[:, 1:]

            output, _ = model(input_data)
            loss = criterion(output, target.reshape(-1))

            # 梯度累积
            loss = loss / max(1, Config.grad_accum_steps)
            loss.backward()
            step_in_epoch += 1

            if step_in_epoch % Config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * max(1, Config.grad_accum_steps)
            batch_count += 1

        # 如果最后还有未step的梯度
        if step_in_epoch % Config.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(batch_count, 1)
        epoch_time = time.time() - epoch_start_time
        print(f"  ✓ Epoch完成 | 平均Loss: {avg_loss:.4f} | 耗时: {epoch_time:.1f}s | 学习率: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        if (epoch + 1) % 2 == 0:
            start_words = '湖光秋月两相和'
            gen_poetry = generate(model, start_words, ix2word, word2ix)
            print(f"  生成示例: {gen_poetry}")
        print('-' * 60)

    total_time = time.time() - train_start_time
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"总训练时间: {total_time/60:.2f} 分钟 ({total_time:.1f} 秒)")
    print(f"平均每轮时间: {total_time/Config.num_epochs:.1f} 秒")
    print(f"{'='*60}")

    checkpoint = {
        'epoch': start_epoch + Config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': Config.embedding_dim,
            'hidden_dim': Config.hidden_dim,
            'num_layers': Config.num_layers,
            'dropout': Config.dropout,
        },
    }
    torch.save(checkpoint, Config.model_path)
    print(f"✓ 模型已保存至 {Config.model_path}")

    return model


# ==================== 加载已训练模型 ====================
def load_model(word2ix):
    """
    加载已训练的模型
    """
    if not os.path.exists(Config.model_path):
        raise FileNotFoundError(f"模型文件 {Config.model_path} 不存在！请先训练模型。")
    
    # 加载checkpoint
    checkpoint = torch.load(Config.model_path, map_location=Config.device)
    
    # 获取模型配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        vocab_size = config['vocab_size']
        embedding_dim = config['embedding_dim']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        dropout = config['dropout']
    else:
        # 兼容旧版本保存格式
        vocab_size = len(word2ix)
        embedding_dim = Config.embedding_dim
        hidden_dim = Config.hidden_dim
        num_layers = Config.num_layers
        dropout = Config.dropout
    
    # 创建模型
    model = PoetryModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(Config.device)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 兼容旧版本
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✓ 模型加载成功！")
    
    return model


# ==================== 模型预测/生成 ====================
def _sample_next_token(logits: torch.Tensor, ix2word, word2ix, blocked_indices=None):
    """根据温度/Top-K/Top-P进行采样，返回采样到的token索引。
    可选 blocked_indices: 在采样时强制屏蔽的一组token索引（例如标点或特殊符号）。
    """
    # 温度缩放
    logits = logits / max(Config.temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)

    vocab_size = probs.size(-1)

    # 禁止某些token（如 <START>）
    if Config.ban_tokens:
        for tok in Config.ban_tokens:
            if tok in word2ix:
                probs[word2ix[tok]] = 0.0

    # 额外屏蔽集合
    if blocked_indices:
        for idx in blocked_indices:
            if 0 <= idx < vocab_size:
                probs[idx] = 0.0

    # Top-K
    if isinstance(Config.top_k, int) and Config.top_k > 0 and Config.top_k < vocab_size:
        topk_probs, topk_idx = torch.topk(probs, Config.top_k)
        mask = torch.zeros_like(probs)
        mask[topk_idx] = topk_probs
        probs = mask

    # Top-P（核采样）
    if isinstance(Config.top_p, float) and 0.0 < Config.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative <= Config.top_p
        # 至少保留一个
        if not torch.any(mask):
            mask[0] = True
        kept_idx = sorted_idx[mask]
        kept_probs = probs[kept_idx]
        probs = torch.zeros_like(probs)
        probs[kept_idx] = kept_probs

    # 归一化并采样
    if torch.sum(probs) <= 0:
        # 退化情况下回退到最大概率
        next_idx = int(torch.argmax(probs).item())
    else:
        probs = probs / torch.sum(probs)
        next_idx = int(torch.multinomial(probs, num_samples=1).item())
    return next_idx


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    生成自由续写诗句（不强制句数/字数）
    start_words: 起始文字，如 "湖光秋月两相和"
    prefix_words: 未使用占位
    """
    results = list(start_words)
    start_words_len = len(start_words)

    input = torch.tensor([word2ix['<START>']], dtype=torch.long, device=Config.device).view(1, 1)
    hidden = None

    model.eval()
    with torch.no_grad():
        for i in range(Config.max_gen_len):
            output, hidden = model(input, hidden)

            if i < start_words_len:
                w = results[i]
                if w not in word2ix:
                    break
                next_idx = word2ix[w]
            else:
                logits = output.data[0]
                next_idx = _sample_next_token(logits, ix2word, word2ix)
                w = ix2word[next_idx]
                results.append(w)

            if w == '<EOP>':
                if len(results) > 0:
                    results.pop()
                break

            input = torch.tensor([next_idx], dtype=torch.long, device=Config.device).view(1, 1)

    return ''.join(results)


def generate_poem_structured(model, start_words, ix2word, word2ix, num_lines=None):
    """
    结构化生成：
    - 句子数量由 num_lines 指定（默认使用 Config.target_num_lines，可为 4/8/12/...）。
    - 每句字数 = 起始句（start_words）的字数。
    - 末尾标点交替：奇数行用 '，'，偶数行用 '。'。
    """
    poem_lines = []

    comma_idx = word2ix.get('，', None)
    period_idx = word2ix.get('。', None)

    # 自动确定句数（优先 4/8，偶尔 12，极少数更长偶数）
    def _choose_num_lines():
        r = random.random()
        if r < 0.50:
            return 4
        elif r < 0.85:
            return 8
        elif r < 0.95:
            return 12
        else:
            # 从 14~24 的偶数里随机挑一个
            candidates = [n for n in range(14, 26) if n % 2 == 0]
            return random.choice(candidates)

    if num_lines is None or num_lines <= 0:
        if getattr(Config, 'target_num_lines', None):
            num_lines = Config.target_num_lines
        else:
            num_lines = _choose_num_lines()

    # 计算起始句有效字数（排除标点）
    def _effective_len(s: str):
        if not s:
            return Config.target_line_len
        banned = set(['，', '。', ',', '.', '！', '？'])
        return max(1, sum(1 for ch in s if ch not in banned))

    line_len = _effective_len(start_words)

    model.eval()
    with torch.no_grad():
        for line_idx in range(num_lines):
            hidden = None
            input = torch.tensor([word2ix['<START>']], dtype=torch.long, device=Config.device).view(1, 1)

            # 前缀：第一句使用 start_words，其余为空
            prefix = start_words if (line_idx == 0 and start_words) else ''
            line_chars = []

            # 先“喂入”前缀字符
            for ch in prefix:
                if ch not in word2ix:
                    continue
                if len(line_chars) < line_len:
                    next_idx = word2ix[ch]
                    line_chars.append(ch)
                    input = torch.tensor([next_idx], dtype=torch.long, device=Config.device).view(1, 1)
                    # 前向走一步以更新隐藏态
                    _, hidden = model(input, hidden)
                else:
                    break

            while len(line_chars) < line_len:
                output, hidden = model(input, hidden)
                logits = output.data[0]

                block = []
                if comma_idx is not None:
                    block.append(comma_idx)
                if period_idx is not None:
                    block.append(period_idx)

                next_idx = _sample_next_token(logits, ix2word, word2ix, blocked_indices=block)
                w = ix2word[next_idx]

                if w in ['<EOP>'] or w.strip() == '':
                    # 重采一次
                    block.append(next_idx)
                    next_idx = _sample_next_token(logits, ix2word, word2ix, blocked_indices=block)
                    w = ix2word[next_idx]

                line_chars.append(w)
                input = torch.tensor([next_idx], dtype=torch.long, device=Config.device).view(1, 1)

            # 结尾标点交替：第1、3、5...行用逗号，第2、4、6...行用句号
            end_punc = '，' if (line_idx % 2 == 0) else '。'
            line = ''.join(line_chars) + end_punc
            poem_lines.append(line)

    return '\n'.join(poem_lines)


def generate_acrostic(model, head_words, ix2word, word2ix):
    """
    生成藏头诗
    head_words: 藏头的字，如 "春夏秋冬"
    """
    result = []

    # 索引列表：用于在未满字数前屏蔽标点，避免提前结束
    comma_idx = word2ix.get('，', None)
    period_idx = word2ix.get('。', None)

    model.eval()
    with torch.no_grad():
        for i, head_word in enumerate(head_words):
            if head_word not in word2ix:
                # 若藏头字不在词表，跳过或用常见字替代
                print(f"[WARN] 藏头字 '{head_word}' 不在词表，跳过该行。")
                continue

            hidden = None
            # 每行从<START>开始
            input = torch.tensor([word2ix['<START>']], dtype=torch.long, device=Config.device).view(1, 1)

            # 先放入藏头字
            sentence_chars = [head_word]
            input = torch.tensor([word2ix[head_word]], dtype=torch.long, device=Config.device).view(1, 1)

            while len(sentence_chars) < Config.target_line_len:
                output, hidden = model(input, hidden)
                logits = output.data[0]

                # 未达标前屏蔽句读标点
                block = []
                if comma_idx is not None:
                    block.append(comma_idx)
                if period_idx is not None:
                    block.append(period_idx)

                next_idx = _sample_next_token(logits, ix2word, word2ix, blocked_indices=block)
                w = ix2word[next_idx]

                # 跳过异常或特殊符号
                if w in ['<EOP>'] or w.strip() == '':
                    # 继续重采
                    # 简单做法：将该idx加入block再采一次
                    block.append(next_idx)
                    next_idx = _sample_next_token(logits, ix2word, word2ix, blocked_indices=block)
                    w = ix2word[next_idx]

                sentence_chars.append(w)
                input = torch.tensor([next_idx], dtype=torch.long, device=Config.device).view(1, 1)

            # 行末标点交替
            sentence = ''.join(sentence_chars)
            end_punc = '，' if (i % 2 == 0) else '。'
            sentence += end_punc
            result.append(sentence)

    return '\n'.join(result)


# ==================== 主函数 ====================
def main():
    """
    主函数：加载数据、训练模型、生成诗句
    """
    print("=" * 60)
    print("实验3: 自动写诗 - 基于LSTM的唐诗生成")
    print("=" * 60)
    
    # 1. 准备数据
    print("\n[1/4] 加载数据...")
    dataloader, ix2word, word2ix = prepare_data(Config.data_path)
    
    # 2. 训练或加载模型
    print("\n[2/4] 准备模型...")
    
    # 检查是否存在已训练模型
    if os.path.exists(Config.model_path) and Config.use_pretrained:
        print(f"发现已训练模型: {Config.model_path}")
        choice = input("是否使用已有模型？(y/n，直接回车默认使用): ").strip().lower()
        
        if choice == '' or choice == 'y':
            model = load_model(word2ix)
            print("跳过训练，直接使用已有模型生成诗句。")
        else:
            print("开始重新训练...")
            model = train(dataloader, word2ix, ix2word)
    else:
        print("未发现已训练模型，开始训练...")
        model = train(dataloader, word2ix, ix2word)
    
    # 3. 生成诗句示例（结构化：控制句数与字数）
    print("\n[3/4] 生成诗句示例...")
    print("-" * 60)
    
    # 普通续写
    test_cases = [
        '湖光秋月两相和',
        '床前明月光',
        '春眠不觉晓',
        '红豆生南国'
    ]
    
    for start_words in test_cases:
        # 不指定句数，自动随机（优先4/8）
        gen_poetry = generate_poem_structured(
            model,
            start_words,
            ix2word,
            word2ix,
            num_lines=None,
        )
        print(f"起始: {start_words}")
        print(f"生成: {gen_poetry}\n")
    
    # 4. 藏头诗（可选）
    print("\n[4/4] 生成藏头诗示例...")
    print("-" * 60)
    
    acrostic_words = ['春', '夏', '秋', '冬']
    acrostic_poetry = generate_acrostic(model, acrostic_words, ix2word, word2ix)
    print(f"藏头字: {''.join(acrostic_words)}")
    print(f"生成藏头诗:\n{acrostic_poetry}")
    
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print(f"\n提示: 模型已保存至 {Config.model_path}")
    print("下次运行时将自动加载已训练模型，无需重新训练。")


if __name__ == '__main__':
    main()
    