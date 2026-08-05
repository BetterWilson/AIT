"""
莎士比亚文本生成 —— 字符级循环神经网络（Char-level RNN 文本生成）
========================================================
本脚本实现基于字符级 RNN 的莎士比亚风格文本生成任务，包括：
1. 数据加载与字符级词典构建（shakespeare.txt，共 65 个字符）
2. 自定义 CharDataset 数据集与 DataLoader（滑动窗口构造输入/目标对）
3. CharRNN 循环神经网络模型构建（Embedding + 多层 RNN + 全连接）
4. 模型训练（CrossEntropyLoss 逐字符预测 + Adam 优化器）
5. 温度系数理解（Multinomial 随机采样 + 不同温度下 Softmax 分布可视化）
6. 温度控制解码与文本生成（generate_text）
7. 模型保存
"""

# ============================================================
# 0. 导入需要的库
# ============================================================

import os  # 操作系统接口，用于创建目录、拼接路径
import numpy as np  # 数值计算库，用于绘制柱状图时的坐标轴
import torch  # PyTorch 核心库，提供张量运算与自动求导
import torch.nn as nn  # 神经网络模块，提供 Embedding、RNN、Linear 等层
import torch.optim as optim  # 优化器模块，提供 Adam 等
from torch.utils.data import Dataset, DataLoader  # Dataset: 自定义数据集基类；DataLoader: 批量加载器
import matplotlib.pyplot as plt  # 绘图库，用于可视化温度分布与训练曲线
from matplotlib import rcParams  # matplotlib 配置字典，用于设置全局绘图参数

# 设置中文字体，防止 matplotlib 中文显示为方块
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号（避免负号显示为方块）


# ============================================================
# 1. 数据加载与字符级词典构建
# ============================================================
# 文本生成任务按“字符”而非“单词”建模：
#   以莎士比亚的原始文本为语料，把每个字符当作一个 token，
#   模型逐字符预测“下一个字符”，从而学会莎士比亚的行文风格。
# 第一步：读取原始文本，统计出现过的所有字符构建词典（本项目共 65 个字符）。

# 读取数据文件（与当前脚本同级的 data 目录下）
data_path = 'data/shakespeare.txt'  # 莎士比亚数据集路径
with open(data_path, 'r', encoding='utf-8') as f:  # 以 utf-8 编码读取全部文本
    text = f.read()  # 整个文本作为一个长字符串

# 构建字符集与映射字典
# set(text) 去重得到所有不重复字符，sorted() 排序保证顺序稳定（结果可复现）
vocab = sorted(list(set(text)))  # 字符列表（共 65 个不同字符）
vocab_size = len(vocab)  # 字符集大小 = 65
# char2idx: 字符 → 整数 ID（供 Embedding 查表使用）
char2idx = {ch: idx for idx, ch in enumerate(vocab)}
# idx2char: 整数 ID → 字符（生成文本时把 ID 还原为字符）
idx2char = {idx: ch for idx, ch in enumerate(vocab)}

print(f"数据集总长度: {len(text)}")  # 1,115,394 个字符
print(f"字符集大小: {vocab_size}")  # 65
print(char2idx)  # 打印字符→ID 映射
print(idx2char)  # 打印 ID→字符 映射


# ============================================================
# 2. Dataset 与 DataLoader
# ============================================================
# 训练数据组织方式 —— 滑动窗口构造 (输入, 目标) 对：
#   对每条长度为 101 的字符窗口，取前 100 个字符作为输入 x，
#   后 100 个字符（整体右移一位）作为目标 y。
#   例如: 输入 "to be or n" → 目标 "o be or no"，实现逐字符错位预测。

# 将整个文本按字符映射为 ID 列表（模型消费的是 ID，不是字符）
data = [char2idx[ch] for ch in text]  # 长度 1,115,394 的字符 ID 列表

class CharDataset(Dataset):
    """
    字符级数据集：以固定窗口滑动截取 (输入序列, 目标序列) 对

    每个样本由 seq_len+1 个连续字符构成:
      输入 x = 窗口的前 seq_len 个字符
      目标 y = 窗口的后 seq_len 个字符（即输入整体右移一位）
    """

    def __init__(self, data, seq_len=100):
        """
        参数:
            data:    全部文本映射后的字符 ID 列表
            seq_len: 每个样本的序列长度，默认 100
        """
        self.data = data  # 保存全部字符的 ID 列表
        self.seq_len = seq_len  # 保存序列长度

    def __len__(self):
        # 每 seq_len+1 个字符构成一个窗口，因此样本总数 = 总长 // 窗口长
        return len(self.data) // (self.seq_len + 1)

    def __getitem__(self, idx):
        # 按窗口起点截取 seq_len+1 个字符
        start = idx * (self.seq_len + 1)  # 当前样本的窗口起点
        end = start + self.seq_len  # 窗口终点（输入序列的终点）
        # 输入 x: 前 seq_len 个字符；目标 y: 后 seq_len 个字符（右移一位）
        x = torch.tensor(self.data[start:end], dtype=torch.long)  # (seq_len,)
        y = torch.tensor(self.data[start + 1:end + 1], dtype=torch.long)  # (seq_len,)
        return x, y  # 返回 (输入序列, 目标序列)


# 创建数据集实例
seq_len = 100  # 序列长度
dataset = CharDataset(data, seq_len=seq_len)  # 字符级数据集

# 创建 DataLoader
batch_size = 64  # 每个 batch 包含 64 条样本
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 打乱顺序增强泛化

# 验证：取一个 batch 查看形状
# iter(data_loader) 把 loader 变成迭代器，next() 取出第一个 batch
x_sample, y_sample = next(iter(data_loader))
print("输入x的shape:", x_sample.shape)  # (batch_size=64, seq_len=100)
print("目标y的shape:", y_sample.shape)  # (64, 100)


# ============================================================
# 3. 搭建模型 —— CharRNN 字符级循环神经网络
# ============================================================
# 数据流: (bs, seq_len) → Embedding → (bs, seq_len, embed_dim)
#         → RNN → (bs, seq_len, hidden_dim) → Linear → (bs, seq_len, vocab_size)
#
# 思路: 每个字符经 Embedding 得到向量，RNN 逐字符读入并维护隐藏状态，
#       每个时间步输出该位置对“下一个字符”的预测 logits。
# RNN 层可以纵向堆叠多层（num_layers），层数越多，模型能捕捉的字符依赖越远。

class CharRNN(nn.Module):
    """
    字符级 RNN 文本生成模型：Embedding + 多层 RNN + 全连接

    结构概览:
      Embedding(vocab, 128) → RNN(128→256, 2层) → Linear(256→vocab)
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        """
        参数:
            vocab_size:    字符集大小（Embedding 表行数 = 输出类别数）
            embedding_dim: 字符向量维度，默认 128
            hidden_dim:    RNN 隐藏状态维度，默认 256
            num_layers:    RNN 层数（纵向堆叠），默认 2
        """
        super().__init__()  # 调用父类 nn.Module 构造函数
        # 字符嵌入层: 把每个字符 ID 查表得到 embedding_dim 维向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 标准 RNN 层（vanilla RNN）: h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b)
        self.rnn = nn.RNN(
            input_size=embedding_dim,  # 每个时间步输入是 embedding_dim 维向量
            hidden_size=hidden_dim,  # 隐藏状态维度
            num_layers=num_layers,  # 纵向堆叠层数
            batch_first=True  # 输入形状为 (batch, seq_len, input_size)
        )
        # 全连接输出层: 把每个时间步的隐藏状态映射为各字符的 logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        前向传播

        参数:
            x:      (batch, seq_len) 的字符 ID 张量
            hidden: 可选的初始隐藏状态（生成文本时需跨时间步传递）
        返回:
            output: (batch, seq_len, vocab_size)，每个位置预测下一字符的 logits
            hidden: 最后一层的隐藏状态 (num_layers, batch, hidden_dim)
        """
        embed = self.embedding(x)  # (batch, seq_len, embedding_dim): 每个字符查表得到向量
        if hidden is None:  # 训练时从头开始，无需传入隐藏状态
            output, hidden = self.rnn(embed)
        else:  # 生成时把上一步的 hidden 传入，实现状态逐字符延续
            output, hidden = self.rnn(embed, hidden)
        # output: (batch, seq_len, hidden_dim)，每个时间步一个隐藏状态
        output = self.fc(output)  # (batch, seq_len, vocab_size)
        return output, hidden  # 返回输出与隐藏状态


# 初始化模型并做前向传播验证
vocab_size = len(char2idx)  # 字符集大小 = 65
model = CharRNN(vocab_size)  # 使用默认超参数创建模型（Embedding 128, RNN 256×2 层）
x_test = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)  # 随机构造一个测试输入 (1, 100)
print(x_test.shape)  # torch.Size([1, 100])
output, hidden = model(x_test)  # 前向传播
print("输出output的shape:", output.shape)  # (1, 100, 65)：每个位置对 65 个字符的预测


# ============================================================
# 4. 训练及预测
# ============================================================
# 训练目标: 对序列的每个位置预测“下一个字符”，逐字符做分类。
# 关键点: CrossEntropyLoss 要求
#   输入 output: (N, C) —— N = batch*seq_len, C = vocab_size
#   目标 target: (N,)   —— 每个位置为字符类别索引
# 因此需要把 output 展平为 (batch*seq_len, vocab_size)，目标展平为 (batch*seq_len)。

def train(model, data_loader, optimizer, criterion, device="cuda", epochs=20, print_every=100):
    """
    训练循环：遍历全部 batch，前向 → 反向 → 更新参数

    参数:
        model:       待训练的 CharRNN 模型
        data_loader: 训练集 DataLoader
        optimizer:   优化器（如 Adam）
        criterion:   损失函数（CrossEntropyLoss）
        device:      训练设备 'cuda' 或 'cpu'
        epochs:      训练轮数
        print_every: 每多少个 batch 打印一次当前损失
    返回:
        epoch_losses: 每轮的平均损失列表（用于绘制训练曲线）
    """
    model = model.to(device)  # 模型移至目标设备
    model.train()  # 切换到训练模式
    epoch_losses = []  # 记录每轮平均损失（用于绘图）
    for epoch in range(epochs):  # 逐轮训练
        epoch_loss = 0  # 当前轮次的累计损失
        for i, (x_batch, y_batch) in enumerate(data_loader):  # 遍历每个 batch
            x_batch = x_batch.to(device)  # 输入 ID 张量移至设备 (batch, seq_len)
            y_batch = y_batch.to(device)  # 目标 ID 张量移至设备 (batch, seq_len)
            optimizer.zero_grad()  # 1. 清空上一步累积的梯度
            output, _ = model(x_batch)  # 2. 前向传播: (batch, seq_len, vocab_size)
            # 3. 展平以适配 CrossEntropyLoss:
            #    output (batch, seq_len, vocab_size) → (batch*seq_len, vocab_size)
            #    y_batch (batch, seq_len) → (batch*seq_len)
            loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
            loss.backward()  # 4. 反向传播求梯度
            optimizer.step()  # 5. 更新参数
            epoch_loss += loss.item()  # 累计当前轮次的损失
            # 每间隔 print_every 个 batch 打印一次该轮的平均损失
            if (i + 1) % print_every == 0:
                avg_loss = epoch_loss / (i + 1)  # 到目前为止的平均损失
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {avg_loss:.4f}")
        # 每轮训练结束，计算并记录该轮平均损失
        avg_epoch_loss = epoch_loss / len(data_loader)  # 本轮平均损失
        epoch_losses.append(avg_epoch_loss)  # 记录到历史列表
        print(f"Epoch [{epoch + 1}/{epochs}] 完成, 平均损失: {avg_epoch_loss:.4f}")
    return epoch_losses  # 返回损失历史


# 定义损失函数与优化器
# torch.nn.CrossEntropyLoss 要求的输入尺寸如下：
#   输入(output): (N, C) —— N 为批量大小×序列长度, C 为类别数（即 vocab_size）
#   目标(target): (N,) —— N 为批量大小×序列长度，每个位置为类别的 index
criterion = torch.nn.CrossEntropyLoss()  # 逐字符交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.005)  # Adam 优化器，学习率 0.005

# 判断可用设备：优先 GPU，否则回退 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择训练设备
print(f"使用设备: {device}")  # 打印当前训练设备

# 开始训练
epochs = 20  # 训练轮数
print(f"\n========== 开始训练 CharRNN (epochs={epochs}) ==========")
epoch_losses = train(model, data_loader, optimizer, criterion, device=device, epochs=epochs, print_every=100)  # 训练

# 绘制训练损失曲线，观察模型收敛情况
plt.figure(figsize=(8, 5))  # 创建 8×5 英寸画布
plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='skyblue')  # 每轮平均损失折线
plt.xlabel('Epoch')  # 横轴: 训练轮数
plt.ylabel('Average Loss')  # 纵轴: 平均损失
plt.title('CharRNN 训练损失曲线')  # 图表标题
plt.grid(True)  # 显示网格
plt.show()  # 显示图像


# ============================================================
# 5. 理解温度设计（Multinomial 随机采样 与 Softmax 温度）
# ============================================================
# 生成文本时若用 argmax 直接选概率最大的字符，会反复输出同一字符、句子毫无多样性。
# 正确做法是“按概率随机采样”：torch.multinomial 让每个类别以 p_i 的概率被抽中。
# 温度 temperature 进一步调节采样的“冒险程度”：
#   温度越低 (<1)：softmax 分布越尖锐（接近 one-hot），生成更确定、更保守
#   温度越高 (>1)：分布越平坦（接近均匀），生成更随机、更有创意
#   温度 = 1：按模型学到的原始概率分布采样

# 5.1 Multinomial 采样示例：验证采样频率趋近于概率分布
probs = torch.tensor([0.1, 0.2, 0.3, 0.4])  # 假设一个 softmax 后的概率分布

# 按该分布随机采样 10 次，观察每次抽中的类别
samples = torch.multinomial(probs, num_samples=10, replacement=True)  # 有放回采样 10 次
print("Sampled indices (10 times):", samples.tolist())  # 打印采样到的类别索引

# 采样 10000 次，统计各类被抽中的频率，应与原始概率近似一致（大数定律）
n_trials = 10000  # 采样次数
sample_result = torch.multinomial(probs, num_samples=n_trials, replacement=True)  # 大量采样
values, counts = torch.unique(sample_result, return_counts=True)  # 统计每个类别出现的次数
print("Sampling frequency for each class:")
for v, c in zip(values, counts):  # 逐类别打印
    print(f"Class {v.item()} : {c.item() / n_trials:.2f}")  # 应接近 0.10 / 0.20 / 0.30 / 0.40

# 可视化：柱状图对比“原始概率”与“采样频率”
plt.bar(range(len(probs)), probs.numpy(), alpha=0.5, label='Original Probabilities')  # 原始概率柱
plt.bar(values.numpy(), (counts.numpy() / n_trials), alpha=0.5, label='Sampled Frequencies')  # 采样频率柱
plt.xticks(range(len(probs)))  # 设置横轴刻度
plt.xlabel('Class')  # 横轴标签
plt.ylabel('Probability/Frequency')  # 纵轴标签
plt.legend()  # 显示图例
plt.title('Multinomial Sampling vs Probability Distribution')  # 图表标题
plt.show()  # 显示图像

# 5.2 不同温度下 Softmax 分布的变化
# 温度作用方式: 先对 logits 除以 temperature，再执行 softmax。
#   除以一个更小的数 → 放大差异 → 分布更尖锐
#   除以一个更大的数 → 缩小差异 → 分布更平坦
logits = torch.tensor([1.0, 2.0, 3.0])  # 假设一个 logits 向量
temperatures = [0.1, 1.0, 100]  # 三种典型温度

# 用 1×3 子图展示不同温度下 softmax 的分布形状
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)  # 一行三列子图，共享纵轴
x = np.arange(len(logits))  # 横轴: 类别索引 [0, 1, 2]

for ax, temp in zip(axes, temperatures):  # 逐个温度绘制
    scaled_logits = logits / temp  # 1. logits 除以温度
    probs = torch.softmax(scaled_logits, dim=0).numpy()  # 2. softmax 得到概率分布
    ax.bar(x, probs, color='skyblue')  # 3. 绘制柱状图
    ax.set_xticks(x)  # 设置横轴刻度
    ax.set_xlabel('Class')  # 横轴标签
    ax.set_ylabel('Probability')  # 纵轴标签
    ax.set_title(f'Temperature={temp}')  # 子图标题

plt.suptitle('Softmax Probabilities at Different Temperatures')  # 总标题
plt.tight_layout(rect=[0, 0, 1, 0.93])  # 调整布局，避免标题重叠
plt.show()  # 显示图像

# 打印各温度下的概率分布数值，直观对比尖锐程度
print("各温度下softmax概率分布示例：")
for temp in temperatures:  # 逐温度打印
    scaled_logits = logits / temp  # 除以温度
    probs = torch.softmax(scaled_logits, dim=0)  # softmax 得到概率
    print(f"温度 {temp}: 概率分布: {[f'{p:.3f}' for p in probs]}")  # 打印概率数组


# ============================================================
# 6. 温度控制解码与文本生成
# ============================================================
# 生成流程: 给定开头序列，逐字符预测并采样：
#   1. 取模型最后一个时间步的 logits
#   2. 除以温度后 softmax 得到概率分布
#   3. multinomial 采样得到下一个字符
#   4. 把新字符拼到生成结果中，并作为下一步的输入继续循环
# 隐藏状态 hidden 逐字符传递，让模型记住已生成的全部上下文。

def generate_text(model, start_seq, char2idx, idx2char, length=200, device="cuda", temperature=1.0):
    """
    基于已训练模型生成文本

    参数:
        model:       训练好的 CharRNN 模型
        start_seq:   起始字符串（种子，如 "hello:"）
        char2idx:    字符→ID 映射字典
        idx2char:    ID→字符 映射字典
        length:      要生成的字符数
        device:      设备 'cuda' 或 'cpu'
        temperature: 温度系数（>0）。越小越确定，越大越随机
    返回:
        generated: 起始串 + 生成字符拼接得到的完整文本
    """
    model.eval()  # 评估模式（不更新参数，关闭梯度）
    # 将起始字符串转换为 ID 序列，并扩展为 (1, seq_len) 形状
    input_seq = [char2idx[char] for char in start_seq]  # 起始串的字符 ID 列表
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)  # (1, len(start_seq))
    hidden = None  # 初始化隐藏状态（生成时逐字符传递，None 表示从零开始）

    generated = list(start_seq)  # 生成序列以起始串开头
    for _ in range(length):  # 循环生成 length 个字符
        with torch.no_grad():  # 生成阶段不需要计算梯度
            output, hidden = model(input_tensor, hidden)  # 前向传播，hidden 跨步传递
            last_logits = output[0, -1, :]  # 取最后一个时间步的 logits (vocab_size,)
            # 应用温度系数：除以 temperature 后 softmax，得到下一个字符的概率分布
            scaled_logits = last_logits / temperature  # 温度缩放
            prob = torch.softmax(scaled_logits, dim=0).cpu()  # 概率分布（转回 CPU）
            next_id = torch.multinomial(prob, num_samples=1).item()  # 按概率采样下一个字符
            generated.append(idx2char[next_id])  # 把字符拼接到生成序列
            # 将新字符作为下一步的输入（形状 (1, 1)）
            input_tensor = torch.tensor([[next_id]], dtype=torch.long).to(device)
    return ''.join(generated)  # 拼接成字符串返回


# 生成示例：以 "hello:" 为种子，温度 0.8 生成 200 个字符
# 温度 0.8 (<1)：分布相对尖锐，生成文本更“保守”，倾向于重复常见句型
start_seq = "hello:"  # 起始字符串
generated_text = generate_text(model, start_seq, char2idx, idx2char, length=200, device=device, temperature=0.8)
print(generated_text)  # 打印生成结果


# ============================================================
# 7. 保存模型
# ============================================================
# 训练结束后把模型权重保存到磁盘，便于以后加载推理（生成文本），无需重新训练。

# 保存目录: data/model（与当前脚本同级的 data 目录下）
save_dir = 'model'  # 模型保存目录
os.makedirs(save_dir, exist_ok=True)  # 目录不存在则创建
save_path = os.path.join(save_dir, 'char_rnn_shakespeare.pth')  # 模型文件路径
torch.save(model.state_dict(), save_path)  # 保存模型权重（state_dict）
print(f"模型已保存到: {save_path}")  # 打印保存路径