"""
IMDB 影评情感二分类 —— Embedding + LSTM（长短期记忆网络文本分类）
======================================================
脚本包含以下内容：
0. 导入所有需要的库
1. HuggingFace datasets 库加载 IMDB 影评数据集（本地缓存，避免重复下载）
2. 文本清洗与词典构建（正则清洗 + Counter 词频统计 + <PAD>/<OOV>/<BOS>/<EOS> 特殊符号）
3. 单词数分布可视化 —— 决定序列最大长度 max_length
4. Tokenizer 分词器类（encode/decode、BOS/EOS/PAD/OOV，支持句首/句尾两种 padding）
5. 自定义 IMDBDataset Dataset 类 + create_collate_fn 组 batch 回调（含 zip(*batch) 拆分技巧）
6. nn.LSTM 层基础 —— 单层单向 / 双向 / 双层 三种配置的参数与输出形状对比
7. 模型：Embedding + LSTM 情感分类器（支持多层/双向扩展）
8. 模型实例化、前向传播验证与参数量统计
9. Trainer 通用训练器类（二分类，BCEWithLogitsLoss，损失与准确率可视化）
10. 训练准备（设备 / 损失函数 / 优化器）
11. 训练模型并绘制训练曲线
12. 总结
"""

# ============================================================
# 0. 导入所有需要的库
# ============================================================

import os  # 操作系统接口，用于路径拼接、缓存目录判断
import re  # 正则表达式库，用于文本清洗
from collections import Counter  # 计数器类，用于统计词频
import numpy as np  # 数值计算库，用于数组化文本与标签
import torch  # PyTorch 核心库，提供张量运算与自动求导
import torch.nn as nn  # 神经网络模块，提供 Embedding、RNN、LSTM、Linear 等层
import torch.optim as optim  # 优化器模块，提供 Adam 等
from torch.utils.data import Dataset, DataLoader  # Dataset: 自定义数据集基类；DataLoader: 批量加载器
import matplotlib.pyplot as plt  # 绘图库，用于训练曲线与直方图可视化
from matplotlib import rcParams  # matplotlib 配置字典，用于设置全局绘图参数（如中文字体）
from datasets import load_dataset  # HuggingFace datasets 库，用于加载 IMDB 数据集（若未安装：pip install datasets）

# 设置中文字体，防止 matplotlib 中文显示为方块
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号（避免负号显示为方块）


# ============================================================
# 1. 数据准备 —— 加载 IMDB 影评数据集
# ============================================================
# IMDB 是一个经典的英文影评情感二分类数据集：
#   - 训练集 25000 条影评（正面 12500 / 负面 12500）
#   - 测试集 25000 条影评（正面 12500 / 负面 12500）
#   - 每条样本: 一段文本（英文影评）+ 一个标签（0=负面, 1=正面）

# 指定数据缓存目录：首次加载会把数据集下载到这里，之后从本地读取，避免重复下载
cache_dir = 'data/hf_cache'  # 缓存目录，与当前文件同级的 data/hf_cache

# 检查本地缓存中是否已存在 IMDB 数据集
# 注: 新版本 datasets 库要求仓库名为 'namespace/name' 格式，因此使用 'stanfordnlp/imdb'
imdb_cache_dir = os.path.join(cache_dir, 'stanfordnlp___imdb')  # 本地缓存中的实际目录名（含数据集版本子目录）
if (os.path.exists(imdb_cache_dir) and  # 缓存目录存在
        os.listdir(imdb_cache_dir)):  # 且目录非空（说明已下载过数据文件）
    print("检测到本地 IMDB 缓存，直接从缓存加载...")  # 提示直接使用本地缓存
else:  # 本地没有缓存
    print("本地无 IMDB 缓存，首次加载需要联网下载（之后会存入 cache_dir）")  # 提示需要下载

# load_dataset: 下载（或从缓存读取）并解析 IMDB 数据集
# 返回 DatasetDict，含 'train' 和 'test' 两个 split
dataset = load_dataset('stanfordnlp/imdb', cache_dir=cache_dir)  # 加载 IMDB 数据集（自动命中本地缓存）

print(type(dataset))  # 打印数据集类型: <class 'datasets.dataset_dict.DatasetDict'>
print(type(dataset['train']['text']))  # 打印训练集 text 列的类型: <class 'list'>

# 将训练集/测试集的内容与标签分别提取为 numpy 数组
# dataset['train']['text'] 返回该 split 下所有样本的 text 字段（Python list）
train_texts = np.array(dataset['train']['text'])  # 训练集影评文本数组 (25000,)
train_labels = np.array(dataset['train']['label'])  # 训练集标签数组 (25000,)，元素为 0/1
test_texts = np.array(dataset['test']['text'])  # 测试集影评文本数组 (25000,)
test_labels = np.array(dataset['test']['label'])  # 测试集标签数组 (25000,)


# ============================================================
# 2. 文本清洗与词典构建
# ============================================================
# 原始影评中含有标点（逗号、句号、HTML 的 <br /> 标签等），会影响分词。
# 我们先把标点替换为空格，再按空白字符切分成单词，最后统计词频构建词典。
# 词典预留 4 个特殊符号：
#   0 → <PAD>  填充符，用于把 batch 内不同长度的序列补齐到等长
#   1 → <OOV>  未登录词（Out Of Vocabulary），词典之外的词统一映射为它
#   2 → <BOS>  句子开始符（Beginning Of Sentence）
#   3 → <EOS>  句子结束符（End Of Sentence）

def clean_text(text):
    """
    文本清洗函数 —— 去掉标点、合并连续空白

    参数:
        text: 原始影评字符串
    返回:
        清洗后的字符串（单词之间用单个空格分隔）
    """
    text = text.replace(',', ' ').replace('.', ' ')  # 将逗号、句号替换为空格（它们不携带情感信息）
    text = re.sub(r'\s+', ' ', text)  # 用正则把连续空白字符（空格/换行/<br/>）压缩为单个空格
    return text.strip()  # 去掉首尾空白后返回

# 对所有训练文本进行清洗
train_texts = np.array([clean_text(t) for t in train_texts])  # 逐个样本清洗，重新组装为数组

# ---- 分词：把每句影评切成单词列表，再统一收集 ----
all_words = []  # 收集训练集所有单词的列表（允许重复，用于统计词频）
for text in train_texts:  # 遍历每条清洗后的影评
    all_words.extend(text.split())  # split() 不带参数：按所有空白字符切分，得到该影评的单词列表

# ---- 统计词频并构建词典 ----
word_counts = Counter(all_words)  # 统计每个单词在训练集中出现的次数

# 限制词典大小：只保留出现频率最高的前 (max_vocab_size - 4) 个词，其余词一律视为未登录词
max_vocab_size = 20000  # 词典总大小（含 4 个特殊符号）
vocab = {'<PAD>': 0, '<OOV>': 1, '<BOS>': 2, '<EOS>': 3}  # 初始化词典，先放入特殊符号

# 选出出现频率最高的 (max_vocab_size - 4) 个词，按词频从高到低分配 ID（从 4 开始）
most_common_words = word_counts.most_common(max_vocab_size - 4)  # 返回 [(单词, 次数), ...]，已按次数降序
start = 4  # 普通词的 ID 从 4 开始（0~3 已被特殊符号占用）
for idx, (word, count) in enumerate(most_common_words, start=start):  # enumerate 从 start=4 开始编号
    vocab[word] = idx  # 将单词映射为对应的整数 ID

print(f"词典大小: {len(vocab)}")  # 应输出 20000（4 个特殊符号 + 19996 个高频词）

# ---- 构建 ID 到 word 的反向词典，便于调试时把 ID 序列还原成文本 ----
id2word = {idx: word for word, idx in vocab.items()}  # 反转字典: {ID: 单词}
for i in range(10):  # 打印前 10 个 ID 对应的词，检查词典是否正确
    print(f"{i}: {id2word[i]}")  # 0:<PAD> 1:<OOV> 2:<BOS> 3:<EOS> 4:the 5:a 6:and ...


# ============================================================
# 3. 单词数分布可视化 —— 决定序列最大长度 max_length
# ============================================================

# 统计每个样本的单词数量（清洗后按空格切分的长度）
sample_word_counts = [len(text.split()) for text in train_texts]  # 每条影评的单词个数

plt.figure(figsize=(8, 4))  # 创建 8×4 英寸的画布
plt.hist(sample_word_counts, bins=50, color='skyblue', edgecolor='black')  # 直方图: 50 个区间
plt.xlabel('单个样本的单词数')  # 横轴: 每条影评包含的单词个数
plt.ylabel('样本数')  # 纵轴: 落在该区间的样本数量
plt.title('train_texts 每个样本单词数直方图')  # 图表标题
plt.show()  # 显示图像

# 从直方图可以看到大多数影评集中在几百个词以内，
# 因此后续 Tokenizer 设 max_length=500：超过 500 词截断，不足则 padding 补齐。


# ============================================================
# 4. Tokenizer 分词器类 —— 文本 ↔ ID 序列的互转
# ============================================================
# 神经网络只能接收数字，不能直接接收文本。Tokenizer 负责:
#   - encode: 文本列表 → 填充好的 ID 序列列表（会加 BOS/EOS，并 padding 到 batch 内等长）
#   - decode: ID 序列 → 文本（便于验证编码是否正确）
#
# 关键设计 —— padding 位置的选择（padding_first 参数）:
#   RNN/LSTM 是从左往右读序列的，如果 padding 在句尾，最后一步的隐藏状态会包含大量
#   padding 信息；把 padding 放到句首，句尾恰好是真实文本，取最后一步 hidden 才有意义。
#   因此本任务统一使用 padding_first=True（PAD 前置）。

class Tokenizer:
    """
    IMDB 文本分词器：完成文本与 ID 序列之间的转换

    特殊符号约定:
      <PAD>=0: 填充符     <OOV>=1: 未登录词
      <BOS>=2: 句首符     <EOS>=3: 句尾符
    """

    def __init__(self, vocab, oov_token='<OOV>', bos_token='<BOS>', eos_token='<EOS>', pad_token='<PAD>'):
        """
        初始化 Tokenizer

        参数:
            vocab:       词典 dict {单词: ID}
            oov_token:   未登录词占位符名称，默认 '<OOV>'
            bos_token:   句首符名称，默认 '<BOS>'
            eos_token:   句尾符名称，默认 '<EOS>'
            pad_token:   填充符名称，默认 '<PAD>'
        """
        self.vocab = vocab  # 保存词典
        self.id2word = {idx: word for word, idx in vocab.items()}  # 构建 ID→单词反向词典
        self.oov_id = vocab.get(oov_token, 1)  # 获取 OOV 的 ID（词典中固定为 1）
        self.bos_id = vocab.get(bos_token, 2)  # 获取 BOS 的 ID（固定为 2）
        self.eos_id = vocab.get(eos_token, 3)  # 获取 EOS 的 ID（固定为 3）
        self.pad_id = vocab.get(pad_token, 0)  # 获取 PAD 的 ID（固定为 0）

    def encode(self, texts, add_bos=False, add_eos=False, max_length=500, padding_first=False):
        """
        文本列表 → 填充后的 ID 序列列表

        参数:
            texts:          文本字符串列表（一个元素一条影评）
            add_bos:        是否在序列开头添加 <BOS>（默认 False）
            add_eos:        是否在序列末尾添加 <EOS>（默认 False）
            max_length:     单条序列最大长度，超过则从右侧截断
            padding_first:  True=PAD 放在句首（RNN 用）；False=PAD 放在句尾（池化用）
        返回:
            padded: list of list of int，已按本 batch 最大长度填充（且不超过 max_length）
        """
        encoded = []  # 存放每个样本的 ID 序列
        seq_lengths = []  # 存放每个样本的原始长度（未 padding 前），用于计算 batch 最大长度
        for text in texts:  # 一个 text 就是一个样本
            ids = []  # 当前样本的 ID 序列
            if add_bos:  # 如果需要句首符
                ids.append(self.bos_id)  # 在序列最前面加入 <BOS>
            for word in text.split():  # 按空白切分出每个单词
                # vocab.get(word, self.oov_id): 单词在词典中取对应 ID，否则取 <OOV> 的 ID
                ids.append(self.vocab.get(word, self.oov_id))  # 查词典得到该词的 ID
            if add_eos:  # 如果需要句尾符
                ids.append(self.eos_id)  # 在序列最后加入 <EOS>
            if len(ids) > max_length:  # 如果序列超过最大长度
                ids = ids[:max_length]  # 截断只保留前 max_length 个 token
            seq_lengths.append(len(ids))  # 记录该样本的序列长度
            encoded.append(ids)  # 把该样本的 ID 序列加入列表

        # 计算本 batch 的最大长度（不超过 max_length；encoded 为空则设为 0 防止除零）
        batch_max_len = min(max(seq_lengths), max_length) if encoded else 0

        # ---- PAD 填充：把所有样本补齐到 batch_max_len 等长 ----
        padded = []  # 存放填充后的等长序列
        for ids in encoded:  # 遍历每个样本的 ID 序列
            pad_length = batch_max_len - len(ids)  # 需要补的 PAD 数量
            if pad_length > 0:  # 该样本比 batch 最长样本短，需要填充
                if padding_first:  # 选择 PAD 前置
                    ids = [self.pad_id] * pad_length + ids  # PAD 加在前面，真实文本靠后
                else:  # 默认 PAD 后置
                    ids = ids + [self.pad_id] * pad_length  # PAD 加在后面，真实文本靠前
            padded.append(ids)  # 把填充后的序列加入结果列表
        return padded  # 返回形状为 (batch_size, batch_max_len) 的等长 ID 序列

    def decode(self, batch_ids, skip_special_tokens=False):
        """
        ID 序列 → 文本，用于验证编码是否正确

        参数:
            batch_ids:              list of list of int，一批 ID 序列
            skip_special_tokens:    True=跳过 <PAD>/<OOV>/<BOS>/<EOS> 等特殊符号
        返回:
            decoded: list of str，还原后的文本列表
        """
        decoded = []  # 存放还原后的文本
        special_ids = {self.oov_id, self.bos_id, self.eos_id, self.pad_id}  # 特殊符号 ID 集合
        for ids in batch_ids:  # 遍历每条 ID 序列
            words = []  # 当前序列还原出的单词列表
            for idx in ids:  # 遍历序列中的每个 ID
                if skip_special_tokens and idx in special_ids:  # 跳过特殊符号
                    continue  # 不加入该 token
                words.append(self.id2word.get(idx, '<OOV>'))  # 反向查词典得到单词（查不到用 <OOV> 兜底）
            decoded.append(' '.join(words))  # 用空格连接单词，组成一句文本
        return decoded  # 返回还原出的文本列表


# 实例化 Tokenizer（传入上一步构建好的词典）
tokenizer = Tokenizer(vocab)  # 创建分词器对象

# ---- 测试 Tokenizer：验证 encode / decode / 两种 padding 模式是否正确 ----
test_sentences = [  # 三个测试句子，其中含一个词典外的未知词
    "i am happy",  # 普通短句
    "this movie is so good",  # 稍长一点的句子
    "unknownword appear"  # 含未知词 unknownword（会映射为 <OOV>）
]

# encode 测试（加 BOS/EOS，padding 默认在后）：
encoded = tokenizer.encode(test_sentences, add_bos=True, add_eos=True)  # 编码（句尾 padding）
print("\n编码后（padding 后置）:")
for i, sample in enumerate(encoded):  # 逐样本打印
    print(f"样本{i + 1}: {sample}")  # 可看到 <PAD>(0) 都补在序列末尾

# encode 测试（padding_first=True，PAD 前置）：
encoded_first = tokenizer.encode(test_sentences, add_bos=True, add_eos=True, padding_first=True)  # 编码（句首 padding）
print("编码后（padding 前置）:")
for i, sample in enumerate(encoded_first):  # 逐样本打印
    print(f"样本{i + 1}: {sample}")  # 可看到 <PAD>(0) 都补在序列开头

# decode 测试（还原文本，保留特殊符号）：
decoded = tokenizer.decode(encoded)  # 解码（保留特殊符号）
print("解码后：")
for i, sample in enumerate(decoded):  # 逐样本打印
    print(f"样本{i + 1}: {sample}")  # 可看到 <BOS>/<EOS>/<PAD> 等符号

# decode 测试（跳过特殊符号，只还原真实单词）：
decoded_skip = tokenizer.decode(encoded, skip_special_tokens=True)  # 解码（跳过特殊符号）
print("解码跳过特殊符号后：")
for i, sample in enumerate(decoded_skip):  # 逐样本打印
    print(f"样本{i + 1}: {sample}")  # 还原出原始句子


# ============================================================
# 5. 自定义 Dataset 类 与 create_collate_fn 组 batch 回调
# ============================================================
# 与图像任务不同，文本长度不一，无法直接堆叠成一个张量。
# 所以我们先让 Dataset 返回原始 (text, label)，再由 DataLoader 的
# collate_fn 统一调用 Tokenizer.encode 完成“文本 → 填充 ID 张量”的转换。

# ---- 先看一个 Python 小技巧: zip(*lists) 按列拆分 ----
# collate_fn 里用 `texts, labels = zip(*batch)` 把一个 batch 的 (text, label)
# 对拆成两个元组，原理与下面这个 demo 完全一致：
list1 = [1, 2]  # 模拟第一个样本的 (特征, 标签)
list2 = [3, 4]  # 模拟第二个样本的 (特征, 标签)
list3 = [5, 6]  # 模拟第三个样本的 (特征, 标签)
list4 = [7, 8]  # 模拟第四个样本的 (特征, 标签)
list5 = [9, 10]  # 模拟第五个样本的 (特征, 标签)
lists = [list1, list2, list3, list4, list5]  # 一批样本 [(特征, 标签), ...]
i, out = zip(*lists)  # 解包: i 收集所有“特征”，out 收集所有“标签”
print(i)  # (1, 3, 5, 7, 9)
print(out)  # (2, 4, 6, 8, 10)


class IMDBDataset(Dataset):
    """
    IMDB 情感分类数据集

    每个样本返回 (原始文本字符串, 整数标签)，不对文本做任何处理。
    真正的 token 化与 padding 工作交给 collate_fn 在组 batch 时完成。
    """

    def __init__(self, texts, labels):
        """
        初始化数据集

        参数:
            texts:  文本数组，每个元素为一条影评字符串
            labels: 标签数组，0=负面，1=正面
        """
        self.texts = texts  # 保存文本数组
        self.labels = labels  # 保存标签数组

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.texts)  # 样本数 = 文本条数

    def __getitem__(self, idx):
        """
        按索引获取单个样本

        参数:
            idx: 样本索引
        返回:
            (text, label): 文本字符串与整数标签
        """
        return self.texts[idx], int(self.labels[idx])  # 返回 (原文, 标签)，标签转成 Python int


def create_collate_fn(tokenizer, add_bos=False, add_eos=False, max_length=500, padding_first=False):
    """
    创建 DataLoader 的 collate_fn 回调函数

    组 batch 时把一批 (text, label) 整理成模型可直接消费的张量:
      input_ids: LongTensor (batch_size, seq_len)，已加 BOS/EOS 并 padding 到等长
      labels:    LongTensor (batch_size,)

    参数:
        tokenizer:      Tokenizer 实例，负责把文本编码为 ID 并 padding
        add_bos:        是否在序列开头加 <BOS>
        add_eos:        是否在序列末尾加 <EOS>
        max_length:     单条序列最大长度
        padding_first:  True=PAD 前置（RNN 用），False=PAD 后置（池化用）
    返回:
        collate_fn: 可供 DataLoader 使用的回调函数
    """

    def collate_fn(batch):
        """
        将一个 batch 的样本整理为模型输入

        参数:
            batch: [(text, label), ...] 列表
        返回:
            input_ids: LongTensor, shape (batch_size, seq_len)
            labels:    LongTensor, shape (batch_size,)
        """
        texts, labels = zip(*batch)  # 把特征和标签拆开（zip(*batch) 按列拆分元组列表）
        texts = list(texts)  # zip 结果转成 list 便于使用
        # 通过 tokenizer 把文本 batch 编码为 ID 并做 padding（返回等长 list）
        input_ids = tokenizer.encode(
            texts,
            add_bos=add_bos,
            add_eos=add_eos,
            max_length=max_length,
            padding_first=padding_first,
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)  # list → LongTensor (batch, seq_len)
        labels = torch.tensor(labels, dtype=torch.long)  # list → LongTensor (batch,)
        return input_ids, labels  # 返回模型输入与标签

    return collate_fn  # 返回闭包作为 DataLoader 的 collate_fn


# 构建训练集与测试集 Dataset
train_dataset = IMDBDataset(train_texts, train_labels)  # 训练集 Dataset
test_dataset = IMDBDataset(test_texts, test_labels)  # 测试集 Dataset

# 创建 collate_fn 回调（RNN 模型统一加 BOS/EOS，PAD 前置，最长 500 token）
collate_fn = create_collate_fn(tokenizer, add_bos=True, add_eos=True, max_length=500, padding_first=True)

# 构建 DataLoader
train_loader = DataLoader(
    train_dataset,  # 训练集 Dataset
    batch_size=32,  # 每批 32 条影评
    shuffle=True,  # 训练集随机打乱，防止模型记忆样本顺序
    collate_fn=collate_fn,  # 组 batch 回调（PAD 前置）
)
test_loader = DataLoader(
    test_dataset,  # 测试集 Dataset
    batch_size=32,  # 每批 32 条影评
    shuffle=False,  # 测试集不打乱，保持评估一致性
    collate_fn=collate_fn,  # 组 batch 回调（PAD 前置）
)

# ---- 验证：取一个 batch 查看形状与内容是否正确 ----
for batch_idx, (input_ids, labels) in enumerate(train_loader):  # 取一个 batch
    print(f"\n========== Batch {batch_idx} 验证 ==========")
    print(f"  input_ids shape: {input_ids.shape}")  # torch.Size([32, 500]) —— (batch, seq_len)
    print(f"  labels shape:    {labels.shape}")  # torch.Size([32]) —— (batch,)
    print(f"  首条 input_ids 前15个:  {input_ids[0][:15].tolist()}")  # PAD(0) 前置在句首
    print(f"  首条 label:      {labels[0].item()}")  # 打印第一个样本的标签
    # 解码验证文本转 ID 是否正确（跳过特殊符号还原出原句）
    decoded = tokenizer.decode([input_ids[0].tolist()], skip_special_tokens=True)  # 解码单条
    print(f"  解码还原:        {decoded[0][:80]}...")  # 打印还原出的文本前 80 字符
    break  # 只验证一个 batch 后跳出


# ============================================================
# 6. nn.LSTM 层基础 —— 单层单向 / 双向 / 双层 对比
# ============================================================
# 循环网络的核心是“按时间步从左到右读取序列”，每个时间步的隐藏状态
# h_t 整合了到当前位置为止的全部上下文。nn.LSTM 相比 nn.RNN 增加了
# 细胞状态 c_t（长期记忆），能更好地缓解梯度消失问题。
#
# 三个正交的超参数：
#   input_size:   每个时间步输入的维度（本任务中为 Embedding 维度）
#   hidden_size:  隐藏状态 h_t 的维度
#   num_layers:   纵向堆叠的层数（上层接收下层的输出）
#   bidirectional:是否横向双向读取（正向一遍 + 反向一遍）
#
# 注意 nn.LSTM 的权重命名规则: weight_ih_l0 表示“第0层 输入→隐藏”权重，
# 维度为 (hidden*4, input)。乘 4 是因为 LSTM 内部有 4 个门（输入门 i、
# 遗忘门 f、输出门 o、候选记忆 g），每个门都有自己的权重。

# ---- 6.1 单层单向 LSTM ----
# 参数形状: weight_ih_l0 (80, 10) —— 4 门 × hidden20 = 80 行，input10 列
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)  # 单层单向 LSTM

print("\nLSTM weights and biases:")  # 打印该 LSTM 层的参数
for name, param in lstm.named_parameters():  # 遍历命名参数
    print(f"  {name}: shape={param.shape}")  # weight_ih_l0 / weight_hh_l0 / bias_ih_l0 / bias_hh_l0

# 随机输入: 假设 batch_size=4, seq_len=7, input_size=10
x = torch.randn(4, 7, 10)  # (batch_size, seq_len, input_size)

# 前向传播: 返回 (output, (hn, cn))
output, (hn, cn) = lstm(x)  # 解包输出

print("output shape:", output.shape)  # (batch_size, seq_len, hidden_size=20)：每个时间步的输出
print("hn shape:", hn.shape)  # (num_layers, batch_size, hidden_size)：最后一个时间步的隐藏状态
print("cn shape:", cn.shape)  # (num_layers, batch_size, hidden_size)：最后一个时间步的细胞状态


# ---- 6.2 双向 LSTM（bidirectional=True）----
# 双向: 同时正向读一遍 + 反向读一遍，两个方向的隐藏状态在输出维度上拼接。
# 关键影响: output 最后一维 = hidden*2；hn/cn 第一维 = num_layers*2。
# 反向权重以 _reverse 后缀命名（如 weight_ih_l0_reverse）。
lstm = nn.LSTM(
    input_size=10,  # 每个时间步输入维度
    hidden_size=20,  # 单方向的隐藏维度
    num_layers=1,  # 层数
    batch_first=True,  # 输入 (batch, seq, input)
    bidirectional=True,  # 开启双向
)

print("\nBidirectional LSTM weights and biases:")  # 打印双向 LSTM 的参数
for name, param in lstm.named_parameters():  # 遍历命名参数
    print(f"  {name}: shape={param.shape}")  # 多出 4 个 _reverse 反向权重

x = torch.randn(4, 7, 10)  # 随机输入 (batch, seq, input)

output, (hn, cn) = lstm(x)  # 前向传播

print("output shape:", output.shape)  # (batch, seq, hidden*2=40)
print("hn shape:", hn.shape)  # (num_layers*2=2, batch, hidden=20)
print("cn shape:", cn.shape)  # (num_layers*2=2, batch, hidden=20)


# ---- 6.3 双层单向 LSTM（num_layers=2）----
# 双层: 第 0 层读原始输入，第 1 层把第 0 层的输出当输入。
# 因此 weight_ih_l1 的输入维度从 input_size 变成 hidden_size。
lstm = nn.LSTM(
    input_size=10,  # 每个时间步输入维度
    hidden_size=20,  # 隐藏维度
    num_layers=2,  # 两层
    batch_first=True,  # 输入 (batch, seq, input)
    bidirectional=False,  # 单向
)

print("\n2-layer unidirectional LSTM weights and biases:")  # 打印双层 LSTM 的参数
for name, param in lstm.named_parameters():  # 遍历命名参数
    print(f"  {name}: shape={param.shape}")  # l1 层的 input 维度是 20（上一层 hidden）

x = torch.randn(4, 7, 10)  # 随机输入 (batch, seq, input)

output, (hn, cn) = lstm(x)  # 前向传播

print("output shape:", output.shape)  # (batch, seq, hidden=20)
print("hn shape:", hn.shape)  # (num_layers=2, batch, hidden=20)
print("cn shape:", cn.shape)  # (num_layers=2, batch, hidden=20)


# ============================================================
# 7. 模型：Embedding + LSTM 情感分类器
# ============================================================
# 数据流: (bs, seq_len) → Embedding → (bs, seq_len, embed_dim)
#         → LSTM → (bs, hidden)（取最后一步隐藏状态）→ 全连接 → (bs, 1)
#
# 思路: LSTM 按时间步从左到右“阅读”整句话，每个时间步的隐藏状态 h_t
#       整合了到当前为止的全部上下文；句子读完时最后一步的隐藏状态
#       就浓缩了整句话的信息，取它做分类。
#
# 与 vanilla RNN 的区别: LSTM 内部多出一个“细胞状态” c_t（长期记忆），
#   通过输入门 / 遗忘门 / 输出门 三个门控，选择性地记住或遗忘信息，
#   能更好地缓解长序列上的梯度消失问题。
#
# 为什么必须 PAD 前置: LSTM 最后一步的隐藏状态取决于序列末尾的内容，
#   如果 PAD 在句尾，最后一步读到的是 PAD，信息被污染；
#   把 PAD 放到句首后，句尾恰好是真实文本，取最后一步才有意义。
#
# 扩展能力: 支持多层（num_layers>1）与双向（bidirectional=True）。
#   本脚本实际训练采用默认的单层单向配置。
#
# 提示: nn.RNN / nn.LSTM / nn.GRU 三者的接口（构造参数与返回结构）完全一致；
#   唯一的区别是 LSTM 前向返回 (output, (hn, cn))，比 RNN 多一个细胞状态 cn。

class LSTMSentimentClassifier(nn.Module):
    """
    Embedding + LSTM 情感分类器（建模词序）

    结构概览:
      Embedding(vocab, 128) → LSTM(128→256) → Linear(256→1)
    参数量: 约 2.96M（LSTM 含 4 组门控权重，比 vanilla RNN 多约 3 倍）
    """

    def __init__(
            self,
            vocab_size,  # 词典大小
            embed_dim=128,  # Embedding 向量维度
            num_class=1,  # 输出类别数（二分类为 1）
            padding_idx=0,  # PAD 的 ID
            lstm_hidden=256,  # LSTM 隐藏状态维度
            lstm_layers=1,  # LSTM 层数
            bidirectional=False  # 是否双向
    ):
        """
        初始化模型

        参数:
            vocab_size:    词典大小（Embedding 表的行数）
            embed_dim:     Embedding 向量维度
            num_class:     输出类别数
            padding_idx:   PAD 的 ID（其 Embedding 恒为 0）
            lstm_hidden:   LSTM 隐藏状态维度，默认 256
            lstm_layers:   LSTM 层数，默认 1
            bidirectional: 是否双向 LSTM，默认 False
        """
        super().__init__()  # 调用父类 nn.Module 构造函数
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)  # 词向量层
        self.lstm = nn.LSTM(  # LSTM 层（含细胞状态 c_t，缓解梯度消失）
            input_size=embed_dim,  # 每个时间步输入是 embed_dim 维的词向量
            hidden_size=lstm_hidden,  # 隐藏状态维度
            num_layers=lstm_layers,  # LSTM 层数
            batch_first=True,  # 输入形状为 (batch, seq_len, input_size) 而非 (seq_len, batch, ...)
            bidirectional=bidirectional  # 是否双向
        )
        self.bidirectional = bidirectional  # 保存是否双向
        self.num_directions = 2 if bidirectional else 1  # 双向 LSTM 输出维度是单向的两倍
        self.lstm_hidden = lstm_hidden  # 保存隐藏维度
        self.lstm_layers = lstm_layers  # 保存层数
        # 全连接层: 输入维度 = hidden × 方向数（双向时拼接两个方向 → hidden*2）
        self.fc = nn.Linear(lstm_hidden * self.num_directions, num_class)

    def forward(self, input_ids):
        """
        前向传播: 读取整句 → 取最后一步隐藏状态 → 全连接分类

        参数:
            input_ids: (batch_size, seq_len) 的 token ID 张量
        返回:
            out: (batch_size, 1) 的 logit 张量
        """
        # (bs, seq_len) → (bs, seq_len, embed_dim): 查表得到每个词的向量
        x = self.embedding(input_ids)
        # LSTM 前向: 输入 (bs, seq_len, embed_dim)
        #   output: (bs, seq_len, hidden*dirs)     每个时间步的输出
        #   hn:     (num_layers*dirs, bs, hidden)  每层最后一个时间步的隐藏状态
        #   cn:     (num_layers*dirs, bs, hidden)  每层最后一个时间步的细胞状态
        output, (hn, cn) = self.lstm(x)
        # 取最后一层的隐藏状态作为整句的语义向量
        if self.bidirectional:  # 双向 LSTM
            # hn[-2] 是正向最后一层最后一步，hn[-1] 是反向最后一层最后一步
            # 把两个方向的向量拼接: (bs, hidden*2)
            last_hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        else:  # 单向 LSTM
            last_hidden = hn[-1]  # (bs, hidden): 最后一层最后一步的隐藏状态
        # 全连接: (bs, hidden*dirs) → (bs, 1) logit
        out = self.fc(last_hidden)
        return out  # 返回 logits


# ============================================================
# 8. 模型实例化、前向传播验证与参数量统计
# ============================================================

vocab_size = len(tokenizer.vocab)  # 词典大小 = 20000（Embedding 表行数）
padding_idx = 0  # PAD 的 ID 为 0

# ---- 实例化模型（默认单层单向）----
model = LSTMSentimentClassifier(
    vocab_size=vocab_size,  # 词典大小
    embed_dim=128,  # 词向量维度
    padding_idx=padding_idx,  # PAD 的 ID
    lstm_layers=1,  # 单层
    bidirectional=False  # 单向
)

# 打印模型结构验证
print("\n========== 单层单向模型结构 ==========")
print(model)

# 打印每层参数的 shape，对照第 6 节的权重命名规则理解
print("\n模型各层参数 shape:")
for name, param in model.named_parameters():  # 遍历所有命名参数
    print(f"Layer: {name}, Shape: {param.shape}")

# ---- 统计模型总参数量 ----
def count_parameters(model):
    """统计模型的可训练参数量（requires_grad=True 的参数总数）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 累加每个参数的元素个数

total_params = count_parameters(model)  # 模型总参数量
print(f"\nTotal parameters in the model: {total_params}")  # 约 2,955,521


# ---- 前向传播验证：模型能接收随机 ID 并输出 (batch, 1) ----
# 模拟一批 token id 序列（单层单向）
batch_size = 4  # 模拟 4 条影评
seq_len = 10  # 模拟每条影评 10 个 token
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))  # 随机生成一批 token ID

print("\n========== 前向传播验证（单层单向）==========")
print("dummy_input.shape:", dummy_input.shape)  # torch.Size([4, 10])
output = model(dummy_input)  # 前向传播
print("output.shape:", output.shape)  # (batch_size, 1)
print("output:", output)  # 打印 logit 值


# ---- 扩展演示一：单层双向 LSTM ----
# 双向时 fc 输入维度翻倍为 hidden*2=512，参数量约翻倍
bi_model = LSTMSentimentClassifier(
    vocab_size=vocab_size,  # 词典大小
    embed_dim=128,  # 词向量维度
    padding_idx=padding_idx,  # PAD 的 ID
    lstm_layers=1,  # 单层
    bidirectional=True  # 开启双向
)

# 打印模型结构验证
print("\n========== 单层双向模型结构 ==========")
print(bi_model)

# 打印每层参数的 shape（可看到 _reverse 反向权重）
for name, param in bi_model.named_parameters():  # 遍历所有命名参数
    print(f"Layer: {name}, Shape: {param.shape}")

# 做一次前向计算，验证模型输出
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))  # 重新生成一批随机 ID
output = bi_model(dummy_input)  # 双向模型前向
print("Bi-directional model output.shape:", output.shape)  # 仍为 (batch_size, 1)
print("output:", output)


# ---- 扩展演示二：双层单向 LSTM ----
two_layer_model = LSTMSentimentClassifier(
    vocab_size=vocab_size,  # 词典大小
    embed_dim=128,  # 词向量维度
    padding_idx=padding_idx,  # PAD 的 ID
    lstm_layers=2,  # 两层
    bidirectional=False  # 单向
)

# 打印模型结构验证
print("\n========== 双层单向模型结构 ==========")
print(two_layer_model)

# 打印每层参数的 shape（可看到 l1 层权重）
for name, param in two_layer_model.named_parameters():  # 遍历所有命名参数
    print(f"Layer: {name}, Shape: {param.shape}")

# 做一次前向计算，验证模型输出
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))  # 重新生成一批随机 ID
output = two_layer_model(dummy_input)  # 双层模型前向
print("Two-layer model output.shape:", output.shape)  # 仍为 (batch_size, 1)
print("output:", output)


# ============================================================
# 9. Trainer 通用训练器类 —— 二分类专用
# ============================================================
# 封装完整的二分类训练流水线：训练循环 + 验证评估 + 曲线绘图。
# 与图像分类（CrossEntropyLoss + argmax）不同，本任务使用:
#   - BCEWithLogitsLoss 二分类损失（模型输出 1 维 logit，>0 判为类别 1）
#   - 准确率: outputs.squeeze() > 0 与标签比较

class Trainer:
    """
    通用训练器（二分类）: 封装训练循环、评估、损失/准确率记录与可视化

    使用方式:
      trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device)
      trainer.fit(epochs)   # 训练 epochs 轮
      trainer.plot()        # 绘制损失与准确率曲线
    """

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device):
        """
        初始化训练器

        参数:
            model:        待训练的 PyTorch 模型（nn.Module）
            train_loader: 训练集 DataLoader
            val_loader:   验证集 DataLoader
            optimizer:    优化器（如 Adam）
            criterion:    损失函数（二分类用 BCEWithLogitsLoss）
            device:       训练设备 'cuda' 或 'cpu'
        """
        self.model = model  # 保存模型
        self.train_loader = train_loader  # 保存训练集加载器
        self.val_loader = val_loader  # 保存验证集加载器
        self.optimizer = optimizer  # 保存优化器
        self.criterion = criterion  # 保存损失函数
        self.device = device  # 保存设备

        # 训练历史记录（每个 epoch 记录一次，用于最终绘图）
        self.train_losses = []  # 每轮训练集平均损失
        self.val_losses = []  # 每轮验证集平均损失
        self.train_accs = []  # 每轮训练集准确率
        self.val_accs = []  # 每轮验证集准确率

    def train_one_epoch(self):
        """
        训练一个 epoch: 遍历全部训练 batch，前向 → 反向 → 更新参数

        返回:
            avg_loss: 本轮平均训练损失
            acc:      本轮训练集准确率（0~1 小数）
        """
        self.model.train()  # 切换到训练模式
        running_loss = 0.0  # 损失累加器（按样本数加权，最后除以总样本数）
        correct = 0  # 预测正确的样本数
        total = 0  # 总样本数

        for inputs, labels in self.train_loader:  # 遍历每个 batch
            inputs = inputs.to(self.device)  # 输入 ID 张量移至设备
            labels = labels.to(self.device)  # 标签移至设备
            self.optimizer.zero_grad()  # 1. 清空上一步累积的梯度
            outputs = self.model(inputs)  # 2. 前向传播，输出形状 (batch, 1) 的 logit
            # 输出层为 1 个神经元，去掉多余的维度 (batch,1)→(batch,)，方便算损失和准确率
            outputs = outputs.squeeze()
            loss = self.criterion(outputs, labels.float())  # 3. 计算二分类损失（标签转 float）
            loss.backward()  # 4. 反向传播求梯度
            self.optimizer.step()  # 5. 更新参数
            running_loss += loss.item() * inputs.size(0)  # 累加损失（乘以样本数做加权，便于求平均）

            # ---- 二分类准确率计算 ----
            preds = outputs > 0  # logit > 0 预测为正类（得到布尔张量）
            correct += (preds == labels).sum().item()  # 预测与真实标签相等的个数
            total += labels.size(0)  # 累加本 batch 样本数

        avg_loss = running_loss / total  # 平均损失 = 加权总损失 / 总样本数
        acc = correct / total  # 准确率 = 正确数 / 总数（0~1）
        return avg_loss, acc  # 返回 (平均损失, 准确率)

    def evaluate(self):
        """
        在验证集上评估: 只做前向，不更新参数

        返回:
            avg_loss: 平均验证损失
            acc:      验证集准确率（0~1 小数）
        """
        self.model.eval()  # 切换到评估模式
        running_loss = 0.0  # 损失累加器
        correct = 0  # 预测正确的样本数
        total = 0  # 总样本数
        with torch.no_grad():  # 禁用梯度计算，节省显存并加速（评估不需要梯度）
            for inputs, labels in self.val_loader:  # 遍历验证集每个 batch
                inputs = inputs.to(self.device)  # 输入移至设备
                labels = labels.to(self.device)  # 标签移至设备
                outputs = self.model(inputs)  # 前向传播，得到 logit (batch,1)
                outputs = outputs.squeeze()  # 去掉最后一维 (batch,1)→(batch,)
                loss = self.criterion(outputs, labels.float())  # 计算二分类损失
                running_loss += loss.item() * inputs.size(0)  # 加权累加损失
                preds = outputs > 0  # logit > 0 判为正类
                correct += (preds == labels).sum().item()  # 累计正确数
                total += labels.size(0)  # 累计样本数

        avg_loss = running_loss / total  # 平均验证损失
        acc = correct / total  # 验证准确率
        return avg_loss, acc  # 返回 (平均损失, 准确率)

    def fit(self, epochs):
        """
        训练主循环: 逐轮调用 train_one_epoch + evaluate，记录历史

        参数:
            epochs: 训练轮数
        """
        for epoch in range(epochs):  # 逐轮训练
            train_loss, train_acc = self.train_one_epoch()  # 训练一个 epoch
            val_loss, val_acc = self.evaluate()  # 在验证集上评估
            self.train_losses.append(train_loss)  # 记录训练损失
            self.val_losses.append(val_loss)  # 记录验证损失
            self.train_accs.append(train_acc)  # 记录训练准确率
            self.val_accs.append(val_acc)  # 记录验证准确率

            # 打印当前 epoch 的损失与准确率
            print(f'Epoch {epoch + 1}/{epochs}: '  # 当前轮数
                  f'Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | '  # 训练指标
                  f'Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}')  # 验证指标

    def plot(self):
        """
        绘制训练/验证的损失曲线与准确率曲线（两张子图并排）
        """
        epochs = range(1, len(self.train_losses) + 1)  # 横轴: epoch 编号（从 1 开始）
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 创建 1 行 2 列的子图，12×5 英寸

        # ---- 左子图: 损失曲线 ----
        axs[0].plot(epochs, self.train_losses, label='Train Loss')  # 训练损失折线
        axs[0].plot(epochs, self.val_losses, label='Validation Loss')  # 验证损失折线
        axs[0].set_title('Loss')  # 子图标题
        axs[0].set_xlabel('Epoch')  # 横轴标签
        axs[0].set_ylabel('Loss')  # 纵轴标签
        axs[0].legend()  # 显示图例
        axs[0].grid(True)  # 显示网格

        # ---- 右子图: 准确率曲线 ----
        axs[1].plot(epochs, self.train_accs, label='Train Accuracy')  # 训练准确率折线
        axs[1].plot(epochs, self.val_accs, label='Validation Accuracy')  # 验证准确率折线
        axs[1].set_title('Accuracy')  # 子图标题
        axs[1].set_xlabel('Epoch')  # 横轴标签
        axs[1].set_ylabel('Accuracy')  # 纵轴标签
        axs[1].legend()  # 显示图例
        axs[1].grid(True)  # 显示网格

        plt.tight_layout()  # 自动调整子图间距，避免重叠
        plt.show()  # 显示图像


# ============================================================
# 10. 训练准备
# ============================================================

# ---- 判断可用设备 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先 GPU，否则 CPU
print(f"\n使用设备: {device}")  # 打印当前训练设备

# 将模型移至目标设备（后续 bi_model / two_layer_model 仅作演示，不参与训练）
model = model.to(device)

# ---- 损失函数 ----
# BCEWithLogitsLoss: 把 Sigmoid 和 BCE 合并成一个函数（数值更稳定）
# 输入: 模型输出的原始 logit（不需要提前 sigmoid）；标签: 0/1 的浮点值
criterion = nn.BCEWithLogitsLoss()  # 二分类损失

# ---- 优化器 ----
lr = 1e-3  # 学习率 0.001，Adam 的推荐默认值
optimizer = optim.Adam(model.parameters(), lr=lr)  # 模型优化器


# ============================================================
# 11. 训练模型并绘图
# ============================================================
# 训练在 train_loader（PAD 前置）上进行，评估在 test_loader（PAD 前置）上进行。
# 参考效果：10 轮后 Train Acc≈0.95，Val Acc≈0.88。
# 注意: 脚本未固定随机种子，每次运行结果会有轻微波动；
#   LSTM 比 vanilla RNN 更稳定，典型 Val Acc 约 0.85~0.88。

epochs = 10  # 训练轮数

print(f"\n========== 开始训练模型：Embedding + LSTM (epochs={epochs}) ==========")

# 创建训练器实例
trainer = Trainer(
    model=model,  # 待训练的模型
    train_loader=train_loader,  # 训练集 DataLoader（PAD 前置）
    val_loader=test_loader,  # 验证集 DataLoader（PAD 前置，使用测试集）
    optimizer=optimizer,  # Adam 优化器
    criterion=criterion,  # BCEWithLogitsLoss 损失函数
    device=device  # 训练设备
)

trainer.fit(epochs)  # 训练 epochs 轮
trainer.plot()  # 绘制训练曲线

# 训练结束后在验证集（测试集）上评估最终准确率
final_loss, final_acc = trainer.evaluate()  # 最终评估
print(f"\n========== 模型最终评估 ==========")  # 分隔标题
print(f"Embedding+LSTM - Val Loss: {final_loss:.4f}, Val Accuracy: {final_acc:.4f}")  # 打印结果


# ============================================================
# 12. 总结
# ============================================================

print("\n" + "=" * 65)  # 打印分隔线
print("========== 总结 ==========")  # 总结标题
print("=" * 65)  # 打印分隔线

print(f"\n模型名称: Embedding + LSTM 情感分类器")  # 模型名称
print(f"总参数量: {total_params:,}")  # 约 2,955,521
print(f"验证集准确率: {final_acc:.4f}")  # 训练后的准确率

print(f"\n关键知识点:")
print(f"  - Embedding 层把离散的 token ID 映射为稠密的词向量，是整个模型的参数量大头")
print(f"  - LSTM 比 vanilla RNN 多一个细胞状态 c_t（长期记忆），靠门控决定记住/遗忘信息")
print(f"  - LSTM 按顺序读取整句，最后一步隐藏状态浓缩了整句上下文，用于分类")
print(f"  - 双向 LSTM 同时读正序与逆序，输出维度翻倍；多层 LSTM 纵向堆叠加深表征能力")
print(f"  - padding 必须前置（padding_first=True）: 句尾的 PAD 会污染最后一步隐藏状态")
print(f"  - nn.RNN / nn.LSTM / nn.GRU 接口一致；LSTM 前向多返回一个细胞状态 cn")