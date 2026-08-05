"""
CIFAR-10 分类 —— VGG11 卷积神经网络 + 模型精调（卷积神经网络CNN3.md）
======================================================
本脚本实现了 CIFAR-10 数据集的 10 分类任务，整合了以下内容：
1. Kaggle 数据准备（PNG 图片 + CSV 标签）
2. 自定义 CIFAR10CustomDataset Dataset 类（根据 CSV 标签读取 PNG 图片）
3. 数据预处理与增强（Resize、RandomRotation、RandomHorizontalFlip、Normalize）
4. VGG11 —— 经典卷积神经网络模型（5 个卷积块 + 全连接分类器）
5. Trainer 通用训练器类（含早停、TensorBoard、绘图、回归支持）
6. 参数统计与前向传播验证
7. 第一阶段训练（从零开始训练 VGG11）与验证集评估
8. 模型精调（Fine-tuning）—— 加载最优权重、重置分类层、分组学习率
9. 第二阶段精调训练与最终评估
10. 模型总结与分析
"""

# ============================================================
# 0. 导入所有需要的库
# ============================================================

import torch  # PyTorch 核心库，提供张量运算与自动求导
import torch.nn as nn  # 神经网络模块，提供 Conv2d、Linear、ReLU、Dropout、MaxPool2d 等层
import torch.optim as optim  # 优化器模块，提供 SGD、Adam 等
from torch.utils.data import Dataset, DataLoader  # Dataset: 自定义数据集基类；DataLoader: 批量加载器
from torchvision import transforms  # 数据预处理模块，提供 Compose、ToTensor、Normalize 等
import matplotlib.pyplot as plt  # 绘图库，用于训练曲线可视化
from matplotlib import rcParams  # matplotlib 配置字典，用于设置全局绘图参数（如中文字体）
import os  # 操作系统接口，用于路径拼接、文件存在检查、目录创建
import pandas as pd  # 数据处理库，用于读取 CSV 标签文件
from PIL import Image  # 图像处理库，用于加载 PNG/JPG 图片
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 写入器，用于记录训练日志

# 设置中文字体，防止 matplotlib 中文显示为方块
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号（避免负号显示为方块）


# ============================================================
# 1. 数据准备 —— 路径与标签 CSV
# ============================================================

# 图像主目录（训练集 PNG 图片存放位置）
data_dir = '../data/cifar-10/train/train'  # 本地路径：训练图片文件夹，内含 50000 张 PNG
# 标签 CSV 文件路径
label_csv = '../data/cifar-10/trainLabels.csv'  # 本地路径：标签 CSV，包含 id 和 label 两列


# ============================================================
# 2. 数据预处理变换定义
# ============================================================

# ---- 训练集 transform：数据增强 + 归一化 ----
# transforms.Compose 将多个变换操作按顺序组合，依次执行
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 统一缩放到 32×32 像素（CIFAR-10 的标准输入尺寸）
    transforms.ToTensor(),  # 将 PIL.Image (0-255 整数) 转为 torch.Tensor (0.0-1.0 浮点)，维度 H×W×C → C×H×W
    transforms.RandomRotation(40),  # 随机旋转：最大旋转角度 ±40 度，模拟不同拍摄角度（数据增强）
    transforms.RandomHorizontalFlip(),  # 随机水平翻转：以 50% 概率将图片左右翻转，增加数据多样性
    transforms.Normalize((0.4915, 0.4821, 0.4464), (0.2472, 0.2437, 0.2617))  # 用 CIFAR-10 训练集的通道均值和标准差做标准化
])

# ---- 验证集 transform：仅基础预处理，不做数据增强 ----
val_transform = transforms.Compose([
    transforms.ToTensor(),  # PIL.Image → Tensor，值域 0~1，维度 HWC → CHW
    transforms.Normalize((0.4915, 0.4821, 0.4464), (0.2472, 0.2437, 0.2617))  # 使用与训练集相同的标准化参数（均值, 标准差）
])


# ============================================================
# 3. 自定义 Dataset 类 —— CIFAR10CustomDataset
# ============================================================
# CIFAR-10 图片按 id.png 命名（如 1.png, 2.png, ...），标签在 trainLabels.csv 中（列: id, label）
# 该类继承 torch.utils.data.Dataset，必须实现 __len__ 和 __getitem__ 两个方法
# CSV 格式示例:
#   id,label
#   1,frog
#   2,truck
#   3,automobile
#   ...


class CIFAR10CustomDataset(Dataset):
    """
    根据 DataFrame 子集读取 CIFAR-10 图片与标签的自定义数据集类

    CIFAR-10 目录结构：
      data_dir/
        1.png      (id=1 的图片，标签在 CSV 中)
        2.png
        ...
        50000.png

    该类的作用: 将分散的 PNG 文件和标签 CSV 文件组织成 PyTorch Dataset，供 DataLoader 使用
    """

    def __init__(self, img_dir, labels_frame, class_to_idx, transform=None):
        """
        初始化 CIFAR-10 数据集

        参数:
            img_dir:       图片文件夹路径，内含 PNG 文件（以数字 id 命名）
            labels_frame:  pandas DataFrame，包含图片 id 和 label 两列
            class_to_idx:  dict，类别名称→整数索引的映射，如 {'airplane':0, 'automobile':1, ...}
            transform:     数据预处理变换（Compose 对象），默认为 None
        """
        self.img_dir = img_dir  # 保存图片目录路径
        self.labels_frame = labels_frame  # 保存标签 DataFrame（含 id 列和 label 列）
        self.class_to_idx = class_to_idx  # 保存类别→索引映射字典
        self.transform = transform  # 保存数据预处理变换（训练用增强，验证仅归一化）

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.labels_frame)  # 样本数 = DataFrame 的行数（即图片张数）

    def __getitem__(self, idx):
        """
        根据索引获取单个样本（图片 + 标签）

        参数:
            idx: 样本索引（0 ~ len-1）
        返回:
            (image, label): image 是预处理后的 Tensor (3,32,32)，label 是 0~9 的整数索引
        """
        img_number = str(self.labels_frame.iloc[idx, 0])  # 获取第 idx 行、第 0 列的图片 id（转为字符串用于拼接路径）
        img_name = os.path.join(self.img_dir, img_number + '.png')  # 拼接完整图片路径: data_dir/id.png
        image = Image.open(img_name).convert('RGB')  # 用 PIL 打开图片并确保为 RGB 三通道模式（丢弃 alpha 通道）
        label_name = self.labels_frame.iloc[idx, 1]  # 获取第 idx 行、第 1 列的类别名称（如 'frog'、'cat' 等）
        label = self.class_to_idx[label_name]  # 将类别名称映射为整数索引 0~9
        if self.transform:  # 如果传入了预处理变换
            image = self.transform(image)  # 对图片应用变换（ToTensor → Normalize 等）
        return image, label  # 返回 (图像Tensor, 标签整数)


# ============================================================
# 4. 数据加载与划分
# ============================================================

# ---- 读取标签 CSV 文件 ----
labels_df = pd.read_csv(label_csv)  # 用 pandas 读取训练标签文件（50000 行 × 2 列: id, label）
print("标签 CSV 总行数:", len(labels_df))  # 应输出 50000

# ---- 划分训练集与验证集 ----
train_size = 45000  # 训练集样本数: 前 45000 张（90%）
val_size = 5000  # 验证集样本数: 后 5000 张（10%）
assert train_size + val_size <= len(labels_df), "数据集图片数量不足！"  # 断言确保数据量足够划分

# iloc 按位置索引切片:
#   [:train_size]       → 前 45000 行作为训练集标签
#   [train_size:train_size+val_size] → 第 45001~50000 行作为验证集标签
# reset_index(drop=True): 重置行索引为 0,1,2,...，丢弃旧索引
train_labels_df = labels_df.iloc[:train_size].reset_index(drop=True)  # 训练集标签 DataFrame，重置行索引
val_labels_df = labels_df.iloc[train_size:train_size + val_size].reset_index(drop=True)  # 验证集标签 DataFrame，重置行索引

# ---- 建立类别映射 ----
# sorted 保证类别按字母序排列，使训练/验证/测试集的类别编号完全一致
class_names = sorted(labels_df['label'].unique())  # 获取所有不重复的类别名称（如 'airplane', 'automobile', ...），排序
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}  # 构建类别名→索引字典: {'airplane':0, 'automobile':1, ..., 'truck':9}
print("类别映射:", class_to_idx)  # 打印类别→索引映射供检查

# ---- 创建 Dataset 实例 ----
train_dataset = CIFAR10CustomDataset(data_dir, train_labels_df, class_to_idx, transform=train_transform)  # 训练集 Dataset（含数据增强）
val_dataset = CIFAR10CustomDataset(data_dir, val_labels_df, class_to_idx, transform=val_transform)  # 验证集 Dataset（仅归一化，无增强）

# ---- 创建 DataLoader ----
# batch_size=32: 每批处理 32 张图片
# shuffle=True: 训练集随机打乱，防止模型记忆样本顺序
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练集 DataLoader
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证集 DataLoader: 不打乱，保持评估一致性


# ============================================================
# 5. 数据集基本信息打印与验证
# ============================================================

print("\n========== 数据集基本信息 ==========")  # 分隔标题
print("训练集样本数:", len(train_dataset))  # 45000
print("验证集样本数:", len(val_dataset))  # 5000
print("类别总数:", len(class_names))  # 10
print("类别名称:", class_names)  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ---- 验证单个样本的形状和标签 ----
sample_img, sample_label = train_dataset[0]  # 获取训练集第一个样本: (图像 Tensor, 标签)
print("单张图片 shape (C, H, W):", sample_img.shape)  # torch.Size([3, 32, 32]) —— 3 通道，32×32 像素
print("第一个样本的标签索引:", sample_label)  # 0~9 的整数（对应类别名称）

# ---- 验证一个 batch 的数据形状 ----
for images, labels in train_loader:  # 取训练集第一个 batch
    print("一个 batch 的图片 shape:", images.shape)  # torch.Size([32, 3, 32, 32]) —— 32 张 3×32×32 图片
    print("一个 batch 的标签 shape:", labels.shape)  # torch.Size([32]) —— 32 个标签
    break  # 只取第一个 batch，立即跳出循环


# ============================================================
# 6. VGG11 卷积神经网络模型定义
# ============================================================
# 整体结构:
#   输入 (3, 32, 32) 彩色图
#   → 5 个卷积块（每块: 卷积 + ReLU + 池化），通道数逐步翻倍 64→128→256→512→512
#   → 特征图尺寸: 32×32 → 16×16 → 8×8 → 4×4 → 2×2 → 1×1
#   → 展平 → 全连接分类器 (512→128→10) 输出 logits
#
# 经典 VGG 设计思想:
#   - 统一使用 3×3 小卷积核堆叠，两个 3×3 的堆叠感受野等同于一个 5×5，但参数量更少
#   - 通道数随网络加深翻倍，深层提取更抽象的语义特征
#   - 第 3~5 卷积块采用双卷积设计，提升非线性表达能力


class VGG11(nn.Module):
    """
    VGG11 卷积神经网络，用于 CIFAR-10 分类（32×32 输入图片）

    结构概览:
      输入 (3, 32, 32) 彩色图
      → features: 5 个卷积块（卷积 + ReLU + MaxPool），输出 (512, 1, 1)
      → 展平: (batch, 512)
      → classifier: Linear(512→128) + ReLU + Dropout + Linear(128→10)

    参数量: 约 9.3M（远大于 InceptionNet 的 318K，适合作为标准的对比模型）
    """

    def __init__(self, num_classes=10):
        """
        初始化 VGG11

        参数:
            num_classes: 输出类别数，默认 10（CIFAR-10 的 10 个类别）
        """
        super().__init__()  # 调用父类 nn.Module 的构造函数

        # ---- 特征提取层：5 个卷积块 ----
        # Sequential 容器按顺序堆叠各层，前向传播时自动依次执行
        self.features = nn.Sequential(
            # --- 卷积块1：3 → 64 通道，32×32 → 16×16 ---
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入 3 通道(RGB) → 64 通道，3×3 卷积核，padding=1 保持尺寸
            nn.ReLU(inplace=True),  # ReLU 激活函数（inplace=True 原地操作节省内存）
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2×2 最大池化，步长 2，特征图尺寸减半: 32×32 → 16×16

            # --- 卷积块2：64 → 128 通道，16×16 → 8×8 ---
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输入 64 通道 → 128 通道
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化: 16×16 → 8×8

            # --- 卷积块3：128 → 256 通道（两层卷积），8×8 → 4×4 ---
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 输入 128 通道 → 256 通道
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # VGG 块内双卷积设计: 提升非线性表达能力
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化: 8×8 → 4×4

            # --- 卷积块4：256 → 512 通道（两层卷积），4×4 → 2×2 ---
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 输入 256 通道 → 512 通道
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # VGG 块内双卷积设计
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化: 4×4 → 2×2

            # --- 卷积块5：512 → 512 通道（两层卷积），2×2 → 1×1 ---
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 输入 512 通道 → 512 通道
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # VGG 块内双卷积设计
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化: 2×2 → 1×1，此时输出为 (512, 1, 1)
        )

        # ---- 分类器（全连接层）：将 512 维特征映射到 10 个类别 ----
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),  # 全连接层: 展平后的 512 维特征 → 128 维隐藏特征
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Dropout(),  # Dropout 层: 训练时以 50% 概率随机丢弃神经元，防止过拟合
            nn.Linear(128, num_classes)  # 全连接层: 128 维 → num_classes(10) 维 logits
        )

        self._initialize_weights()  # 调用自定义方法对所有权重进行初始化

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状 (batch_size, 3, 32, 32)
        返回:
            logits: 形状 (batch_size, num_classes)，即 (batch_size, 10)
        """
        x = self.features(x)  # 通过 5 个卷积块提取特征，输出形状 (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # 将特征图展平为二维向量: (batch, 512, 1, 1) → (batch, 512)
        x = self.classifier(x)  # 通过全连接分类器得到各类别的原始得分 (batch, 10)
        return x  # 返回 logits（未经 softmax，CrossEntropyLoss 内部会自动做 softmax）

    def _initialize_weights(self):
        """
        自定义权重初始化方法（好的初始化能加速收敛、提升精度）
        """
        for m in self.modules():  # 遍历模型中的所有子模块（递归遍历）
            if isinstance(m, nn.Conv2d):  # 如果当前模块是二维卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 正态初始化卷积权重（专为 ReLU 设计）
                if m.bias is not None:  # 如果卷积层有偏置参数
                    nn.init.zeros_(m.bias)  # 将偏置初始化为全零
            elif isinstance(m, nn.Linear):  # 如果当前模块是全连接层
                nn.init.xavier_uniform_(m.weight)  # Xavier 均匀分布初始化全连接层权重（保持输入输出方差一致）
                nn.init.zeros_(m.bias)  # 将偏置初始化为全零


# ============================================================
# 7. Trainer 通用训练器类
# ============================================================
# 封装完整的训练流水线:
#   训练循环 + 验证评估 + 早停 + 最优模型保存 + TensorBoard + 绘图
# 同时支持分类任务（记录准确率）与回归任务（仅记录损失）


class Trainer:
    """
    通用训练器：封装训练循环、评估、早停、模型保存与可视化

    支持两种任务:
      - 分类: 使用 train() + evaluating()，记录损失与准确率
      - 回归: 使用 regression_train() + regression_evaluating()，仅记录损失

    早停机制:
      - 监控验证集指标（损失或准确率），连续 patience 轮未提升则自动停止
      - 停止后自动恢复到训练过程中保存的最优模型权重
    """

    def __init__(
            self,
            model,  # 待训练的 PyTorch 模型实例（nn.Module）
            trainloader,  # 训练集 DataLoader
            valloader,  # 验证集 DataLoader
            criterion,  # 损失函数（如 CrossEntropyLoss、MSELoss）
            optimizer,  # 优化器（如 Adam、SGD）
            device='cuda',  # 训练设备: 'cuda'（GPU）或 'cpu'
            epochs=10,  # 最大训练轮数，默认 10 轮
            early_stopping=True,  # 是否启用早停机制，默认开启
            patience=5,  # 早停容忍度: 连续 patience 轮指标未提升则停止训练
            save_path="best_model.pth",  # 最优模型权重保存路径
            early_stop_mode="loss",  # 早停监控指标: "loss"（损失越小越好）或 "acc"（准确率）
            maximize_acc=True,  # early_stop_mode="acc" 时: True=准确率越大越好, False=越小越好
            use_tensorboard=False,  # 是否启用 TensorBoard 可视化日志
            log_dir='tensorboard_logs'  # TensorBoard 日志存放目录
    ):
        """初始化训练器，保存所有配置并创建 TensorBoard 写入器"""
        self.model = model  # 保存模型实例
        self.trainloader = trainloader  # 保存训练集加载器
        self.valloader = valloader  # 保存验证集加载器
        self.criterion = criterion  # 保存损失函数
        self.optimizer = optimizer  # 保存优化器
        self.device = device  # 保存训练设备（GPU/CPU）
        self.epochs = epochs  # 保存最大训练轮数

        # 训练历史记录列表（每个元素对应一个 epoch，用于最终绘图）
        self.train_losses = []  # 每轮训练集平均损失
        self.val_losses = []  # 每轮验证集平均损失
        self.train_accuracies = []  # 每轮训练集准确率（%）
        self.val_accuracies = []  # 每轮验证集准确率（%）

        # 早停相关配置
        self.early_stopping = early_stopping  # 是否启用早停
        self.patience = patience  # 早停容忍度（连续未提升多少轮就停）
        self.save_path = save_path  # 最优模型权重保存路径
        self.early_stop_mode = early_stop_mode  # 早停监控模式: "loss" 或 "acc"
        self.maximize_acc = maximize_acc  # acc 模式下: True=准确率越大越好, False=越小越好

        # 早停运行状态变量
        self.best_metric = None  # 历史最优度量值（初始为 None，第一个 epoch 自动记录）
        self.early_stop_counter = 0  # 连续未提升的轮数计数器
        self.best_epoch = 0  # 取得最优度量值时的 epoch 编号

        # TensorBoard 日志配置
        self.use_tensorboard = use_tensorboard  # 是否使用 TensorBoard
        self._writer = None  # TensorBoard SummaryWriter 句柄，初始为 None
        if self.use_tensorboard:  # 如果启用了 TensorBoard
            if not os.path.exists(log_dir):  # 检查日志目录是否存在
                os.makedirs(log_dir)  # 不存在则递归创建目录
            self._writer = SummaryWriter(log_dir)  # 创建 SummaryWriter 实例，绑定日志目录

    def evaluating(self, dataloader):
        """
        分类任务评估函数 —— 在给定数据上计算平均损失和准确率

        参数:
            dataloader: 待评估的数据加载器（验证集或测试集）
        返回:
            avg_loss: 平均损失（所有 batch 损失的平均）
            acc:      准确率（%，正确预测数 / 总样本数 × 100）
        """
        self.model.eval()  # 切换到评估模式: 关闭 Dropout、冻结 BatchNorm 运行统计量
        correct = 0  # 累计预测正确的样本数
        total = 0  # 累计总样本数
        running_loss = 0.0  # 累计总损失（用于求平均）

        with torch.no_grad():  # 禁用梯度计算，大幅节省显存和计算量（推理无需梯度）
            for images, labels in dataloader:  # 逐 batch 遍历
                images = images.to(self.device)  # 将当前 batch 图片数据移至 GPU/CPU
                labels = labels.to(self.device)  # 将当前 batch 标签数据移至 GPU/CPU
                outputs = self.model(images)  # 前向传播: 得到每类 logits，形状 (batch, 10)
                loss = self.criterion(outputs, labels)  # 计算当前 batch 的分类损失
                running_loss += loss.item()  # 累加损失（.item() 将 0 维 Tensor 转为 Python float）
                # torch.argmax(outputs, dim=1): 沿类别维度 (dim=1) 取最大值的索引作为预测类别
                predicted = torch.argmax(outputs, dim=1)  # 获取每个样本的预测类别 (0~9)
                total += labels.size(0)  # 累加当前 batch 的样本数（通常是 batch_size，最后一个 batch 可能更少）
                correct += (predicted == labels).sum().item()  # 累加预测正确的样本数（比较后求和再转 float）

        acc = 100 * correct / total if total > 0 else 0  # 准确率转为百分比（%），分母防零
        avg_loss = running_loss / len(dataloader)  # 平均损失 = 总损失 / batch 数量
        return avg_loss, acc  # 返回二元组 (平均损失, 准确率%)

    def regression_evaluating(self, dataloader):
        """
        回归任务评估函数 —— 只返回平均损失，不计算准确率

        参数:
            dataloader: 数据加载器
        返回:
            avg_loss: 平均损失
        """
        self.model.eval()  # 切换到评估模式
        running_loss = 0.0  # 累计损失初始化为 0
        with torch.no_grad():  # 禁用梯度计算
            for data, target in dataloader:  # 遍历每个 batch
                data = data.to(self.device)  # 输入数据移至设备
                target = target.to(self.device)  # 目标值移至设备
                output = self.model(data)  # 前向传播得到预测值
                loss = self.criterion(output, target)  # 计算损失（如 MSE）
                running_loss += loss.item()  # 累加损失
        avg_loss = running_loss / len(dataloader)  # 计算平均损失
        return avg_loss  # 返回平均损失（标量 float）

    def regression_train(self):
        """
        回归任务训练循环 —— 仅记录损失，不计算准确率

        与 train() 的区别: 评估时不计算准确率，只使用验证损失作为早停指标
        适用于房价预测、温度预测等连续值回归任务
        """
        self.model.to(self.device)  # 将模型参数迁移到目标设备
        for epoch in range(self.epochs):  # 逐轮训练，共 epochs 轮
            self.model.train()  # 切换到训练模式: 启用 Dropout、让 BatchNorm 更新统计量
            running_loss = 0.0  # 本轮损失累加器清零

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):  # 遍历训练集每个 batch
                inputs = inputs.to(self.device)  # 输入数据移至设备
                targets = targets.to(self.device)  # 目标数据移至设备
                self.optimizer.zero_grad()  # 清空上一轮的梯度（PyTorch 默认累加梯度）
                outputs = self.model(inputs)  # 前向传播
                loss = self.criterion(outputs, targets)  # 计算损失
                loss.backward()  # 反向传播求梯度
                self.optimizer.step()  # 优化器更新参数: θ = θ - lr × ∇loss
                running_loss += loss.item()  # 累加损失值

                if (batch_idx + 1) % 100 == 0:  # 每 100 个 batch 打印一次当前进度
                    print(f"[Regression] Epoch [{epoch + 1}/{self.epochs}], "
                          f"Step [{batch_idx + 1}/{len(self.trainloader)}], Loss: {loss.item():.4f}")

            avg_train_loss = running_loss / len(self.trainloader)  # 本轮平均训练损失（batch 级）
            train_loss = self.regression_evaluating(self.trainloader)  # 在训练集上评估（得到更准确的损失）
            val_loss = self.regression_evaluating(self.valloader)  # 在验证集上评估
            self.train_losses.append(train_loss)  # 记录训练损失
            self.val_losses.append(val_loss)  # 记录验证损失
            print(f"[Regression] Epoch [{epoch + 1}/{self.epochs}], "
                  f"Loss: {avg_train_loss:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # ---- TensorBoard 日志记录 ----
            if self.use_tensorboard and self._writer is not None:
                self._writer.add_scalar('Train/Loss', train_loss, epoch + 1)  # 记录训练损失曲线
                self._writer.add_scalar('Val/Loss', val_loss, epoch + 1)  # 记录验证损失曲线
                for i, param_group in enumerate(self.optimizer.param_groups):  # 遍历所有参数组（通常是 1 组）
                    self._writer.add_scalar(f'LR/group_{i}', param_group['lr'], epoch + 1)  # 记录各组学习率

            # ---- 早停与模型保存 ----
            metric = val_loss  # 回归任务只用验证损失作为评估指标（越小越好）
            if self.early_stopping:  # 如果开启早停
                if self.best_metric is None or metric < self.best_metric:  # 首次记录（None）或当前损失更低 → 有提升
                    self.best_metric = metric  # 更新历史最优损失值
                    self.early_stop_counter = 0  # 重置早停计数器
                    self.best_epoch = epoch + 1  # 记录最优 epoch 编号（从 1 开始）
                    torch.save(self.model.state_dict(), self.save_path)  # 保存当前最优模型权重到文件
                    print(f"[Info][Regression] Model improved at epoch {epoch + 1}, saving to {self.save_path}")
                else:  # 损失未下降 → 无提升
                    self.early_stop_counter += 1  # 早停计数器 +1
                    print(f"[Info][Regression] Early stop counter: {self.early_stop_counter}/{self.patience}")
                    if self.early_stop_counter >= self.patience:  # 连续 patience 轮未提升
                        print(f"[Regression] Early stopping triggered at epoch {epoch + 1}. "
                              f"Best epoch: {self.best_epoch}, Best Loss: {self.best_metric:.4f}")
                        if os.path.isfile(self.save_path):  # 如果最优权重文件存在
                            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))  # 恢复最优权重
                        if self.use_tensorboard and self._writer is not None:
                            self._writer.close()  # 关闭 TensorBoard 写入器
                        return  # 提前结束训练

        # 全部 epoch 跑完且未触发早停: 加载训练过程中保存的最优权重
        if self.early_stopping and self.best_metric is not None:
            print(f"[Regression] Training finished. Loading best model from {self.save_path}")
            if os.path.isfile(self.save_path):  # 检查权重文件是否存在
                self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))  # 恢复到最优状态
        if self.use_tensorboard and self._writer is not None:
            self._writer.close()  # 关闭 TensorBoard 写入器

    def _is_improvement(self, metric):
        """
        根据早停模式判断当前度量值是否优于历史最优

        参数:
            metric: 当前 epoch 的度量值（损失或准确率）
        返回:
            True = 有提升（优于历史最优）, False = 未提升
        """
        if self.best_metric is None:  # 尚无历史最优记录（第一个 epoch）
            return True  # 始终视为有提升（记作当前最优）
        if self.early_stop_mode == "loss":  # 损失模式: 损失越小越好
            return metric < self.best_metric  # 当前损失 < 历史最优损失 → 有提升
        elif self.early_stop_mode == "acc":  # 准确率模式
            if self.maximize_acc:  # 准确率越大越好（常规情况）
                return metric > self.best_metric  # 当前准确率 > 历史最优准确率 → 有提升
            else:  # 准确率越小越好（不常见，某些特殊指标）
                return metric < self.best_metric
        else:
            raise ValueError("Unknown early_stop_mode: {}".format(self.early_stop_mode))  # 未知模式抛出异常

    def _get_val_metric(self, val_loss, val_acc):
        """
        根据早停模式返回用于比较的度量值

        参数:
            val_loss: 当前验证集平均损失
            val_acc:  当前验证集准确率（%）
        返回:
            用于早停判断的度量值（损失值或准确率值）
        """
        if self.early_stop_mode == "loss":  # 以验证损失为早停依据
            return val_loss
        elif self.early_stop_mode == "acc":  # 以验证准确率为早停依据
            return val_acc
        else:
            raise ValueError("Unknown early_stop_mode: {}".format(self.early_stop_mode))

    def train(self):
        """
        分类任务训练主循环

        每个 epoch 的流程:
          1. 遍历训练集所有 batch，前向传播 → 计算损失 → 反向传播 → 更新参数
          2. 在训练集和验证集上分别评估损失与准确率
          3. 记录 TensorBoard 日志（如启用）
          4. 早停判断：若验证指标连续 patience 轮未提升则停止训练
          5. 每个 epoch 保存最优模型权重
        """
        self.model.to(self.device)  # 将模型参数迁移到目标设备 (GPU/CPU)

        for epoch in range(self.epochs):  # 逐轮训练，共 epochs 轮
            self.model.train()  # 切换到训练模式: 启用 Dropout、BatchNorm 统计量更新等
            running_loss = 0.0  # 当前 epoch 的损失累加器（batch 级别，用于实时显示）

            for batch_idx, (images, labels) in enumerate(self.trainloader):  # 遍历训练集每个 batch
                images = images.to(self.device)  # 将图片数据移到 GPU/CPU
                labels = labels.to(self.device)  # 将标签数据移到 GPU/CPU

                # ---- 核心训练五步（深度学习训练的标准范式） ----
                self.optimizer.zero_grad()  # 1. 清空上一轮累积的梯度（PyTorch 默认累加梯度，不手动清零会叠加）
                outputs = self.model(images)  # 2. 前向传播: 输入图片 → 模型 → 得到各类别 logits
                loss = self.criterion(outputs, labels)  # 3. 计算损失: logits 与真实标签之间的交叉熵
                loss.backward()  # 4. 反向传播: 根据损失自动计算所有 requires_grad=True 的参数的梯度
                self.optimizer.step()  # 5. 优化器更新参数: θ = θ - lr × ∇loss（对 Adam 会更复杂）

                running_loss += loss.item()  # 累加损失值（.item() 将标量张量提取为 Python float）

                if (batch_idx + 1) % 100 == 0:  # 每 100 个 batch 打印一次当前进度
                    print(f'Epoch [{epoch + 1}/{self.epochs}], '
                          f'Step [{batch_idx + 1}/{len(self.trainloader)}], Loss: {loss.item():.4f}')

            # ---- epoch 结束后的评估 ----
            avg_train_loss = running_loss / len(self.trainloader)  # 本轮平均训练损失（batch 级算数平均）
            train_loss, train_acc = self.evaluating(self.trainloader)  # 在训练集上评估: 获得更精确的平均损失与准确率
            val_loss, val_acc = self.evaluating(self.valloader)  # 在验证集上评估: 获得泛化能力的指标

            # 记录历史数据（每个 epoch 记录一次，用于后续绘图）
            self.train_losses.append(train_loss)  # 保存训练集平均损失
            self.val_losses.append(val_loss)  # 保存验证集平均损失
            self.train_accuracies.append(train_acc)  # 保存训练集准确率
            self.val_accuracies.append(val_acc)  # 保存验证集准确率

            print(f'Epoch [{epoch + 1}/{self.epochs}], '
                  f'Loss: {avg_train_loss:.4f}, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

            # ---- TensorBoard 日志记录 ----
            if self.use_tensorboard and self._writer is not None:
                self._writer.add_scalar('Train/Loss', train_loss, epoch + 1)  # 训练损失随 epoch 变化曲线
                self._writer.add_scalar('Train/Accuracy', train_acc, epoch + 1)  # 训练准确率随 epoch 变化曲线
                self._writer.add_scalar('Val/Loss', val_loss, epoch + 1)  # 验证损失随 epoch 变化曲线
                self._writer.add_scalar('Val/Accuracy', val_acc, epoch + 1)  # 验证准确率随 epoch 变化曲线
                for i, param_group in enumerate(self.optimizer.param_groups):  # 遍历优化器中的参数组
                    self._writer.add_scalar(f'LR/group_{i}', param_group['lr'], epoch + 1)  # 记录各组学习率

            # ---- 早停判断与最优模型保存 ----
            metric = self._get_val_metric(val_loss, val_acc)  # 根据早停模式获取用于比较的度量值（loss 或 acc）
            if self.early_stopping:  # 如果启用了早停机制
                if self._is_improvement(metric):  # 当前度量值优于历史最优 → 模型有提升
                    self.best_metric = metric  # 更新历史最优度量值
                    self.early_stop_counter = 0  # 重置早停计数器
                    self.best_epoch = epoch + 1  # 记录当前 epoch 为最优 epoch
                    torch.save(self.model.state_dict(), self.save_path)  # 将模型权重保存到文件（仅保存参数，不含结构）
                    print(f"[Info] Model improved at epoch {epoch + 1}, saving to {self.save_path}")
                else:  # 当前度量未提升 → 模型可能过拟合或已收敛
                    self.early_stop_counter += 1  # 早停计数器 +1
                    print(f"[Info] Early stop counter: {self.early_stop_counter}/{self.patience}")
                    if self.early_stop_counter >= self.patience:  # 连续 patience 轮没有提升
                        print(f"Early stopping triggered at epoch {epoch + 1}. "
                              f"Best epoch: {self.best_epoch}, Best metric: {self.best_metric:.4f}")
                        if os.path.isfile(self.save_path):  # 如果之前保存过最优权重
                            # 加载最优模型权重以恢复到最佳泛化状态
                            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
                        if self.use_tensorboard and self._writer is not None:
                            self._writer.close()  # 关闭 TensorBoard 写入器
                        return  # 提前结束训练

        # 所有 epoch 完成且未触发早停: 加载训练过程中保存的最优模型（兜底）
        if self.early_stopping and self.best_metric is not None:
            print(f"Training finished. Loading best model from {self.save_path}")
            if os.path.isfile(self.save_path):  # 验证权重文件存在
                self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))  # 恢复最优权重
        if self.use_tensorboard and self._writer is not None:
            self._writer.close()  # 关闭 TensorBoard 写入器

    def plot(self, acc=True):
        """
        可视化训练过程中的损失与准确率曲线

        参数:
            acc: True = 绘制损失+准确率双图（分类任务），False = 仅绘制损失曲线（回归任务）
        """
        epochs_range = range(1, len(self.train_losses) + 1)  # 横轴: epoch 编号（从 1 开始到训练结束）

        if acc:  # 分类任务: 绘制损失和准确率两张子图（并排）
            plt.figure(figsize=(14, 5))  # 创建宽 14、高 5 英寸的画布

            # ---- 子图 1: 训练/验证损失曲线 ----
            plt.subplot(1, 2, 1)  # 1 行 2 列的第 1 个位置
            plt.plot(epochs_range, self.train_losses, label='Train Loss')  # 绘制训练损失折线（蓝色）
            plt.plot(epochs_range, self.val_losses, label='Validation Loss')  # 绘制验证损失折线（橙色）
            plt.xlabel('Epoch')  # 横轴标签: 训练轮数
            plt.ylabel('Loss')  # 纵轴标签: 损失值
            plt.title('Training and Validation Loss')  # 子图标题
            plt.legend()  # 显示图例（Train Loss / Validation Loss）
            plt.grid(True)  # 显示网格线，便于读取数值

            # ---- 子图 2: 训练/验证准确率曲线 ----
            plt.subplot(1, 2, 2)  # 1 行 2 列的第 2 个位置
            plt.plot(epochs_range, self.train_accuracies, label='Train Accuracy')  # 训练准确率折线
            plt.plot(epochs_range, self.val_accuracies, label='Validation Accuracy')  # 验证准确率折线
            plt.xlabel('Epoch')  # 横轴标签
            plt.ylabel('Accuracy (%)')  # 纵轴标签（百分比）
            plt.title('Training and Validation Accuracy')  # 子图标题
            plt.legend()  # 显示图例
            plt.grid(True)  # 显示网格线

            plt.tight_layout()  # 自动调整子图间距，防止标题和内容重叠
            plt.show()  # 显示图像窗口

        else:  # 回归任务: 只绘制损失曲线（无准确率）
            plt.figure(figsize=(7, 5))  # 创建 7×5 英寸的画布（单图，不用太宽）
            plt.plot(epochs_range, self.train_losses, label='Train Loss')  # 训练损失曲线
            plt.plot(epochs_range, self.val_losses, label='Validation Loss')  # 验证损失曲线
            plt.xlabel('Epoch')  # 横轴标签
            plt.ylabel('Loss')  # 纵轴标签
            plt.title('Training and Validation Loss')  # 标题
            plt.legend()  # 显示图例
            plt.grid(True)  # 显示网格线
            plt.tight_layout()  # 自动调整间距
            plt.show()  # 显示图像


# ============================================================
# 8. 模型实例化与前向传播验证
# ============================================================

# ---- 实例化 VGG11 模型 ----
model = VGG11(num_classes=10)  # 创建 VGG11 模型实例，10 分类
print("\n========== VGG11 模型结构 ==========")  # 分隔标题
print(model)  # 打印模型结构概览（包含各层的名称、参数和连接关系）

# ---- 前向传播验证 ----
# 用随机生成的虚拟数据（batch_size=4, 3 通道, 32×32）测试模型输入输出尺寸是否正确
dummy_input = torch.randn(4, 3, 32, 32)  # 模拟 4 张 CIFAR-10 大小的 RGB 图片
output = model(dummy_input)  # 前向传播: 输入随机张量，得到 logits
print(f"\n模型输入 shape: {dummy_input.shape}")  # torch.Size([4, 3, 32, 32])
print(f"模型输出 shape: {output.shape}")  # torch.Size([4, 10])，说明前向传播正确，输出 10 类 logits


# ============================================================
# 9. 模型参数量统计
# ============================================================

print("\n========== VGG11 参数统计 ==========")  # 分隔标题

def count_parameters(model):
    """统计模型的可训练参数量（即 requires_grad=True 的参数总数）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 遍历所有参数，筛选可训练的，累加元素个数

total_params = count_parameters(model)  # 调用统计函数，获取可训练参数总量
print(f"模型可训练参数总数: {total_params:,}")  # 约 9,287,434（千分位格式化）

# 打印各层参数量明细
print("\n各层参数量明细:")  # 明细标题
for name, param in model.named_parameters():  # 遍历所有命名参数（如 'features.0.weight', 'classifier.0.weight'）
    num_params = param.numel()  # .numel() 返回张量中的元素个数（Num Elements）
    print(f"  {name}: {num_params:,}")  # 打印参数名和参数量（千分位格式）


# ============================================================
# 10. 训练准备
# ============================================================

# ---- 判断可用设备 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU CUDA，否则回退到 CPU
print(f"\n使用设备: {device}")  # 打印当前训练设备

# ---- 训练超参数 ----
epochs = 30  # 第一阶段训练轮数: VGG11 参数量约 9.3M，30 轮足够收敛
lr = 0.001  # 学习率: Adam 的推荐默认值 0.001

# ---- 损失函数 ----
# CrossEntropyLoss: 内部自动完成 softmax + 负对数似然 NLLLoss
# 输入: 模型输出的原始 logits（不需要提前 softmax）
# 标签: 整数索引（0~9），不需要 one-hot 编码
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失，默认返回 batch 的平均损失

# ---- 优化器 ----
# Adam (Adaptive Moment Estimation): 结合了 Momentum 和 RMSProp 的优点
# 自适应调整每个参数的学习率，收敛快且调参简单
optimizer = optim.Adam(model.parameters(), lr=lr)  # 将模型所有 requires_grad=True 的参数传给优化器

# ---- 将模型移至设备 ----
model = model.to(device)  # 将模型的所有参数和 buffer 迁移到目标设备


# ============================================================
# 11. 开始训练（第一阶段：从零开始训练 VGG11）
# ============================================================

print(f"\n========== 开始 VGG11 第一阶段训练 (epochs={epochs}) ==========")

# 创建 Trainer 实例，封装完整训练流程
trainer = Trainer(
    model=model,  # 待训练的 VGG11 模型
    trainloader=train_loader,  # 训练集 DataLoader（含数据增强）
    valloader=val_loader,  # 验证集 DataLoader（仅归一化）
    criterion=criterion,  # 损失函数（交叉熵）
    optimizer=optimizer,  # 优化器（Adam, lr=0.001）
    device=device,  # 训练设备（GPU 或 CPU）
    epochs=epochs,  # 最大训练轮数 30
    early_stopping=True,  # 启用早停: 验证指标不再提升时自动停止
    patience=5,  # 早停容忍度: 连续 5 轮验证指标未提升则停止训练
    save_path="best_model.pth",  # 第一阶段最优权重保存路径
    early_stop_mode="loss",  # 早停依据: 以验证集损失为监控指标（越小越好）
    maximize_acc=True,  # acc 模式下准确率越大越好（此处用 loss 模式，该参数不生效）
    use_tensorboard=False  # 不使用 TensorBoard（若需要可视化日志可设为 True + 改 log_dir）
)

trainer.train()  # 开始训练: 包含前向、反向、评估、早停、保存最优模型等完整流程
trainer.plot(acc=True)  # 绘制训练/验证损失和准确率曲线（双图并排）


# ============================================================
# 12. 第一阶段验证集评估
# ============================================================

# 使用训练好的最优模型（Trainer 内部已恢复到 best_epoch 的权重）在验证集上评估
stage1_val_loss, stage1_val_acc = trainer.evaluating(val_loader)  # 调用 evaluating 获得 (平均损失, 准确率%)
print(f"\n========== VGG11 第一阶段验证集评估结果 ==========")  # 分隔标题
print(f"VGG11 第一阶段 - Val Loss: {stage1_val_loss:.4f}, Val Accuracy: {stage1_val_acc:.2f}%")  # 打印验证结果


# ============================================================
# 13. 模型精调（Fine-tuning）—— 加载最优权重并重置分类层
# ============================================================
# 精调策略: 在已训练模型的基础上继续训练
#  1. 加载第一阶段保存的最优模型权重（保留已学到的特征提取能力）
#  2. 重置最后一层分类器（为当前任务重新学习决策边界）
#  3. 分组学习率: 特征提取层用小学习率微调，分类层用大学习率重新学习

# ---- 加载第一阶段保存的最优模型权重 ----
best_model_path = 'best_model.pth'  # 第一阶段最优模型权重的文件路径（与 train 阶段的 save_path 保持一致）
model.load_state_dict(torch.load(best_model_path, map_location=device))  # 将保存的权重加载到模型中（map_location 确保在正确设备上加载）

# ---- 重新初始化分类器的最后一层（为精调任务重置决策边界） ----
if hasattr(model, 'classifier'):  # 检查模型是否具有 'classifier' 属性（VGG11 的分类层名称）
    if isinstance(model.classifier, nn.Sequential):  # 如果 classifier 是 Sequential 容器
        if hasattr(model.classifier[-1], 'reset_parameters'):  # 如果最后一层（通常是 Linear）有 reset_parameters 方法
            model.classifier[-1].reset_parameters()  # 调用 PyTorch 内置方法重置最后一层的权重和偏置
            print('模型最后一层重新初始化成功')  # 打印成功提示
        else:  # 如果最后一层没有 reset_parameters 方法（兜底方案）
            for param in model.classifier[-1].parameters():  # 遍历最后一层的所有参数
                if param.dim() >= 2:  # 参数维度 >= 2 说明是权重矩阵（而非偏置向量）
                    nn.init.kaiming_normal_(param)  # 用 Kaiming 正态分布重新初始化权重
                else:  # 参数维度 < 2 说明是偏置向量
                    nn.init.zeros_(param)  # 将偏置初始化为零
    elif hasattr(model.classifier, 'reset_parameters'):  # 如果 classifier 本身就有 reset_parameters（非 Sequential 的情况）
        model.classifier.reset_parameters()  # 直接调用重置方法
elif hasattr(model, 'fc'):  # 如果模型的分类层叫 'fc'（全连接层的常见命名）
    model.fc.reset_parameters()  # 重置 fc 层参数
elif hasattr(model, 'head'):  # 如果模型的分类层叫 'head'（某些预训练模型的命名习惯）
    model.head.reset_parameters()  # 重置 head 层参数
elif hasattr(model, 'cls'):  # 如果模型的分类层叫 'cls'（精调任务中常见的缩写命名）
    model.cls.reset_parameters()  # 重置 cls 层参数

model = model.to(device)  # 确保模型在参数修改后仍然位于正确的设备上


# ============================================================
# 14. 分组学习率优化器（精调的核心策略之一）
# ============================================================
# 不同层使用不同的学习率:
#   - 特征提取层（features.*）: 已学习到通用特征，用较小学习率微调（lr=0.0001），防止破坏已有表示
#   - 分类层（classifier.*）: 新重置的决策层，用较大学习率重新学习（lr=0.0005）
optimizer = optim.Adam([
    {  # 第一组: 特征提取层 —— 使用较小学习率
        "params": [value for key, value in model.named_parameters() if "classifier" not in key],  # 筛选参数名不含 "classifier" 的参数（特征提取层）
        "lr": 0.0001  # 特征层学习率设为 0.0001（较小的 lr，防止破坏预训练好的特征表示）
    },
    {  # 第二组: 分类层 —— 使用较大学习率
        "params": [value for key, value in model.named_parameters() if "classifier" in key],  # 筛选参数名含 "classifier" 的参数（新初始化的分类层）
        "lr": 0.0005  # 分类层学习率设为 0.0005（较大的 lr，加速学习新的分类决策边界）
    },
])


# ============================================================
# 15. 精调训练（Fine-tuning）
# ============================================================

print("\n========== 开始 VGG11 精调训练 ==========")

epochs_finetune = 10  # 精调阶段训练轮数（通常比第一阶段少，因为模型已有较好的特征提取能力）
trainer_finetune = Trainer(
    model=model,  # 传入已加载最优权重并重置了分类层的模型
    trainloader=train_loader,  # 训练集 DataLoader
    valloader=val_loader,  # 验证集 DataLoader
    criterion=criterion,  # 损失函数（与第一阶段相同）
    optimizer=optimizer,  # 传入分组学习率的优化器
    device=device,  # 训练设备（GPU 或 CPU）
    epochs=epochs_finetune,  # 精调训练轮数 10
    early_stopping=True,  # 启用早停机制
    patience=5,  # 早停容忍度
    save_path="best_model.pth",  # 精调最优权重保存路径（与第一阶段相同，覆盖最优权重）
    early_stop_mode="loss",  # 早停依据: 以验证集损失为监控指标
    maximize_acc=True,  # acc 模式下准确率越大越好
    use_tensorboard=False  # 不使用 TensorBoard
)

trainer_finetune.train()  # 启动精调训练循环
trainer_finetune.plot(acc=True)  # 绘制精调过程的训练曲线


# ============================================================
# 16. 精调后验证集评估
# ============================================================

finetune_val_loss, finetune_val_acc = trainer_finetune.evaluating(val_loader)  # 在验证集上评估精调后模型的最终性能
print(f"\n========== VGG11 精调后验证集评估结果 ==========")  # 分隔标题
print(f"VGG11 精调后 - Val Loss: {finetune_val_loss:.4f}, Val Accuracy: {finetune_val_acc:.2f}%")  # 打印最终结果


# ============================================================
# 17. VGG11 模型总结与分析
# ============================================================

print("\n" + "=" * 65)  # 打印分隔线
print("========== VGG11 模型总结 ==========")  # 总结标题
print("=" * 65)  # 打印分隔线

# ---- 模型关键指标 ----
print(f"\n模型名称: VGG11")  # 模型名称
print(f"总参数量: {total_params:,}")  # 约 9,287,434
print(f"输入尺寸: (3, 32, 32)")  # CIFAR-10 标准输入尺寸
print(f"输出类别数: 10")  # CIFAR-10 的 10 个类别
print(f"第一阶段验证集准确率: {stage1_val_acc:.2f}%")  # 从零训练后的验证集准确率
print(f"精调后验证集准确率: {finetune_val_acc:.2f}%")  # 精调后的验证集准确率

# ---- 模型架构分析 ----
print(f"\n架构特点:")  # 架构特点标题
print(f"  1. 统一的 3×3 小卷积核:")  # 核心设计
print(f"     - 全部使用 3×3 卷积，两个 3×3 堆叠的感受野等同于一个 5×5，但参数量更少")  # 小卷积核优势
print(f"     - padding=1 使特征图尺寸在卷积后保持不变，降采样完全由池化层负责")  # padding 作用
print(f"  2. 通道数逐块翻倍:")  # 通道设计
print(f"     - 64 → 128 → 256 → 512 → 512，深层提取更抽象的语义特征")  # 通道变化
print(f"  3. 特征图尺寸逐块减半:")  # 尺寸变化
print(f"     - 32×32 → 16×16 → 8×8 → 4×4 → 2×2 → 1×1，逐步压缩空间信息")  # 池化降采样
print(f"  4. 全连接分类器 + Dropout:")  # 分类器设计
print(f"     - Dropout 以 50% 概率随机丢弃神经元，防止过拟合")  # Dropout 作用

# ---- 与其它模型的对比 ----
print(f"\n与其他模型对比:")  # 对比标题
print(f"  模型            总参数量      特点")
print(f"  ───────────────────────────────────────────────")
print(f"  VGG11           ~9.3M        纯串联，结构简单，标准对比模型")
print(f"  InceptionNet    ~318K        多分支并行，轻量高效")
print(f"  ResNet18        ~11.2M       残差连接，深层可训练")

# ---- 改进建议 ----
print(f"\n改进方向:")  # 改进标题
print(f"  - 在各卷积块后添加 BatchNorm 层以加速收敛、稳定训练")
print(f"  - 使用学习率调度器（如 CosineAnnealingLR）动态调整学习率")
print(f"  - 结合更强的数据增强策略（Cutout、Mixup、AutoAugment）")
print(f"  - 尝试更深的 VGG 变体（VGG13/16/19）或 ResNet 残差结构")
