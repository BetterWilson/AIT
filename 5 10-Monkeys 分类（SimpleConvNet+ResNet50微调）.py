"""
10-Monkeys 分类 —— SimpleConvNet + ResNet50 微调（卷积神经网络CNN1.md）
======================================================
本脚本实现了 10-Monkeys 数据集的 10 分类任务，包括：
1. 数据加载与预处理（自定义 ImageFolder 数据集类）
2. 数据可视化
3. Trainer 通用训练器类（含早停、TensorBoard、绘图）
4. SimpleConvNet 自定义 CNN 模型构建、训练与评估
5. ResNet50 预训练模型微调（只解冻 layer4.2.conv3 + fc）
6. 双模型对比总结
"""

import torch  # PyTorch 核心库，提供张量运算与自动求导
import torch.nn as nn  # 神经网络模块，提供 Conv2d、Linear、ReLU 等层
import torch.nn.functional as F  # 神经网络函数式 API，提供 relu、max_pool2d 等函数
import torch.optim as optim  # 优化器模块，提供 SGD、Adam 等
from torchvision import datasets, transforms, models  # datasets: 数据集加载；transforms: 数据预处理；models: 预训练模型
from torch.utils.data import DataLoader  # DataLoader: 批量加载器，支持 shuffle、多线程加载
import matplotlib.pyplot as plt  # 绘图库，用于数据可视化与训练曲线绘制
from matplotlib import rcParams  # matplotlib 配置字典，用于设置全局绘图参数
import os  # 操作系统接口，用于创建目录、判断文件是否存在等
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 写入器，用于记录训练日志

# 设置中文字体，防止 matplotlib 中文显示为方块
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号（避免负号显示为方块）


# ============================================================
# 1. 自定义 Dataset 类 —— 继承 ImageFolder
# ============================================================
# ImageFolder 要求数据按以下目录结构组织:
#   root/
#     class_0/  (如 n0/)
#       img1.jpg
#       img2.jpg
#       ...
#     class_1/  (如 n1/)
#       ...
# ImageFolder 自动将子文件夹名作为类别标签，按字母序编号 0, 1, 2, ...


class MonkeyImageFolderDataset(datasets.ImageFolder):
    """
    自定义数据集类，继承自 torchvision 的 ImageFolder

    ImageFolder 自动完成:
      1. 扫描 root 下的子文件夹，每个子文件夹对应一个类别
      2. 将文件夹名按字母序映射为 0~N-1 的整数标签
      3. 通过 self.classes 可获取类别名列表
      4. 通过 self.class_to_idx 可获取类别名→索引的映射字典

    继承此类便于后续扩展自定义行为（如自定义采样策略）
    """

    def __init__(self, root, transform=None):
        """
        初始化数据集

        参数:
            root:      数据集根目录路径，其下每个子文件夹代表一个类别
            transform: 数据预处理变换（Compose 对象），默认为 None
        """
        super().__init__(root=root, transform=transform)  # 调用父类 ImageFolder 的构造函数


# ============================================================
# 2. 数据预处理定义
# ============================================================

# transforms.Compose: 将多个 transform 操作组合在一起，按顺序依次执行
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # 将图片缩放到 128×128 像素（模型输入要求固定尺寸）
    transforms.ToTensor(),  # 将 PIL.Image (0-255) 转为 torch.Tensor (0.0-1.0)，并将 H×W×C → C×H×W
    # 注: 此处未使用 Normalize，训练时可根据需要取消下面的注释
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet 标准化参数
])

# ============================================================
# 3. 数据集路径与加载
# ============================================================

# 数据集目录结构:
#   data/archive/training/training/  ← 训练集根目录（内层 training 是实际类别文件夹所在）
#     n0/  n1/  n2/  ...  n9/       ← 10 个类别的子文件夹
#   data/archive/validation/validation/  ← 验证集根目录
#     n0/  n1/  n2/  ...  n9/

# 训练集根目录路径
train_dir = 'data/archive/training/training'  # 训练集路径，其下有 n0~n9 十个类别文件夹
# 验证集根目录路径
val_dir = 'data/archive/validation/validation'  # 验证集路径，其下有 n0~n9 十个类别文件夹

# 创建训练集 Dataset 实例
# ImageFolder 会自动扫描 train_dir 下的子文件夹并分配标签
train_dataset = MonkeyImageFolderDataset(root=train_dir, transform=data_transforms)  # 训练集
# 创建验证集 Dataset 实例
val_dataset = MonkeyImageFolderDataset(root=val_dir, transform=data_transforms)  # 验证集

# ============================================================
# 4. 创建 DataLoader
# ============================================================

batch_size = 32  # 批量大小: 每次送入模型的图片数量（根据显存大小可调整）

# 训练集 DataLoader: shuffle=True 打乱顺序，防止模型记忆样本顺序
train_loader = DataLoader(
    train_dataset,  # 训练集 Dataset
    batch_size=batch_size,  # 每批 32 张图片
    shuffle=True  # 每个 epoch 随机打乱样本顺序
)

# 验证集 DataLoader: shuffle=False 不打乱，保持评估的一致性
val_loader = DataLoader(
    val_dataset,  # 验证集 Dataset
    batch_size=batch_size,  # 每批 32 张图片
    shuffle=False  # 验证/测试时不需打乱
)

# ============================================================
# 5. 数据集基本信息
# ============================================================

# 打印各数据集样本数
print("训练集样本数：", len(train_dataset))  # 训练集总样本数
print("验证集样本数：", len(val_dataset))  # 验证集总样本数

# 查看单个样本的形状: (C, H, W) = (3, 128, 128)
print("单张图片 shape (C, H, W):", train_dataset[0][0].shape)  # torch.Size([3, 128, 128])
# 查看单个样本的标签: 0~9 的整数
print("第一张图片的标签编号:", train_dataset[0][1])  # 0~9

# 查看类别名称（由 ImageFolder 自动从文件夹名解析）
class_names = train_dataset.classes  # 获取所有类别名称列表，如 ['n0','n1',...,'n9']
print("类别名称:", class_names)  # 打印类别名称
num_classes = len(class_names)  # 类别总数: 10
print("类别总数:", num_classes)  # 10

# 查看一个 batch 的数据形状
for imgs, labels in train_loader:  # 取训练集第一个 batch
    print("一个 batch 的图片 shape:", imgs.shape)  # torch.Size([32, 3, 128, 128])
    print("一个 batch 的标签:", labels)  # tensor([...]) 32 个 0-9 的整数
    break  # 只取第一个 batch

# ============================================================
# 6. 数据可视化
# ============================================================

# 可视化训练集前 15 个样本
fig, axs = plt.subplots(3, 5, figsize=(15, 10))  # 创建 3×5 子图布局，画布尺寸 15×10 英寸
axs = axs.flatten()  # 将 2D 轴数组展平为 1D 列表，方便通过索引访问

for i in range(15):  # 遍历前 15 个样本
    img, label = train_dataset[i]  # 获取第 i 个样本: img 形状 (3, 128, 128), label 是 0-9 的整数
    # permute(1, 2, 0): 将 (C,H,W) 转换为 (H,W,C)，matplotlib 需要 H×W×C 格式
    img = img.permute(1, 2, 0).numpy()  # 转换维度顺序后转为 numpy 数组
    axs[i].imshow(img)  # 显示彩色图片（RGB 三通道）
    axs[i].set_title(class_names[label])  # 标题设为对应的类别名称
    axs[i].axis('off')  # 隐藏坐标轴，使图像更清晰

plt.tight_layout()  # 自动调整子图间距，避免标题和图片重叠
plt.savefig('可视化train_dataset前15个样本_monkeys.png')  # 保存为 PNG 图片
plt.show()  # 显示图像窗口


# ============================================================
# 7. Trainer 通用训练器类
# ============================================================
# 该类封装了完整的训练流水线: 训练循环 + 验证评估 + 早停 + 模型保存 + TensorBoard + 绘图
# 同时支持分类任务（带准确率）与回归任务（仅损失）


class Trainer:
    """
    通用训练器：封装训练循环、评估、早停、模型保存与可视化

    支持两种任务:
      - 分类: 使用 train() + evaluating()，记录损失与准确率
      - 回归: 使用 regression_train() + regression_evaluating()，仅记录损失
    """

    def __init__(
            self,
            model,  # 待训练的 PyTorch 模型实例
            trainloader,  # 训练集 DataLoader
            valloader,  # 验证集 DataLoader
            criterion,  # 损失函数（如 CrossEntropyLoss）
            optimizer,  # 优化器（如 Adam、SGD）
            device='cuda',  # 训练设备: 'cuda'（GPU）或 'cpu'
            epochs=10,  # 最大训练轮数，默认 10
            early_stopping=True,  # 是否启用早停机制
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
        self.device = device  # 保存训练设备
        self.epochs = epochs  # 保存最大训练轮数

        # 训练历史记录列表（用于绘图）
        self.train_losses = []  # 每轮训练集平均损失
        self.val_losses = []  # 每轮验证集平均损失
        self.train_accuracies = []  # 每轮训练集准确率（%）
        self.val_accuracies = []  # 每轮验证集准确率（%）

        # 早停相关配置
        self.early_stopping = early_stopping  # 是否启用早停
        self.patience = patience  # 早停容忍度
        self.save_path = save_path  # 最优模型保存路径
        self.early_stop_mode = early_stop_mode  # 早停监控模式: "loss" 或 "acc"
        self.maximize_acc = maximize_acc  # acc 模式下: True=越大越好

        # 早停运行状态变量
        self.best_metric = None  # 历史最优度量值（初始为 None）
        self.early_stop_counter = 0  # 连续未提升的轮数计数器
        self.best_epoch = 0  # 取得最优度量值时的 epoch 编号

        # TensorBoard 日志配置
        self.use_tensorboard = use_tensorboard  # 是否使用 TensorBoard
        self._writer = None  # TensorBoard SummaryWriter 句柄，初始为 None
        if self.use_tensorboard:  # 如果启用了 TensorBoard
            if not os.path.exists(log_dir):  # 检查日志目录是否存在
                os.makedirs(log_dir)  # 不存在则递归创建
            self._writer = SummaryWriter(log_dir)  # 创建 SummaryWriter 实例

    def evaluating(self, dataloader):
        """
        分类任务评估函数

        参数:
            dataloader: 待评估的数据加载器（验证集或测试集）
        返回:
            avg_loss: 平均损失
            acc:      准确率（%）
        """
        self.model.eval()  # 切换到评估模式: 关闭 Dropout、冻结 BatchNorm 统计量
        correct = 0  # 累计预测正确的样本数
        total = 0  # 累计总样本数
        running_loss = 0.0  # 累计总损失

        with torch.no_grad():  # 禁用梯度计算，大幅节省显存和计算量
            for images, labels in dataloader:  # 逐 batch 遍历
                images = images.to(self.device)  # 将图片数据移至 GPU/CPU
                labels = labels.to(self.device)  # 将标签数据移至 GPU/CPU
                outputs = self.model(images)  # 前向传播得到 logits
                loss = self.criterion(outputs, labels)  # 计算当前 batch 的损失
                running_loss += loss.item()  # 累加损失（.item() 将标量张量转 Python float）
                # torch.argmax(outputs, dim=1): 沿类别维度取最大值的索引作为预测类别
                predicted = torch.argmax(outputs, dim=1)  # 获取每个样本的预测类别 (0~9)
                total += labels.size(0)  # 累加当前 batch 的样本数
                correct += (predicted == labels).sum().item()  # 累加预测正确的样本数

        acc = 100 * correct / total if total > 0 else 0  # 准确率转为百分比（%）
        avg_loss = running_loss / len(dataloader)  # 平均损失 = 总损失 / batch 数
        return avg_loss, acc  # 返回 (平均损失, 准确率%)

    def regression_evaluating(self, dataloader):
        """
        回归任务评估函数: 只返回平均损失

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
                output = self.model(data)  # 前向传播
                loss = self.criterion(output, target)  # 计算损失
                running_loss += loss.item()  # 累加损失
        avg_loss = running_loss / len(dataloader)  # 计算平均损失
        return avg_loss  # 返回平均损失

    def regression_train(self):
        """
        回归任务训练循环: 仅记录损失，不计算准确率

        与 train() 的区别: 评估时不计算准确率，只使用验证损失作为早停指标
        """
        self.model.to(self.device)  # 将模型移至目标设备
        for epoch in range(self.epochs):  # 逐轮训练
            self.model.train()  # 切换到训练模式: 启用 Dropout 等
            running_loss = 0.0  # 本轮损失累加器清零

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):  # 遍历训练集
                inputs = inputs.to(self.device)  # 输入移至设备
                targets = targets.to(self.device)  # 目标移至设备
                self.optimizer.zero_grad()  # 清空上一轮梯度
                outputs = self.model(inputs)  # 前向传播
                loss = self.criterion(outputs, targets)  # 计算损失
                loss.backward()  # 反向传播求梯度
                self.optimizer.step()  # 优化器更新参数
                running_loss += loss.item()  # 累加损失

                if (batch_idx + 1) % 100 == 0:  # 每 100 个 batch 打印一次当前损失
                    print(f"[Regression] Epoch [{epoch + 1}/{self.epochs}], "
                          f"Step [{batch_idx + 1}/{len(self.trainloader)}], Loss: {loss.item():.4f}")

            avg_train_loss = running_loss / len(self.trainloader)  # 本轮平均训练损失
            train_loss = self.regression_evaluating(self.trainloader)  # 评估训练集损失
            val_loss = self.regression_evaluating(self.valloader)  # 评估验证集损失
            self.train_losses.append(train_loss)  # 记录训练损失
            self.val_losses.append(val_loss)  # 记录验证损失
            print(f"[Regression] Epoch [{epoch + 1}/{self.epochs}], "
                  f"Loss: {avg_train_loss:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # ---- TensorBoard 日志记录 ----
            if self.use_tensorboard and self._writer is not None:
                self._writer.add_scalar('Train/Loss', train_loss, epoch + 1)  # 记录训练损失曲线
                self._writer.add_scalar('Val/Loss', val_loss, epoch + 1)  # 记录验证损失曲线
                for i, param_group in enumerate(self.optimizer.param_groups):  # 遍历所有参数组
                    self._writer.add_scalar(f'LR/group_{i}', param_group['lr'], epoch + 1)  # 记录学习率

            # ---- 早停与模型保存 ----
            metric = val_loss  # 回归任务只用验证损失作为评估指标
            if self.early_stopping:  # 如果开启早停
                if self.best_metric is None or metric < self.best_metric:  # 首次记录或损失下降
                    self.best_metric = metric  # 更新最优损失值
                    self.early_stop_counter = 0  # 重置早停计数器
                    self.best_epoch = epoch + 1  # 记录最优 epoch
                    torch.save(self.model.state_dict(), self.save_path)  # 保存最优模型权重
                    print(f"[Info][Regression] Model improved at epoch {epoch + 1}, saving to {self.save_path}")
                else:  # 损失未下降
                    self.early_stop_counter += 1  # 早停计数器 +1
                    print(f"[Info][Regression] Early stop counter: {self.early_stop_counter}/{self.patience}")
                    if self.early_stop_counter >= self.patience:  # 超过容忍度
                        print(f"[Regression] Early stopping triggered at epoch {epoch + 1}. "
                              f"Best epoch: {self.best_epoch}, Best Loss: {self.best_metric:.4f}")
                        if os.path.isfile(self.save_path):  # 如果最优权重文件存在
                            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))  # 恢复最优权重
                        if self.use_tensorboard and self._writer is not None:
                            self._writer.close()  # 关闭 TensorBoard 写入器
                        return  # 结束训练

        # 全部 epoch 跑完且未触发早停: 加载训练过程中保存的最优权重
        if self.early_stopping and self.best_metric is not None:
            print(f"[Regression] Training finished. Loading best model from {self.save_path}")
            if os.path.isfile(self.save_path):  # 检查权重文件是否存在
                self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))  # 恢复最优权重
        if self.use_tensorboard and self._writer is not None:
            self._writer.close()  # 关闭 TensorBoard 写入器

    def _is_improvement(self, metric):
        """
        根据早停模式判断当前度量值是否优于历史最优

        参数:
            metric: 当前 epoch 的度量值（损失或准确率）
        返回:
            True=有提升, False=未提升
        """
        if self.best_metric is None:  # 尚无历史最优记录（第一个 epoch）
            return True  # 视为提升
        if self.early_stop_mode == "loss":  # 损失模式: 越小越好
            return metric < self.best_metric  # 当前损失 < 历史最优损失 → 提升
        elif self.early_stop_mode == "acc":  # 准确率模式
            if self.maximize_acc:  # 准确率越大越好
                return metric > self.best_metric  # 当前准确率 > 历史最优准确率 → 提升
            else:  # 准确率越小越好（不常见）
                return metric < self.best_metric
        else:
            raise ValueError("Unknown early_stop_mode: {}".format(self.early_stop_mode))  # 未知模式报错

    def _get_val_metric(self, val_loss, val_acc):
        """
        根据早停模式返回用于比较的度量值

        参数:
            val_loss: 当前验证集平均损失
            val_acc:  当前验证集准确率（%）
        返回:
            用于早停判断的度量值（损失或准确率）
        """
        if self.early_stop_mode == "loss":  # 以损失为早停依据
            return val_loss
        elif self.early_stop_mode == "acc":  # 以准确率为早停依据
            return val_acc
        else:
            raise ValueError("Unknown early_stop_mode: {}".format(self.early_stop_mode))

    def train(self):
        """
        分类任务训练主循环

        每个 epoch 的流程:
          1. 遍历训练集 batch，前向 → 损失 → 反向 → 更新
          2. 在训练集和验证集上评估损失与准确率
          3. 记录 TensorBoard 日志
          4. 早停判断与最优模型保存
        """
        self.model.to(self.device)  # 将模型参数迁移到目标设备 (GPU/CPU)

        for epoch in range(self.epochs):  # 逐轮训练，共 epochs 轮
            self.model.train()  # 切换到训练模式: 启用 Dropout、BatchNorm 更新等
            running_loss = 0.0  # 当前 epoch 的损失累加器（用于显示）

            for batch_idx, (images, labels) in enumerate(self.trainloader):  # 遍历训练集每个 batch
                images = images.to(self.device)  # 图片数据移到设备
                labels = labels.to(self.device)  # 标签数据移到设备

                # ---- 核心训练五步 ----
                self.optimizer.zero_grad()  # 1. 清空上一轮的梯度（PyTorch 默认累加梯度）
                outputs = self.model(images)  # 2. 前向传播，得到预测 logits
                loss = self.criterion(outputs, labels)  # 3. 计算损失
                loss.backward()  # 4. 反向传播，计算梯度
                self.optimizer.step()  # 5. 更新参数: θ = θ - lr × ∇loss

                running_loss += loss.item()  # 累加损失值（.item() 提取 Python float）

                if (batch_idx + 1) % 100 == 0:  # 每 100 个 batch 打印一次进度
                    print(f'Epoch [{epoch + 1}/{self.epochs}], '
                          f'Step [{batch_idx + 1}/{len(self.trainloader)}], Loss: {loss.item():.4f}')

            # ---- epoch 结束后的评估 ----
            avg_train_loss = running_loss / len(self.trainloader)  # 本轮平均训练损失（batch 级）
            train_loss, train_acc = self.evaluating(self.trainloader)  # 训练集评估: 获得平均损失与准确率
            val_loss, val_acc = self.evaluating(self.valloader)  # 验证集评估: 获得平均损失与准确率

            # 记录历史数据（用于绘图）
            self.train_losses.append(train_loss)  # 保存训练损失
            self.val_losses.append(val_loss)  # 保存验证损失
            self.train_accuracies.append(train_acc)  # 保存训练准确率
            self.val_accuracies.append(val_acc)  # 保存验证准确率

            print(f'Epoch [{epoch + 1}/{self.epochs}], '
                  f'Loss: {avg_train_loss:.4f}, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

            # ---- TensorBoard 日志记录 ----
            if self.use_tensorboard and self._writer is not None:
                self._writer.add_scalar('Train/Loss', train_loss, epoch + 1)  # 训练损失曲线
                self._writer.add_scalar('Train/Accuracy', train_acc, epoch + 1)  # 训练准确率曲线
                self._writer.add_scalar('Val/Loss', val_loss, epoch + 1)  # 验证损失曲线
                self._writer.add_scalar('Val/Accuracy', val_acc, epoch + 1)  # 验证准确率曲线
                for i, param_group in enumerate(self.optimizer.param_groups):  # 遍历优化器中的参数组
                    self._writer.add_scalar(f'LR/group_{i}', param_group['lr'], epoch + 1)  # 记录学习率

            # ---- 早停判断与最优模型保存 ----
            metric = self._get_val_metric(val_loss, val_acc)  # 根据模式获取用于比较的度量值
            if self.early_stopping:  # 如果启用了早停机制
                if self._is_improvement(metric):  # 当前度量优于历史最优 → 提升
                    self.best_metric = metric  # 更新历史最优度量值
                    self.early_stop_counter = 0  # 重置早停计数器
                    self.best_epoch = epoch + 1  # 记录最优 epoch 编号
                    torch.save(self.model.state_dict(), self.save_path)  # 保存最优模型权重到文件
                    print(f"[Info] Model improved at epoch {epoch + 1}, saving to {self.save_path}")
                else:  # 未提升
                    self.early_stop_counter += 1  # 早停计数器 +1
                    print(f"[Info] Early stop counter: {self.early_stop_counter}/{self.patience}")
                    if self.early_stop_counter >= self.patience:  # 连续 patience 轮未提升
                        print(f"Early stopping triggered at epoch {epoch + 1}. "
                              f"Best epoch: {self.best_epoch}, Best metric: {self.best_metric:.4f}")
                        if os.path.isfile(self.save_path):  # 如果之前保存过最优权重
                            # 加载最优模型权重以恢复到最佳状态
                            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
                        if self.use_tensorboard and self._writer is not None:
                            self._writer.close()  # 关闭 TensorBoard 写入器
                        return  # 结束训练

        # 所有 epoch 完成且未触发早停: 加载训练过程中保存的最优模型
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
            acc: True=绘制损失+准确率双图（分类），False=仅绘制损失曲线（回归）
        """
        epochs_range = range(1, len(self.train_losses) + 1)  # 横轴: epoch 编号（从 1 开始）

        if acc:  # 分类任务: 绘制损失和准确率两张子图
            plt.figure(figsize=(14, 5))  # 创建宽 14、高 5 英寸的画布

            # 子图 1: 训练/验证损失曲线
            plt.subplot(1, 2, 1)  # 1 行 2 列的第 1 个
            plt.plot(epochs_range, self.train_losses, label='Train Loss')  # 训练损失折线
            plt.plot(epochs_range, self.val_losses, label='Validation Loss')  # 验证损失折线
            plt.xlabel('Epoch')  # 横轴标签
            plt.ylabel('Loss')  # 纵轴标签
            plt.title('Training and Validation Loss')  # 子图标题
            plt.legend()  # 显示图例
            plt.grid(True)  # 显示网格线

            # 子图 2: 训练/验证准确率曲线
            plt.subplot(1, 2, 2)  # 1 行 2 列的第 2 个
            plt.plot(epochs_range, self.train_accuracies, label='Train Accuracy')  # 训练准确率折线
            plt.plot(epochs_range, self.val_accuracies, label='Validation Accuracy')  # 验证准确率折线
            plt.xlabel('Epoch')  # 横轴标签
            plt.ylabel('Accuracy (%)')  # 纵轴标签（百分比）
            plt.title('Training and Validation Accuracy')  # 子图标题
            plt.legend()  # 显示图例
            plt.grid(True)  # 显示网格线

            plt.tight_layout()  # 自动调整子图间距，防止重叠
            plt.show()  # 显示图像

        else:  # 回归任务: 只绘制损失曲线
            plt.figure(figsize=(7, 5))  # 创建 7×5 英寸的画布
            plt.plot(epochs_range, self.train_losses, label='Train Loss')  # 训练损失
            plt.plot(epochs_range, self.val_losses, label='Validation Loss')  # 验证损失
            plt.xlabel('Epoch')  # 横轴标签
            plt.ylabel('Loss')  # 纵轴标签
            plt.title('Training and Validation Loss')  # 标题
            plt.legend()  # 显示图例
            plt.grid(True)  # 显示网格线
            plt.tight_layout()  # 自动调整间距
            plt.show()  # 显示图像


# ============================================================
# 8. 模型一: SimpleConvNet —— 自定义小型卷积神经网络
# ============================================================
# 结构: 三组 (Conv→ReLU→Conv→ReLU→MaxPool) + Flatten + FC→ReLU→FC
# 输入: (batch, 3, 128, 128) 彩色图片
# 输出: (batch, 10) 类别 logits


class SimpleConvNet(nn.Module):
    """
    简单卷积神经网络 —— 用于 10-Monkeys 分类（彩色 128×128 输入）

    结构概览:
      输入 (3, 128, 128) 彩色图
      → 第一组: Conv(3→32,3×3)→ReLU→Conv(32→32,3×3)→ReLU→MaxPool (128→64)
      → 第二组: Conv(32→64,3×3)→ReLU→Conv(64→64,3×3)→ReLU→MaxPool (64→32)
      → 第三组: Conv(64→128,3×3)→ReLU→Conv(128→128,3×3)→ReLU→MaxPool (32→16)
      → 展平 → FC(128×16×16, 128)→ReLU → FC(128, 10)

    参数量计算:
      conv1_1: 3×32×3×3 + 32 = 864 + 32 = 896
      conv1_2: 32×32×3×3 + 32 = 9,216 + 32 = 9,248
      conv2_1: 32×64×3×3 + 64 = 18,432 + 64 = 18,496
      conv2_2: 64×64×3×3 + 64 = 36,864 + 64 = 36,928
      conv3_1: 64×128×3×3 + 128 = 73,728 + 128 = 73,856
      conv3_2: 128×128×3×3 + 128 = 147,456 + 128 = 147,584
      fc1:     128×16×16×128 + 128 = 4,194,304 + 128 = 4,194,432
      fc2:     128×10 + 10 = 1,280 + 10 = 1,290
      总计: 约 4,482,730
    """

    def __init__(self, num_classes=10):
        """
        初始化 SimpleConvNet

        参数:
            num_classes: 输出类别数，默认 10（10 种猴子）
        """
        super(SimpleConvNet, self).__init__()  # 调用父类 nn.Module 的构造函数

        # ====== 第一组卷积 + 池化 (128→64) ======
        # nn.Conv2d(in_channels, out_channels, kernel_size, padding): 二维卷积层
        # padding=1: 在输入四周各补一圈 0，使输出尺寸与输入相同
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 第一层: (3,128,128)→(32,128,128)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # 第二层: (32,128,128)→(32,128,128)
        # nn.MaxPool2d(kernel_size=2, stride=2): 2×2 最大池化，尺寸减半
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (32,128,128)→(32,64,64)

        # ====== 第二组卷积 + 池化 (64→32) ======
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (32,64,64)→(64,64,64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # (64,64,64)→(64,64,64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (64,64,64)→(64,32,32)

        # ====== 第三组卷积 + 池化 (32→16) ======
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (64,32,32)→(128,32,32)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # (128,32,32)→(128,32,32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (128,32,32)→(128,16,16)

        # ====== 全连接分类器 ======
        # 三次池化后特征图尺寸: 128 通道 × 16 × 16 = 32768 维
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # 全连接: 32768 → 128
        self.fc2 = nn.Linear(128, num_classes)  # 输出层: 128 → 10（logits，不加 Softmax）

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状 (batch_size, 3, 128, 128)
        返回:
            logits: 形状 (batch_size, 10)
        """
        # ====== 第一组: Conv→ReLU→Conv→ReLU→MaxPool ======
        x = F.relu(self.conv1_1(x))  # 卷积 + ReLU: (batch,3,128,128)→(batch,32,128,128)
        x = F.relu(self.conv1_2(x))  # 卷积 + ReLU: (batch,32,128,128)→(batch,32,128,128)
        x = self.pool1(x)  # 最大池化降采样: (batch,32,128,128)→(batch,32,64,64)

        # ====== 第二组: Conv→ReLU→Conv→ReLU→MaxPool ======
        x = F.relu(self.conv2_1(x))  # 卷积 + ReLU: (batch,32,64,64)→(batch,64,64,64)
        x = F.relu(self.conv2_2(x))  # 卷积 + ReLU: (batch,64,64,64)→(batch,64,64,64)
        x = self.pool2(x)  # 最大池化降采样: (batch,64,64,64)→(batch,64,32,32)

        # ====== 第三组: Conv→ReLU→Conv→ReLU→MaxPool ======
        x = F.relu(self.conv3_1(x))  # 卷积 + ReLU: (batch,64,32,32)→(batch,128,32,32)
        x = F.relu(self.conv3_2(x))  # 卷积 + ReLU: (batch,128,32,32)→(batch,128,32,32)
        x = self.pool3(x)  # 最大池化降采样: (batch,128,32,32)→(batch,128,16,16)

        # ====== 展平 + 全连接 ======
        x = torch.flatten(x, 1)  # 展平: (batch,128,16,16)→(batch,32768)
        x = F.relu(self.fc1(x))  # 全连接 + ReLU: (batch,32768)→(batch,128)
        x = self.fc2(x)  # 输出层: (batch,128)→(batch,10) logits
        return x  # 返回 10 个类别的原始分数


# 实例化 SimpleConvNet 模型
model_simple = SimpleConvNet(num_classes=num_classes)  # 创建简单 CNN 模型实例
print(model_simple)  # 打印模型结构概览

# ============================================================
# 9. SimpleConvNet 模型参数统计
# ============================================================

print("\n========== SimpleConvNet 参数统计 ==========")  # 打印分隔标题

# 统计模型的总参数量和可训练参数量
total_params_simple = sum(p.numel() for p in model_simple.parameters())  # 总参数量（所有参数的元素总和）
trainable_params_simple = sum(p.numel() for p in model_simple.parameters() if p.requires_grad)  # 可训练参数量
print(f"SimpleConvNet 模型总参数量: {total_params_simple:,}")  # 约 4,482,730
print(f"SimpleConvNet 可训练参数量: {trainable_params_simple:,}")  # 应等于总参数量（无冻结层）

# 打印各层参数量明细
print("\n各层参数量明细:")  # 标题
for name, param in model_simple.named_parameters():  # 遍历所有命名参数
    num_params = param.numel()  # .numel() 返回张量中元素个数
    print(f"  {name}: {num_params:,}")  # 打印参数名和参数量（千分位格式）

# ============================================================
# 10. SimpleConvNet 前向传播验证
# ============================================================

# 用随机生成的单张虚拟图片测试前向传播
x_test = torch.randn(1, 3, 128, 128)  # batch_size=1, 3 通道, 128×128 的随机张量
output_simple = model_simple(x_test)  # 前向传播
print(f"\nSimpleConvNet 正向传播输出 shape: {output_simple.shape}")  # torch.Size([1, 10])

# ============================================================
# 11. SimpleConvNet 训练准备
# ============================================================

# 判断可用设备: 优先使用 GPU (CUDA)，不可用则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测设备
print(f"\n使用设备: {device}")  # 打印当前训练设备

# 将模型移至设备
model_simple = model_simple.to(device)  # 模型参数迁移到 GPU/CPU

# 损失函数: 交叉熵损失 CrossEntropyLoss
# 内部自动完成 softmax + 负对数似然，输入应为原始 logits
criterion_simple = nn.CrossEntropyLoss()  # 默认返回 batch 的平均损失

# 优化器: Adam (Adaptive Moment Estimation)
# Adam 结合了 Momentum 和 RMSProp 的优点，自适应调整学习率
# lr=0.001: 学习率，Adam 的推荐默认值
optimizer_simple = optim.Adam(model_simple.parameters(), lr=0.001)  # 创建 Adam 优化器

# 训练超参数
epochs_simple = 10  # 训练轮数: SimpleConvNet 参数较多，10 轮为基础训练量

# ============================================================
# 12. SimpleConvNet 训练
# ============================================================

print(f"\n========== 开始 SimpleConvNet 训练 (epochs={epochs_simple}) ==========")

# 使用 Trainer 封装训练流程: 早停 + 保存最优模型 + 绘图
trainer_simple = Trainer(
    model=model_simple,  # 待训练的 SimpleConvNet 模型
    trainloader=train_loader,  # 训练集 DataLoader
    valloader=val_loader,  # 验证集 DataLoader
    criterion=criterion_simple,  # 损失函数（交叉熵）
    optimizer=optimizer_simple,  # 优化器（Adam）
    device=device,  # 训练设备
    epochs=epochs_simple,  # 训练轮数
    early_stopping=True,  # 启用早停: 验证集准确率不再提升时自动停止
    patience=5,  # 早停容忍度: 连续 5 轮准确率未提升则停止
    save_path="best_model_simple_cnn.pth",  # 最优权重保存路径
    early_stop_mode="acc",  # 早停依据: 以验证集准确率为监控指标
    maximize_acc=True,  # 准确率越大越好
    use_tensorboard=False  # 不使用 TensorBoard（若需要可视化可设为 True）
)

trainer_simple.train()  # 开始训练
trainer_simple.plot(acc=True)  # 绘制训练/验证损失和准确率曲线

# ============================================================
# 13. SimpleConvNet 验证集评估
# ============================================================

# 使用最终模型在验证集上评估（注: 此数据无独立测试集，用验证集代替）
test_loss_simple, test_acc_simple = trainer_simple.evaluating(val_loader)  # 在验证集上评估
print(f"\n========== SimpleConvNet 验证集评估结果 ==========")  # 打印标题
print(f"SimpleConvNet - Val Loss: {test_loss_simple:.4f}, Val Accuracy: {test_acc_simple:.2f}%")  # 打印结果

print()  # 空行分隔，使输出更清晰


# ============================================================
# 14. 模型二: CustomResNet50 —— ResNet50 预训练模型微调
# ============================================================
# ResNet50: 50 层残差网络，在 ImageNet (1000 类) 上预训练
# 微调策略:
#   1. 加载预训练权重（IMAGENET1K_V2）
#   2. 冻结所有卷积层参数
#   3. 只解冻 layer4.2.conv3（最后一个 bottleneck 的第三卷积层）+ fc 全连接层
#   4. 替换 fc 为 10 分类输出
#
# 为什么只解冻最后一层的部分参数？
#   - 预训练的低层特征（边缘/纹理等）具有良好的泛化能力，无需重新学习
#   - 高层语义特征需要微调以适配新任务（猴子分类 vs ImageNet 通用分类）
#   - 参数少 → 训练快、过拟合风险低


class CustomResNet50(nn.Module):
    """
    自定义 ResNet50 分类器 —— ImageNet 预训练 + 部分层微调

    参数量:
      总参数:   约 23,528,522（含冻结层）
      可训练:   约 1,069,066（layer4.2.conv3 + fc）
      冻结:     约 22,459,456

    解冻策略:
      - layer4.2.conv3: ResNet50 最后一个 bottleneck block 的 conv3（1×1 卷积）
      - fc (全连接层): 从 1000 类替换为 10 类，必须解冻
      - 其余所有层: 冻结（参数不更新）
    """

    def __init__(self, num_classes=10, weights=models.ResNet50_Weights.IMAGENET1K_V2):
        """
        初始化 CustomResNet50

        参数:
            num_classes: 输出类别数，默认 10
            weights:     预训练权重版本
                          IMAGENET1K_V1 = 老版 ImageNet 训练
                          IMAGENET1K_V2 = 新版 ImageNet 训练（更强的数据增强，准确率更高）
        """
        super().__init__()  # 调用父类 nn.Module 的构造函数

        # 加载预训练的 ResNet50 模型
        self.resnet = models.resnet50(weights=weights)  # 下载（首次）并加载 ImageNet 预训练权重

        # 修改最后的全连接层 (fc) 为 10 分类
        in_features = self.resnet.fc.in_features  # 获取原始 fc 的输入特征维度: 2048
        self.resnet.fc = nn.Linear(in_features, num_classes)  # (2048 → 10)，替换原始 (2048 → 1000)

        # ---- 冻结所有参数 ----
        for param in self.resnet.parameters():  # 遍历 ResNet50 所有参数
            param.requires_grad = False  # 关闭梯度计算 → 参数将被冻结，不参与更新

        # ---- 只解冻 layer4.2.conv3 ----
        # layer4 是最后一个残差阶段，包含 3 个 Bottleneck block (layer4.0, layer4.1, layer4.2)
        # 每个 Bottleneck 有 conv1(1×1), conv2(3×3), conv3(1×1) 三层卷积
        # 这里只解冻最后一个 block 的 conv3（1×1 卷积，用于通道变换）
        for name, module in self.resnet.named_modules():  # 遍历所有命名子模块
            # name 示例: 'layer4.2.conv3', 'layer4.0.conv1', 'fc' 等
            if name == "layer4.2.conv3":  # 找到目标层: layer4 的第 3 个 bottleneck 的 conv3
                for param in module.parameters():  # 遍历该层的所有权重和偏置
                    param.requires_grad = True  # 解冻 → 允许反向传播更新

        # ---- 解冻全连接层 fc ----
        # fc 是新增的，必须训练，否则无法适配 10 分类任务
        for param in self.resnet.fc.parameters():  # 遍历 fc 层的参数
            param.requires_grad = True  # 解冻

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状 (batch_size, 3, 128, 128)
        返回:
            logits: 形状 (batch_size, 10)
        """
        return self.resnet(x)  # 直接调用 ResNet50 的前向传播


# 创建 CustomResNet50 模型实例
model_resnet = CustomResNet50(num_classes=num_classes)  # 实例化 ResNet50 微调模型
print(model_resnet)  # 打印模型结构概览（确认解冻层和 fc 替换是否正确）

# ============================================================
# 15. CustomResNet50 参数统计
# ============================================================

print("\n========== CustomResNet50 参数统计 ==========")  # 打印分隔标题

# 统计总参数量与可训练参数量
total_params_resnet = sum(p.numel() for p in model_resnet.parameters())  # 模型所有参数：约 23,528,522
trainable_params_resnet = sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)  # 可训练参数：约 1,069,066
frozen_params_resnet = total_params_resnet - trainable_params_resnet  # 冻结参数: 约 22,459,456

print(f"CustomResNet50 模型总参数量: {total_params_resnet:,}")  # 约 23.5M
print(f"CustomResNet50 可训练参数量: {trainable_params_resnet:,}")  # 约 1.07M
print(f"CustomResNet50 冻结参数量: {frozen_params_resnet:,}")  # 约 22.46M
print(f"可训练参数占比: {trainable_params_resnet / total_params_resnet * 100:.1f}%")  # 约 4.5%

# 打印各层可训练状态（只显示可训练的层，避免输出过多）
print("\n可训练层明细:")  # 标题
for name, param in model_resnet.named_parameters():  # 遍历所有参数
    if param.requires_grad:  # 只打印需要梯度更新的参数
        print(f"  {name}: {param.numel():,} (requires_grad=True)")  # 参数名、参数量、可训练标记

# ============================================================
# 16. CustomResNet50 前向传播验证
# ============================================================

# 用随机生成的单张虚拟图片测试前向传播
x_test = torch.randn(1, 3, 128, 128)  # batch_size=1, 3 通道, 128×128
output_resnet = model_resnet(x_test)  # 前向传播
print(f"\nCustomResNet50 正向传播输出 shape: {output_resnet.shape}")  # torch.Size([1, 10])

# ============================================================
# 17. CustomResNet50 训练准备
# ============================================================

# 将模型移至设备（GPU 或 CPU）
model_resnet = model_resnet.to(device)  # 将参数迁移到目标设备

# 损失函数: 交叉熵损失（与 SimpleConvNet 相同）
criterion_resnet = nn.CrossEntropyLoss()  # 多分类交叉熵

# 优化器: Adam
# 注意: 只有 requires_grad=True 的参数会被优化器更新
# lr=0.001: 微调任务中常用较小的学习率（预训练权重已有较好初始化）
optimizer_resnet = optim.Adam(model_resnet.parameters(), lr=0.001)  # 创建 Adam 优化器

# 训练超参数
epochs_resnet = 10  # 训练轮数: 微调通常 5~10 轮即可收敛

# ============================================================
# 18. CustomResNet50 训练
# ============================================================

print(f"\n========== 开始 CustomResNet50 训练 (epochs={epochs_resnet}) ==========")

# 使用 Trainer 封装训练流程
trainer_resnet = Trainer(
    model=model_resnet,  # 待训练的 ResNet50 微调模型
    trainloader=train_loader,  # 训练集 DataLoader
    valloader=val_loader,  # 验证集 DataLoader
    criterion=criterion_resnet,  # 损失函数（交叉熵）
    optimizer=optimizer_resnet,  # 优化器（Adam）
    device=device,  # 训练设备
    epochs=epochs_resnet,  # 训练轮数
    early_stopping=True,  # 启用早停
    patience=5,  # 早停容忍度
    save_path="best_model_resnet50.pth",  # 最优权重保存路径（带 resnet50 后缀区分）
    early_stop_mode="acc",  # 早停依据: 验证集准确率
    maximize_acc=True,  # 准确率越大越好
    use_tensorboard=False  # 不使用 TensorBoard
)

trainer_resnet.train()  # 开始微调训练
trainer_resnet.plot(acc=True)  # 绘制训练曲线

# ============================================================
# 19. CustomResNet50 验证集评估
# ============================================================

# 使用最终模型在验证集上评估
test_loss_resnet, test_acc_resnet = trainer_resnet.evaluating(val_loader)  # 在验证集上评估
print(f"\n========== CustomResNet50 验证集评估结果 ==========")  # 打印标题
print(f"CustomResNet50 - Val Loss: {test_loss_resnet:.4f}, Val Accuracy: {test_acc_resnet:.2f}%")  # 打印结果

# ============================================================
# 20. SimpleConvNet vs CustomResNet50 对比总结
# ============================================================

print("\n" + "=" * 65)  # 打印分隔线
print("========== SimpleConvNet vs CustomResNet50 对比总结 ==========")  # 对比标题
print("=" * 65)  # 打印分隔线

# 构建对比表格
print(f"{'模型':<22} {'总参数':<15} {'可训练参数':<15} {'验证准确率':<12}")  # 表头
print("-" * 65)  # 分隔线
print(
    f"{'SimpleConvNet':<22} {total_params_simple:<15,} {trainable_params_simple:<15,} {test_acc_simple:<12.2f}%")  # 简单 CNN
print(
    f"{'CustomResNet50':<22} {total_params_resnet:<15,} {trainable_params_resnet:<15,} {test_acc_resnet:<12.2f}%")  # ResNet50 微调
print("-" * 65)  # 分隔线

# 打印详细分析
print("\n结论分析:")  # 分析标题
print(f"  1. 参数量对比:")  # 参数量子标题
print(f"     SimpleConvNet 总参数: {total_params_simple:,}（全部可训练）")  # ~4.48M
print(f"     CustomResNet50 总参数: {total_params_resnet:,}（可训练: {trainable_params_resnet:,}）")  # ~23.5M / ~1.07M
print(f"  2. 训练效率对比:")  # 效率子标题
print(f"     SimpleConvNet: 从零训练 {total_params_simple:,} 个参数，需要更多 epoch 和时间")
print(f"     CustomResNet50: 只微调 {trainable_params_resnet:,} 个参数，利用预训练权重快速收敛")
print(f"  3. 迁移学习优势:")  # 迁移学习子标题
print(f"     - ResNet50 在 ImageNet 上预训练，低层已学会通用的边缘/纹理/形状特征")
print(f"     - 只需微调高层语义特征和分类头即可适配新任务")
print(f"     - 数据量较少时（如每类几百张），迁移学习通常优于从头训练")
print(f"  4. 适用场景:")  # 场景子标题
print(f"     SimpleConvNet: 适合小数据集、低计算资源、需要小型模型的场景")
print(f"     CustomResNet50: 适合追求高准确率、有 GPU 加速、数据量与 ImageNet 相似的场景")
