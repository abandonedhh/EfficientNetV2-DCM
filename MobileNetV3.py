"""
煤粉浓度检测深度学习模型
- 输入：煤粉图像（RGB）
- 输出：预测浓度值（mg/m³）及所属分段
- 创新点：
  1. MobileNetV3-Small + 改进CBAM注意力（增强煤粉颗粒特征）
  2. 专业煤粉图像预处理（CLAHE + 非局部均值去噪）
  3. 轻量化设计（模型<5MB，适合矿用设备）
  4. 分段误差分析（按10个浓度段验证性能）

数据集要求：
- 根目录结构: root/1.50/157.6/Image_20250714161545445_159.6[.扩展名]
- 自动提取瞬时浓度作为标签（从文件名解析）
- 保证每风速1500张，浓度0-1000mg/m³分10段（每段150张）

注意：Windows系统默认隐藏文件扩展名，本代码已智能处理此问题
"""

import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import time
import sys
from pathlib import Path
from typing import List, Tuple, Dict


# ==============================
# 模块1: 配置参数 (关键: 适配您的数据集)
# ==============================
class Config:
    """全局配置参数 - 请根据您的实际情况调整"""

    # 数据集路径 (必须修改!)
    DATA_ROOT = r"D:\Project\MobileNetV3+CBAM\data"  # 例如: r"D:\coal_dust_data"

    # 图像处理参数
    IMG_SIZE = 224  # MobileNetV3输入尺寸
    CLAHE_CLIP = 3.0  # CLAHE对比度增强阈值
    DENOISE_H = 10  # 非局部均值去噪强度

    # 模型与训练参数
    BATCH_SIZE = 32  # 轻量化设计，适合小GPU
    LEARNING_RATE = 1e-3  # 初始学习率
    EPOCHS = 50  # 训练轮数
    VAL_SPLIT = 0.2  # 验证集比例
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 浓度范围与分段 (固定为您的要求)
    CONC_MIN = 0.0  # mg/m³
    CONC_MAX = 1000.0  # mg/m³
    NUM_SEGMENTS = 10  # 10个浓度段 (0-100, 100-200, ..., 900-1000)
    SEGMENT_WIDTH = (CONC_MAX - CONC_MIN) / NUM_SEGMENTS  # 每段100 mg/m³

    # 模型保存
    MODEL_SAVE_DIR = "models"
    MODEL_SAVE_NAME = "coal_concentration_model.pth"

    # 日志配置
    LOG_DIR = "logs"
    LOG_FILE = "training.log"

    # 风速列表 (自动检测，此处仅作说明)
    WIND_SPEED_PATTERN = r'^\d+\.\d+$'  # 匹配"1.50"这样的格式

    # 支持的图像扩展名 (关键: Windows隐藏扩展名处理)
    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    # 验证设置
    SEGMENT_ERROR_REPORT = True  # 是否生成分段误差报告


# 初始化配置
config = Config()

# 创建必要目录
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, config.LOG_FILE)),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==============================
# 模块2: 数据集诊断与验证
# ==============================
def diagnose_dataset():
    """诊断数据集结构，特别处理Windows隐藏扩展名问题"""
    logger.info(f"开始诊断数据集: {config.DATA_ROOT}")

    # 检查根目录是否存在
    if not os.path.exists(config.DATA_ROOT):
        raise FileNotFoundError(f"数据集根目录不存在: {config.DATA_ROOT}")

    # 检查是否为目录
    if not os.path.isdir(config.DATA_ROOT):
        raise NotADirectoryError(f"指定路径不是目录: {config.DATA_ROOT}")

    # 检测Windows是否隐藏文件扩展名
    windows_hide_ext = False
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced")
        value, _ = winreg.QueryValueEx(key, "HideFileExt")
        winreg.CloseKey(key)
        windows_hide_ext = (value == 1)
    except:
        pass  # 非Windows系统忽略

    # 扫描风速文件夹
    wind_speed_folders = [
        d for d in os.listdir(config.DATA_ROOT)
        if os.path.isdir(os.path.join(config.DATA_ROOT, d)) and
           re.match(config.WIND_SPEED_PATTERN, d)
    ]

    if not wind_speed_folders:
        # 详细诊断
        all_items = os.listdir(config.DATA_ROOT)
        folders = [f for f in all_items if os.path.isdir(os.path.join(config.DATA_ROOT, f))]

        error_msg = (
            f"诊断失败: 未找到有效的风速文件夹!\n"
            f"- 扫描路径: {config.DATA_ROOT}\n"
            f"- 实际找到的目录: {folders}\n"
            f"- 期望格式: 匹配正则表达式 '{config.WIND_SPEED_PATTERN}' (如 '1.50', '2.00')\n"
            f"解决方案:\n"
            f"1. 确认数据集路径正确\n"
            f"2. 确认风速文件夹命名格式正确\n"
        )

        if windows_hide_ext:
            error_msg += (
                f"3. Windows系统可能隐藏了文件扩展名 - 请按以下步骤显示:\n"
                f"   a) 打开文件资源管理器\n"
                f"   b) 点击'查看'选项卡\n"
                f"   c) 勾选'文件扩展名'选项\n"
            )

        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"找到 {len(wind_speed_folders)} 个有效风速文件夹: {wind_speed_folders}")

    # 检测扩展名
    sample_path = os.path.join(config.DATA_ROOT, wind_speed_folders[0])
    sample_files = []
    for root, _, files in os.walk(sample_path):
        if files:
            sample_files = files
            break

    detected_exts = set()
    for file in sample_files:
        ext = os.path.splitext(file)[1].lower()
        if ext:
            detected_exts.add(ext)

    if detected_exts:
        logger.info(f"检测到的图像扩展名: {', '.join(detected_exts)}")
        logger.info(f"支持的扩展名列表: {config.SUPPORTED_EXTENSIONS}")
    else:
        logger.warning("无法检测到文件扩展名 - 可能是Windows隐藏了扩展名")
        if windows_hide_ext:
            logger.warning("提示: Windows系统可能隐藏了文件扩展名，请按说明显示")

    return wind_speed_folders, bool(detected_exts)


# ==============================
# 模块3: 数据集处理 (关键: 适配您的特殊文件结构)
# ==============================
class CoalDustDataset(Dataset):
    """煤粉浓度数据集类 - 智能处理Windows隐藏扩展名问题"""

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: 数据集根目录
            transform: 图像转换操作
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []  # 瞬时浓度 (mg/m³)

        # 步骤1: 扫描所有风速文件夹 (5个)
        wind_speeds = [d for d in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, d)) and
                       re.match(config.WIND_SPEED_PATTERN, d)]

        logger.info(f"找到 {len(wind_speeds)} 个风速文件夹: {wind_speeds}")

        # 步骤2: 遍历每个风速文件夹
        for wind_speed in wind_speeds:
            wind_path = os.path.join(root_dir, wind_speed)

            # 遍历平均浓度子文件夹 (如 "157.6")
            for avg_conc_folder in os.listdir(wind_path):
                avg_conc_path = os.path.join(wind_path, avg_conc_folder)
                if not os.path.isdir(avg_conc_path):
                    continue

                # 步骤3: 遍历子文件夹内所有可能的图像文件
                for img_file in os.listdir(avg_conc_path):
                    # 尝试解析文件名 (关键: 处理可能的扩展名)
                    base_name, ext = os.path.splitext(img_file)

                    # 跳过非图像文件
                    if ext.lower() not in config.SUPPORTED_EXTENSIONS and ext != '':
                        continue

                    # 检查是否符合"Image_时间戳_浓度"格式
                    if not base_name.startswith('Image_'):
                        continue

                    # 从文件名提取瞬时浓度
                    try:
                        # 提取最后一个下划线后的部分
                        conc_str = base_name.split('_')[-1]
                        conc = float(conc_str)

                        # 验证浓度范围 (0-1000)
                        if config.CONC_MIN <= conc <= config.CONC_MAX:
                            img_path = os.path.join(avg_conc_path, img_file)
                            self.image_paths.append(img_path)
                            self.labels.append(conc)
                        else:
                            logger.debug(f"跳过浓度超出范围的文件: {img_file} ({conc})")
                    except (ValueError, IndexError) as e:
                        logger.debug(f"跳过文件名解析失败的文件: {img_file} - {str(e)}")
                        continue

        if len(self.image_paths) == 0:
            # 详细诊断
            diag_msg = (
                f"数据集加载失败诊断:\n"
                f"- 扫描目录: {root_dir}\n"
                f"- 风速文件夹: {wind_speeds}\n"
                f"- 支持的扩展名: {config.SUPPORTED_EXTENSIONS}\n"
                f"请检查:\n"
                f"1. 文件是否实际是图像格式\n"
                f"2. 文件名格式是否严格匹配 'Image_时间戳_浓度'\n"
            )

            if sys.platform.startswith('win'):
                diag_msg += (
                    f"3. Windows可能隐藏了文件扩展名 - 请显示文件扩展名:\n"
                    f"   a) 打开文件资源管理器\n"
                    f"   b) 点击'查看'选项卡\n"
                    f"   c) 勾选'文件扩展名'选项\n"
                )

            logger.error(diag_msg)
            raise ValueError(f"未找到有效图像文件!\n{diag_msg}")

        logger.info(f"成功加载 {len(self.image_paths)} 个有效样本")
        logger.info(f"浓度范围: {min(self.labels):.1f}-{max(self.labels):.1f} mg/m³")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)

        if img is None:
            # 尝试添加常见扩展名重试 (针对Windows隐藏扩展名问题)
            base_path = os.path.splitext(img_path)[0]
            for ext in config.SUPPORTED_EXTENSIONS:
                alt_path = base_path + ext
                if os.path.exists(alt_path):
                    img = cv2.imread(alt_path)
                    if img is not None:
                        break

            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")

        # 创新点1: 图像预处理 (提升煤粉图像质量)
        img = self._enhance_coal_image(img)

        # 转为RGB (OpenCV默认BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 转为PIL用于torchvision transforms
        img = transforms.ToPILImage()(img)

        # 应用标准转换
        if self.transform:
            img = self.transform(img)

        # 标签归一化: [0, 1000] -> [0, 1] (稳定训练)
        label = self.labels[idx] / config.CONC_MAX
        return img, torch.tensor(label, dtype=torch.float32)

    def _enhance_coal_image(self, img: np.ndarray) -> np.ndarray:
        """煤粉专用图像增强 (创新点)"""
        # 步骤1: 转为灰度 (煤粉浓度主要依赖灰度变化)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 步骤2: 非局部均值去噪 (减少传感器噪声)
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=config.DENOISE_H,
            templateWindowSize=7,
            searchWindowSize=21
        )

        # 步骤3: CLAHE对比度增强 (突出煤粉颗粒细节)
        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP,
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(denoised)

        # 步骤4: 转回3通道 (适配RGB输入模型)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ==============================
# 模块4: 模型定义 (MobileNetV3 + 创新修改)
# ==============================
class CBAM(nn.Module):
    """Convolutional Block Attention Module (轻量化实现)"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_att(x)
        x = x * ca

        # 空间注意力
        max_pool = torch.max(x, 1, keepdim=True)[0]
        avg_pool = torch.mean(x, 1, keepdim=True)
        spatial = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_att(spatial)
        x = x * sa
        return x


class CoalConcentrationModel(nn.Module):
    """煤粉浓度检测模型 (轻量化+创新设计)"""

    def __init__(self):
        super().__init__()
        # 步骤1: 加载预训练MobileNetV3-Small (轻量SOTA)
        base_model = models.mobilenet_v3_small(pretrained=True)

        # 步骤2: 移除分类层，保留特征提取器
        self.features = base_model.features  # 输出: [batch, 576, 7, 7]

        # 创新点2: 在特征图末尾添加CBAM
        self.cbam = CBAM(channels=576)

        # 步骤3: 自定义回归头 (轻量化设计)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 -> [batch, 576, 1, 1]
            nn.Flatten(),  # -> [batch, 576]
            nn.Linear(576, 128),  # 轻量全连接
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 输出单个浓度值
        )

        # 冻结部分基础层 (加速训练，适合小数据集)
        for param in self.features[:5].parameters():  # 仅微调高层特征
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = self.cbam(x)  # 注意力增强 (关键创新)
        x = self.regressor(x)  # 回归预测
        return x

    def get_model_size(self) -> float:
        """计算模型大小(MB)"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb


# ==============================
# 模块5: 分段误差分析
# ==============================
def calculate_segment_errors(
        true_conc: np.ndarray,
        pred_conc: np.ndarray,
        segments: int = config.NUM_SEGMENTS
) -> dict:
    """
    计算每个浓度段的MAE

    Args:
        true_conc: 真实浓度值数组 (mg/m³)
        pred_conc: 预测浓度值数组 (mg/m³)
        segments: 浓度分段数

    Returns:
        dict: 各段的MAE值
    """
    segment_errors = {}
    segment_counts = {}

    for i in range(segments):
        low = i * config.SEGMENT_WIDTH
        high = (i + 1) * config.SEGMENT_WIDTH

        # 筛选该分段的样本
        mask = (true_conc >= low) & (true_conc < high)
        if np.any(mask):
            mae = mean_absolute_error(true_conc[mask], pred_conc[mask])
            segment_errors[f"{int(low)}-{int(high)}"] = mae
            segment_counts[f"{int(low)}-{int(high)}"] = int(mask.sum())
        else:
            segment_errors[f"{int(low)}-{int(high)}"] = float('nan')
            segment_counts[f"{int(low)}-{int(high)}"] = 0

    return segment_errors, segment_counts


def plot_segment_errors(segment_errors: dict, segment_counts: dict, epoch: int, save_dir: str = "segment_errors"):
    """可视化分段误差 (用于验证报告)"""
    os.makedirs(save_dir, exist_ok=True)

    segments = list(segment_errors.keys())
    errors = [segment_errors[s] for s in segments]
    counts = [segment_counts[s] for s in segments]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(segments, errors, color='skyblue')

    # 添加样本数量标签
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'n={count}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.xlabel('Concentration Segment (mg/m³)')
    plt.ylabel('MAE (mg/m³)')
    plt.title(f'Concentration Segment Error Analysis (Epoch {epoch})')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'segment_errors_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()

    return save_path


# ==============================
# 模块6: 训练与验证逻辑
# ==============================
def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """创建训练/验证DataLoader"""
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图像单通道
    ])

    # 加载数据集
    dataset = CoalDustDataset(config.DATA_ROOT, transform=transform)

    # 划分训练/验证集
    val_size = int(len(dataset) * config.VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # 确保batch size一致
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"数据集划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion, device):
    """单轮训练"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()

        # 确保形状匹配
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(loss=loss.item())

    return total_loss / total_samples


def validate(model, loader, criterion, device) -> Dict:
    """验证 + 分段误差分析"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()

            # 处理单样本batch
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            # 计算损失
            loss = criterion(outputs, labels)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # 保存原始数据 (用于分段分析)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 转为NumPy (浓度反归一化)
    all_preds = np.array(all_preds) * config.CONC_MAX
    all_labels = np.array(all_labels) * config.CONC_MAX

    # 分段误差计算
    segment_errors, segment_counts = calculate_segment_errors(all_labels, all_preds)

    # 整体MAE
    overall_mae = mean_absolute_error(all_labels, all_preds)

    return {
        "loss": total_loss / total_samples,
        "overall_mae": overall_mae,
        "segment_errors": segment_errors,
        "segment_counts": segment_counts,
        "num_samples": total_samples
    }


# ==============================
# 模块7: 主训练流程
# ==============================
def main():
    # 步骤0: 数据集诊断
    logger.info("=" * 50)
    logger.info("煤粉浓度检测模型训练系统启动")
    logger.info("=" * 50)

    try:
        wind_speeds, ext_detected = diagnose_dataset()
        logger.info(f"数据集诊断完成! 风速: {wind_speeds}")
    except Exception as e:
        logger.exception("数据集诊断失败!")
        logger.info("\n===== 数据集问题解决方案 =====")
        logger.info("1. 确认数据集路径正确 (修改Config.DATA_ROOT)")
        logger.info("2. 确认风速文件夹命名格式 (如'1.50', '2.00')")
        logger.info("3. 确认文件名格式: Image_时间戳_浓度[.扩展名]")
        if sys.platform.startswith('win'):
            logger.info("4. Windows用户: 显示文件扩展名 (查看 -> 文件扩展名)")
        logger.info("=" * 50)
        raise

    # 步骤1: 准备数据
    logger.info("\n=> 准备数据集...")
    try:
        train_loader, val_loader = create_dataloaders(config)
    except Exception as e:
        logger.exception("数据集准备失败!")
        raise

    # 步骤2: 初始化模型
    logger.info("\n=> 初始化模型...")
    model = CoalConcentrationModel().to(config.DEVICE)

    # 显示模型信息
    model_size = model.get_model_size()
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"模型大小: {model_size:.2f} MB")
    if model_size > 5.0:
        logger.warning("警告: 模型大小超过5MB，可能不适合矿用设备!")

    # 步骤3: 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, #verbose=True
    )

    # 步骤4: 训练循环
    logger.info(f"\n=> 开始训练 ({config.EPOCHS} epochs)...")
    best_mae = float('inf')
    train_losses, val_maes = [], []

    for epoch in range(1, config.EPOCHS + 1):
        start_time = time.time()

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        train_losses.append(train_loss)

        # 验证
        val_results = validate(model, val_loader, criterion, config.DEVICE)
        val_maes.append(val_results["overall_mae"])

        # 学习率调整
        scheduler.step(val_results["loss"])

        # 保存最佳模型
        if val_results["overall_mae"] < best_mae:
            best_mae = val_results["overall_mae"]
            save_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_SAVE_NAME)
            torch.save(model.state_dict(), save_path)

        # 记录时间
        epoch_time = time.time() - start_time

        # 打印进度
        logger.info(f"Epoch {epoch:2d}/{config.EPOCHS} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val MAE: {val_results['overall_mae']:.2f} mg/m³ | "
                    f"Best MAE: {best_mae:.2f} mg/m³ | "
                    f"Time: {epoch_time:.1f}s")

        # 打印分段误差报告
        if config.SEGMENT_ERROR_REPORT:
            logger.info("\n  Concentration Segment Error Analysis:")
            for seg, err in val_results["segment_errors"].items():
                count = val_results["segment_counts"][seg]
                if not np.isnan(err):
                    logger.info(f"    {seg} mg/m³: MAE = {err:.2f} mg/m³ (n={count})")
                else:
                    logger.info(f"    {seg} mg/m³: No samples (n={count})")

            # 生成分段误差可视化
            plot_path = plot_segment_errors(
                val_results["segment_errors"],
                val_results["segment_counts"],
                epoch
            )
            logger.info(f"  Segment error plot saved to: {plot_path}")

    # 步骤5: 训练后分析
    logger.info("\n" + "=" * 50)
    logger.info("训练完成!")
    logger.info(f"最佳验证MAE: {best_mae:.2f} mg/m³")
    logger.info(f"模型已保存至: {os.path.join(config.MODEL_SAVE_DIR, config.MODEL_SAVE_NAME)}")
    logger.info("=" * 50)

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_maes, label='Validation MAE (mg/m³)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss & Validation MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.LOG_DIR, 'training_curve.png'))
    logger.info("Training curve saved to logs/training_curve.png")


# ==============================
# 执行入口
# ==============================
if __name__ == "__main__":
    # 关键安全检查
    if config.DATA_ROOT == r"D:\your\dataset\path":
        logger.error("ERROR: 请设置Config.DATA_ROOT为您的数据集路径!")
        logger.info("示例: DATA_ROOT = r\"D:\\coal_dust_data\"")
        sys.exit(1)

    # 设置随机种子 (确保可复现)
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        main()
    except Exception as e:
        logger.exception("训练过程中发生未处理异常")
        logger.info("\n===== 常见问题解决方案 =====")
        logger.info("1. 数据集路径问题: 确认DATA_ROOT设置正确")
        logger.info("2. 文件扩展名问题: Windows用户请显示文件扩展名")
        logger.info("3. 文件名格式: 确认格式为 'Image_时间戳_浓度[.扩展名]'")
        logger.info("4. 权限问题: 以管理员身份运行或检查文件权限")
        sys.exit(1)
