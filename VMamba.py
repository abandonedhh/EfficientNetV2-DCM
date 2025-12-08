# 改进的煤粉浓度检测系统

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import json
import time
import warnings
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# 1. 增强的数据加载与预处理
class CoalDustDataset(data.Dataset):
    """
    煤粉浓度数据集加载器
    处理以风速和浓度分层的图像数据
    """

    def __init__(self, root_dir, transform=None, concentration_bins=None, is_train=True, train_ratio=0.8):
        """
        初始化数据集

        参数:
            root_dir: 数据集根目录，包含以风速命名的子文件夹
            transform: 数据增强/预处理变换
            concentration_bins: 浓度分段边界，如[0, 100, 200, ..., 1000]
            is_train: 是否为训练集
            train_ratio: 训练集比例
        """
        self.root_dir = root_dir
        self.transform = transform
        self.concentration_bins = concentration_bins if concentration_bins is not None else [i * 100 for i in range(11)]
        self.is_train = is_train
        self.train_ratio = train_ratio

        # 收集所有图像路径和对应的元数据
        self.image_paths = []
        self.concentrations = []
        self.wind_speeds = []
        self.bin_labels = []

        # 遍历所有风速文件夹
        wind_speed_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        for ws_folder in wind_speed_folders:
            ws_path = os.path.join(root_dir, ws_folder)
            try:
                ws_value = float(ws_folder)  # 将文件夹名转换为浮点数风速值
            except ValueError:
                print(f"警告: 无法将 '{ws_folder}' 转换为浮点数，跳过此文件夹")
                continue

            # 遍历浓度文件夹
            conc_folders = [f for f in os.listdir(ws_path) if os.path.isdir(os.path.join(ws_path, f))]

            for conc_folder in conc_folders:
                conc_path = os.path.join(ws_path, conc_folder)
                try:
                    avg_concentration = float(conc_folder)  # 平均浓度值
                except ValueError:
                    print(f"警告: 无法将 '{conc_folder}' 转换为浮点数，跳过此文件夹")
                    continue

                # 获取所有图像文件
                image_files = [f for f in os.listdir(conc_path) if f.endswith('.jpg')]

                if not image_files:
                    print(f"警告: 文件夹 '{conc_path}' 中没有找到jpg图像，跳过")
                    continue

                # 计算训练/验证分割点
                split_idx = int(len(image_files) * train_ratio)

                if is_train:
                    selected_files = image_files[:split_idx]
                else:
                    selected_files = image_files[split_idx:]

                # 处理每个图像文件
                for img_file in selected_files:
                    img_path = os.path.join(conc_path, img_file)

                    # 从文件名解析瞬时浓度
                    try:
                        # 格式: Image_时间戳_瞬时浓度.jpg
                        parts = img_file.split('_')
                        if len(parts) < 3:
                            instant_conc = avg_concentration
                        else:
                            instant_conc_str = parts[2].split('.')[0]
                            instant_conc = float(instant_conc_str)
                    except (ValueError, IndexError):
                        instant_conc = avg_concentration  # 如果解析失败，使用文件夹的平均浓度

                    # 确定浓度分段标签
                    bin_idx = np.digitize(instant_conc, self.concentration_bins) - 1
                    bin_idx = max(0, min(len(self.concentration_bins) - 2, bin_idx))  # 确保在范围内

                    self.image_paths.append(img_path)
                    self.concentrations.append(instant_conc)
                    self.wind_speeds.append(ws_value)
                    self.bin_labels.append(bin_idx)

        # 打印数据集统计信息
        if len(self.image_paths) > 0:
            print(f"{'训练' if is_train else '验证'}集大小: {len(self.image_paths)}")
            print(f"浓度范围: {min(self.concentrations):.2f} - {max(self.concentrations):.2f}")
            print(f"风速值: {set(self.wind_speeds)}")

            # 计算每个分段的样本数量
            bin_counts = np.bincount(self.bin_labels, minlength=len(self.concentration_bins) - 1)
            for i, count in enumerate(bin_counts):
                lower = self.concentration_bins[i]
                upper = self.concentration_bins[i + 1]
                print(f"浓度段 [{lower}-{upper}): {count}个样本")
        else:
            print(f"警告: {'训练' if is_train else '验证'}集为空!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        concentration = self.concentrations[idx]
        wind_speed = self.wind_speeds[idx]
        bin_label = self.bin_labels[idx]

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            # 返回一个黑色图像作为替代
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        # 将风速信息作为额外特征
        wind_tensor = torch.tensor([wind_speed], dtype=torch.float32)

        return image, wind_tensor, torch.tensor(concentration, dtype=torch.float32), torch.tensor(bin_label,
                                                                                                  dtype=torch.long)


# 2. 改进的模型架构
class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class AttentionModule(nn.Module):
    """注意力模块"""

    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EnhancedCoalDustModel(nn.Module):
    """
    改进的煤粉浓度检测模型
    使用残差连接和注意力机制
    """

    def __init__(self, num_bins=10, wind_embed_dim=32, pretrained=True):
        super(EnhancedCoalDustModel, self).__init__()

        # 使用预训练的ResNet作为特征提取器
        if pretrained:
            self.feature_extractor = models.resnet34(pretrained=True)
            # 替换最后一层全连接层
            num_features = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()  # 移除最后的全连接层
        else:
            # 自定义特征提取器
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                ResidualBlock(64, 64),
                ResidualBlock(64, 64),

                ResidualBlock(64, 128, stride=2),
                ResidualBlock(128, 128),

                ResidualBlock(128, 256, stride=2),
                ResidualBlock(256, 256),

                ResidualBlock(256, 512, stride=2),
                ResidualBlock(512, 512),

                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            num_features = 512

        # 风速信息处理
        self.wind_embed = nn.Sequential(
            nn.Linear(1, wind_embed_dim),
            nn.ReLU(),
            nn.Linear(wind_embed_dim, wind_embed_dim),
            nn.ReLU()
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(num_features + wind_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_features + wind_embed_dim),
            nn.Sigmoid()
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(num_features + wind_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 分类头（用于分段评估）
        self.classifier = nn.Sequential(
            nn.Linear(num_features + wind_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_bins)
        )

    def forward(self, x, wind_speed):
        # 提取图像特征
        img_features = self.feature_extractor(x)

        # 处理风速信息
        wind_features = self.wind_embed(wind_speed)

        # 合并特征
        combined = torch.cat([img_features, wind_features], dim=1)

        # 应用注意力机制
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights

        # 回归预测
        concentration_pred = self.regressor(attended_features).squeeze()

        # 分类预测
        bin_pred = self.classifier(attended_features)

        return concentration_pred, bin_pred


# 3. 增强的训练与评估
class EnhancedCoalDustTrainer:
    """改进的煤粉浓度检测模型训练器"""

    def __init__(self, model, train_loader, val_loader, concentration_bins, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.concentration_bins = concentration_bins
        self.num_bins = len(concentration_bins) - 1

        # 损失函数
        self.regression_criterion = nn.HuberLoss()  # 使用HuberLoss替代MSE，对异常值更鲁棒
        self.classification_criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        # 记录训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_rmse': [], 'val_rmse': [],
            'train_r2': [], 'val_r2': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': [],
            'learning_rate': []
        }

        # 记录每个分段的指标
        self.bin_metrics = {
            'train': {i: {'mae': [], 'rmse': [], 'r2': []} for i in range(self.num_bins)},
            'val': {i: {'mae': [], 'rmse': [], 'r2': []} for i in range(self.num_bins)}
        }

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        bin_preds = []
        bin_targets = []

        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')

        for batch_idx, (images, winds, concentrations, bin_labels) in enumerate(pbar):
            # 确保所有数据都在同一设备上
            images = images.to(self.device)
            winds = winds.to(self.device)
            concentrations = concentrations.to(self.device)
            bin_labels = bin_labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            conc_pred, bin_pred = self.model(images, winds)

            # 计算损失
            regression_loss = self.regression_criterion(conc_pred, concentrations)
            classification_loss = self.classification_criterion(bin_pred, bin_labels)

            # 动态调整损失权重
            alpha = min(1.0, epoch / 10)  # 前10个epoch逐渐增加分类损失权重
            loss = regression_loss + alpha * classification_loss

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 记录指标
            running_loss += loss.item()
            all_preds.extend(conc_pred.detach().cpu().numpy())
            all_targets.extend(concentrations.cpu().numpy())
            bin_preds.extend(torch.argmax(bin_pred, dim=1).cpu().numpy())
            bin_targets.extend(bin_labels.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': self.optimizer.param_groups[0]['lr']
            })

        # 计算 epoch 指标
        epoch_loss = running_loss / len(self.train_loader)
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)

        # 计算分类指标
        accuracy = accuracy_score(bin_targets, bin_preds)
        precision = precision_score(bin_targets, bin_preds, average='weighted', zero_division=0)
        recall = recall_score(bin_targets, bin_preds, average='weighted', zero_division=0)
        f1 = f1_score(bin_targets, bin_preds, average='weighted', zero_division=0)

        # 记录历史
        self.history['train_loss'].append(epoch_loss)
        self.history['train_mae'].append(mae)
        self.history['train_rmse'].append(rmse)
        self.history['train_r2'].append(r2)
        self.history['train_acc'].append(accuracy)
        self.history['train_precision'].append(precision)
        self.history['train_recall'].append(recall)
        self.history['train_f1'].append(f1)
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

        # 计算每个分段的指标
        bin_errors = self._calculate_bin_metrics(all_preds, all_targets, bin_targets, 'train')

        return epoch_loss, mae, rmse, r2, accuracy, precision, recall, f1, bin_errors

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        bin_preds = []
        bin_targets = []

        # 按浓度分段存储预测和真实值
        bin_predictions = {i: {'preds': [], 'targets': []} for i in range(self.num_bins)}

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Validate Epoch {epoch}')

            for images, winds, concentrations, bin_labels in pbar:
                # 确保所有数据都在同一设备上
                images = images.to(self.device)
                winds = winds.to(self.device)
                concentrations = concentrations.to(self.device)
                bin_labels = bin_labels.to(self.device)

                # 前向传播
                conc_pred, bin_pred = self.model(images, winds)

                # 计算损失
                regression_loss = self.regression_criterion(conc_pred, concentrations)
                classification_loss = self.classification_criterion(bin_pred, bin_labels)

                # 动态调整损失权重
                alpha = min(1.0, epoch / 10)
                loss = regression_loss + alpha * classification_loss

                # 记录指标
                running_loss += loss.item()
                all_preds.extend(conc_pred.cpu().numpy())
                all_targets.extend(concentrations.cpu().numpy())

                bin_pred_classes = torch.argmax(bin_pred, dim=1)
                bin_preds.extend(bin_pred_classes.cpu().numpy())
                bin_targets.extend(bin_labels.cpu().numpy())

                # 按分段存储预测结果
                for i in range(len(conc_pred)):
                    bin_idx = bin_labels[i].item()
                    bin_predictions[bin_idx]['preds'].append(conc_pred[i].item())
                    bin_predictions[bin_idx]['targets'].append(concentrations[i].item())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 计算整体验证指标
        epoch_loss = running_loss / len(self.val_loader)
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)

        # 计算分类指标
        accuracy = accuracy_score(bin_targets, bin_preds)
        precision = precision_score(bin_targets, bin_preds, average='weighted', zero_division=0)
        recall = recall_score(bin_targets, bin_preds, average='weighted', zero_division=0)
        f1 = f1_score(bin_targets, bin_preds, average='weighted', zero_division=0)

        # 记录历史
        self.history['val_loss'].append(epoch_loss)
        self.history['val_mae'].append(mae)
        self.history['val_rmse'].append(rmse)
        self.history['val_r2'].append(r2)
        self.history['val_acc'].append(accuracy)
        self.history['val_precision'].append(precision)
        self.history['val_recall'].append(recall)
        self.history['val_f1'].append(f1)

        # 计算每个分段的指标
        bin_errors = self._calculate_bin_metrics(all_preds, all_targets, bin_targets, 'val')

        return epoch_loss, mae, rmse, r2, accuracy, precision, recall, f1, bin_errors

    def _calculate_bin_metrics(self, preds, targets, bin_targets, phase):
        """计算每个浓度分段的指标"""
        bin_errors = {}
        for bin_idx in range(self.num_bins):
            # 获取当前分段的预测和真实值
            bin_mask = np.array(bin_targets) == bin_idx
            if np.sum(bin_mask) > 0:
                bin_pred = np.array(preds)[bin_mask]
                bin_true = np.array(targets)[bin_mask]

                # 计算指标
                bin_mae = mean_absolute_error(bin_true, bin_pred)
                bin_rmse = np.sqrt(mean_squared_error(bin_true, bin_pred))
                bin_r2 = r2_score(bin_true, bin_pred) if len(bin_true) > 1 else 0

                bin_errors[bin_idx] = {
                    'mae': bin_mae,
                    'rmse': bin_rmse,
                    'r2': bin_r2,
                    'count': np.sum(bin_mask)
                }

                # 记录到历史中
                self.bin_metrics[phase][bin_idx]['mae'].append(bin_mae)
                self.bin_metrics[phase][bin_idx]['rmse'].append(bin_rmse)
                self.bin_metrics[phase][bin_idx]['r2'].append(bin_r2)

        return bin_errors

    def train(self, num_epochs):
        best_val_loss = float('inf')
        best_val_r2 = -float('inf')

        for epoch in range(1, num_epochs + 1):
            # 训练一个epoch
            train_loss, train_mae, train_rmse, train_r2, train_acc, train_precision, train_recall, train_f1, train_bin_errors = self.train_epoch(
                epoch)

            # 验证
            val_loss, val_mae, val_rmse, val_r2, val_acc, val_precision, val_recall, val_f1, val_bin_errors = self.validate(
                epoch)

            # 更新学习率
            self.scheduler.step()

            # 打印结果
            print(f'\nEpoch {epoch}/{num_epochs}:')
            print(f'Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}')
            print(
                f'Train - Acc: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')
            print(f'Val - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, R²: {val_r2:.4f}')
            print(
                f'Val - Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

            # 打印分段误差
            print("\n验证集分段误差分析:")
            for bin_idx, errors in val_bin_errors.items():
                lower_bound = self.concentration_bins[bin_idx]
                upper_bound = self.concentration_bins[bin_idx + 1]
                print(
                    f"浓度段 [{lower_bound}-{upper_bound}): MAE={errors['mae']:.2f}, RMSE={errors['rmse']:.2f}, R²={errors['r2']:.4f}, 样本数={errors['count']}")

            # 保存最佳模型
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': val_loss,
                    'r2': val_r2,
                    'history': self.history,
                    'bin_metrics': self.bin_metrics
                }, 'best_model.pth')
                print(f"保存了最佳模型! R²: {val_r2:.4f}")

        return self.history, self.bin_metrics

    def plot_training_history(self):
        """绘制训练历史图表"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='训练损失', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='验证损失', color='red')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MAE曲线
        axes[0, 1].plot(self.history['train_mae'], label='训练MAE', color='blue')
        axes[0, 1].plot(self.history['val_mae'], label='验证MAE', color='red')
        axes[0, 1].set_title('平均绝对误差(MAE)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE值')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # RMSE曲线
        axes[0, 2].plot(self.history['train_rmse'], label='训练RMSE', color='blue')
        axes[0, 2].plot(self.history['val_rmse'], label='验证RMSE', color='red')
        axes[0, 2].set_title('均方根误差(RMSE)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('RMSE值')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # R²曲线
        axes[1, 0].plot(self.history['train_r2'], label='训练R²', color='blue')
        axes[1, 0].plot(self.history['val_r2'], label='验证R²', color='red')
        axes[1, 0].set_title('决定系数(R²)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²值')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 准确率曲线
        axes[1, 1].plot(self.history['train_acc'], label='训练准确率', color='blue')
        axes[1, 1].plot(self.history['val_acc'], label='验证准确率', color='red')
        axes[1, 1].set_title('分类准确率')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('准确率')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # F1分数曲线
        axes[1, 2].plot(self.history['train_f1'], label='训练F1', color='blue')
        axes[1, 2].plot(self.history['val_f1'], label='验证F1', color='red')
        axes[1, 2].set_title('F1分数')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1值')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        # 学习率
        axes[2, 0].plot(self.history['learning_rate'], label='学习率', color='green')
        axes[2, 0].set_title('学习率变化')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('学习率')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # 精确率
        axes[2, 1].plot(self.history['train_precision'], label='训练精确率', color='blue')
        axes[2, 1].plot(self.history['val_precision'], label='验证精确率', color='red')
        axes[2, 1].set_title('精确率')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('精确率')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # 召回率
        axes[2, 2].plot(self.history['train_recall'], label='训练召回率', color='blue')
        axes[2, 2].plot(self.history['val_recall'], label='验证召回率', color='red')
        axes[2, 2].set_title('召回率')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('召回率')
        axes[2, 2].legend()
        axes[2, 2].grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_bin_metrics(self):
        """绘制每个浓度分段的指标"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 获取最后一个epoch的分段指标
        last_epoch = len(self.history['train_loss']) - 1

        # 准备数据
        bin_ranges = [f"{self.concentration_bins[i]}-{self.concentration_bins[i + 1]}"
                      for i in range(self.num_bins)]

        train_mae = [self.bin_metrics['train'][i]['mae'][last_epoch] for i in range(self.num_bins)]
        val_mae = [self.bin_metrics['val'][i]['mae'][last_epoch] for i in range(self.num_bins)]

        train_rmse = [self.bin_metrics['train'][i]['rmse'][last_epoch] for i in range(self.num_bins)]
        val_rmse = [self.bin_metrics['val'][i]['rmse'][last_epoch] for i in range(self.num_bins)]

        train_r2 = [self.bin_metrics['train'][i]['r2'][last_epoch] for i in range(self.num_bins)]
        val_r2 = [self.bin_metrics['val'][i]['r2'][last_epoch] for i in range(self.num_bins)]

        # MAE对比
        x = np.arange(len(bin_ranges))
        width = 0.35
        axes[0, 0].bar(x - width / 2, train_mae, width, label='训练集')
        axes[0, 0].bar(x + width / 2, val_mae, width, label='验证集')
        axes[0, 0].set_xlabel('浓度分段 (mg/m³)')
        axes[0, 0].set_ylabel('MAE值')
        axes[0, 0].set_title('各浓度分段的MAE对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(bin_ranges, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, axis='y')

        # RMSE对比
        axes[0, 1].bar(x - width / 2, train_rmse, width, label='训练集')
        axes[0, 1].bar(x + width / 2, val_rmse, width, label='验证集')
        axes[0, 1].set_xlabel('浓度分段 (mg/m³)')
        axes[0, 1].set_ylabel('RMSE值')
        axes[0, 1].set_title('各浓度分段的RMSE对比')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(bin_ranges, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, axis='y')

        # R²对比
        axes[1, 0].bar(x - width / 2, train_r2, width, label='训练集')
        axes[1, 0].bar(x + width / 2, val_r2, width, label='验证集')
        axes[1, 0].set_xlabel('浓度分段 (mg/m³)')
        axes[1, 0].set_ylabel('R²值')
        axes[1, 0].set_title('各浓度分段的R²对比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(bin_ranges, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, axis='y')

        # 样本数量分布
        sample_counts = [self.bin_metrics['val'][i]['count'] for i in range(self.num_bins)]
        axes[1, 1].bar(bin_ranges, sample_counts, color='orange')
        axes[1, 1].set_xlabel('浓度分段 (mg/m³)')
        axes[1, 1].set_ylabel('样本数量')
        axes[1, 1].set_title('验证集各浓度分段的样本分布')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, axis='y')

        plt.tight_layout()
        plt.savefig('bin_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()


# 4. 模型部署与实时检测
class EnhancedCoalDustDetector:
    """改进的煤粉浓度实时检测器"""

    def __init__(self, model_path, concentration_bins, device='cuda'):
        self.concentration_bins = concentration_bins
        self.num_bins = len(concentration_bins) - 1
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = EnhancedCoalDustModel(num_bins=self.num_bins)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"模型加载成功，使用设备: {self.device}")

    def predict(self, image, wind_speed):
        """
        预测单张图像的煤粉浓度

        参数:
            image: PIL Image 或 numpy array
            wind_speed: 风速值 (float)

        返回:
            concentration: 预测浓度值
            bin_idx: 浓度分段索引
            bin_range: 浓度分段范围
            confidence: 预测置信度
        """
        # 转换图像
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        wind_tensor = torch.tensor([[wind_speed]], dtype=torch.float32).to(self.device)

        # 预测
        with torch.no_grad():
            concentration_pred, bin_pred = self.model(image_tensor, wind_tensor)
            concentration = concentration_pred.item()
            bin_probs = torch.softmax(bin_pred, dim=1)
            confidence, bin_idx = torch.max(bin_probs, dim=1)
            confidence = confidence.item()
            bin_idx = bin_idx.item()

        # 获取分段范围
        lower_bound = self.concentration_bins[bin_idx]
        upper_bound = self.concentration_bins[bin_idx + 1]
        bin_range = f"{lower_bound}-{upper_bound}"

        return concentration, bin_idx, bin_range, confidence


# 5. 主程序
def main():
    """主函数"""
    # 配置参数
    data_dir = "D:\Project\VMamba\data"  # 更改为您的数据路径
    concentration_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    batch_size = 32
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备: {device}")

    # 数据预处理 - 增强数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集和数据加载器
    train_dataset = CoalDustDataset(
        root_dir=data_dir,
        transform=train_transform,
        concentration_bins=concentration_bins,
        is_train=True
    )

    val_dataset = CoalDustDataset(
        root_dir=data_dir,
        transform=val_transform,
        concentration_bins=concentration_bins,
        is_train=False
    )

    # 检查数据集是否为空
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("错误: 数据集为空，请检查数据路径和格式")
        return

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 创建模型
    model = EnhancedCoalDustModel(num_bins=len(concentration_bins) - 1, pretrained=True)

    # 创建训练器
    trainer = EnhancedCoalDustTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        concentration_bins=concentration_bins,
        device=device
    )

    # 开始训练
    print("开始训练...")
    history, bin_metrics = trainer.train(num_epochs)

    # 绘制训练历史
    trainer.plot_training_history()

    # 绘制分段指标
    trainer.plot_bin_metrics()

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'bin_metrics': bin_metrics,
        'concentration_bins': concentration_bins
    }, 'final_model.pth')
    print("训练完成，模型已保存!")

    # 保存训练历史为JSON
    with open('training_history.json', 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        json_history = {}
        for k, v in history.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.floating, float)):
                json_history[k] = [float(x) for x in v]
            else:
                json_history[k] = v
        json.dump(json_history, f, indent=4)

    # 保存分段指标
    with open('bin_metrics.json', 'w') as f:
        json_bin_metrics = {}
        for phase, metrics in bin_metrics.items():
            json_bin_metrics[phase] = {}
            for bin_idx, bin_data in metrics.items():
                json_bin_metrics[phase][str(bin_idx)] = {
                    'mae': [float(x) for x in bin_data['mae']],
                    'rmse': [float(x) for x in bin_data['rmse']],
                    'r2': [float(x) for x in bin_data['r2']]
                }
        json.dump(json_bin_metrics, f, indent=4)


# 测试函数
def test_model():
    """测试模型是否正确加载和运行"""
    # 创建一个随机测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试使用设备: {device}")

    # 创建模型
    model = EnhancedCoalDustModel(num_bins=10, pretrained=True)
    model.to(device)
    model.eval()

    # 创建测试输入
    test_image = torch.randn(1, 3, 224, 224).to(device)
    test_wind = torch.tensor([[1.5]], dtype=torch.float32).to(device)

    # 测试前向传播
    with torch.no_grad():
        conc_pred, bin_pred = model(test_image, test_wind)
        print(f"测试成功! 浓度预测: {conc_pred.item()}, 分段预测形状: {bin_pred.shape}")

    # 测试数据加载器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建一个小的测试数据集
    test_dataset = CoalDustDataset(
        root_dir="D:\Project\VMamba\data",
        transform=transform,
        concentration_bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        is_train=True
    )

    if len(test_dataset) > 0:
        test_loader = data.DataLoader(test_dataset, batch_size=2, shuffle=True)
        images, winds, concentrations, bin_labels = next(iter(test_loader))

        # 移动到设备
        images = images.to(device)
        winds = winds.to(device)
        concentrations = concentrations.to(device)
        bin_labels = bin_labels.to(device)

        # 测试模型
        with torch.no_grad():
            conc_pred, bin_pred = model(images, winds)
            print(f"批量测试成功! 浓度预测形状: {conc_pred.shape}, 分段预测形状: {bin_pred.shape}")


if __name__ == "__main__":
    # 先测试模型是否能正常运行
    test_model()

    # 运行主程序
    main()