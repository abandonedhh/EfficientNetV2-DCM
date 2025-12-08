import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
if torch.cuda.is_available():
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB')


# ====================== 数据预处理模块 ======================
class CoalDustDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        自定义数据集类
        Args:
            root_dir: 数据集根目录
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # 解析数据集结构
        wind_speeds = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))], key=float)
        for wind in wind_speeds:
            wind_path = os.path.join(root_dir, wind)
            # 获取所有浓度文件夹
            conc_folders = sorted([d for d in os.listdir(wind_path) if os.path.isdir(os.path.join(wind_path, d))],
                                  key=float)

            for conc_folder in conc_folders:
                conc_path = os.path.join(wind_path, conc_folder)
                # 获取所有图像文件
                image_files = [f for f in os.listdir(conc_path) if f.endswith('.jpg')]

                for img_file in image_files:
                    # 从文件名解析瞬时浓度
                    parts = img_file.split('_')
                    if len(parts) < 3:
                        continue

                    try:
                        # 提取瞬时浓度（去掉扩展名）
                        instant_conc = float(parts[-1].split('.')[0])
                    except ValueError:
                        continue

                    # 风速转换为浮点数
                    wind_speed = float(wind)

                    img_path = os.path.join(conc_path, img_file)
                    self.samples.append((img_path, instant_conc, wind_speed))

        print(f'总共加载 {len(self.samples)} 个样本')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, concentration, wind_speed = self.samples[idx]

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"无法加载图像: {img_path}")
            # 返回黑色图像作为占位符
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        # 将风速转换为张量
        wind_speed = torch.tensor([wind_speed], dtype=torch.float32)

        # 计算浓度段 (0-9)
        conc_segment = min(int(concentration // 100), 9)

        return image, wind_speed, concentration, conc_segment


# 图像预处理变换 - 更丰富的数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),  # 调整为较大尺寸
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),  # 调整为较大尺寸
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化
    ]),
}


# ====================== 注意力机制模块 ======================
class EfficientChannelAttention(nn.Module):
    """高效通道注意力模块 (ECA) - 更轻量高效"""

    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# ====================== 风速融合模块 ======================
class WindFusion(nn.Module):
    """增强的风速信息融合模块"""

    def __init__(self, in_features, wind_dim):
        super(WindFusion, self).__init__()
        self.wind_encoder = nn.Sequential(
            nn.Linear(wind_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features + 256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_features, wind_speed):
        # 编码风速信息
        wind_features = self.wind_encoder(wind_speed)
        # 拼接图像特征和风速信息
        combined = torch.cat((img_features, wind_features), dim=1)
        return self.fc(combined)


# ====================== 主要模型架构 ======================
class EnhancedCoalDustModel(nn.Module):
    """增强的煤尘浓度检测模型"""

    def __init__(self, num_classes=1):
        super(EnhancedCoalDustModel, self).__init__()

        # 使用更大的ConvNeXt V2 Base模型
        self.base_model = timm.create_model('convnextv2_base', pretrained=True, features_only=True)

        # 特征层通道数
        self.channels = [128, 256, 512, 1024]

        # 添加ECA注意力模块到每个阶段
        self.attention_modules = nn.ModuleList([
            EfficientChannelAttention(ch) for ch in self.channels
        ])

        # 特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(sum(self.channels), 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            EfficientChannelAttention(1024)
        )

        # 风速融合模块
        self.wind_fusion = WindFusion(in_features=1024, wind_dim=1)

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # 分类头（用于浓度段分类）
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10)  # 10个浓度段
        )

        # 自适应池化
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images, wind_speeds):
        # 提取图像特征
        features = self.base_model(images)

        # 应用注意力模块并调整尺寸
        attended_features = []
        for feat, att in zip(features, self.attention_modules):
            feat = att(feat)
            # 上采样到统一尺寸
            feat = nn.functional.interpolate(feat, size=(14, 14), mode='bilinear', align_corners=False)
            attended_features.append(feat)

        # 融合多尺度特征
        fused_features = torch.cat(attended_features, dim=1)
        fused_features = self.feature_fusion(fused_features)

        # 全局平均池化
        x = self.pool(fused_features)
        x = torch.flatten(x, 1)

        # 融合风速信息
        fused_with_wind = self.wind_fusion(x, wind_speeds)

        # 浓度回归
        concentration = self.regressor(fused_with_wind)

        # 浓度段分类
        segment = self.classifier(fused_with_wind)

        return concentration.squeeze(), segment


# ====================== 训练和评估函数 ======================
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=50):
    """模型训练函数"""
    train_losses, val_losses = [], []
    best_model_wts = None
    best_loss = float('inf')
    history = {'train_mae': [], 'val_mae': [], 'train_rmse': [], 'val_rmse': [], 'train_r2': [], 'val_r2': []}

    for epoch in range(num_epochs):
        print(f'纪元 {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_reg_loss = 0.0
            running_cls_loss = 0.0

            # 用于计算指标
            all_outputs = []
            all_labels = []

            # 遍历数据
            for inputs, wind_speeds, labels, segments in dataloaders[phase]:
                inputs = inputs.to(device)
                wind_speeds = wind_speeds.to(device)
                labels = labels.to(device).float()
                segments = segments.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, seg_preds = model(inputs, wind_speeds)

                    # 计算回归损失
                    reg_loss = criterion['reg'](outputs, labels)

                    # 计算分类损失
                    cls_loss = criterion['cls'](seg_preds, segments)

                    # 总损失 = 回归损失 + 分类损失
                    total_loss = reg_loss + 0.3 * cls_loss

                    # 反向传播+优化仅在训练阶段
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                # 统计指标
                running_loss += total_loss.item() * inputs.size(0)
                running_reg_loss += reg_loss.item() * inputs.size(0)
                running_cls_loss += cls_loss.item() * inputs.size(0)

                # 收集预测结果用于计算指标
                all_outputs.append(outputs.detach().cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            # 计算指标
            all_outputs = np.concatenate(all_outputs)
            all_labels = np.concatenate(all_labels)

            mae = np.mean(np.abs(all_outputs - all_labels))
            rmse = np.sqrt(np.mean((all_outputs - all_labels) ** 2))
            r2 = r2_score(all_labels, all_outputs)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_reg_loss = running_reg_loss / len(dataloaders[phase].dataset)
            epoch_cls_loss = running_cls_loss / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                history['train_mae'].append(mae)
                history['train_rmse'].append(rmse)
                history['train_r2'].append(r2)
            else:
                val_losses.append(epoch_loss)
                history['val_mae'].append(mae)
                history['val_rmse'].append(rmse)
                history['val_r2'].append(r2)

            print(f'{phase} 总损失: {epoch_loss:.4f} | 回归损失: {epoch_reg_loss:.4f} | 分类损失: {epoch_cls_loss:.4f}')
            print(f'{phase} MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}')

            # 深拷贝模型（如果是最佳模型）
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), 'best_model.pth')
                print('==> 保存最佳模型')

        # 更新学习率
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(history['val_mae'][-1])
            else:
                scheduler.step()

    print(f'最佳验证损失: {best_loss:.4f}')
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_by_segment(model, dataloader, segment_ranges):
    """按浓度分段评估模型"""
    model.eval()
    segment_metrics = {}

    # 初始化每个分段的指标
    for i in range(len(segment_ranges) - 1):
        segment_metrics[i] = {
            'mae': [],
            'rmse': [],
            'samples': 0,
            'correct': 0
        }

    with torch.no_grad():
        for inputs, wind_speeds, labels, segments in dataloader:
            inputs = inputs.to(device)
            wind_speeds = wind_speeds.to(device)
            labels = labels.cpu().numpy()
            segments = segments.cpu().numpy()

            outputs, seg_preds = model(inputs, wind_speeds)
            outputs = outputs.cpu().numpy()

            # 获取分类预测
            seg_preds = seg_preds.argmax(dim=1).cpu().numpy()

            # 计算每个样本的误差
            abs_errors = np.abs(outputs - labels)
            sq_errors = (outputs - labels) ** 2

            # 将样本分配到对应的浓度段
            for i in range(len(segment_ranges) - 1):
                mask = (labels >= segment_ranges[i]) & (labels < segment_ranges[i + 1])
                n_samples = np.sum(mask)

                if n_samples > 0:
                    segment_metrics[i]['mae'].extend(abs_errors[mask])
                    segment_metrics[i]['rmse'].extend(sq_errors[mask])
                    segment_metrics[i]['samples'] += n_samples

                    # 计算分类准确率
                    correct = np.sum(seg_preds[mask] == segments[mask])
                    segment_metrics[i]['correct'] += correct

    # 计算每个分段的指标
    segment_results = {}
    for i in range(len(segment_ranges) - 1):
        seg_data = segment_metrics[i]
        n_samples = seg_data['samples']

        if n_samples > 0:
            mae = np.mean(seg_data['mae']) if seg_data['mae'] else 0
            rmse = np.sqrt(np.mean(seg_data['rmse'])) if seg_data['rmse'] else 0
            accuracy = seg_data['correct'] / n_samples
        else:
            mae = rmse = accuracy = float('nan')

        seg_range = f'{segment_ranges[i]}-{segment_ranges[i + 1]}mg/m³'
        segment_results[seg_range] = {
            'MAE': mae,
            'RMSE': rmse,
            'Accuracy': accuracy,
            'Samples': n_samples
        }

    return segment_results


def plot_metrics(history):
    """绘制训练指标"""
    plt.figure(figsize=(15, 10))

    # 绘制损失
    plt.subplot(2, 2, 1)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # 绘制RMSE
    plt.subplot(2, 2, 2)
    plt.plot(history['train_rmse'], label='Train RMSE')
    plt.plot(history['val_rmse'], label='Validation RMSE')
    plt.title('RMSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()

    # 绘制R²
    plt.subplot(2, 2, 3)
    plt.plot(history['train_r2'], label='Train R²')
    plt.plot(history['val_r2'], label='Validation R²')
    plt.title('R² over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()

    # 绘制总损失
    plt.subplot(2, 2, 4)
    plt.plot(range(len(history['train_mae'])), [h for h in history['train_mae']], label='Train Loss')
    plt.plot(range(len(history['val_mae'])), [h for h in history['val_mae']], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()


# ====================== 主程序 ======================
if __name__ == '__main__':
    # 1. 准备数据集
    dataset = CoalDustDataset(root_dir='D:\Project\ConvNeXt V2\data', transform=data_transforms['train'])

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    batch_size = 16  # 减小batch size以适应更大的模型
    num_workers = 6 if torch.cuda.is_available() else 2

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    dataloaders = {'train': train_loader, 'val': val_loader}

    # 2. 初始化模型
    model = EnhancedCoalDustModel().to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数: {total_params:,} | 可训练参数: {trainable_params:,}')

    # 3. 设置训练参数
    criterion = {
        'reg': nn.HuberLoss(),  # 回归损失
        'cls': nn.CrossEntropyLoss()  # 分类损失
    }

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 4. 训练模型
    trained_model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs=100
    )

    # 5. 绘制训练指标
    plot_metrics(history)

    # 6. 按浓度分段评估
    segment_ranges = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    segment_results = evaluate_by_segment(trained_model, val_loader, segment_ranges)

    print("\n分段评估结果:")
    overall_mae, overall_rmse, total_samples = 0, 0, 0
    for segment, metrics in segment_results.items():
        if metrics['Samples'] > 0:
            print(f"浓度段 {segment}:")
            print(f"  样本数: {metrics['Samples']}")
            print(f"  MAE: {metrics['MAE']:.2f}")
            print(f"  RMSE: {metrics['RMSE']:.2f}")
            print(f"  准确率: {metrics['Accuracy']:.2%}")

            # 计算整体指标
            overall_mae += metrics['MAE'] * metrics['Samples']
            overall_rmse += metrics['RMSE'] * metrics['Samples']
            total_samples += metrics['Samples']

    # 计算整体MAE和RMSE
    if total_samples > 0:
        overall_mae /= total_samples
        overall_rmse /= total_samples
        print(f"\n整体MAE: {overall_mae:.2f}")
        print(f"整体RMSE: {overall_rmse:.2f}")

    # 7. 保存最终模型
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'segment_ranges': segment_ranges,
        'history': history
    }, 'enhanced_coal_dust_model.pth')
    print("模型已保存为 'enhanced_coal_dust_model.pth'")