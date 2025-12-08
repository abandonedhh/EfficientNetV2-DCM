import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from timm.models.swin_transformer_v2 import SwinTransformerV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, \
    classification_report
from PIL import Image

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)  # 保证可重复性


# 自定义数据集类
class CoalDustDataset(Dataset):
    def __init__(self, root_dir, transform=None, wind_speeds=None):
        """
        初始化煤粉数据集
        Args:
            root_dir: 数据集根目录
            transform: 图像变换
            wind_speeds: 可用风速列表
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.wind_speeds = wind_speeds if wind_speeds else sorted(
            [float(d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        # 创建风速到索引的映射
        self.wind_to_idx = {speed: idx for idx, speed in enumerate(self.wind_speeds)}

        # 收集所有样本
        for wind_dir in os.listdir(root_dir):
            wind_path = os.path.join(root_dir, wind_dir)
            if not os.path.isdir(wind_path):
                continue

            wind_speed = float(wind_dir)
            for conc_dir in os.listdir(wind_path):
                conc_path = os.path.join(wind_path, conc_dir)
                if not os.path.isdir(conc_path):
                    continue

                avg_concentration = float(conc_dir)
                for img_file in os.listdir(conc_path):
                    if img_file.endswith('.jpg'):
                        # 从文件名解析瞬时浓度
                        parts = img_file.split('_')
                        instant_conc = float(parts[-1].split('.')[0])

                        # 计算浓度段 (0-9)
                        conc_segment = min(int(instant_conc // 100), 9)

                        img_path = os.path.join(conc_path, img_file)
                        self.samples.append({
                            'image_path': img_path,
                            'instant_conc': instant_conc,
                            'avg_conc': avg_concentration,
                            'wind_speed': wind_speed,
                            'conc_segment': conc_segment
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 风速编码
        wind_speed = sample['wind_speed']
        wind_idx = self.wind_to_idx[wind_speed]
        wind_tensor = torch.tensor(wind_idx, dtype=torch.long)

        return {
            'image': image,
            'instant_conc': torch.tensor(sample['instant_conc'], dtype=torch.float32),
            'wind_speed': wind_tensor,
            'conc_segment': torch.tensor(sample['conc_segment'], dtype=torch.long)
        }


# 通道-空间注意力模块 (CBAM)
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_att(x)
        x_channel = x * channel_att

        # 空间注意力
        spatial_avg = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_att = self.spatial_att(spatial_att)
        x_out = x_channel * spatial_att

        return x_out


# 风速自适应池化模块
class WindAdaptivePooling(nn.Module):
    def __init__(self, feature_dim, num_winds):
        super(WindAdaptivePooling, self).__init__()
        # 为每个风速创建独立的自适应平均池化
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)) for _ in range(num_winds)
        ])

        # 风速权重学习
        self.wind_weights = nn.Parameter(torch.ones(num_winds))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, wind_indices):
        batch_size = x.size(0)
        pooled_features = []

        # 为每个样本应用对应的池化
        for i in range(batch_size):
            wind_idx = wind_indices[i]
            pooled = self.pools[wind_idx](x[i].unsqueeze(0))
            pooled_features.append(pooled)

        # 合并结果
        x_pooled = torch.cat(pooled_features, dim=0)

        # 应用风速权重
        weights = self.softmax(self.wind_weights)
        weighted_pool = torch.zeros_like(x_pooled)
        for i in range(batch_size):
            wind_idx = wind_indices[i]
            weighted_pool[i] = x_pooled[i] * weights[wind_idx]

        return weighted_pool.squeeze(-1).squeeze(-1)


# 主模型：增强型Swin Transformer V2
class EnhancedSwinDustModel(nn.Module):
    def __init__(self, num_winds=5, num_segments=10):
        super(EnhancedSwinDustModel, self).__init__()
        # 骨干网络: Swin Transformer V2 Base
        self.backbone = SwinTransformerV2(
            img_size=256,
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=16,
            pretrained_window_sizes=[12, 12, 12, 6]
        )

        # 获取骨干网络特征维度
        self.feature_dim = self.backbone.num_features

        # 注意力增强模块
        self.cbam = CBAM(self.feature_dim)

        # 风速自适应池化
        self.wind_pool = WindAdaptivePooling(self.feature_dim, num_winds)

        # 风速嵌入
        self.wind_embedding = nn.Embedding(num_winds, 64)

        # 回归头 (浓度值预测)
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 分类头 (浓度段预测)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_segments)
        )

    def forward(self, x, wind_speeds):
        # 骨干网络特征提取
        features = self.backbone.forward_features(x)

        # 调整特征维度 [B, C, H, W]
        features = features.permute(0, 3, 1, 2)

        # 应用CBAM注意力
        features = self.cbam(features)

        # 风速自适应池化
        pooled_features = self.wind_pool(features, wind_speeds)

        # 风速嵌入
        wind_emb = self.wind_embedding(wind_speeds)

        # 合并特征
        combined = torch.cat([pooled_features, wind_emb], dim=1)

        # 回归输出 (浓度值)
        conc_value = self.regressor(combined)

        # 分类输出 (浓度段)
        conc_segment = self.classifier(combined)

        return conc_value.squeeze(1), conc_segment


# 训练函数
def train_model(model, dataloaders, criterion_reg, criterion_cls, optimizer, num_epochs=50):
    best_loss = float('inf')
    history = {'train': {'loss': [], 'reg_loss': [], 'cls_loss': [], 'mae': []},
               'val': {'loss': [], 'reg_loss': [], 'cls_loss': [], 'mae': []}}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()  # 评估模式

            running_loss = 0.0
            running_reg_loss = 0.0
            running_cls_loss = 0.0
            running_mae = 0.0

            # 使用tqdm显示进度条
            pbar = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch + 1}')

            # 迭代数据
            for batch in pbar:
                inputs = batch['image'].to(device)
                conc_values = batch['instant_conc'].to(device)
                wind_speeds = batch['wind_speed'].to(device)
                conc_segments = batch['conc_segment'].to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    pred_values, pred_segments = model(inputs, wind_speeds)

                    # 计算损失
                    reg_loss = criterion_reg(pred_values, conc_values)
                    cls_loss = criterion_cls(pred_segments, conc_segments)
                    loss = 0.7 * reg_loss + 0.3 * cls_loss  # 加权损失

                    # 后向传播 + 优化 (仅训练阶段)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计指标
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_reg_loss += reg_loss.item() * batch_size
                running_cls_loss += cls_loss.item() * batch_size
                running_mae += mean_absolute_error(
                    conc_values.cpu().numpy(),
                    pred_values.detach().cpu().numpy()
                ) * batch_size

                # 更新进度条
                pbar.set_postfix({
                    'Loss': loss.item(),
                    'RegLoss': reg_loss.item(),
                    'ClsLoss': cls_loss.item()
                })

            # 计算epoch指标
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_reg_loss = running_reg_loss / len(dataloaders[phase].dataset)
            epoch_cls_loss = running_cls_loss / len(dataloaders[phase].dataset)
            epoch_mae = running_mae / len(dataloaders[phase].dataset)

            # 保存历史
            history[phase]['loss'].append(epoch_loss)
            history[phase]['reg_loss'].append(epoch_reg_loss)
            history[phase]['cls_loss'].append(epoch_cls_loss)
            history[phase]['mae'].append(epoch_mae)

            print(
                f'{phase} Loss: {epoch_loss:.4f} | RegLoss: {epoch_reg_loss:.4f} | ClsLoss: {epoch_cls_loss:.4f} | MAE: {epoch_mae:.2f}')

            # 保存最佳模型
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'Saved best model with val loss: {best_loss:.4f}')

    print('Training complete')
    return history


# 评估函数
def evaluate_model(model, dataloader, num_segments=10):
    model.eval()
    all_values = []
    all_pred_values = []
    all_segments = []
    all_pred_segments = []
    all_wind_speeds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            inputs = batch['image'].to(device)
            conc_values = batch['instant_conc'].cpu().numpy()
            wind_speeds = batch['wind_speed'].cpu().numpy()
            conc_segments = batch['conc_segment'].cpu().numpy()

            # 预测
            pred_values, pred_segments = model(inputs, batch['wind_speed'].to(device))

            # 收集结果
            all_values.extend(conc_values)
            all_pred_values.extend(pred_values.cpu().numpy())
            all_segments.extend(conc_segments)
            all_pred_segments.extend(torch.argmax(pred_segments, dim=1).cpu().numpy())
            all_wind_speeds.extend(wind_speeds)

    # 转换为numpy数组
    all_values = np.array(all_values)
    all_pred_values = np.array(all_pred_values)
    all_segments = np.array(all_segments)
    all_pred_segments = np.array(all_pred_segments)
    all_wind_speeds = np.array(all_wind_speeds)

    # 计算回归指标
    mae = mean_absolute_error(all_values, all_pred_values)
    rmse = np.sqrt(mean_squared_error(all_values, all_pred_values))
    r2 = r2_score(all_values, all_pred_values)

    # 计算分类指标
    accuracy = accuracy_score(all_segments, all_pred_segments)
    conf_matrix = confusion_matrix(all_segments, all_pred_segments)
    class_report = classification_report(all_segments, all_pred_segments, digits=4)

    # 按风速分组计算指标
    wind_metrics = {}
    unique_winds = np.unique(all_wind_speeds)
    for wind in unique_winds:
        wind_mask = (all_wind_speeds == wind)
        wind_mae = mean_absolute_error(all_values[wind_mask], all_pred_values[wind_mask])
        wind_rmse = np.sqrt(mean_squared_error(all_values[wind_mask], all_pred_values[wind_mask]))
        wind_r2 = r2_score(all_values[wind_mask], all_pred_values[wind_mask])
        wind_acc = accuracy_score(all_segments[wind_mask], all_pred_segments[wind_mask])

        wind_metrics[wind] = {
            'MAE': wind_mae,
            'RMSE': wind_rmse,
            'R2': wind_r2,
            'Accuracy': wind_acc
        }

    # 按浓度段分组计算指标
    segment_metrics = {}
    for seg in range(num_segments):
        seg_mask = (all_segments == seg)
        if np.sum(seg_mask) > 0:
            seg_mae = mean_absolute_error(all_values[seg_mask], all_pred_values[seg_mask])
            seg_rmse = np.sqrt(mean_squared_error(all_values[seg_mask], all_pred_values[seg_mask]))
            seg_r2 = r2_score(all_values[seg_mask], all_pred_values[seg_mask])
            seg_acc = accuracy_score(all_segments[seg_mask], all_pred_segments[seg_mask])

            segment_metrics[seg] = {
                'MAE': seg_mae,
                'RMSE': seg_rmse,
                'R2': seg_r2,
                'Accuracy': seg_acc,
                'Range': f'{seg * 100}-{(seg + 1) * 100}mg/m³'
            }

    return {
        'overall': {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Accuracy': accuracy
        },
        'wind_metrics': wind_metrics,
        'segment_metrics': segment_metrics,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }


# 主函数
def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    # 创建数据集
    dataset = CoalDustDataset(root_dir='D:\Project\Swin Transformer V2\data', transform=transform)

    # 划分训练集和验证集 (80%训练, 20%验证)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # 初始化模型
    model = EnhancedSwinDustModel(num_winds=5, num_segments=10)
    model = model.to(device)

    # 损失函数和优化器
    criterion_reg = nn.SmoothL1Loss()  # 回归损失
    criterion_cls = nn.CrossEntropyLoss()  # 分类损失

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练模型
    history = train_model(
        model, dataloaders, criterion_reg, criterion_cls,
        optimizer, num_epochs=100
    )

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 评估模型
    eval_results = evaluate_model(model, val_loader)

    # 打印评估结果
    print("\n===== Overall Metrics =====")
    print(f"MAE: {eval_results['overall']['MAE']:.2f}")
    print(f"RMSE: {eval_results['overall']['RMSE']:.2f}")
    print(f"R²: {eval_results['overall']['R2']:.4f}")
    print(f"Accuracy: {eval_results['overall']['Accuracy']:.4f}")

    print("\n===== Wind Speed Metrics =====")
    for wind, metrics in eval_results['wind_metrics'].items():
        print(f"Wind {wind}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, "
              f"R²={metrics['R2']:.4f}, Acc={metrics['Accuracy']:.4f}")

    print("\n===== Concentration Segment Metrics =====")
    for seg, metrics in eval_results['segment_metrics'].items():
        print(f"Segment {metrics['Range']}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, "
              f"R²={metrics['R2']:.4f}, Acc={metrics['Accuracy']:.4f}")

    print("\n===== Classification Report =====")
    print(eval_results['classification_report'])

    # 保存完整结果
    torch.save({
        'model_state': model.state_dict(),
        'history': history,
        'eval_results': eval_results
    }, 'full_results.pth')


if __name__ == '__main__':
    main()