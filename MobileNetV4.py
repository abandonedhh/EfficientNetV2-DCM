import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import timm

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 自定义数据集类
class CoalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化煤粉数据集
        root_dir: 数据集根目录
        transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # 遍历风速文件夹
        wind_folders = sorted(glob.glob(os.path.join(root_dir, '*')))
        for wind_dir in wind_folders:
            try:
                wind_speed = float(os.path.basename(wind_dir))
            except ValueError:
                continue

            # 遍历浓度子文件夹
            conc_folders = sorted(glob.glob(os.path.join(wind_dir, '*')))
            for conc_dir in conc_folders:
                try:
                    avg_concentration = float(os.path.basename(conc_dir))
                except ValueError:
                    continue

                # 收集所有图像文件
                img_files = glob.glob(os.path.join(conc_dir, '*.jpg')) + \
                            glob.glob(os.path.join(conc_dir, '*.png'))

                for img_path in img_files:
                    # 更健壮的文件名解析
                    filename = os.path.basename(img_path)
                    # 移除扩展名
                    filename_no_ext = os.path.splitext(filename)[0]

                    # 分割文件名各部分
                    parts = filename_no_ext.split('_')

                    # 确保有足够的部分
                    if len(parts) < 3:
                        print(f"警告: 文件名格式异常 - {filename}，跳过")
                        continue

                    # 尝试解析瞬时浓度
                    try:
                        # 最后一部分应该是瞬时浓度
                        conc_str = parts[-1]
                        # 处理可能的小数点
                        if '.' in conc_str:
                            instant_conc = float(conc_str)
                        else:
                            # 尝试解析整数
                            instant_conc = float(conc_str)
                    except ValueError:
                        # 尝试从倒数第二部分解析
                        try:
                            conc_str = parts[-2]
                            if '.' in conc_str:
                                instant_conc = float(conc_str)
                            else:
                                instant_conc = float(conc_str)
                        except (ValueError, IndexError):
                            print(f"警告: 无法解析浓度值 - {filename}，跳过")
                            continue

                    self.samples.append((img_path, instant_conc, wind_speed))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, concentration, wind_speed = self.samples[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 归一化浓度 (0-1000 -> 0-1)
        normalized_conc = concentration / 1000.0

        # 转换为张量
        wind_speed_tensor = torch.tensor(wind_speed, dtype=torch.float32).unsqueeze(0)

        return image, normalized_conc, wind_speed_tensor


# 增强的CBAM注意力模块
class EnhancedCBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_att(x)
        x_channel = x * channel_att

        # 空间注意力
        max_pool = torch.max(x_channel, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        x_out = x_channel * spatial_att

        return x_out


# 风速融合模块
class WindFusion(nn.Module):
    def __init__(self, in_features, wind_dim):
        super().__init__()
        self.wind_fc = nn.Sequential(
            nn.Linear(wind_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_features),
            nn.Sigmoid()  # 门控机制
        )

    def forward(self, features, wind_speed):
        # 处理风速
        wind_emb = self.wind_fc(wind_speed)

        # 特征融合 (门控机制)
        fused_features = features * wind_emb.unsqueeze(-1).unsqueeze(-1)

        return fused_features


# 改进的模型（兼容MobileNetV4或EfficientNetV2）
class CoalNet(nn.Module):
    def __init__(self, num_classes=1, wind_dim=1, model_name='mobilenetv4_conv_small'):
        super().__init__()
        # 创建基础模型
        if model_name.startswith('mobilenetv4'):
            # 尝试创建MobileNetV4
            try:
                self.base_model = timm.create_model(
                    model_name,
                    pretrained=True,
                    features_only=True,
                    out_indices=(1, 2, 3, 4)
                )
                print(f"成功创建MobileNetV4: {model_name}")
            except:
                # 如果失败则使用EfficientNetV2作为替代
                print("MobileNetV4创建失败，使用EfficientNetV2作为替代")
                self.base_model = timm.create_model(
                    'tf_efficientnetv2_s',
                    pretrained=True,
                    features_only=True,
                    out_indices=(1, 2, 3, 4)
                )
        else:
            # 使用EfficientNetV2
            self.base_model = timm.create_model(
                model_name,
                pretrained=True,
                features_only=True,
                out_indices=(1, 2, 3, 4)
            )

        # 获取特征通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.base_model(dummy)
            channels = [f.shape[1] for f in features]

        # 添加注意力模块
        self.attentions = nn.ModuleList([
            EnhancedCBAM(channels[0]),
            EnhancedCBAM(channels[1]),
            EnhancedCBAM(channels[2]),
            EnhancedCBAM(channels[3])
        ])

        # 风速融合模块
        self.wind_fusion = WindFusion(channels[-1], wind_dim)

        # 回归头
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, wind_speed):
        # 特征提取
        features = self.base_model(x)

        # 应用注意力机制
        for i in range(len(features)):
            features[i] = self.attentions[i](features[i])

        # 取最高层特征
        x = features[-1]

        # 风速融合
        x = self.wind_fusion(x, wind_speed)

        # 回归预测
        conc_pred = self.regressor(x)

        return conc_pred


# 多尺度图像增强
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ], p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# 按浓度分段计算指标
def segment_evaluation(y_true, y_pred):
    # 创建浓度分段 (0-1000mg/m3)
    segments = np.arange(0, 1001, 100)
    segment_results = []

    for i in range(len(segments) - 1):
        low = segments[i]
        high = segments[i + 1]

        # 获取当前分段内的样本
        mask = (y_true >= low) & (y_true < high)
        if np.sum(mask) == 0:
            continue

        seg_true = y_true[mask]
        seg_pred = y_pred[mask]

        # 计算指标
        mae = mean_absolute_error(seg_true, seg_pred)
        rmse = np.sqrt(np.mean((seg_true - seg_pred) ** 2))
        r2 = r2_score(seg_true, seg_pred)

        segment_results.append({
            'segment': f"{low}-{high}",
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'count': len(seg_true)
        })

    # 整体指标
    total_mae = mean_absolute_error(y_true, y_pred)
    total_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    total_r2 = r2_score(y_true, y_pred)

    return segment_results, total_mae, total_rmse, total_r2


# 主训练函数
def main():
    # 数据集参数
    data_dir = "D:\Project\MobileNetV4\data"  # 修改为你的数据集路径
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-4

    # 创建数据集
    train_transform, val_transform = get_transforms()
    full_dataset = CoalDataset(data_dir, transform=train_transform)

    # 数据集划分 (80%训练, 20%验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 设置验证集使用val_transform
    val_dataset.dataset.transform = val_transform

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = CoalNet(model_name='mobilenetv4_conv_small').to(device)

    # 损失函数和优化器
    criterion = nn.SmoothL1Loss()  # 对回归任务鲁棒的损失函数
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练循环
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0

        for images, concs, winds in train_loader:
            images = images.to(device)
            concs = concs.to(device).view(-1, 1)  # 确保正确形状
            winds = winds.to(device)

            # 前向传播
            outputs = model(images, winds)
            loss = criterion(outputs, concs)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, concs, winds in val_loader:
                images = images.to(device)
                concs = concs.to(device).view(-1, 1)
                winds = winds.to(device)

                outputs = model(images, winds)
                loss = criterion(outputs, concs)

                val_loss += loss.item() * images.size(0)

                # 收集预测结果
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(concs.cpu().numpy().flatten())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # 更新学习率
        scheduler.step(epoch_val_loss)

        # 打印统计信息
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.6f}, "
              f"Val Loss: {epoch_val_loss:.6f}")

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_coal_model.pth')
            print("保存新最佳模型")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()

    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('best_coal_model.pth'))
    model.eval()

    all_preds = []
    all_targets = []
    wind_speeds = []

    with torch.no_grad():
        for images, concs, winds in val_loader:
            images = images.to(device)
            concs = concs.cpu().numpy().flatten() * 1000  # 转换为实际浓度
            winds = winds.cpu().numpy().flatten()

            outputs = model(images, winds)
            preds = outputs.cpu().numpy().flatten() * 1000  # 转换为实际浓度

            all_preds.extend(preds)
            all_targets.extend(concs)
            wind_speeds.extend(winds)

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    wind_speeds = np.array(wind_speeds)

    # 整体评估
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(np.mean((all_targets - all_preds) ** 2))
    r2 = r2_score(all_targets, all_preds)

    print(f"\n最终评估:")
    print(f"整体MAE: {mae:.2f} mg/m³")
    print(f"整体RMSE: {rmse:.2f} mg/m³")
    print(f"整体R²: {r2:.4f}")

    # 分段评估
    segment_results, total_mae, total_rmse, total_r2 = segment_evaluation(all_targets, all_preds)

    print("\n按浓度分段评估:")
    for result in segment_results:
        print(f"分段 {result['segment']} (n={result['count']}): "
              f"MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}, R²={result['r2']:.4f}")

    # 风速影响分析
    print("\n风速影响分析:")
    unique_winds = np.unique(wind_speeds)
    for wind in unique_winds:
        mask = wind_speeds == wind
        wind_mae = mean_absolute_error(all_targets[mask], all_preds[mask])
        print(f"风速 {wind}m/s: MAE={wind_mae:.2f} mg/m³")

    # 可视化预测结果
    plt.figure(figsize=(10, 8))
    plt.scatter(all_targets, all_preds, alpha=0.6)
    plt.plot([0, 1000], [0, 1000], 'r--')
    plt.xlabel('实际浓度 (mg/m³)')
    plt.ylabel('预测浓度 (mg/m³)')
    plt.title('预测浓度 vs 实际浓度')
    plt.grid(True)
    plt.savefig('prediction_scatter.png')
    plt.close()


if __name__ == "__main__":
    main()