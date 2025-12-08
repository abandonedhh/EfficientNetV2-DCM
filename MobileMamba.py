import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import re
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import multiprocessing

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 自定义数据集类
class CoalDustDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_wind_speed=2.0, max_concentration=1000.0):
        self.root_dir = root_dir
        self.transform = transform
        self.max_wind_speed = max_wind_speed
        self.max_concentration = max_concentration
        self.data = []

        # 遍历目录结构收集数据
        wind_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for wind_dir in wind_dirs:
            wind_speed = float(wind_dir)
            wind_path = os.path.join(root_dir, wind_dir)

            conc_dirs = [d for d in os.listdir(wind_path) if os.path.isdir(os.path.join(wind_path, d))]
            for conc_dir in conc_dirs:
                conc_path = os.path.join(wind_path, conc_dir)

                img_files = [f for f in os.listdir(conc_path) if f.endswith('.jpg')]
                for img_file in img_files:
                    # 从文件名解析瞬时浓度
                    match = re.search(r'_(\d+\.\d+)\.jpg$', img_file)
                    if match:
                        instant_conc = float(match.group(1))
                        img_path = os.path.join(conc_path, img_file)

                        # 存储路径、风速和浓度
                        self.data.append({
                            'image_path': img_path,
                            'wind_speed': wind_speed,
                            'concentration': instant_conc
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')

        # 归一化风速和浓度
        wind_speed = sample['wind_speed'] / self.max_wind_speed
        concentration = sample['concentration'] / self.max_concentration

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 计算浓度分段 (0-9)
        conc_segment = min(int(sample['concentration'] // 100), 9)

        return {
            'image': image,
            'wind_speed': torch.tensor(wind_speed, dtype=torch.float32),
            'concentration': torch.tensor(concentration, dtype=torch.float32),
            'conc_segment': torch.tensor(conc_segment, dtype=torch.long)
        }


# 自定义MobileMamba模块
class MobileMambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_ratio=4):
        super().__init__()
        expanded_channels = in_channels * expansion_ratio

        # 深度可分离卷积
        self.dw_conv = nn.Conv2d(
            in_channels, expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(expanded_channels)

        # 通道混合卷积
        self.pw_conv1 = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.dw_conv(x)
        out = F.relu6(self.bn1(out))

        out = self.pw_conv1(out)
        out = self.bn2(out)

        # 应用通道注意力
        att = self.attention(out)
        out = out * att

        # 添加残差连接
        out += self.shortcut(residual)
        return F.relu6(out)


# 带风速融合的MobileMamba模型
class WindConditionedMobileMamba(nn.Module):
    def __init__(self, num_segments=10):
        super().__init__()

        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        # MobileMamba块序列
        self.blocks = nn.Sequential(
            MobileMambaBlock(32, 64, stride=2),  # /4
            MobileMambaBlock(64, 128, stride=2),  # /8
            MobileMambaBlock(128, 256, stride=2),  # /16
            MobileMambaBlock(256, 512, stride=2),  # /32
        )

        # 风速处理分支
        self.wind_branch = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # 图像特征处理
        self.image_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 融合分支
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 128),  # 图像特征 + 风速特征
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 回归头（浓度预测）
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 分类头（浓度分段）
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_segments)
        )

        # 图像预处理分支（多尺度特征）
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, image, wind_speed):
        # 图像预处理
        preprocessed = self.preprocess(image)

        # 主特征提取
        x = self.initial_conv(image)
        x = self.blocks(x)
        img_features = self.image_features(x)

        # 风速特征
        wind_features = self.wind_branch(wind_speed.unsqueeze(1))

        # 特征融合
        combined = torch.cat((img_features, wind_features), dim=1)
        fused = self.fusion(combined)

        # 双任务输出
        concentration = self.regression_head(fused)
        segment = self.classification_head(fused)

        return concentration, segment


if __name__ == '__main__':
    # 解决多进程启动问题
    multiprocessing.freeze_support()

    # 数据增强和转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集 - 替换为您的实际路径
    dataset = CoalDustDataset(root_dir='D:\Project\MobileMamba\data', transform=transform)

    # 数据集分割
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 创建数据加载器 - 减少num_workers避免多进程问题
    batch_size = 64
    num_workers = 0  # 设置为0避免多进程问题

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    # 初始化模型
    model = WindConditionedMobileMamba(num_segments=10).to(device)

    # 损失函数和优化器
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 混合精度训练
    scaler = GradScaler()


    # 训练函数
    def train_epoch(model, loader, optimizer, epoch):
        model.train()
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        total_loss = 0.0

        progress_bar = tqdm(loader, desc=f'Epoch {epoch + 1} [Train]', leave=False)
        for batch in progress_bar:
            images = batch['image'].to(device)
            wind_speeds = batch['wind_speed'].to(device)
            concentrations = batch['concentration'].to(device).unsqueeze(1)
            segments = batch['conc_segment'].to(device)

            optimizer.zero_grad()

            with autocast():
                pred_conc, pred_segments = model(images, wind_speeds)
                reg_loss = reg_criterion(pred_conc, concentrations)
                cls_loss = cls_criterion(pred_segments, segments)
                loss = reg_loss + cls_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_reg_loss += reg_loss.item() * images.size(0)
            total_cls_loss += cls_loss.item() * images.size(0)
            total_loss += loss.item() * images.size(0)

            progress_bar.set_postfix({
                'RegLoss': f'{reg_loss.item():.4f}',
                'ClsLoss': f'{cls_loss.item():.4f}'
            })

        avg_reg_loss = total_reg_loss / len(loader.dataset)
        avg_cls_loss = total_cls_loss / len(loader.dataset)
        avg_loss = total_loss / len(loader.dataset)

        return avg_reg_loss, avg_cls_loss, avg_loss


    # 验证函数
    def validate(model, loader):
        model.eval()
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        total_loss = 0.0

        all_concentrations = []
        all_pred_concentrations = []
        all_segments = []
        all_pred_segments = []

        with torch.no_grad():
            progress_bar = tqdm(loader, desc='[Validation]', leave=False)
            for batch in progress_bar:
                images = batch['image'].to(device)
                wind_speeds = batch['wind_speed'].to(device)
                concentrations = batch['concentration'].to(device).unsqueeze(1)
                segments = batch['conc_segment'].to(device)

                pred_conc, pred_segments = model(images, wind_speeds)

                reg_loss = reg_criterion(pred_conc, concentrations)
                cls_loss = cls_criterion(pred_segments, segments)
                loss = reg_loss + cls_loss

                total_reg_loss += reg_loss.item() * images.size(0)
                total_cls_loss += cls_loss.item() * images.size(0)
                total_loss += loss.item() * images.size(0)

                # 收集预测结果
                all_concentrations.extend(concentrations.cpu().numpy().flatten())
                all_pred_concentrations.extend(pred_conc.cpu().numpy().flatten())
                all_segments.extend(segments.cpu().numpy())
                all_pred_segments.extend(torch.argmax(pred_segments, dim=1).cpu().numpy())

        avg_reg_loss = total_reg_loss / len(loader.dataset)
        avg_cls_loss = total_cls_loss / len(loader.dataset)
        avg_loss = total_loss / len(loader.dataset)

        # 计算回归指标
        mse = mean_squared_error(all_concentrations, all_pred_concentrations)
        mae = mean_absolute_error(all_concentrations, all_pred_concentrations)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_concentrations, all_pred_concentrations)

        # 计算分类指标
        conf_matrix = confusion_matrix(all_segments, all_pred_segments)
        segment_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

        return (avg_reg_loss, avg_cls_loss, avg_loss,
                mse, mae, rmse, r2,
                segment_accuracy, conf_matrix)


    # 训练循环
    num_epochs = 100
    best_val_loss = float('inf')
    history = {
        'train_reg_loss': [], 'train_cls_loss': [], 'train_loss': [],
        'val_reg_loss': [], 'val_cls_loss': [], 'val_loss': [],
        'val_mse': [], 'val_mae': [], 'val_rmse': [], 'val_r2': [],
        'val_segment_acc': []
    }

    for epoch in range(num_epochs):
        # 训练
        train_reg_loss, train_cls_loss, train_loss = train_epoch(model, train_loader, optimizer, epoch)

        # 验证
        val_results = validate(model, val_loader)
        val_reg_loss, val_cls_loss, val_loss, val_mse, val_mae, val_rmse, val_r2, val_seg_acc, conf_matrix = val_results

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        history['train_reg_loss'].append(train_reg_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_loss'].append(train_loss)
        history['val_reg_loss'].append(val_reg_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        history['val_segment_acc'].append(val_seg_acc)

        # 打印结果
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} (Reg: {train_reg_loss:.4f}, Cls: {train_cls_loss:.4f})')
        print(f'Val Loss: {val_loss:.4f} (Reg: {val_reg_loss:.4f}, Cls: {val_cls_loss:.4f})')
        print(f'Val Metrics: MSE={val_mse:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R²={val_r2:.4f}')
        print(f'Segment Accuracy: {val_seg_acc:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved best model')

    # 在测试集上评估最佳模型
    print('Testing best model...')
    model.load_state_dict(torch.load('best_model.pth'))
    test_results = validate(model, test_loader)
    _, _, test_loss, test_mse, test_mae, test_rmse, test_r2, test_seg_acc, test_conf_matrix = test_results

    print('\nFinal Test Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}')
    print(f'Segment Accuracy: {test_seg_acc:.4f}')

    # 可视化训练过程
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 回归指标
    plt.subplot(2, 2, 2)
    plt.plot(history['val_mse'], label='MSE')
    plt.plot(history['val_mae'], label='MAE')
    plt.plot(history['val_rmse'], label='RMSE')
    plt.title('Validation Regression Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    # R² 分数
    plt.subplot(2, 2, 3)
    plt.plot(history['val_r2'], label='R² Score', color='green')
    plt.title('Validation R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.ylim(0, 1)
    plt.legend()

    # 分段准确率
    plt.subplot(2, 2, 4)
    plt.plot(history['val_segment_acc'], label='Segment Accuracy', color='purple')
    plt.title('Validation Segment Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=test_conf_matrix,
        display_labels=[f'{i * 100}-{(i + 1) * 100}' for i in range(10)]
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Concentration Segment Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # 预测vs真实值散点图
    test_concentrations = []
    test_pred_concentrations = []

    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Collecting test predictions')
        for batch in progress_bar:
            images = batch['image'].to(device)
            wind_speeds = batch['wind_speed'].to(device)

            pred_conc, _ = model(images, wind_speeds)

            test_concentrations.extend(batch['concentration'].numpy().flatten() * 1000)
            test_pred_concentrations.extend(pred_conc.cpu().numpy().flatten() * 1000)

    plt.figure(figsize=(10, 8))
    plt.scatter(test_concentrations, test_pred_concentrations, alpha=0.5)
    plt.plot([0, 1000], [0, 1000], 'r--')
    plt.title('Actual vs Predicted Concentration')
    plt.xlabel('Actual Concentration (mg/m³)')
    plt.ylabel('Predicted Concentration (mg/m³)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.show()