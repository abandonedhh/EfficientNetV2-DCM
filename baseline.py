import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn.functional as F

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 数据集参数
DATA_ROOT = 'D:\Project\EfficientNetV2WithWind\data'  # 修改为您的数据集路径
WINDOW_SPEEDS = ['0.23', '0.52', '1.07', '1.48','1.97' ]  # 根据实际情况修改
CONCENTRATION_BINS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
BIN_LABELS = [f'{i}-{i + 100}' for i in range(0, 1000, 100)]

# 超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
IMG_SIZE = 224

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 自定义数据集类
class CoalDustDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # 遍历所有风速文件夹
        for wind_speed in WINDOW_SPEEDS:
            wind_path = os.path.join(root_dir, wind_speed)

            # 遍历所有浓度文件夹
            for conc_folder in os.listdir(wind_path):
                conc_path = os.path.join(wind_path, conc_folder)
                if not os.path.isdir(conc_path):
                    continue

                # 提取平均浓度值
                avg_concentration = float(conc_folder)

                # 遍历所有图像
                for img_name in os.listdir(conc_path):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(conc_path, img_name)

                        # 从文件名中提取瞬时浓度
                        parts = img_name.split('_')
                        instant_conc = float(parts[-1].split('.')[0])

                        # 将风速转换为数值
                        wind_value = float(wind_speed)

                        # 确定浓度区间
                        bin_idx = np.digitize(avg_concentration, CONCENTRATION_BINS) - 1
                        bin_idx = max(0, min(len(BIN_LABELS) - 1, bin_idx))

                        self.data.append({
                            'img_path': img_path,
                            'wind_speed': wind_value,
                            'avg_concentration': avg_concentration,
                            'instant_concentration': instant_conc,
                            'bin_idx': bin_idx
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 加载图像
        image = Image.open(item['img_path']).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 获取风速并转换为张量
        wind_speed = torch.tensor([item['wind_speed']], dtype=torch.float32)

        # 获取浓度值
        concentration = torch.tensor([item['avg_concentration']], dtype=torch.float32)

        # 获取浓度区间标签
        bin_label = torch.tensor(item['bin_idx'], dtype=torch.long)

        return image, wind_speed, concentration, bin_label


# 创建数据集
dataset = CoalDustDataset(DATA_ROOT, transform=transform)

# 划分数据集 (8:1:1)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f'训练集大小: {len(train_dataset)}')
print(f'验证集大小: {len(val_dataset)}')
print(f'测试集大小: {len(test_dataset)}')

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# 定义模型
class EfficientNetV2WithWind(nn.Module):
    def __init__(self, num_bins=10):
        super(EfficientNetV2WithWind, self).__init__()

        # 加载预训练的EfficientNetV2-S
        self.backbone = models.efficientnet_v2_s(pretrained=True)

        # 获取特征提取器的输出维度
        in_features = self.backbone.classifier[1].in_features

        # 移除原始分类头
        self.backbone.classifier = nn.Identity()

        # 风速处理分支
        self.wind_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 融合特征后的分类头
        self.fc = nn.Sequential(
            nn.Linear(in_features + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 回归输出浓度值
        )

        # 分类头（用于浓度区间分类）
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_bins)
        )

    def forward(self, x, wind_speed):
        # 提取图像特征
        img_features = self.backbone(x)

        # 处理风速特征
        wind_features = self.wind_fc(wind_speed)

        # 融合特征
        combined = torch.cat([img_features, wind_features], dim=1)

        # 回归输出
        regression_output = self.fc(combined)

        # 分类输出
        classification_output = self.classifier(combined)

        return regression_output, classification_output


# 初始化模型
model = EfficientNetV2WithWind(num_bins=len(BIN_LABELS)).to(device)

# 定义损失函数和优化器
criterion_reg = nn.MSELoss()  # 回归任务使用MSE损失
criterion_cls = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


# 训练函数
def train_model(model, train_loader, val_loader, criterion_reg, criterion_cls, optimizer, scheduler, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_reg_loss = 0.0
        train_cls_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [训练]')
        for images, wind_speeds, concentrations, bin_labels in progress_bar:
            images = images.to(device)
            wind_speeds = wind_speeds.to(device)
            concentrations = concentrations.to(device)
            bin_labels = bin_labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            reg_output, cls_output = model(images, wind_speeds)

            # 计算损失
            loss_reg = criterion_reg(reg_output, concentrations)
            loss_cls = criterion_cls(cls_output, bin_labels)
            loss = loss_reg + 0.5 * loss_cls  # 组合损失

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_reg_loss += loss_reg.item() * images.size(0)
            train_cls_loss += loss_cls.item() * images.size(0)

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'RegLoss': f'{loss_reg.item():.4f}',
                'ClsLoss': f'{loss_cls.item():.4f}'
            })

        epoch_loss = running_loss / len(train_loader.dataset)
        train_reg_loss = train_reg_loss / len(train_loader.dataset)
        train_cls_loss = train_cls_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_reg_loss = 0.0
        val_cls_loss = 0.0

        with torch.no_grad():
            for images, wind_speeds, concentrations, bin_labels in val_loader:
                images = images.to(device)
                wind_speeds = wind_speeds.to(device)
                concentrations = concentrations.to(device)
                bin_labels = bin_labels.to(device)

                reg_output, cls_output = model(images, wind_speeds)

                loss_reg = criterion_reg(reg_output, concentrations)
                loss_cls = criterion_cls(cls_output, bin_labels)
                loss = loss_reg + 0.5 * loss_cls

                val_loss += loss.item() * images.size(0)
                val_reg_loss += loss_reg.item() * images.size(0)
                val_cls_loss += loss_cls.item() * images.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_reg_loss = val_reg_loss / len(val_loader.dataset)
        val_cls_loss = val_cls_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # 学习率调整
        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'训练损失: {epoch_loss:.4f} (回归: {train_reg_loss:.4f}, 分类: {train_cls_loss:.4f})')
        print(f'验证损失: {val_loss:.4f} (回归: {val_reg_loss:.4f}, 分类: {val_cls_loss:.4f})')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('模型已保存!')
        print()

    return train_losses, val_losses


# 训练模型
print("开始训练模型...")
train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion_reg, criterion_cls,
    optimizer, scheduler, NUM_EPOCHS
)

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.title('训练和验证损失曲线')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()


# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_bin_preds = []
    all_bin_labels = []
    all_wind_speeds = []

    with torch.no_grad():
        for images, wind_speeds, concentrations, bin_labels in tqdm(test_loader, desc='测试中'):
            images = images.to(device)
            wind_speeds = wind_speeds.to(device)

            reg_output, cls_output = model(images, wind_speeds)

            # 获取预测结果
            preds = reg_output.cpu().numpy().flatten()
            bin_preds = torch.argmax(cls_output, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(concentrations.numpy().flatten())
            all_bin_preds.extend(bin_preds)
            all_bin_labels.extend(bin_labels.numpy())
            all_wind_speeds.extend(wind_speeds.cpu().numpy().flatten())

    return np.array(all_preds), np.array(all_labels), np.array(all_bin_preds), np.array(all_bin_labels), np.array(
        all_wind_speeds)


# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 评估模型
print("开始评估模型...")
preds, labels, bin_preds, bin_labels, wind_speeds = evaluate_model(model, test_loader)

# 计算总体评估指标
mae = mean_absolute_error(labels, preds)
mse = mean_squared_error(labels, preds)
rmse = np.sqrt(mse)
r2 = r2_score(labels, preds)

print(f'总体评估指标:')
print(f'MAE: {mae:.2f} mg/m³')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f} mg/m³')
print(f'R²: {r2:.4f}')

# 计算分类准确率
bin_accuracy = np.mean(bin_preds == bin_labels)
print(f'浓度区间分类准确率: {bin_accuracy:.4f}')

# 计算各浓度区间的评估指标
bin_metrics = {}
for bin_idx in range(len(BIN_LABELS)):
    mask = bin_labels == bin_idx
    if np.sum(mask) > 0:
        bin_mae = mean_absolute_error(labels[mask], preds[mask])
        bin_rmse = np.sqrt(mean_squared_error(labels[mask], preds[mask]))
        bin_r2 = r2_score(labels[mask], preds[mask])
        bin_accuracy = np.mean(bin_preds[mask] == bin_labels[mask])

        bin_metrics[bin_idx] = {
            'MAE': bin_mae,
            'RMSE': bin_rmse,
            'R2': bin_r2,
            'Accuracy': bin_accuracy,
            'Samples': np.sum(mask)
        }

# 打印各浓度区间的评估指标
print('\n各浓度区间评估指标:')
for bin_idx, metrics in bin_metrics.items():
    print(f'{BIN_LABELS[bin_idx]} mg/m³ ({metrics["Samples"]}个样本):')
    print(
        f'  MAE: {metrics["MAE"]:.2f}, RMSE: {metrics["RMSE"]:.2f}, R²: {metrics["R2"]:.4f}, 准确率: {metrics["Accuracy"]:.4f}')

# 可视化预测结果 vs 真实值
plt.figure(figsize=(10, 8))
plt.scatter(labels, preds, alpha=0.6, c=wind_speeds, cmap='viridis')
plt.colorbar(label='Wind speed(m/s)')
plt.plot([0, 1000], [0, 1000], 'r--', label='Ideal value')
plt.xlabel('True concentration(mg/m³)')
plt.ylabel('Regression concentration(mg/m³)')
#plt.title('预测浓度 vs 真实浓度')
plt.legend()
plt.grid(True)
plt.savefig('predictions_vs_actuals.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制各浓度区间的MAE
plt.figure(figsize=(12, 6))
bin_names = [BIN_LABELS[i] for i in bin_metrics.keys()]
mae_values = [metrics['MAE'] for metrics in bin_metrics.values()]
sample_counts = [metrics['Samples'] for metrics in bin_metrics.values()]

plt.bar(bin_names, mae_values, alpha=0.7)
plt.xlabel('Concentration range(mg/m³)')
plt.ylabel('MAE (mg/m³)')
#plt.title('各浓度区间的平均绝对误差(MAE)')
plt.xticks(rotation=45)

# 在柱状图上添加样本数量标注
for i, count in enumerate(sample_counts):
    plt.text(i, mae_values[i] + 5, f'n={count}', ha='center')

plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('mae_by_bin.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制各浓度区间的准确率
plt.figure(figsize=(12, 6))
accuracy_values = [metrics['Accuracy'] for metrics in bin_metrics.values()]

plt.bar(bin_names, accuracy_values, alpha=0.7, color='green')
plt.xlabel('Concentration range(mg/m³)')
plt.ylabel('Accuracy')
#plt.title('各浓度区间的分类准确率')
plt.xticks(rotation=45)

# 在柱状图上添加样本数量标注
for i, count in enumerate(sample_counts):
    plt.text(i, accuracy_values[i] + 0.01, f'n={count}', ha='center')

plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('accuracy_by_bin.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存评估结果到CSV文件
results_df = pd.DataFrame({
    'True concentration': labels,
    'Regression concentration': preds,
    'Concentration range label': bin_labels,
    'Regression concentration range': bin_preds,
    'Wind speed': wind_speeds
})
results_df.to_csv('evaluation_results.csv', index=False)

print("评估完成!结果已保存到evaluation_results.csv")