
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image
from timm.models.maxxvit import maxvit_base_tf_224
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler



# 配置参数
class Config:
    data_root = "D:\Project\MaxViT\data"  # 替换为你的数据集路径
    wind_speeds = [0.23, 0.52, 1.07, 1.48, 1.97]  # 根据实际风速修改
    batch_size = 32
    num_workers = 4
    lr = 1e-4
    epochs = 50
    image_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "best_model.pth"
    num_bins = 10  # 浓度分段数量
    bin_edges = np.linspace(0, 1000, num_bins + 1)  # 0-1000mg/m³分10段


# 自定义数据集类
class CoalDustDataset(Dataset):
    def __init__(self, root, wind_speeds, transform=None):
        self.data = []
        self.transform = transform
        self.wind_speeds = wind_speeds
        self.wind_scaler = StandardScaler()
        wind_values = []

        # 遍历所有风速文件夹
        for wind_speed in wind_speeds:
            wind_dir = os.path.join(root, str(wind_speed))

            # 遍历浓度子文件夹
            for conc_dir in os.listdir(wind_dir):
                conc_path = os.path.join(wind_dir, conc_dir)
                if not os.path.isdir(conc_path):
                    continue

                # 提取平均浓度值
                avg_concentration = float(conc_dir)

                # 处理每张图片
                for img_name in os.listdir(conc_path):
                    if img_name.endswith(".jpg"):
                        img_path = os.path.join(conc_path, img_name)

                        # 从文件名提取瞬时浓度
                        match = re.search(r'_(\d+\.\d+)\.jpg$', img_name)
                        if match:
                            inst_concentration = float(match.group(1))

                            # 存储样本信息
                            self.data.append({
                                "image_path": img_path,
                                "wind_speed": wind_speed,
                                "concentration": inst_concentration,
                                "avg_concentration": avg_concentration
                            })
                            wind_values.append([wind_speed])

        # 训练风速标准化器
        if wind_values:
            self.wind_scaler.fit(np.array(wind_values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 加载图像
        image = read_image(sample["image_path"]).float() / 255.0

        # 数据增强
        if self.transform:
            image = self.transform(image)

        # 风速处理 (标准化)
        wind_speed = torch.tensor(
            self.wind_scaler.transform([[sample["wind_speed"]]]),
            dtype=torch.float32
        )

        # 浓度值 (归一化到0-1)
        concentration = torch.tensor(
            sample["concentration"] / 1000.0,  # 标量值，不是向量
            dtype=torch.float32
        )

        return image, wind_speed, concentration


# 带风速融合的MaxViT模型
class MaxViTWithWind(nn.Module):
    def __init__(self, num_wind_features=1, num_outputs=1):
        super().__init__()

        # 图像特征提取 (使用预训练的MaxViT)
        self.img_backbone = maxvit_base_tf_224(pretrained=True)

        # 移除原始分类头并获取特征维度
        self.img_backbone.reset_classifier(0)
        self.img_feature_dim = self.img_backbone.num_features  # 动态获取特征维度

        # 风速特征处理
        self.wind_fc = nn.Sequential(
            nn.Linear(num_wind_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # 特征融合与回归
        self.fusion = nn.Sequential(
            nn.Linear(self.img_feature_dim + 256, 512),  # 使用动态特征维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
            nn.Sigmoid()  # 输出在0-1之间
        )

    def forward(self, img, wind):
        if wind.dim() == 3:  # 如果是 [B, 1, 1]
            wind = wind.squeeze(2)  # → [B, 1]
        elif wind.dim() == 1:  # 如果是 [B]（batch size=1 时可能）
            wind = wind.unsqueeze(1)  # → [B, 1]

            # 现在 wind 保证是 [B, 1]
        assert wind.dim() == 2, f"Wind must be 2D [B, 1], got {wind.shape}"
        # 提取图像特征
        img_features = self.img_backbone(img)

        # 处理风速特征
        wind_features = self.wind_fc(wind)

        # 特征融合
        combined = torch.cat((img_features, wind_features), dim=1)

        # 浓度预测
        concentration_pred = self.fusion(combined)

        return concentration_pred


# 计算分段指标
def calculate_bin_metrics(y_true, y_pred, bin_edges):
    bin_metrics = []
    true_bins = np.digitize(y_true, bin_edges) - 1
    true_bins = np.clip(true_bins, 0, len(bin_edges) - 2)

    for bin_idx in range(len(bin_edges) - 1):
        mask = (true_bins == bin_idx)
        if np.sum(mask) > 0:
            bin_true = y_true[mask]
            bin_pred = y_pred[mask]
            mae = mean_absolute_error(bin_true, bin_pred)
            rmse = np.sqrt(mean_squared_error(bin_true, bin_pred))
            r2 = r2_score(bin_true, bin_pred)
            relative_error = np.mean(np.abs((bin_true - bin_pred) / np.maximum(bin_true, 1e-6)))
            bin_metrics.append({
                "bin": f"{bin_edges[bin_idx]:.0f}-{bin_edges[bin_idx + 1]:.0f}",
                "samples": len(bin_true),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "relative_error": relative_error
            })

    return pd.DataFrame(bin_metrics)


# 可视化结果
def visualize_results(results, bin_metrics, save_path="results.png"):
    plt.figure(figsize=(18, 12))

    # 真实值 vs 预测值散点图
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=results["true"], y=results["pred"], hue=results["bin"], palette="viridis", alpha=0.6)
    plt.plot([0, 1000], [0, 1000], 'r--')
    plt.xlabel("True Concentration (mg/m³)")
    plt.ylabel("Predicted Concentration (mg/m³)")
    plt.title("True vs Predicted Concentration")
    plt.grid(True)

    # 误差分布图
    plt.subplot(2, 2, 2)
    error = results["pred"] - results["true"]
    sns.histplot(error, kde=True, bins=50)
    plt.xlabel("Prediction Error (mg/m³)")
    plt.title("Error Distribution")
    plt.grid(True)

    # 分段MAE
    plt.subplot(2, 2, 3)
    sns.barplot(x="bin", y="mae", data=bin_metrics, palette="coolwarm")
    plt.xticks(rotation=45)
    plt.xlabel("Concentration Bin (mg/m³)")
    plt.ylabel("MAE (mg/m³)")
    plt.title("MAE per Concentration Bin")
    plt.grid(True)

    # 分段R²
    plt.subplot(2, 2, 4)
    sns.barplot(x="bin", y="r2", data=bin_metrics, palette="RdYlGn")
    plt.xticks(rotation=45)
    plt.xlabel("Concentration Bin (mg/m³)")
    plt.ylabel("R² Score")
    plt.title("R² per Concentration Bin")
    plt.ylim(0, 1)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # 额外可视化：相对误差
    plt.figure(figsize=(10, 6))
    sns.barplot(x="bin", y="relative_error", data=bin_metrics, palette="YlOrRd")
    plt.xticks(rotation=45)
    plt.xlabel("Concentration Bin (mg/m³)")
    plt.ylabel("Relative Error")
    plt.title("Relative Error per Concentration Bin")
    plt.grid(True)
    plt.savefig("relative_error.png", dpi=300)
    plt.close()


# 训练函数
def train_model(model, train_loader, val_loader, config):
    # 调试：检查第一个batch的维度
    sample = next(iter(train_loader))
    print("\nDEBUG: First batch shapes")
    print(f"Images: {sample[0].shape}  # Should be [B, 3, 224, 224]")
    print(f"Wind:   {sample[1].shape}  # Should be [B, 1]")
    print(f"Conc:   {sample[2].shape}  # Should be [B]\n")

    model.to(config.device)
    criterion = nn.L1Loss()  # MAE损失更适合回归任务
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]")

        for images, wind_speeds, concentrations in train_bar:
            images = images.to(config.device)
            wind_speeds = wind_speeds.to(config.device)
            concentrations = concentrations.to(config.device)

            optimizer.zero_grad()
            outputs = model(images, wind_speeds)

            # 确保输出和标签维度一致
            outputs = outputs.squeeze()
            concentrations = concentrations * 1000.0  # 恢复原始范围

            loss = criterion(outputs, concentrations)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Val]")

        with torch.no_grad():
            for images, wind_speeds, concentrations in val_bar:
                images = images.to(config.device)
                wind_speeds = wind_speeds.to(config.device)
                concentrations = concentrations.to(config.device)

                outputs = model(images, wind_speeds)

                # 确保输出和标签维度一致
                outputs = outputs.view(-1)
                concentrations = concentrations * 1000.0

                loss = criterion(outputs, concentrations)
                val_loss += loss.item() * images.size(0)
                val_bar.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # 学习率调整
        scheduler.step(epoch_val_loss)

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), config.save_path)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")

        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MAE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()

    return model


# 测试函数
def test_model(model, test_loader, config):
    model.load_state_dict(torch.load(config.save_path))
    model.to(config.device)
    model.eval()

    all_preds = []
    all_trues = []
    all_winds = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing")
        for images, wind_speeds, concentrations in test_bar:
            images = images.to(config.device)
            wind_speeds = wind_speeds.to(config.device)
            concentrations = concentrations.to(config.device)

            outputs = model(images, wind_speeds)

            # 确保输出和标签维度一致
            outputs = outputs.squeeze().cpu().numpy() * 1000.0  # 恢复原始范围
            concentrations = concentrations.cpu().numpy() * 1000.0

            all_preds.extend(outputs)
            all_trues.extend(concentrations)
            all_winds.extend(wind_speeds.cpu().numpy().flatten())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    # 总体指标
    overall_mae = mean_absolute_error(all_trues, all_preds)
    overall_rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
    overall_r2 = r2_score(all_trues, all_preds)
    overall_relative_error = np.mean(np.abs((all_trues - all_preds) / np.maximum(all_trues, 1e-6)))

    print(f"\nOverall Metrics:")
    print(f"MAE: {overall_mae:.2f} mg/m³")
    print(f"RMSE: {overall_rmse:.2f} mg/m³")
    print(f"R²: {overall_r2:.4f}")
    print(f"Relative Error: {overall_relative_error:.4f}")

    # 分段指标
    bin_metrics = calculate_bin_metrics(all_trues, all_preds, config.bin_edges)
    print("\nBin-wise Metrics:")
    print(bin_metrics)

    # 准备可视化数据
    results_df = pd.DataFrame({
        "true": all_trues,
        "pred": all_preds,
        "wind": all_winds,
        "bin": pd.cut(all_trues, bins=config.bin_edges, include_lowest=True)
    })

    # 可视化
    visualize_results(results_df, bin_metrics, "concentration_results.png")

    # 保存预测结果
    results_df.to_csv("prediction_results.csv", index=False)
    bin_metrics.to_csv("bin_metrics.csv", index=False)

    return overall_mae, overall_rmse, overall_r2


# 主函数
def main():
    config = Config()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集
    print("Loading dataset...")
    full_dataset = CoalDustDataset(
        root=config.data_root,
        wind_speeds=config.wind_speeds,
        transform=transform
    )

    # 数据集划分 (80%训练, 10%验证, 10%测试)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print(f"Dataset loaded: {len(full_dataset)} samples")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 初始化模型
    model = MaxViTWithWind()
    print(f"Model created: {type(model).__name__}")
    print(f"Image feature dimension: {model.img_feature_dim}")

    # 测试模型维度
    test_img = torch.randn(1, 3, config.image_size, config.image_size)
    test_wind = torch.randn(1)
    test_output = model(test_img, test_wind)
    print(f"Test output shape: {test_output.shape}")

    # 训练模型
    print("\nStarting training...")
    trained_model = train_model(model, train_loader, val_loader, config)

    # 测试模型
    print("\nStarting testing...")
    test_model(trained_model, test_loader, config)


if __name__ == "__main__":
    main()