import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke


plt.rcParams['font.family'] = 'Times New Roman'

# ===========================
# 用户配置
# ===========================
csv_path = "alertscore_results.csv"   # ← 你的测试结果 CSV
k_value = 1.5                         # AlertScore = ŷ + k·σ 中的 k
num_levels = 4                        # 预警等级数量
save_path = "alertscore_results_2.csv"  # 输出文件

# ===========================
# 1. 读取 CSV 文件
# ===========================
df = pd.read_csv(csv_path)

# 假设列顺序如下
y_true = df.iloc[:, 0].to_numpy()
y_pred = df.iloc[:, 1].to_numpy()
sigma = df.iloc[:, 2].to_numpy()

# ===========================
# 2. 计算 AlertScore
# ===========================
AlertScore = y_pred + k_value * sigma
df["AlertScore"] = AlertScore

print("\n示例 AlertScore:")
print(AlertScore[:10])

# ===========================
# 3. 使用 K-means 自动学习阈值
# ===========================
kmeans = KMeans(n_clusters=num_levels, random_state=42)
labels = kmeans.fit_predict(AlertScore.reshape(-1, 1))

# 聚类中心（升序）
centers = np.sort(kmeans.cluster_centers_.flatten())

# 计算阈值（相邻中心点的中点）
thresholds = []
for i in range(len(centers) - 1):
    thresholds.append((centers[i] + centers[i+1]) / 2)

print("\n==============================")
print("自动学习的预警等级阈值 AlertScore：")
for i, t in enumerate(thresholds, 1):
    print(f"Level {i} → Level {i+1} 分界点 = {t:.2f}")
print("==============================\n")

# ===========================
# 4. 根据阈值生成预警等级
# ===========================
def assign_level(score, thresholds):
    level = 1
    for t in thresholds:
        if score > t:
            level += 1
        else:
            break
    return level

alert_levels = [assign_level(s, thresholds) for s in AlertScore]
df["AlertLevel"] = alert_levels

# ===========================
# 5. 保存结果
# ===========================
df.to_csv(save_path, index=False)
print(f"已保存结果到: {save_path}")

print("\n输出数据样例：")
print(df.head())


# ======================================================
# ========== 可视化 1：K-means 聚类结果可视化 ==========
# ======================================================
plt.figure(figsize=(10, 6))
plt.scatter(range(len(AlertScore)), AlertScore, c=labels, cmap='viridis', s=12)
plt.colorbar(label="Cluster Label")
#plt.title("K-means Clustering on AlertScore")
plt.xlabel("Sample Index")
plt.ylabel("AlertScore")

# 绘制聚类中心（红线）
for c in centers:
    plt.axhline(c, color='red', linestyle='--', linewidth=1.8)
    plt.text(len(AlertScore) * 1.02,            # ← 稍微超出一点点，使其紧贴右边
             c,
             f"Center={c:.1f}",
             ha='right', va='bottom', color='red', fontsize=11,
             path_effects=[withStroke(foreground="white", linewidth=3)])

# 绘制阈值线（黑线）
for t in thresholds:
    plt.axhline(t, color='black', linestyle='-.', linewidth=1.8)
    plt.text(-len(AlertScore) * 0.02,           # ← 稍微向左延伸，让文字贴左侧
             t,
             f"Threshold={t:.1f}",
             ha='left', va='bottom', color='black', fontsize=11,
             path_effects=[withStroke(foreground="white", linewidth=3)])


'''
# 绘制聚类中心
for c in centers:
    plt.axhline(c, color='red', linestyle='--', linewidth=1)
    plt.text(len(AlertScore)*0.99, c, f"Center={c:.1f}", ha='right', va='bottom', color='red')

# 绘制阈值线
for t in thresholds:
    plt.axhline(t, color='black', linestyle='-.', linewidth=1)
    plt.text(len(AlertScore)*0.01, t, f"Threshold={t:.1f}", ha='left', va='bottom', color='black')
'''
plt.tight_layout()
plt.savefig("kmeans_alertscore_visualization.png", dpi=300)
plt.show()




# ======================================================
# ========== 可视化 2：预警等级可视化 ==========
# ======================================================
plt.figure(figsize=(12, 6))
plt.plot(AlertScore, label="AlertScore", color="blue", linewidth=1)

# 绘制等级区间背景色块
colors = ["#d0f0c0", "#fef3bd", "#ffe4b5", "#ffb3b3"]  # level 1~4 对应色块，可修改
start = 0
max_val = max(AlertScore)

# 绘制每段颜色
prev = -1e9
for i, t in enumerate(thresholds + [max_val]):
    plt.axhspan(prev, t, facecolor=colors[i], alpha=0.3)
    prev = t

# 绘制阈值线
for t in thresholds:
    plt.axhline(t, color="black", linestyle="--")
    plt.text(0, t, f"{t:.1f}", va="bottom", ha="left", fontsize=9)

plt.title("AlertScore-based Automatic Warning Level Segmentation")
plt.xlabel("Sample Index")
plt.ylabel("AlertScore")
plt.legend()
plt.tight_layout()
plt.savefig("alertscore_levels_visualization.png", dpi=300)
plt.show()
