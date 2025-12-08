import matplotlib.pyplot as plt
import numpy as np


# ==============================
# ① 模型与实验数据（可修改）
# ==============================
models = ["EffNetV2-S", "EffNetV2-M", "EffNetV2-L"]

MAE = [52.64, 51.36, 50.07]
RMSE = [70.08, 67.57, 66.39]
R2 = [0.8693, 0.9012, 0.9239]
Params = [21.5, 54.1, 118.5]  # 单位：M 参数
GFLOPs = [8.37, 24.58, 23.8]  # 单位：GFLOPs
PET = [0.6000, 0.3984, 0.4144]  # PET 分数

metrics = [MAE, RMSE, R2, Params, GFLOPs, PET]
metric_names = ["MAE", "RMSE", "R²", "Params (M)", "GFLOPs", "PET"]

# ==============================
# ② 图像与配色设置（可调整）
# ==============================
colors = ["#94BFF2", "#D1D1D1", "#E59874"]  # 蓝 / 灰 / 橙
#plt.rcParams['font.family'] = 'Arial'  # 字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴粗细
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150

# ==============================
# ③ 绘制 2×3 合并图
# ==============================
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
axes = axes.flatten()  # 拉平方便循环

for i, (ax, values, name) in enumerate(zip(axes, metrics, metric_names)):
    x = np.arange(len(models))
    bars = ax.bar(x, values, color=colors, width=0.6)

    # 顶部数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if height < 1:
            label = f"{height:.4f}"
        elif height < 100:
            label = f"{height:.2f}"
        else:
            label = f"{height:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, height * 1.02,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 横坐标
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)

    # 纵坐标标题
    ax.set_ylabel(name, fontsize=10,)
    ax.set_title(metric_names[i], fontweight='bold', pad=6)
    # 坐标与网格
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # 自动调整纵轴范围
    ax.set_ylim(0, max(values) * 1.25)

# ==============================
# ④ 全局美化与保存
# ==============================
#fig.suptitle("Performance Comparison of EfficientNetV2 Variants", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)

plt.savefig("EffNetV2_Comparison_2x3.png", bbox_inches='tight', facecolor='white')
plt.show()
