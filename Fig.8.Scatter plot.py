import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from matplotlib.lines import Line2D

# ==========================
# 全局绘图设置（论文级）
# ==========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600       # 显示分辨率
plt.rcParams['savefig.dpi'] = 600      # 保存分辨率
plt.rcParams['font.size'] = 12

# ==========================
# 读取数据
# ==========================

df = pd.read_csv(
    "evaluation_results_uncertainty.csv",
    encoding="latin1",  # ISO-8859-1，万能兜底
    engine="python",
)

true = df.iloc[:, 0].to_numpy()       # True concentration
pred = df.iloc[:, 1].to_numpy()       # Regression concentration
sigma = df.iloc[:, 2].to_numpy()      # σ
upper = df.iloc[:, 3].to_numpy()      # y_hat + σ
lower = df.iloc[:, 4].to_numpy()      # y_hat - σ
wind = df.iloc[:, 7].to_numpy()       # Wind speed (m/s)

# ==========================
# 线性回归拟合 + 95%置信区间
# ==========================
X = sm.add_constant(true)
model = sm.OLS(pred, X).fit()
pred_line = model.predict(X)

pred_summary = model.get_prediction(X).summary_frame(alpha=0.05)
ci_lower = pred_summary["obs_ci_lower"].to_numpy()
ci_upper = pred_summary["obs_ci_upper"].to_numpy()

# 按 true 值排序（防止置信带乱序）
sorted_idx = np.argsort(true)
true_sorted = true[sorted_idx]
pred_line_sorted = pred_line[sorted_idx]
ci_lower_sorted = ci_lower[sorted_idx]
ci_upper_sorted = ci_upper[sorted_idx]

# ==========================
# 绘图（高分辨率版本）
# ==========================
plt.figure(figsize=(10, 8))

# ---------------------------------
# ★ 不确定性区间竖线：ŷ − σ → ŷ + σ
# ---------------------------------
plt.vlines(true, ymin=lower, ymax=upper, color='gray', alpha=0.4, linewidth=1)

# 散点图（风速上色）
sc = plt.scatter(true, pred, c=wind, cmap='viridis', alpha=0.85, s=40, edgecolors='none')

# 理想线 y = x
plt.plot([0, 1000], [0, 1000], 'r--', linewidth=1.5, label='Ideal value')

# 回归线
plt.plot(true_sorted, pred_line_sorted, color='blue', linewidth=2.5, label='Regression line')

# 95%置信区间
plt.fill_between(true_sorted, ci_lower_sorted, ci_upper_sorted, color='gray', alpha=0.25, label='95% Confidence intervals')

# 颜色条
cbar = plt.colorbar(sc)
cbar.set_label('Wind speed (m/s)', fontsize=13)
cbar.ax.tick_params(labelsize=11)

# ==========================
# ★ 加入竖线图例元素
# ==========================
uncertainty_legend = Line2D(
    [0], [0],
    color='gray',
    linewidth=1,
    alpha=0.6,
    linestyle='-',
    label='Uncertainty interval (σ)'
)

# 坐标与标签
plt.xlabel('True concentration (mg/m³)', fontsize=14)
plt.ylabel('Regression concentration (mg/m³)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 将竖线图例加入整体图例
plt.legend(
    handles=[uncertainty_legend] + plt.gca().get_legend_handles_labels()[0],
    fontsize=12,
    loc='upper left',
    frameon=True
)

plt.grid(True, linestyle='--', alpha=0.6)

plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.tight_layout()

# 保存图片
plt.savefig("regression_with_uncertainty.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()

# 打印回归结果
print(model.summary())
