
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke

plt.rcParams['font.family'] = 'Times New Roman'
# ========== 读取数据 ==========
# 请确保文件名与路径正确
df = pd.read_csv("alertscore_results.csv")

true = df.iloc[:, 0].to_numpy()   # 真实浓度
pred = df.iloc[:, 3].to_numpy()   # AlertScore
# wind = df.iloc[:, 4].to_numpy() # 本图按论文样式不使用颜色映射（如果需要可恢复）

# ========== 线性回归拟合 ==========
X = sm.add_constant(true)
model = sm.OLS(pred, X).fit()
pred_line = model.predict(X)

# ========== Level 边界 ==========
# 题主定义的区间边界（严格按 0,100,500,800,1000）
boundaries = [0, 315.16, 580.55, 817.10, 1000]
level_labels = ["Level I", "Level II", "Level III", "Level IV"]
# 从内到外的颜色（浅→深），便于区分
level_colors = ["#cdebc8", "#f7eca8", "#b2baec", "#f1b6af"]
#level_colors = ["#f7f9e6", "#e2f0c7", "#c7e6a3", "#9ec373"]

# ========== 绘图 ==========
fig, ax = plt.subplots(figsize=(10, 8))

# 背景略微浅色
#ax.set_facecolor("#fbfcee")

# 绘制嵌套矩形（每个矩形左下角固定为 (0,0)，右上角为 (boundary, boundary)）
# 注意：为了让颜色分明，应先画外层，再画内层（这里我们从外到内绘制，确保内层覆盖外层）
for i in reversed(range(4)):
    b = boundaries[i+1]
    rect = Rectangle((0, 0), b, b,
                     facecolor=level_colors[i], edgecolor='none',
                     alpha=1.0, zorder=0)
    ax.add_patch(rect)
    # 在矩形边界处画蓝色虚线框（四边）
    ax.plot([0, b], [0, 0], linestyle='--', color='none', linewidth=1.0, zorder=1)   # bottom
    ax.plot([0, b], [b, b], linestyle='--', color='none', linewidth=1.0, zorder=1)   # top
    ax.plot([0, 0], [0, b], linestyle='--', color='none', linewidth=1.0, zorder=1)   # left
    ax.plot([b, b], [0, b], linestyle='--', color='none', linewidth=1.0, zorder=1)   # right

# 为了和论文图风格一致，也在每个等级的“中间偏左”处写上 Level 标签
label_positions = [(10, 10), (10, 325), (10, 590), (10, 827)]
for i, lab in enumerate(level_labels):
    x_text, y_text = label_positions[i]
    ax.text(x_text, y_text, lab, fontsize=14, color='black', weight='bold', va='bottom')

# ========== 绘制散点（空心绿色圆）和回归实线 ==========
# 空心圆： facecolors='none', edgecolors='green'
ax.scatter(true, pred, s=40, facecolors='none', edgecolors='black', linewidths=0.8, alpha=0.5, zorder=3)

# 回归线（按 true 值排序再画以保证线条连续）
sort_idx = np.argsort(true)
ax.plot(true[sort_idx], pred_line[sort_idx], color='green', linewidth=2.0, label='Linear fitting', zorder=4)

# 理想 y=x 参考红虚线（如果需要保留）
#ax.plot([0, 1000], [0, 1000], 'r--', linewidth=1.2, label='Ideal value', zorder=2)


# ========== 图形格式设置 ==========
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_xlabel('True concentration', fontsize=14)
ax.set_ylabel('AlertScore', fontsize=14)

# 网格、刻度和去掉上右边框以更接近论文风格
ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例（放在左上）
ax.legend(loc='upper left', fontsize=11, frameon=True)


"""
# ========== 可选：在右下角嵌入四象限小图（模仿论文右下圆形示意） ==========
# 这里给出简单示意：若不需要可注释掉
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax, width="18%", height="18%", loc='lower right', borderpad=2)
axins.set_xticks([])
axins.set_yticks([])
# 画一个圆并分成四象限
circle = plt.Circle((0.5, 0.5), 0.45, transform=axins.transAxes, facecolor='#7bbf4a', edgecolor='black')
axins.add_artist(circle)
# 四个象限标注
axins.text(0.32, 0.6, 'Category\n2', transform=axins.transAxes, fontsize=8, ha='center')
axins.text(0.68, 0.6, 'Category\n3', transform=axins.transAxes, fontsize=8, ha='center')
axins.text(0.32, 0.35, 'Category\n1', transform=axins.transAxes, fontsize=8, ha='center')
axins.text(0.68, 0.35, 'Category\n4', transform=axins.transAxes, fontsize=8, ha='center')
axins.set_facecolor('none')
for spine in axins.spines.values():
    spine.set_visible(False)
"""


plt.savefig("confusion_matrix_highres.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.tight_layout()
plt.show()


