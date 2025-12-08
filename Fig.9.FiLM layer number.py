# plot_film_metrics_v9.py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
# ---------------------------
# =========== USER INPUT ===========
vector_metrics = {
    "MAE(mg/m³)":   [43.21,42.05,41.07,41.53,41.88],
    "RMSE(mg/m³)":  [61.98,61.04,57.71,60.63,60.97],
    "R²":    [0.9215,0.9357,0.9598,0.9468,0.9443],
    "Params(M)":[22.20,22.35,22.50,22.65,22.80],
    "GFLOPs":[8.92,8.98,9.04,9.10,9.16],
}

feature_metrics = {
    "MAE(mg/m³)":   [43.87,42.92,42.28,42.36,42.45],
    "RMSE(mg/m³)":  [62.05,61.44,61.01,61.13,61.25],
    "R²":    [0.9199,0.9232,0.9350,0.9321,0.9289],
    "Params(M)":[22.32,22.47,22.62,22.77,22.92],
    "GFLOPs":[8.97,9.03,9.09,9.15,9.21],
}
# ---------------------------

metrics = ["MAE(mg/m³)", "RMSE(mg/m³)", "R²", "Params(M)", "GFLOPs"]

col_colors = {
    0: "#4C86F2",
    1: "#2FC18C",
    2: "#F3B000",
    3: "#8A5BE8",
    4: "#F25C54",
}
col_markers = {
    0: "o",
    1: "s",
    2: "^",
    3: "D",
    4: "+",
}

# Layout：紧凑
ncols, nrows = 5, 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5.6), dpi=300)
plt.subplots_adjust(wspace=0.25, hspace=0.22, left=0.05, right=0.98, top=0.90, bottom=0.10)

x = np.arange(1, 6)  # FiLM 层数 1~5


# ---------- 通用绘图函数 ----------
def plot_small(ax, y, colidx, ylabel=None, title=None, metric_name=None,
               custom_yticks=None, shift_up=False):
    color = col_colors[colidx]
    marker = col_markers[colidx]

    ax.set_facecolor("#f7f7f7")
    ax.plot(x, y, marker=marker, markersize=7, linewidth=2.0,
            color=color, markeredgecolor=color, markerfacecolor=color)

    baseline = np.mean(y)
    ax.axhline(baseline, linestyle="--", linewidth=1.2, color="black")

    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#dcdcdc')
    ax.set_xticks(x)
    ax.set_xlim(0.6, 5.4)
    ax.tick_params(axis='both', which='major', labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=5)
    ax.tick_params(axis='x', pad=5)

    # 紧凑纵轴范围
    ymin, ymax = np.min(y), np.max(y)
    pad = (ymax - ymin) * 0.15 if ymax != ymin else 0.1

    # === 特殊处理：Feature-wise FiLM 的 MAE 上浮 ===
    if shift_up:
        ymin -= pad * 0.5   # 增加下限 margin，使整体上浮
        ymax += pad * 0.2

    ax.set_ylim(ymin - pad, ymax + pad)

    # 自定义纵坐标刻度（仅用于特定子图）
    if custom_yticks is not None:
        ax.set_yticks(custom_yticks)

    # 格式化纵坐标显示
    if metric_name == "R²":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2f}"))
    elif "MAE" in metric_name:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0f}"))


# ---------- 上行：Vector-wise FiLM ----------
for col_idx, metric in enumerate(metrics):
    y = np.array(vector_metrics[metric])
    ax = axes[0, col_idx]
    ylabel = "Vector-wise FiLM" if col_idx == 0 else None

    # 自定义 Vector-wise MAE 的刻度
    if metric == "MAE(mg/m³)":
        custom_yticks = [41, 42, 43]
    else:
        custom_yticks = None

    plot_small(ax, y, col_idx, ylabel=ylabel, title=metric,
               metric_name=metric, custom_yticks=custom_yticks)

# ---------- 下行：Feature-wise FiLM ----------
for col_idx, metric in enumerate(metrics):
    y = np.array(feature_metrics[metric])
    ax = axes[1, col_idx]
    ylabel = "Feature-wise FiLM" if col_idx == 0 else None

    # 针对性定制纵坐标刻度
    if metric == "MAE(mg/m³)":
        custom_yticks = [42, 43, 44]
        shift_up = True
    elif metric == "RMSE(mg/m³)":
        custom_yticks = [61, 62]
        shift_up = False
    elif metric == "R²":
        custom_yticks = [0.92, 0.93]
        shift_up = False
    else:
        custom_yticks = None
        shift_up = False

    plot_small(ax, y, col_idx, ylabel=ylabel, title=None,
               metric_name=metric, custom_yticks=custom_yticks, shift_up=shift_up)

# 保存与展示
plt.savefig("film_metrics_comparison_v9.png", bbox_inches="tight", dpi=300)
plt.show()
print("✅ 图已保存为 film_metrics_comparison_v9.png")
