import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# === 设置全局字体为 Times New Roman ===
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示异常

# 读取数据
#df = pd.read_csv("evaluation_results _test.csv")
df = pd.read_csv(
    "evaluation_results _test.csv",
    encoding="latin1",  # ISO-8859-1，万能兜底
    engine="python",
)
# 提取真实与预测标签
y_true = df["Level"]
y_pred = df["AlertLevel"]

# 明确标签顺序
labels_true = sorted(y_true.unique())
labels_pred = sorted(y_pred.unique())

# 生成计数矩阵
cm_counts = pd.crosstab(index=y_true, columns=y_pred)
cm_counts = cm_counts.reindex(index=labels_true, columns=labels_pred, fill_value=0)

# 归一化（按行）
cm_normalized = cm_counts.div(cm_counts.sum(axis=1), axis=0).fillna(0).values
cm_counts_values = cm_counts.values

# 注释文本：比例 + 样本数
annot_text = np.empty(cm_counts_values.shape, dtype=object)
for i in range(cm_counts_values.shape[0]):
    for j in range(cm_counts_values.shape[1]):
        annot_text[i, j] = f"{cm_normalized[i, j]:.2f}\n({cm_counts_values[i, j]})"

print("Support (每个真实类别的样本数)：")
print(cm_counts.sum(axis=1))

# === 绘图 ===
plt.figure(figsize=(8, 6), dpi=1200)  # 高分辨率输出
sns.heatmap(cm_normalized,
            annot=annot_text,
            fmt="",
            cmap="Blues",
            xticklabels=labels_pred,
            yticklabels=labels_true,
            cbar=True,
            annot_kws={"size": 10})

plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()

# === 保存高分辨率图片 ===
plt.savefig("confusion_matrix_highres.png", dpi=1200, bbox_inches='tight', pad_inches=0.1)
plt.show()

# === 分类性能报告 ===
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, digits=3))
