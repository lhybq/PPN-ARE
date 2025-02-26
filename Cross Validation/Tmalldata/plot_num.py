import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取 Excel 文件
file_path = "user_interaction_distribution_TandB.xlsx"  # 请替换为你的文件路径
df = pd.read_excel(file_path)

# 提取数据
tmall_interactions = df.iloc[:, 0]  # First column: Interaction counts (Tmall)
tmall_users = df.iloc[:, 1]         # Second column: User counts (Tmall)

beibei_interactions = df.iloc[:, 2]  # Third column: Interaction counts (Beibei)
beibei_users = df.iloc[:, 3]         # Fourth column: User counts (Beibei)

# 绘制直方图
plt.figure(figsize=(10, 6))

plt.bar(tmall_interactions, tmall_users, width=1, alpha=0.8, label="Tmall", color='blue')
plt.bar(beibei_interactions, beibei_users, width=1, alpha=0.5, label="Beibei", color='red')

plt.xscale('log')  # Use logarithmic scale
plt.yscale('log')

# 设置 X 轴和 Y 轴刻度，使其更加细化
x_ticks = np.logspace(np.log10(min(tmall_interactions.min(), beibei_interactions.min())),
                      np.log10(max(tmall_interactions.max(), beibei_interactions.max())),
                      num=10)  # 生成 10 个对数均匀分布的刻度
y_ticks = np.logspace(np.log10(min(tmall_users.min(), beibei_users.min())),
                      np.log10(max(tmall_users.max(), beibei_users.max())),
                      num=10)

plt.xticks(x_ticks, labels=[f"{int(x)}" for x in x_ticks], fontsize=12)
plt.yticks(y_ticks, labels=[f"{int(y)}" for y in y_ticks], fontsize=12)

plt.xlabel("Interaction Count", fontsize=16)
plt.ylabel("Number of Users", fontsize=16)
plt.title("User Interaction Distribution: Tmall and Beibei", fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

plt.show()
