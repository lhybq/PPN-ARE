
import json
import pandas as pd
from collections import defaultdict, Counter

# 定义文件名
file_paths = ["test_dict.txt", "train_dict.txt", "validation_dict.txt"]

# 初始化用户交互字典
user_interactions = defaultdict(set)

# 读取并合并数据
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 读取 JSON 格式的字典
        for user, items in data.items():
            user_interactions[user].update(items)  # 合并物品ID，去重

# 统计每个用户的交互数
interaction_counts = [len(items) for items in user_interactions.values()]

# 统计每个交互次数的用户数量
interaction_distribution = Counter(interaction_counts)

# 转换为 DataFrame
df = pd.DataFrame(list(interaction_distribution.items()), columns=["交互次数", "用户数量"])

# 按交互次数升序排序
df.sort_values(by="交互次数", ascending=True, inplace=True)

# 保存到 Excel 文件
df.to_excel("user_interaction_distribution.xlsx", index=False, engine="openpyxl")

print("统计完成，数据已保存为 user_interaction_distribution.xlsx")
