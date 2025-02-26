import pandas as pd
from collections import defaultdict

def count_user_interactions(file_path, output_excel="user_interaction_distribution.xlsx"):
    user_counts = defaultdict(int)

    # 读取文件并统计每个用户的交互次数
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            user, _ = map(int, line.strip().split())
            user_counts[user] += 1

    # 统计不同交互次数的用户数量
    interaction_distribution = defaultdict(int)
    for count in user_counts.values():
        interaction_distribution[count] += 1

    # 按交互次数排序
    sorted_distribution = sorted(interaction_distribution.items())

    # 转换为 DataFrame 并保存为 Excel
    df = pd.DataFrame(sorted_distribution, columns=["交互次数", "用户数量"])
    df.to_excel(output_excel, index=False)

    print(f"统计结果已保存至 {output_excel}")

# 示例调用
file_path = "merged_sorted_unique.txt"  # 替换为你的数据文件
count_user_interactions(file_path)
