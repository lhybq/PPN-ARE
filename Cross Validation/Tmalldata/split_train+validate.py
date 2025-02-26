import random
from collections import defaultdict


def split_train_valid(file_path, valid_ratio=0.34):
    user_interactions = defaultdict(list)

    # 读取交互数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            user, item = map(int, line.strip().split())
            user_interactions[user].append(item)

    train_set = []
    valid_set = []

    for user, items in user_interactions.items():
        num_items = len(items)

        if num_items >= 2:  # 正常处理 (20% 规则)
            valid_count = max(1, int(num_items * valid_ratio))  # 选 20%
            valid_items = random.sample(items, valid_count)  # 随机抽样作为验证集

            for item in valid_items:
                valid_set.append((user, item))
                items.remove(item)  # 其余数据留给训练集

        elif num_items == 1:  # 仅 1 条交互
            if random.random() < 0.5:  # 50% 概率进入 valid_set
                valid_set.append((user, items[0]))
                items = []  # 训练集中无数据

        # if num_items >= 3:  # 正常处理 (20% 规则)
        #     valid_count = max(1, int(num_items * valid_ratio))  # 选 20%
        #     valid_items = random.sample(items, valid_count)  # 随机抽样作为验证集
        #
        #     for item in valid_items:
        #         valid_set.append((user, item))
        #         items.remove(item)  # 其余数据留给训练集
        #
        # # elif num_items == 2:  # 2 条交互的情况
        # #     if random.random() < 1:  # 20% 概率将 1 条交互放入 valid_set
        # #         valid_item = random.choice(items)
        # #         valid_set.append((user, valid_item))
        # #         items.remove(valid_item)
        # elif num_items == 2:  # 2 条交互的情况
        #     p = random.random()
        #     if p < 1:  # 50% 概率两条都放进验证集
        #         valid_set.extend([(user, item) for item in items])
        #         items.clear()  # 训练集中无剩余项
        #     elif p < 1:  # 30% 概率随机选 1 条放进验证集
        #         valid_item = random.choice(items)
        #         valid_set.append((user, valid_item))
        #         items.remove(valid_item)  # 训练集中留 1 条
        #     # 剩下 20% 概率，所有数据都留在训练集中
        # elif num_items == 1:  # 1 条交互的情况
        #     if random.random() < 1:  # 50% 概率进入 valid_set
        #         valid_set.append((user, items[0]))
        #         items = []

        # 剩余数据留作训练集
        for item in items:
            train_set.append((user, item))

    return train_set, valid_set


# 读取原始数据并进行划分
file_path = "merged_sorted_unique_train+validate.txt"
train_data, valid_data = split_train_valid(file_path)

# 保存最终的训练、验证集
with open("Tmall01/train.txt", "w") as f:
    for u, i in train_data:
        f.write(f"{u} {i}\n")

with open("Tmall01/buy.txt", "w") as f:
    for u, i in train_data:
        f.write(f"{u} {i}\n")

with open("Tmall01/validation.txt", "w") as f:
    for u, i in valid_data:
        f.write(f"{u} {i}\n")

print(f"划分完成！训练集: {len(train_data)}, 验证集: {len(valid_data)}")
