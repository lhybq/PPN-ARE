import random
import math
from collections import defaultdict


def split_leave_one_with_prob(file_path, lambda_factor=4):
    user_interactions = defaultdict(list)

    # 读取交互数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            user, item = map(int, line.strip().split())
            user_interactions[user].append(item)

    train_set = []
    test_set = []

    for user, items in user_interactions.items():
        num_interactions = len(items)

        # # 计算该用户进入测试集的概率
        # test_prob = 1 - math.exp(-num_interactions / lambda_factor)  # 指数增长策略
        #
        # if random.random() < test_prob:  # 以概率 test_prob 选择该用户进入测试集
        #     test_item = random.choice(items)  # 随机选择 1 条交互放入测试集
        #     test_set.append((user, test_item))
        #     items.remove(test_item)  # 其余交互进入训练集
        test_item = random.choice(items)  # 随机选择 1 条交互放入测试集
        test_set.append((user, test_item))
        items.remove(test_item)  # 其余交互进入训练集

        # 剩下的全部进入初步的训练集
        for item in items:
            train_set.append((user, item))

    return train_set, test_set


def split_train_valid(train_data, valid_ratio=0.34):
    user_data = defaultdict(list)

    # 将训练数据按用户分组
    for user, item in train_data:
        user_data[user].append(item)

    new_train_set = []
    valid_set = []

    for user, items in user_data.items():
        # num_items = len(items)
        valid_item = random.choice(items)
        valid_set.append((user, valid_item))
        items.remove(valid_item)  # 训练集中留 1 条

        # 剩余数据留作训练集
        for item in items:
            new_train_set.append((user, item))

    return new_train_set, valid_set


# 读取原始数据并进行划分
file_path = "merged_sorted_unique.txt"
train_data, test_data = split_leave_one_with_prob(file_path)
train_data, valid_data = split_train_valid(train_data)

# # 保存最终的训练、验证和测试集
# with open("Tmall1/train.txt", "w") as f:
#     for u, i in train_data:
#         f.write(f"{u} {i}\n")
# with open("Tmall1/buy.txt", "w") as f:
#     for u, i in train_data:
#         f.write(f"{u} {i}\n")
# with open("Tmall1/validation.txt", "w") as f:
#     for u, i in valid_data:
#         f.write(f"{u} {i}\n")
# with open("Tmall1/test.txt", "w") as f:
#     for u, i in test_data:
#         f.write(f"{u} {i}\n")
#


# 验证集和测试集交换

with open("Tmall1/train.txt", "w") as f:
    for u, i in train_data:
        f.write(f"{u} {i}\n")
with open("Tmall1/buy.txt", "w") as f:
    for u, i in train_data:
        f.write(f"{u} {i}\n")
with open("Tmall1/validation.txt", "w") as f:
    for u, i in test_data:
        f.write(f"{u} {i}\n")
with open("Tmall1/test.txt", "w") as f:
    for u, i in valid_data:
        f.write(f"{u} {i}\n")

print(f"划分完成！训练集: {len(train_data)}, 验证集: {len(valid_data)}, 测试集: {len(test_data)}")
