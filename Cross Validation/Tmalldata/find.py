import pandas as pd

# 读取 txt 文件
file_path = "merged_sorted_unique.txt"  # 替换为你的文件名
df = pd.read_csv(file_path, sep=" ", header=None, names=["col1", "col2"])

# 查找完全重复的行
duplicates = df[df.duplicated(subset=["col1", "col2"], keep=False)]

if not duplicates.empty:
    print("存在完全相同的两行：")
    print(duplicates)
else:
    print("不存在完全相同的两行")
