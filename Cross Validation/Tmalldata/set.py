import pandas as pd
import glob

# 获取所有 .txt 文件的文件名
txt_files = ["train.txt", "validation.txt"]  # 替换为你的文件名

# 读取所有文件并合并
df_list = [pd.read_csv(f, sep=" ", header=None, names=["col1", "col2"]) for f in txt_files]
df = pd.concat(df_list, ignore_index=True)

# 按第一列排序
df_sorted = df.sort_values(by=["col1", "col2"])  # 按两列排序

# 查找完全重复的行
duplicates = df_sorted[df_sorted.duplicated(subset=["col1", "col2"], keep=False)]
if not duplicates.empty:
    print("重复项如下:")
    print(duplicates)

# 去重后保存
df_unique = df_sorted.drop_duplicates(subset=["col1", "col2"], keep="first")
df_unique.to_csv("merged_sorted_unique_train+validate.txt", sep=" ", index=False, header=False)

print("合并完成，去重后结果保存在 merged_sorted_unique.txt")
