import os

import numpy as np
import pandas as pd

path = "./data/merge.xlsx"
df = pd.read_excel(path)

# 假设病理号列的名称为 '病理号'
if '病理号' in df.columns:
    # 使用 explode 函数拆分列
    df['病理号'] = df['病理号'].astype(str).str.split(r'[;/]')
    df = df.explode('病理号').reset_index(drop=True)

# 保存结果到新的 Excel 文件
df.to_excel("./data/merge_split.xlsx", index=False)
