import os

import numpy as np
import pandas as pd

path1 = "./data/TT-LBM.xlsx"
path2 = "./data/LBM.xlsx"
# 根据path1的住院号 == path2的patient，合并两个表格,path2只保留status
df1 = pd.read_excel(path1)
df2 = pd.read_excel(path2)
df2 = df2[['patient', 'status', "病理id"]]
df = pd.merge(df1, df2, left_on='住院号', right_on='patient', how='left')
# 保存到本地
df.to_excel('./data/merge.xlsx', index=False)
