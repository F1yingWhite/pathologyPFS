import matplotlib.pyplot as plt
import numpy
import pandas as pd
from lifelines import KaplanMeierFitter

# Load the data
path = "./data/merge.xlsx"
df = pd.read_excel(path)
# 需要的数据列为OS和status
df = df[['PFS', 'status']].dropna()

# Kaplan-Meier生存曲线拟合
kmf = KaplanMeierFitter()
kmf.fit(durations=df['PFS'], event_observed=df['status'])

# 绘制生存曲线
plt.figure(figsize=(8, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Duration')
plt.ylabel('Survival Probability')
plt.grid(True)
# 保存到本地
plt.savefig('./data/Kaplan-Meier Survival Curve.png')
