import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

if __name__ == "__main__":
    # 读取数据
    data = pd.read_excel("./data/merge.xlsx")
    data = data[["OS", "最后随访状态"]]
    data = data.dropna()

    # 分为进展为1和进展为0两组
    T1 = data.loc[data["最后随访状态"] == 1, "OS"]
    T2 = data.loc[data["最后随访状态"] == 0, "OS"]
    E1 = np.ones(len(T1))
    E2 = np.ones(len(T2))

    # 创建Kaplan-Meier估计对象并绘制生存曲线
    kmf = KaplanMeierFitter()
    kmf.fit(T1, event_observed=E1, label="No Progress")
    ax = kmf.plot()

    kmf.fit(T2, event_observed=E2, label="Progress")
    ax = kmf.plot(ax=ax)

    # 进行log-rank检验
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    p_value = results.p_value
    print("Log-rank test p-value:", p_value)

    # 在图像上添加p值
    text_x_position = max(max(T1), max(T2)) * 0.7
    text_y_position = 0.7
    ax.text(text_x_position, text_y_position, f"p-value: {p_value:.4f}", fontsize=12, bbox=dict(facecolor="white", alpha=0.5))

    # 保存图像
    ax.figure.savefig("progress.png")
    # 显示图像
    plt.show()
