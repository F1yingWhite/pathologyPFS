import torch
import torch.nn as nn
from pycox.models.loss import CoxPHLoss

# 示例用法
if __name__ == "__main__":
    # 假设有以下示例数据
    risk_scores = torch.tensor([-0.2, 2.5, 0.8, 1.0], dtype=torch.float32)
    events = torch.tensor([1, 0, 1, 1], dtype=torch.float32)  # 事件发生（1 表示事件发生，0 表示未发生）
    durations = torch.tensor([4, 3, 2, 1], dtype=torch.float32)  # 事件发生时间或随访时间

    # 创建 CoxPHLoss 实例
    coxph_loss = CoxPHLoss()

    # 计算损失
    loss = coxph_loss(risk_scores, events, durations)
    print(f"CoxPH Loss: {loss.item()}")
