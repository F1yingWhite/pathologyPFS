import torch
import torch.nn as nn
from pycox.models.loss import CoxPHLoss

# 示例用法
if __name__ == "__main__":
    # 假设有以下示例数据
    risk_scores = torch.tensor([-0.5303, -0.3923, -0.0447], device='cpu', dtype=torch.float32)
    events = torch.tensor([0, 0, 0], device='cpu', dtype=torch.float32)
    durations = torch.tensor([27.8333, 137.2667, 18.1333], device='cpu', dtype=torch.float32)
  # 事件发生时间或随访时间

    # 创建 CoxPHLoss 实例
    coxph_loss = CoxPHLoss()

    # 计算损失
    try:
        loss = coxph_loss(risk_scores, events, durations)
        print(f"CoxPH Loss: {loss.item()}")
    except Exception as e:
        print(f"An error occurred: {e}")
