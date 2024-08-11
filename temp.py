import torch
from numpy import float32
from pycox.models.loss import CoxPHLoss

loss = CoxPHLoss()
predic_risk = torch.tensor([0, 0, 0.], dtype=torch.float32)  # .to("cuda")
target = torch.tensor([0, 0, 0], dtype=torch.float32)  # .to("cuda")
os = torch.tensor([13.7, 1.2, 12.7], dtype=torch.float32)  # .to("cuda")
print(loss(predic_risk, target, os))
