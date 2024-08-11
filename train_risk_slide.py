import os
from datetime import datetime
from math import nan

import numpy as np
import torch
from pycox.models.loss import CoxPHLoss
from sklearn.metrics import auc, roc_curve
from torch.utils.tensorboard import SummaryWriter

from dataset.slide_embedding_dataset import get_slide_embedding_dataloader
from model.regression_model import RegressionModel


def main():
    # * model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionModel(768, 2048, 1).to(device)
    model.train()
    # * config
    max_epoch = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CoxPHLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{current_time}"
    writer = SummaryWriter(log_dir)
    model_save_path = f"{writer.log_dir}/checkpoints/model"
    if not os.path.exists(model_save_path):
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # * dataset
    train_loader, valid_loader, test_loader = get_slide_embedding_dataloader()
    # * train
    best_valid_loss = float("inf")
    for epoch in range(max_epoch):
        model.train()
        for i, (embedding, OS, death) in enumerate(train_loader):
            embedding, OS, death = embedding.to(device), OS.to(device), death.to(device)
            optimizer.zero_grad()
            y_pred = model(embedding)
            loss = criterion(y_pred, OS, death)
            if loss == nan:
                print("loss==nan!")
                exit(-1)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss {loss}")
            writer.add_scalar("Loss/train", loss, epoch * len(train_loader) + i)
        torch.save(
            model.state_dict(),
            model_save_path + f"{epoch}.pth",
        )
        # * valid
        # model.eval()
        # total_loss = 0
        # with torch.no_grad():
        #     total_loss = 0

        #     for i, (embedding, OS, death) in enumerate(valid_loader):
        #         embedding, OS, death = embedding.to(device), OS.to(device), death.to(device)
        #         y_pred = model(embedding)

        #         loss = criterion(y_pred, OS, death)
        #         if loss == nan:
        #             print("loss==nan!")
        #             exit(-1)
        #         total_loss += loss.item()

        #     total_loss /= len(valid_loader)
        #     writer.add_scalar("Loss/valid", total_loss, epoch)
        #     print(f"Epoch {epoch}, Valid Loss {total_loss}")
        #     if total_loss < best_valid_loss:
        #         best_valid_loss = total_loss
        #         torch.save(
        #             model.state_dict(),
        #             model_save_path,
        #         )
        #         print(f"Save model and best threshold at epoch {epoch}")

        scheduler.step()


def test(model_path):
    # 载入模型参数进行预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionModel(768, 2048, 1).to(device)
    model.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    main()
