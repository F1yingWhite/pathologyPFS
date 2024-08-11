import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from dataset.binary_dataset import get_binary_dataloader
from model.regression_model import RegressionModel


def main():
    # * model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionModel(768, 2048, 2).to(device)
    model.train()
    # * config
    max_epoch = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f'logs/{current_time}'
    writer = SummaryWriter(log_dir)
    model_save_path = f"{writer.log_dir}/checkpoints/best_model.pth"
    if not os.path.exists(model_save_path):
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # * dataset
    train_loader, valid_loader, test_loader = get_binary_dataloader()
    # * train
    best_valid_loss = float("inf")
    for epoch in range(max_epoch):
        model.train()
        for i, (embedding, death) in enumerate(train_loader):
            embedding, death = embedding.to(device), death.to(device)
            optimizer.zero_grad()
            y_pred = model(embedding)
            loss = criterion(y_pred, death)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss {loss}")
            writer.add_scalar("Loss/train", loss, epoch * len(train_loader) + i)
        # * valid
        model.eval()
        total_loss = 0
        with torch.no_grad():
            y_true = []
            y_scores = []
            total_loss = 0

            for i, (embedding, death) in enumerate(valid_loader):
                embedding, death = embedding.to(device), death.to(device)
                y_pred = model(embedding)

                loss = criterion(y_pred, death)
                total_loss += loss.item()

                # 将标签和预测分数保存以用于计算ROC曲线
                y_true.extend(death.cpu().numpy())
                y_scores.extend(torch.softmax(y_pred, dim=1)[:, 1].cpu().numpy())  # 假设目标类是第1类

            total_loss /= len(valid_loader)
            writer.add_scalar("Loss/valid", total_loss, epoch)
            print(f"Epoch {epoch}, Valid Loss {total_loss}")

            # 计算ROC曲线和AUC
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            writer.add_scalar("AUC/valid", roc_auc, epoch)

            # 计算最佳阈值
            youden_index = tpr - fpr
            best_threshold = thresholds[np.argmax(youden_index)]
            print(f"Best Threshold at Epoch {epoch}: {best_threshold}")

            if total_loss < best_valid_loss:
                best_valid_loss = total_loss
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'best_threshold': best_threshold,
                    },
                    model_save_path,
                )
                print(f"Save model and best threshold at epoch {epoch}")

        scheduler.step()


if __name__ == "__main__":
    main()
