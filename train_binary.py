import os
from datetime import datetime

import numpy as np
import torch
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
        for i, (embedding, jinzhan) in enumerate(train_loader):
            embedding, jinzhan = embedding.to(device), jinzhan.to(device)
            optimizer.zero_grad()
            y_pred = model(embedding)
            loss = criterion(y_pred, jinzhan)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss {loss}")
            writer.add_scalar("Loss/train", loss, epoch * len(train_loader) + i)
        # * valid
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (embedding, jinzhan) in enumerate(valid_loader):
                embedding, jinzhan = embedding.to(device), jinzhan.to(device)
                y_pred = model(embedding)
                loss = criterion(y_pred, jinzhan)
                total_loss += loss
            total_loss /= len(valid_loader)
            writer.add_scalar("Loss/valid", total_loss, epoch)
            print(f"Epoch {epoch}, Valid Loss {total_loss}")
            if total_loss < best_valid_loss:
                best_valid_loss = total_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"Save model at epoch {epoch}")
        scheduler.step()


if __name__ == "__main__":
    main()
