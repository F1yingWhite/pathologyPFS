import os
import sched
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.slide_embedding_dataset import SlideEmbeddingDataset
from model.regression_model import RegressionModel

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def main():
    model = RegressionModel(768, 2048, 1)
    model.to("cuda")
    dataset = SlideEmbeddingDataset("./data/merge.xlsx", "./data/slide_level_embedding/Path")

    # 计算训练集和测试集的大小
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 划分数据集
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")
    train_dataloader, valid_dataloader, test_dataloader = (
        DataLoader(train_dataset, batch_size=16, shuffle=True),
        DataLoader(valid_dataset, batch_size=16, shuffle=False),
        DataLoader(test_dataset, batch_size=16, shuffle=False),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
    log_dir = f"./logs/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    best_loss = float("inf")
    model_save_path = os.path.join(writer.log_dir, "best_model.pth")
    for epoch in range(100):
        for i, (embedding, pfs) in enumerate(train_dataloader):
            embedding, pfs = embedding.to("cuda"), pfs.to("cuda")
            optimizer.zero_grad()
            output = model(embedding)
            loss = criterion(output, pfs)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), epoch * len(train_dataloader) + i)
        with torch.no_grad():
            total_loss = 0
            for i, (embedding, pfs) in enumerate(valid_dataloader):
                embedding, pfs = embedding.to("cuda"), pfs.to("cuda")
                output = model(embedding)
                loss = criterion(output, pfs)
                total_loss += loss.item()
            total_loss /= len(valid_dataloader)
            print(f"Validation Loss: {total_loss}")
            writer.add_scalar("valid_loss", total_loss, epoch)
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), model_save_path)
        scheduler.step()


if __name__ == "__main__":
    main()
