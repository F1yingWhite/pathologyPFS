import os
import sched
from datetime import datetime

import numpy as np
import torch
from pycox.models.loss import CoxPHLoss
from pyzstd import train_dict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.slide_embedding_dataset import SlideEmbeddingDataset
from dataset.tile_embedding_dataset import get_tile_embedding_dataloader
from model.regression_model import MySurvivalPredictionModel


def main():
    # * model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MySurvivalPredictionModel(2048, 1, prov_decoder_path="./checkpoints/pretrain/prov-gigapath/slide_encoder.pth").to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 2])
    model.train()
    # * config
    max_epoch = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = CoxPHLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{current_time}"
    writer = SummaryWriter(log_dir)
    # * dataset
    train_loader, valid_loader, test_loader = get_tile_embedding_dataloader(batch_size=2)
    # * train
    best_valid_loss = float("inf")
    for epoch in range(max_epoch):
        model.train()
        for i, (embedding, position, pfs, jinzhan) in enumerate(train_loader):
            embedding, position, pfs, jinzhan = embedding.to(device), position.to(device), pfs.to(device), jinzhan.to(device)
            optimizer.zero_grad()
            y_pred = model(embedding, position)
            loss = criterion(y_pred, pfs, jinzhan)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss {loss}")
            writer.add_scalar("Loss/train", loss, epoch * len(train_loader) + i)
        # * valid
        model.eval()
        with torch.no_grad():
            for i, (embedding, pfs, jinzhan) in enumerate(valid_loader):
                embedding, position, pfs, jinzhan = embedding.to(device), position.to(device), pfs.to(device), jinzhan.to(device)
                y_pred = model(embedding)
                loss = criterion(y_pred, pfs, jinzhan)
                writer.add_scalar("Loss/valid", loss, epoch * len(valid_loader) + i)
                if loss < best_valid_loss:
                    best_valid_loss = loss
                    torch.save(model.state_dict(), f"checkpoints/{current_time}.pth")
        scheduler.step()


if __name__ == "__main__":
    main()
