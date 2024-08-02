import os
import glob
import torch
import torch.nn as nn
import torch.utils.data as data
import timm
import warnings
from torchvision import transforms
from PIL import Image
import numpy as np


class PathologyTileDataset(data.Dataset):
    def __init__(self, img_path, transform=None):
        self.img_list = []
        self.coordinates_list = []

        # 使用 glob 过滤出特定格式的图像文件（例如 jpg 和 png）
        for img_file in glob.glob(os.path.join(img_path, '*.jpg')) + glob.glob(os.path.join(img_path, '*.png')):
            self.img_list.append(img_file)
            filename = os.path.basename(img_file).split(".")[0]
            coordinates = [int(coord) for coord in filename.split("_")[:2]]
            self.coordinates_list.append(coordinates)

        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')  # 确保图像是 RGB 格式
        if self.transform:
            img = self.transform(img)
        return img, self.coordinates_list[index]
