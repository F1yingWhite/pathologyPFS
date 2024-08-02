import glob
import os

import torch
import torch.utils.data as data
from PIL import Image


class PathologyTileDataset(data.Dataset):
    def __init__(self, img_path, transform):
        assert transform is not None, "transform is required"
        self.transform = transform
        self.img_list = []
        self.coordinates_list = []

        for img_file in glob.glob(os.path.join(img_path, '*.jpeg')):
            self.img_list.append(img_file)
            filename = os.path.basename(img_file).split(".")[0]
            coordinates = [int(coord) for coord in filename.split("_")[:2]]
            self.coordinates_list.append(coordinates)

        assert len(self.img_list) > 0, "No images found in the given path"

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.coordinates_list[index])
