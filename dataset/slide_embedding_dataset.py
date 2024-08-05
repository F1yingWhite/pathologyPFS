import glob
import os

import torch
import torch.utils.data as data
from PIL import Image


class SlideEmbeddingDataset(data.Dataset):
    def __init__(self, embedding_path):
        self.embedding_list = []
        for pt_file in glob.glob(os.path.join(embedding_path, '*.pt')):
            self.embedding_list.append(pt_file)

    def __len__(self):
        return len(self.embedding_list)

    def __getitem__(self, index):
        return self.embedding_list[index]
