import glob
import os
import re
import warnings

import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image

warnings.filterwarnings("ignore")


class TileEmbeddingDataset(data.Dataset):
    def __init__(self, excel_path="../data/merge.xlsx", embedding_dir="../data/tile224embedding"):
        df = pd.read_excel(excel_path)
        df = df[["病理id", "PFS"]].dropna()
        self.embedding_dir = embedding_dir
        self.embedding_list = []
        for index, row in df.iterrows():
            slide_id = str(int(row["病理id"]))
            pfs = row["PFS"]
            # 查找嵌入目录中包含slide_id的文件
            matching_files = [f for f in os.listdir(embedding_dir) if re.search(slide_id, f)]
            for file_name in matching_files:
                embedding_path = os.path.join(embedding_dir, file_name)
                self.embedding_list.append((embedding_path, pfs))

    def __len__(self):
        return len(self.embedding_list)

    def __getitem__(self, index):
        embedding_path, pfs = self.embedding_list[index]
        embedding = torch.load(embedding_path)
        return embedding.float(), torch.tensor(pfs).float()


if __name__ == "__main__":
    tile = TileEmbeddingDataset()
    print(len(tile))
