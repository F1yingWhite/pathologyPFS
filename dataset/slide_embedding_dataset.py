import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


class SlideEmbeddingDataset(data.Dataset):
    def __init__(self, excel_path="../data/merge.xlsx", embedding_dir="../data/slide224_level_embedding/Path"):
        df = pd.read_excel(excel_path)
        df = df[["病理id", "OS", "最后随访状态"]].dropna()
        self.embedding_dir = embedding_dir
        self.embedding_list = []
        for index, row in df.iterrows():
            slide_id = row["病理id"]
            pfs = row["OS"]
            if not os.path.exists(os.path.join(embedding_dir, f"{int(slide_id)}.pt")):
                continue
            self.embedding_list.append((int(slide_id), pfs, 1 - int(row["最后随访状态"])))

    def __len__(self):
        return len(self.embedding_list)

    def __getitem__(self, index):
        embedding, pfs, dead = self.embedding_list[index]
        embedding_path = os.path.join(self.embedding_dir, f"{embedding}.pt")
        embedding = torch.load(embedding_path)
        return embedding.float().flatten(), torch.tensor(pfs).float(), torch.tensor(dead).float()


def get_slide_embedding_dataloader(excel_path="./data/merge.xlsx", embedding_dir="./data/slide224_level_embedding/Path", batch_size=32, split=[0.7, 0.15, 0.15], shuffle=[True, False, False]):
    dataset = SlideEmbeddingDataset(excel_path, embedding_dir)
    # Calculate dataset sizes
    train_size = int(split[0] * len(dataset))
    valid_size = int(split[1] * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split dataset
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle[0])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle[1])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle[2])

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    dataset = SlideEmbeddingDataset()
    print(len(dataset))
    # # 计算训练集和测试集的大小
    # train_size = int(0.7 * len(dataset))
    # valid_size = int(0.15 * len(dataset))
    # test_size = len(dataset) - train_size - valid_size
    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # # 划分数据集
    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    # print(f"训练集大小: {len(train_dataset)}")
    # print(f"测试集大小: {len(test_dataset)}")
    # print(f"验证集大小: {len(valid_dataset)}")
    embeddingLists = []
    pfsLists = []
    jinzhanLists = []
    print(dataset[0][0].shape)
    for i in range(len(dataset)):
        embedding, pfs, jinzhan = dataset[i]
        embeddingLists.append(embedding.cpu().reshape(-1).numpy())
        pfsLists.append(pfs)
        jinzhanLists.append(jinzhan)
    # 使用kmeans聚类成为两类,并且按照对应的index聚类pfs
    kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddingLists)
    pfs1 = []
    pfs2 = []
    jinzhan1 = []
    jinzhan2 = []
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] == 0:
            pfs1.append(pfsLists[i])
            jinzhan1.append(jinzhanLists[i])
        else:
            pfs2.append(pfsLists[i])
            jinzhan2.append(jinzhanLists[i])
    # 使用 Kaplan-Meier 估计生存曲线
    kmf = KaplanMeierFitter()
    # 绘制第一个生存曲线
    kmf.fit(pfs1, jinzhan1, label="cluster1")
    ax = kmf.plot()

    # 绘制第二个生存曲线
    kmf.fit(pfs2, jinzhan2, label="cluster2")
    kmf.plot(ax=ax)

    # 使用 logrank_test 计算 p 值
    results = logrank_test(pfs1, pfs2, event_observed_A=jinzhan1, event_observed_B=jinzhan2)
    p_value = results.p_value
    print("Log-rank test p-value:", p_value)

    # 在图像上添加 p 值
    text_x_position = max(max(pfs1), max(pfs2)) * 0.7
    text_y_position = 0.7
    ax.text(text_x_position, text_y_position, f"p-value: {p_value:.4f}", fontsize=12, bbox=dict(facecolor="white", alpha=0.5))

    # 保存到本地
    ax.get_figure().savefig("survival_curve_with_p_value.png")

    # 显示图像
    plt.show()
