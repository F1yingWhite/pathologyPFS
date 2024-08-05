import glob
import os

import torch
import tqdm
from torchvision import transforms

from dataset.pathology_dataset import PathologyTileDataset
from model.prov_path import Prov_decoder, Prov_encoder
from utils.get_pt import read_coordinate_pt


def compute_tile_feature():
    original_path = "./data/tile224/"
    target_path = "./data/tile224embedding"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # 获得original_path下的所有文件夹
    folders = [f for f in glob.glob(os.path.join(original_path, "*")) if os.path.isdir(f)]
    encoder = Prov_encoder(model_path="./checkpoints/pretrain/prov-gigapath/pytorch_model.bin").half().to("cuda")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    encoder.eval()
    num_workers = os.cpu_count()  # 获取 CPU 核心数
    print(f"Using {num_workers} workers for data loading.")
    with torch.no_grad():
        for folder in tqdm.tqdm(folders):
            try:
                embeddings = []
                coordinations = []
                dataset = PathologyTileDataset(folder, transform=transform)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=num_workers)
                for i, (image, coordination) in enumerate(dataloader):
                    image = image.half().to("cuda")
                    embedding = encoder(image)
                    coordinations.append(coordination)
                    embeddings.append(embedding.detach().cpu())
                embeddings = torch.cat(embeddings, dim=0)
                coordinations = torch.cat(coordinations, dim=0)
                concatenated = torch.cat((embeddings, coordinations), dim=1)
                # 把embedding和coordinate拼接起来并且保存到pt
                torch.save(
                    concatenated,
                    os.path.join(target_path, os.path.basename(folder) + ".pt"),
                )
            except Exception as e:
                print("Error in folder:", folder)
                print("contine...")


def compute_slide_embedding():
    original_path = "./data/tile224embedding"
    target_path = "./data/slide_level_embedding"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    pt_list = glob.glob(os.path.join(original_path, "*.pt"))
    decoder = Prov_decoder(model_path="./checkpoints/pretrain/prov-gigapath/slide_encoder.pth").half().to("cuda")
    decoder.eval()
    with torch.no_grad():
        for pt in tqdm.tqdm(pt_list):
            embeddings, coordinate = read_coordinate_pt(pt)
            embeddings, coordinate = embeddings.half().unsqueeze(0).to("cuda"), coordinate.half().unsqueeze(0).to("cuda")
            output = decoder(embeddings, coordinate)[0]
            torch.save(output, os.path.join(target_path, os.path.basename(pt)))


if __name__ == "__main__":
    compute_tile_feature()
    compute_slide_embedding()
