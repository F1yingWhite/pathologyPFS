import glob
import os
import shlex
from re import L

import torch
import tqdm
from torchvision import transforms

from dataset.pathology_dataset import PathologyTileDataset
from model.prov_path import Prov_decoder, Prov_encoder
from utils.get_pt import read_coordinate_pt


def compute_tile_feature(original_path="./data/tile224/", target_path="./data/tile224embedding"):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # 获得original_path下的所有文件夹
    folders = [f for f in glob.glob(os.path.join(original_path, "*")) if os.path.isdir(f)]
    encoder = Prov_encoder(model_path="./checkpoints/pretrain/prov-gigapath/pytorch_model.bin").half().to("cuda")
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
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
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=num_workers)
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


def compute_slide_embedding(original_path="./data/tile224embedding", target_path="./data/slide224_level_embedding"):
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


def remake_dir(target_slide_level_path="./data/slide224_level_embedding"):
    path = "./data/WSI/"
    for file in os.listdir(target_slide_level_path):
        file = file[:-3]
        print(file)
        for pos in ["转移瘤2SVS", "转移瘤ES-SVS", "Path"]:
            if glob.glob(os.path.join(path, pos, file + ".*")):
                os.makedirs(os.path.join(target_slide_level_path, pos), exist_ok=True)
                # 把文件拷贝到对应的文件夹
                source = os.path.join(target_slide_level_path, file + ".pt")
                destination = os.path.join(target_slide_level_path, pos)
                os.system("cp {} {}".format(shlex.quote(source), shlex.quote(destination)))


def name_format(prefix="./data/slide224_level_embedding/"):
    path1 = os.path.join(prefix, "转移瘤2SVS")
    # 所有的名称去掉-前面的内容
    for file in os.listdir(path1):
        os.rename(os.path.join(path1, file), os.path.join(path1, file.split("-")[-1]))
    # 删除文件名前缀的0
    for file in os.listdir(path1):
        new_name = file.lstrip('0')
        os.rename(os.path.join(path1, file), os.path.join(path1, new_name))

    path2 = os.path.join(prefix, "转移瘤ES-SVS")
    for file in os.listdir(path2):
        os.rename(os.path.join(path2, file), os.path.join(path2, file.split("-")[-1]))
    # 删除文件名前缀的0
    for file in os.listdir(path2):
        new_name = file.lstrip('0')
        os.rename(os.path.join(path2, file), os.path.join(path2, new_name))

    path3 = os.path.join(prefix, "Path")

    def process_filename(filename):
        # 去掉文件扩展名
        base_name, _ = os.path.splitext(filename)
        # 如果存在'-'，则提取'-'前面的内容
        if '-' in base_name:
            base_name = base_name.split('-')[0].strip()
        else:
            base_name = base_name.split()[0]
        # 返回处理后的内容
        return base_name.replace(' ', '') + ".pt"

    for file in os.listdir(path3):
        new_name = process_filename(file)
        os.rename(os.path.join(path3, file), os.path.join(path3, new_name))


if __name__ == "__main__":
    compute_tile_feature("./data/tile256/", "./data/tile256embedding")
    compute_slide_embedding("./data/tile256embedding", "./data/slide256_level_embedding")
    remake_dir("./data/slide256_level_embedding")
    name_format("./data/slide256_level_embedding")
