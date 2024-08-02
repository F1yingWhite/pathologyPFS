import torch


def read_coordinate_pt(file_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """读取保存的pt文件

    Args:
        file_path (str): pt文件路径

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 分别返回embedding和coordinate
    """
    data = torch.load(file_path, weights_only=True)
    embedding = data[:, :-2]
    coordinate = data[:, -2:].int()
    return embedding, coordinate


if __name__ == "__main__":
    embedding, coordinate = read_coordinate_pt("../data/tile224embedding/115787-2 - 2023-10-19 19.pt")
    print(embedding.shape)
    print(coordinate.shape)
    print(embedding)
    print(coordinate)
