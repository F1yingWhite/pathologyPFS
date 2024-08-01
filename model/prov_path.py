import torch
import timm
import warnings
from torchvision import transforms
from gigapath import slide_encoder

warnings.filterwarnings('ignore')


# 模型的transform
# transform = transforms.Compose(
#     [
#         transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ]
# )
class Prov_encoder(torch.nn.Module):
    def __init__(self, model_path="../checkpoints/pretrain/prov-gigapath/pytorch_model.bin"):
        super(Prov_encoder, self).__init__()
        # 创建 ViT 模型
        self.model = timm.create_model(
            'vit_giant_patch14_dinov2',  # 指定模型架构
            pretrained=False,  # 是否加载预训练权重
            num_classes=0,  # 分类数目
            img_size=224,  # 输入图像大小
            in_chans=3,  # 输入通道数
            patch_size=16,  # Patch 大小
            embed_dim=1536,  # 嵌入维度
            depth=40,  # Transformer 的深度
            num_heads=24,  # 多头注意力机制的头数
            mlp_ratio=5.33334,  # MLP 层的维度扩展比
            global_pool='token',  # 全局池化方式
            init_values=1e-05,  # 初始化值
        )

        self.model.load_state_dict(torch.load(model_path))

    def forward(self, x):
        return self.model(x)

    def get_shape(self):
        return 1536


class Prov_decoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.slide_encoder = slide_encoder.create_model("../checkpoints/pretrain/prov-gigapath/slide_encoder.pth", "gigapath_slide_enc12l768d", 1536)

    def forward(self, x, coords, all_layer_embed=False):
        """
        The forward pass of the model

        Arguments:
        ----------
        x: torch.Tensor
            The input tile embeddings, of shape [N, L, D]
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        all_layer_embed: bool
            Whether to return embeddings from all layers or not
        """
        return self.slide_encoder(x, coords, all_layer_embed)


if __name__ == "__main__":
    encoder = Prov_encoder()
    decoder = Prov_decoder()
