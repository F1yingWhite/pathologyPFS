import timm
import torch
import os
import torchsummary
import warnings

warnings.filterwarnings("ignore")


class UNIPretrain(torch.nn.Module):
    def __init__(self, model_path="../checkpoints/pretrain/UNI/pytorch_model.bin"):
        super(UNIPretrain, self).__init__()
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.model.load_state_dict(
            torch.load(model_path),
            strict=True,
        )
        self.output_shape = 1024

    def forward(self, x):
        """模型输入维度

        Args:
            x (float_tensor): (batch,3,224,224)

        Returns:
            embedding of x: 输入图片的EMbedding
        """
        return self.model(x)

    def get_shape(self):
        """返回模型的输出维度

        Returns:
            int: 模型的返回维度大小
        """
        return self.output_shape


if __name__ == "__main__":
    model = UNIPretrain()
    model = model.to("cuda")
    torchsummary.summary(model, (3, 224, 224))
    print(model(torch.rand(1, 3, 224, 224).to("cuda")).shape)
