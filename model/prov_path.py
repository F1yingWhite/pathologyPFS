import os
import torch
import timm
import warnings
from torchvision import transforms
from gigapath import slide_encoder
from PIL import Image
import numpy as np
from dataset.pathology_dataset import PathologyTileDataset

warnings.filterwarnings('ignore')


class Prov_encoder(torch.nn.Module):
    def __init__(self, model_path="../checkpoints/pretrain/prov-gigapath/pytorch_model.bin"):
        super(Prov_encoder, self).__init__()
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.half()  # Convert model to half precision

    def forward(self, x):
        x = x.half()  # Convert input to half precision
        return self.model(x)

    def get_shape(self):
        return 1536


class Prov_decoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.slide_encoder = slide_encoder.create_model("../checkpoints/pretrain/prov-gigapath/slide_encoder.pth", "gigapath_slide_enc12l768d", 1536)
        self.slide_encoder = self.slide_encoder.half()  # Convert slide_encoder to half precision

    def forward(self, x, coords, all_layer_embed=False):
        x = x.half()  # Convert input to half precision
        return self.slide_encoder(x, coords, all_layer_embed)


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    encoder = Prov_encoder().to("cuda").half()
    encoder.eval()
    decoder = Prov_decoder().to("cuda").half()
    decoder.eval()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_dataset = PathologyTileDataset("../data/tile224/2011-38911", transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, pin_memory=True, shuffle=False, num_workers=32, multiprocessing_context='spawn')
    coordinates_list = []
    img_embeddings = []
    with torch.no_grad():
        for img, coordinates in test_loader:
            img = img.to("cuda").half()  # Ensure images are in half precision
            coordinates_list.extend(coordinates)
            img_embeddings.append(encoder(img).cpu().numpy())
        coordinates_list = np.concatenate(coordinates_list, axis=0)
        img_embeddings = np.concatenate(img_embeddings, axis=0)
        coordinates_list = torch.from_numpy(coordinates_list).unsqueeze(0).to("cuda").half()  # Convert to half precision
        img_embeddings = torch.from_numpy(img_embeddings).unsqueeze(0).to("cuda").half()  # Convert to half precision
        output = decoder(img_embeddings, coordinates_list)
        print(output)
