import torch
from torch import nn

from model.prov_path import Prov_decoder


class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MySurvivalPredictionModel(nn.Module):
    def __init__(self, hidden_size, output_size, prov_decoder_path="./checkpoints/pretrain/prov-gigapath/slide_encoder.pth"):
        super(MySurvivalPredictionModel, self).__init__()
        self.prov_decoder = Prov_decoder(model_path=prov_decoder_path).half()
        self.regresson = RegressionModel(1536, hidden_size, output_size)

    def forward(self, x, position):
        out = self.prov_decoder(x, position)
        out = self.regresson(out)
        return out


if __name__ == "__main__":
    model = MySurvivalPredictionModel(2048, 1)
    x = torch.randn(1, 3, 224, 224).half()
    y = model(x)
    print(y)
