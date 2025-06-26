import torch
from torch import nn


class ClassificationModule(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone_model = backbone
        self.loss_fn = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = self.backbone_model(x)
        return self.loss_fn(y_pred, y_true)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_model(x)

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = self.predict(x)
        return torch.argmax(y_pred, dim=1)
