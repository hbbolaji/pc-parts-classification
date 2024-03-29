import torch.nn as nn
from torchvision import models


class BaseModel(nn.Module):
  def __init__(self,  output_feature: int):
    super(BaseModel, self).__init__()
    self.weights = models.EfficientNet_B0_Weights.DEFAULT
    self.transform = self.weights.transforms()
    self.model = models.efficientnet_b0(weights=self.weights)
    print(self.transform)

    for params in self.model.features.parameters():
      params.requires_grad = False

    self.model.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(1280, output_feature)
    )

  def forward(self, x):
    return self.model(x)