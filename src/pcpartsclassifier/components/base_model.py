import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

from pcpartsclassifier.entity.config_entity import BaseModelConfig
from pcpartsclassifier.utils.common import create_directories


class BaseModel(nn.Module):
  def __init__(self,  config: BaseModelConfig):
    super(BaseModel, self).__init__()
    self.config = config
    self.weights = models.EfficientNet_B0_Weights.DEFAULT
    self.transform = self.weights.transforms()
    self.model = models.efficientnet_b0(weights=self.weights)

    for params in self.model.features.parameters():
      params.requires_grad = False

    self.model.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(1280, self.config.classes)
    )

  def forward(self, x):
    return self.model(x)
  
  def save_untrained_model(self) -> None:
    create_directories([self.config.root_dir])
    torch.save(self.model.state_dict(), self.config.untrained_base_model)

  @staticmethod
  def save_model(model:nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)