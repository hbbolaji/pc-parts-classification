import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, models
from pcpartsclassifier import logger
from pcpartsclassifier.entity.config_entity import TrainingConfig, BaseModelConfig
from pcpartsclassifier.utils.common import create_directories
from pcpartsclassifier.components.base_model import BaseModel
from pcpartsclassifier.config.configuration import ConfigurationManager

class Training:
  def __init__(self, config: TrainingConfig) -> None:
    self.config = config
    self.weights = models.EfficientNet_B0_Weights.DEFAULT
    self.transform = self.weights.transforms()
    self.data_dir = os.path.join(self.config.data_dir, self.config.data_name)
    base_config = ConfigurationManager()
    base_model_config = base_config.get_base_model_config()
    self.model = BaseModel(config=base_model_config)
    
  
  def data_preparation(self):
    dataset = datasets.ImageFolder(root=self.data_dir,
                                   transform=self.transform)
    train_set, test_set = random_split(dataset=dataset,
                                       lengths=[0.7, 0.3])
    self.train_loader = DataLoader(dataset=train_set,
                              batch_size=self.config.batch_size,
                              shuffle=True)
    self.test_loader = DataLoader(dataset=test_set,
                            batch_size=self.config.batch_size,
                            shuffle=False)
    self.classes = dataset.classes
  
  def accuracy(self, logits, labels):
    prediction = torch.argmax(logits, dim=1)
    correction = (prediction == labels).sum()
    return correction / len(logits)
  
  def train_loop(self, optimizer, criterion: nn.Module):
    train_loss, train_acc = 0, 0
    for _, (images, labels) in enumerate(self.train_loader):
      self.model.train()
      # forward propagation and loss
      logits = self.model(images)
      loss = criterion(logits, labels)
      acc = self.accuracy(logits=logits, labels=labels)
      train_loss += loss.item()
      train_acc += acc
      # zero_grad
      optimizer.zero_grad()

      # backward propagation
      loss.backward()

      # step
      optimizer.step()
    return train_loss, train_acc

  def test_loop(self, criterion: nn.Module):
    self.model.eval()
    with torch.inference_mode():
      # forward pass
      test_loss, test_acc = 0, 0
      for _, (images, labels) in enumerate(self.test_loader):
        logits = self.model(images)
        loss = criterion(logits, labels)
        test_loss += loss.item()
        acc = self.accuracy(logits=logits, labels=labels)
        test_acc += acc
    return test_loss, test_acc

  def train(self):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=self.model.parameters(),
                                 lr=self.config.learning_rate)
    self.data_preparation()
    for epoch in range(self.config.epochs):
      train_loss, train_acc = self.train_loop(
                      optimizer=optimizer,
                      criterion=criterion)
      train_loss /= len(self.train_loader)
      train_acc /= len(self.train_loader)

      logger.info(f'Train - Epoch: {epoch + 1} / {self.config.epochs} | Accuracy: {100 * train_acc:.2f}% | Loss: {train_loss:.3f}')

      test_loss, test_acc = self.test_loop(
                     criterion=criterion)
      test_loss /= len(self.test_loader)
      test_ac = test_acc / len(self.test_loader)

      logger.info(f'Test - Epoch: {epoch + 1} / {self.config.epochs} | Accuracy: {100 * test_ac:.2f}% | Loss: {test_loss:.3f}')
  
  def save_model(self):
    create_directories([self.config.root_dir])
    torch.save(self.model.state_dict(), self.config.trained_model_path)
      
