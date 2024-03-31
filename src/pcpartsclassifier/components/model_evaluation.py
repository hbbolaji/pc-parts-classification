# Evaluation Component
import torch
import mlflow
from urllib.parse import urlparse
from pathlib import Path

from pcpartsclassifier.config.configuration import ConfigurationManager as CM
from pcpartsclassifier.components.base_model import BaseModel
from pcpartsclassifier.components.model_trainer import Training
from pcpartsclassifier.utils.common import save_json
from pcpartsclassifier.entity.config_entity import EvaluationConfig

class ModelEvaluation:
  def __init__(self, config: EvaluationConfig) -> None:
    self.config = config

    self.config_manager = CM()
  
  def get_eval_data(self):
    training_config = self.config_manager.get_training_config()
    training = Training(config=training_config)
    training.data_preparation()
    self.accuracy = training.accuracy
    self.validation_loader = training.validation_loader
    
  @staticmethod
  def load_model(model_path: Path, model_config):
    model = BaseModel(config=model_config)
    model.load_state_dict(torch.load(model_path))
    return model
  
  def evaluate(self):
    model_config = self.config_manager.get_base_model_config()
    criterion = torch.nn.CrossEntropyLoss()
    self.model = self.load_model(model_path=self.config.model_path,
                                 model_config=model_config)
    self.get_eval_data()
    self.model.eval()
    with torch.inference_mode():
      eval_loss, eval_acc = 0, 0
      for i, (images, labels) in enumerate(self.validation_loader):
        logits = self.model(images)
        loss = criterion(logits, labels)
        acc = self.accuracy(logits, labels)

        eval_loss += loss.item()
        eval_acc += acc.item()

      eval_loss /= len(self.validation_loader)
      eval_acc /= len(self.validation_loader)

      self.scores = eval_loss, 100 * eval_acc
  
  def save_score(self):
    scores = {'Loss': self.scores[0], 'Accuracy': self.scores[1]}
    save_json(path=Path('scores.json'), data=scores)

  def log_into_mlflow(self):
    tracking_uri = self.config.mlflow_uri
    mlflow.set_registry_uri(tracking_uri)

    tracking_uri_type_store = urlparse(mlflow.get_registry_uri()).scheme

    with mlflow.start_run():
      mlflow.log_params(self.config.all_params)
      mlflow.log_metrics({'loss': self.scores[0], 'accuracy': self.scores[1]})

      # model registery is not tracked with file store
      if tracking_uri_type_store != "file":
        mlflow.pytorch.log_model(self.model, 'model', registered_model_name="efficientnet_b0")
      else:
        mlflow.pytorch.log_model(self.model, 'model')