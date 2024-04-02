import io
import torch
from pathlib import Path
from PIL import Image

from pcpartsclassifier.components.base_model import BaseModel
from pcpartsclassifier.config.configuration import ConfigurationManager
from pcpartsclassifier.components.model_trainer import Training
from pcpartsclassifier.components.model_evaluation import ModelEvaluation



class PredictionPipeline:
  def __init__(self) -> None:
    self.config_manager = ConfigurationManager()
    pass

  def transform_image(self, image_bytes):
    training = Training(config=self.config_manager.get_training_config())
    training.data_preparation()
    transform = training.transform
    # print(transform)
    self.classes = training.classes
    # print(self.classes)
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image)
    self.image = image.unsqueeze(dim=0)
  
  def get_prediction(self):
    model = BaseModel(config=self.config_manager.get_base_model_config())
    model.load_state_dict(torch.load('artifacts/training/trained_model.pth'))
    model.eval()
    logits = model(self.image)
    preds = torch.argmax(logits, dim=1)

    # confidence
    conf = torch.max(torch.softmax(logits, dim=1))
    conf = round(100 * conf.item(), 2)
    return {'prediction': self.classes[preds.item()],
            'confidence': conf}

# classifier = PredictionPipeline()
# classifier.transform_image()