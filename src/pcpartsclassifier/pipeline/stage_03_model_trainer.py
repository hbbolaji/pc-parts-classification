import torch
from pcpartsclassifier.config.configuration import ConfigurationManager
from pcpartsclassifier.components.model_trainer import Training
from pcpartsclassifier import logger
from pcpartsclassifier.components.base_model import BaseModel

STAGE_NAME = 'Training'

class ModelTrainerTrainingPipeline:
  def __init__(self) -> None:
    pass

  def main(self):
    config = ConfigurationManager()
    training_config = config.get_training_config()
    training = Training(config=training_config)
    training.train()
    training.save_model()


if __name__ == '__main__':
  try:
    logger.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<< \n x===================x')
  except Exception as e:
    raise e

