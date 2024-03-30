from pcpartsclassifier.components.base_model import BaseModel
from pcpartsclassifier import logger
from pcpartsclassifier.config.configuration import ConfigurationManager

STAGE_NAME = 'Base Model'

class BaseModelTrainingPipeline:
  def __init__(self) -> None:
    pass
  def main(self):
    config = ConfigurationManager()
    base_model_config = config.get_base_model_config()
    base_model = BaseModel(config=base_model_config)
    base_model.save_untrained_model()
    return base_model

if __name__ == '__main__':
  try:
    logger.info(f'>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<')
    model = BaseModelTrainingPipeline()
    model.main()
    logger.info(f'>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<< \n x===================x')
  except Exception as e:
    logger.exception(e)
    raise e