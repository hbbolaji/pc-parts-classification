from pcpartsclassifier.components.base_model import BaseModel
from pcpartsclassifier import logger

STAGE_NAME = 'Base Model'

class BaseModelTrainingPipeline:
  def __init__(self) -> None:
    pass
  def main():
    model = BaseModel(14)
    return model

if __name__ == '__main__':
  try:
    logger.info(f'>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<')
    model = BaseModelTrainingPipeline()
    model.main()
    logger.info(f'>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<< \n x===================x')
  except Exception as e:
    logger.exception(e)
    raise e