from pcpartsclassifier import logger
from pcpartsclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from pcpartsclassifier.pipeline.stage_02_base_model import BaseModelTrainingPipeline
from pcpartsclassifier.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline
from pcpartsclassifier.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"

if __name__ == '__main__':
  try:
    logger.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<< \n x===================x')
  except Exception as e:
    logger.exception(e)
    raise e
  

STAGE_NAME = 'Base Model'

if __name__ == '__main__':
  try:
    logger.info(f'>>>>>>>>> {STAGE_NAME} started <<<<<<<<<<')
    model = BaseModelTrainingPipeline()
    model.main()
    logger.info(f'>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<< \n x===================x')
  except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = 'Training'

if __name__ == '__main__':
  try:
    logger.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<<< \n x===================x')
  except Exception as e:
    raise e
  
  
STAGE_NAME = 'Model Evaluation'

if __name__ == '__main__':
  try:
    logger.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<< \n x===================x')
  except Exception as e:
    logger.exception(e)
    raise e