from pcpartsclassifier import logger
from pcpartsclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STATE_NAME = "Data Ingestion Stage"

if __name__ == '__main__':
  try:
    logger.info(f'>>>>>>> {STATE_NAME} started <<<<<<<')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> {STATE_NAME} completed <<<<<<< \n \n x===================x')
  except Exception as e:
    logger.exception(e)
    raise e