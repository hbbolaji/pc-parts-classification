from pcpartsclassifier.config.configuration import ConfigurationManager
from pcpartsclassifier.components.data_ingestion import DataIngestion
from pcpartsclassifier import logger

STATE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
  def __init__(self):
    pass

  def main(self):
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    # data_ingestion.download_data()
    data_ingestion.extract_zip_file()

if __name__ == '__main__':
  try:
    logger.info(f'>>>>>>> {STATE_NAME} started <<<<<<<')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> {STATE_NAME} completed <<<<<<< \n \n x===================x')
  except Exception as e:
    logger.exception(e)
    raise e