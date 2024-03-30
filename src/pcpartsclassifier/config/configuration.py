from pcpartsclassifier.constants import *
from pcpartsclassifier.utils.common import read_yaml, create_directories
from pcpartsclassifier.entity.config_entity import DataIngestionConfig, BaseModelConfig

class ConfigurationManager:
  def __init__(self,
               config_path = CONFIG_FILE_PATH,
               params_path = PARAMS_FILE_PATH):
    self.config = read_yaml(config_path)
    self.params = read_yaml(params_path)
    create_directories([self.config.artifacts_root])

  def get_data_ingestion_config(self) -> DataIngestionConfig:
    config = self.config.data_ingestion
    create_directories([config.root_dir])

    data_ingestion_config = DataIngestionConfig(
      root_dir=config.root_dir,
      source_url=config.source_url,
      local_zip_file=config.local_zip_file,
      unzip_dir=config.unzip_dir
    )

    return data_ingestion_config
  
  def get_base_model_config(self) -> BaseModelConfig:
    config = self.config.base_model
    base_model_config = BaseModelConfig(
      root_dir=config.root_dir,
      untrained_base_model=config.untrained_base_model,
      trained_base_model=config.trained_base_model,
      classes = self.params.CLASSES
    )
    return base_model_config