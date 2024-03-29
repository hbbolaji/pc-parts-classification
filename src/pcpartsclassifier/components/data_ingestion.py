import requests
import logging
import zipfile
import os
from pathlib import Path
from pcpartsclassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
  def __init__(self, config: DataIngestionConfig):
    self.config = config

  def download_data(self):
    try:
      zip_download_dir = self.config.local_zip_file
      dataset_url = self.config.source_url
      # os.makedirs(zip_download_dir, exist_ok=True)
      logging.info(f'Downloading data from {dataset_url} to {zip_download_dir}')
      response = requests.get(dataset_url)
      with open(zip_download_dir, 'wb') as f:
        f.write(response.content)
      logging.info(f'Downloaded data from {dataset_url} to {zip_download_dir}')
    except Exception as e:
      raise e
    return
  
  def extract_zip_file(self):
    unzip_path = self.config.unzip_dir
    os.makedirs(unzip_path, exist_ok=True)
    print(self.config.local_zip_file)
    print(zipfile.is_zipfile(self.config.local_zip_file))
    with zipfile.ZipFile(Path(self.config.local_zip_file)) as zip_ref:
      zip_ref.extractall(unzip_path)