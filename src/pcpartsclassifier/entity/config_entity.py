from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig: 
  root_dir: Path
  source_url: str
  local_zip_file: Path
  unzip_dir: Path

@dataclass(frozen=True)
class BaseModelConfig:
  root_dir: Path
  untrained_base_model: Path
  trained_base_model: Path
  classes: int

@dataclass(frozen=True)
class TrainingConfig:
  root_dir: Path
  data_dir: Path
  trained_model_path: Path
  epochs: int
  learning_rate: float
  batch_size: int
  data_name: str