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