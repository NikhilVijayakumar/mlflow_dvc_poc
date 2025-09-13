import os
from pathlib import Path
import yaml
from pydantic import BaseModel, DirectoryPath, FilePath
from pydantic_settings import BaseSettings

# Define project's root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class PathsConfig(BaseModel):
    """Pydantic model for file paths."""
    raw_data: str
    processed_data: str
    train_data: str
    test_data: str
    model: str
    reports: str

class TrainingConfig(BaseModel):
    model_name: str
    max_iter: int
    test_size: float
    random_state: int
    registered_model_name: str

class MinioConfig(BaseModel):
    endpoint: str
    bucket_name: str

class MlflowConfig(BaseModel):
    experiment_name: str

class AppConfig(BaseModel):
    paths: PathsConfig
    training: TrainingConfig
    mlflow: MlflowConfig  # <-- This is the newly added line
    minio: MinioConfig

class EnvConfig(BaseSettings):
    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str

    class Config:
        # Specifies the .env file to load
        env_file = ROOT_DIR / "config" / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Ignore extra variables in the .env file

def load_config(config_path: Path = ROOT_DIR / "config" / "config.yaml") -> AppConfig:
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return AppConfig(**config_data)
    except FileNotFoundError:
        raise Exception(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise Exception(f"Error parsing YAML configuration: {e}")


app_config = load_config()
env_config = EnvConfig()

