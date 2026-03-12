import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    CUDA: str = "cuda" if torch.cuda.is_available() else "cpu"
    STREAM_URL: str


Settings = Settings()  # type: ignore
