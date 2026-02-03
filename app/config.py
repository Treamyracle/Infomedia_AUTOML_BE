import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Infomedia AutoML Backend"
    API_V1_STR: str = "/api/v1"
    
    # CORS (Frontend URL)
    # Ganti '*' dengan 'http://localhost:5173' untuk keamanan lebih di production
    BACKEND_CORS_ORIGINS: list = ["*"] 
    
    # Path Config
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    MODEL_DIR: str = os.path.join(BASE_DIR, "models")
    
    # API Keys (Load from .env)
    GEMINI_API_KEY: str | None = None

    class Config:
        env_file = ".env"
        case_sensitive = True

# Instance global settings
settings = Settings()

# Pastikan folder ada saat aplikasi start
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)