"""
Configurações centralizadas para todo o sistema
"""
import os
from dataclasses import dataclass
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """Configurações do banco de dados PostgreSQL"""
    DB_NAME: str = os.getenv("DB_NAME", "faceshield")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "root")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")

    @classmethod
    def to_dict(cls):
        """Converte configuração para dicionário"""
        return {
            "dbname": cls.DB_NAME,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD,
            "host": cls.DB_HOST,
            "port": cls.DB_PORT
        }

@dataclass
class ModelConfig:
    """Configurações do modelo de reconhecimento facial - COMPATÍVEL ENTRE CADASTRO E RECONHECIMENTO"""
    MODEL_NAME: str = "VGG-Face"
    EMBEDDING_DIMENSION: int = 2622
    DISTANCE_THRESHOLD: float = 0.60
    MIN_FACE_SIZE: Tuple[int, int] = (100, 100)
    DETECTOR_BACKEND: str = "skip"
    MIN_CONFIDENCE: float = 0.70
    MARGIN_REQUIREMENT: float = 0.015

@dataclass
class SecurityConfig:
    """Configurações de segurança"""
    SECRET_KEY: str = os.getenv("SECRET_KEY", "face-recognition-secure-key-change-in-production")
    TOKEN_EXPIRATION_HOURS: int = 24
    ALGORITHM: str = "HS256"
    MAX_FILE_SIZE: int = 16 * 1024 * 1024  # 16MB

@dataclass
class AppConfig:
    """Configurações gerais da aplicação"""
    MIN_PHOTOS_REQUIRED: int = 8
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    SERVER_PORT_CADASTRO: int = int(os.getenv("SERVER_PORT_CADASTRO", "7001"))
    SERVER_PORT_RECONHECIMENTO: int = int(os.getenv("SERVER_PORT_RECONHECIMENTO", "5005"))

# Instâncias globais
DATABASE_CONFIG = DatabaseConfig()
MODEL_CONFIG = ModelConfig()
SECURITY_CONFIG = SecurityConfig()
APP_CONFIG = AppConfig()