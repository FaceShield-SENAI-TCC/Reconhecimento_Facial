"""
Configuracoes centralizadas para todo o sistema
"""
import os
from dataclasses import dataclass
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """Configuracoes do banco de dados PostgreSQL"""
    DB_NAME: str = os.getenv("DB_NAME", "faceshield")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "root")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")

    @classmethod
    def to_dict(cls):
        """Converte configuracao para dicionario"""
        return {
            "dbname": cls.DB_NAME,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD,
            "host": cls.DB_HOST,
            "port": cls.DB_PORT
        }

@dataclass
class ModelConfig:
    """Configuracoes do modelo de reconhecimento facial - CRITERIOS RESTRITIVOS"""
    MODEL_NAME: str = "VGG-Face"
    EMBEDDING_DIMENSION: int = 2622
    DISTANCE_THRESHOLD: float = 0.15  #  ALTERADO: 0.15 = 85% de confiança mínima
    MIN_FACE_SIZE: Tuple[int, int] = (100, 100)  #  AUMENTADO: Rosto maior
    DETECTOR_BACKEND: str = "opencv"
    MIN_CONFIDENCE: float = 0.85  #  ALTERADO: 85% mínimo
    MARGIN_REQUIREMENT: float = 0.01  #  AUMENTADO: Margem mais restritiva

@dataclass
class SecurityConfig:
    """Configuracoes de seguranca"""
    SECRET_KEY: str = os.getenv("SECRET_KEY", "face-recognition-secure-key-change-in-production")
    TOKEN_EXPIRATION_HOURS: int = 24
    ALGORITHM: str = "HS256"
    MAX_FILE_SIZE: int = 16 * 1024 * 1024

@dataclass
class AppConfig:
    """Configuracoes gerais da aplicacao"""
    MIN_PHOTOS_REQUIRED: int = 8
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    SERVER_PORT_CADASTRO: int = int(os.getenv("SERVER_PORT_CADASTRO", "7001"))
    SERVER_PORT_RECONHECIMENTO: int = int(os.getenv("SERVER_PORT_RECONHECIMENTO", "5005"))

@dataclass
class WebSocketConfig:
    """Configuracoes WebSocket para ESP32"""

    ESP32_IP: str = os.getenv("ESP32_IP", "10.110.22.11")
    ESP32_PORT: int = int(os.getenv("ESP32_PORT", "8765"))
    TIMEOUT_ABERTURA: int = 180
    TIMEOUT_FECHAMENTO: int = 10
    RECONNECT_ATTEMPTS: int = 3

# Instancias globais
DATABASE_CONFIG = DatabaseConfig()
MODEL_CONFIG = ModelConfig()
SECURITY_CONFIG = SecurityConfig()
APP_CONFIG = AppConfig()
WEBSOCKET_CONFIG = WebSocketConfig()