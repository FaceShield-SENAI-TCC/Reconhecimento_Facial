"""
Modelos de dados compartilhados entre cadastro e reconhecimento
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class UserInfo:
    """Informações do usuário"""
    nome: str
    sobrenome: str
    turma: str
    tipo: str = "aluno"
    display_name: Optional[str] = None

    def __post_init__(self):
        if not self.display_name:
            self.display_name = f"{self.nome} {self.sobrenome}"


@dataclass
class FacialEmbedding:
    """Embedding facial com metadados"""
    embedding: List[float]
    user_info: UserInfo
    quality_score: float
    timestamp: datetime


@dataclass
class RecognitionResult:
    """Resultado do reconhecimento facial"""
    authenticated: bool
    user: Optional[str] = None
    confidence: float = 0.0
    distance: float = 1.0
    user_info: Optional[UserInfo] = None
    message: str = ""
    timestamp: Optional[str] = None


@dataclass
class DatabaseStatus:
    """Status do banco de dados"""
    status: str
    user_count: int
    total_embeddings: int
    last_update: Optional[float] = None
    monitoring_active: bool = False
    database_type: str = "PostgreSQL"


@dataclass
class SystemMetrics:
    """Métricas de desempenho do sistema"""
    total_attempts: int
    successful_recognitions: int
    failed_recognitions: int
    no_face_detected: int
    average_processing_time: float
    success_rate: float
    database_reloads: int = 0