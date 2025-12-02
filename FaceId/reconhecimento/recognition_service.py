"""
Serviço Central de Reconhecimento Facial - VERSÃO CORRIGIDA
"""
import logging
import time
import threading
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import cv2

# Módulos compartilhados
from common.config import MODEL_CONFIG
from common.exceptions import ImageValidationError, FaceRecognitionServiceError
from common.image_utils import ImageValidator, FaceQualityValidator
from face_recognition_ia import FaceRecognizer

logger = logging.getLogger(__name__)

@dataclass
class RecognitionResult:
    """Resultado do reconhecimento facial com campos completos"""
    authenticated: bool
    confidence: float
    message: str
    timestamp: str
    user: Optional[str] = None
    distance: Optional[float] = None
    user_info: Optional[Dict[str, Any]] = None
    id: Optional[int] = None
    username: Optional[str] = None
    tipo_usuario: Optional[str] = None
    nome: Optional[str] = None
    sobrenome: Optional[str] = None
    turma: Optional[str] = None

class RecognitionMetrics:
    """Métricas de desempenho do reconhecimento"""
    def __init__(self):
        self.total_attempts = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0
        self.no_face_detected = 0
        self.processing_times = []
        self._lock = threading.RLock()

class RecognitionService:
    """Serviço principal de orquestração do reconhecimento facial"""

    def __init__(self):
        self.face_recognizer = FaceRecognizer()
        self.quality_validator = FaceQualityValidator(
            min_face_size=(50, 50),
            min_sharpness=80.0
        )
        self.metrics = RecognitionMetrics()

    def initialize(self) -> bool:
        """Inicializa o serviço"""
        logger.info("Inicializando Serviço de Reconhecimento Facial...")
        return self.face_recognizer.initialize()

    def process_face_login(self, image_data: str) -> Dict[str, Any]:
        """
        Processa o login facial a partir de dados de imagem base64

        Args:
            image_data: Imagem em formato base64

        Returns:
            Dict: Resultado do reconhecimento
        """
        start_time = time.time()

        with self.metrics._lock:
            self.metrics.total_attempts += 1

        try:
            # 1. Decodificar imagem usando common.image_utils
            frame = ImageValidator.decode_base64_image(image_data)
            if frame is None:
                raise ImageValidationError("Dados de imagem inválidos")

            # 2. Pré-processar imagem
            frame = ImageValidator.preprocess_image(frame)

            # 3. Verificar se o banco está vazio
            if not self.face_recognizer.facial_database:
                logger.warning("Banco de dados vazio - tentando recarregar...")
                self.face_recognizer.load_facial_database()

                if not self.face_recognizer.facial_database:
                    with self.metrics._lock:
                        self.metrics.failed_recognitions += 1

                    return RecognitionResult(
                        authenticated=False,
                        confidence=0.0,
                        message="Nenhum usuário cadastrado no sistema.",
                        timestamp=self.get_current_timestamp()
                    ).__dict__

            # 4. Detectar rosto
            face_data = self.face_recognizer._detect_face(frame)
            if not face_data:
                with self.metrics._lock:
                    self.metrics.failed_recognitions += 1
                    self.metrics.no_face_detected += 1

                return RecognitionResult(
                    authenticated=False,
                    confidence=0.0,
                    message="Nenhum rosto detectado - posicione-se melhor na frente da câmera",
                    timestamp=self.get_current_timestamp()
                ).__dict__

            # 5. Extrair ROI do rosto
            face_area = face_data["facial_area"]
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            face_roi = frame[y:y+h, x:x+w]

            # 6. Validar qualidade do rosto
            is_valid, validation_msg = self.quality_validator.validate_face_image(face_roi)
            if not is_valid:
                with self.metrics._lock:
                    self.metrics.failed_recognitions += 1

                return RecognitionResult(
                    authenticated=False,
                    confidence=0.0,
                    message=f"Qualidade do rosto insuficiente: {validation_msg}",
                    timestamp=self.get_current_timestamp()
                ).__dict__

            # 7. Reconhecer rosto usando IA
            user, distance, user_data = self.face_recognizer.recognize_face(face_roi)

            # 8. Processar resultado
            processing_time = time.time() - start_time
            with self.metrics._lock:
                self.metrics.processing_times.append(processing_time)

                if user and user_data:
                    self.metrics.successful_recognitions += 1
                    confidence = 1 - distance

                    # Garantir que tipo_usuario está em maiúsculas
                    tipo_usuario = user_data['tipo_usuario'].upper() if user_data.get('tipo_usuario') else None

                    # Mensagens personalizadas
                    if tipo_usuario == "PROFESSOR":
                        if user_data.get('username'):
                            message = f"Bem-vindo(a), Professor(a) {user_data['nome']}!"
                        else:
                            message = f"Bem-vindo(a), Professor(a) {user_data['nome']}!"
                    else:
                        message = f"Bem-vindo(a), {user_data['nome']}!"

                    # Log detalhado
                    logger.info(f"RECONHECIMENTO BEM SUCEDIDO - Tipo: {tipo_usuario}, Nome: {user_data.get('nome')}, Tempo: {processing_time:.3f}s")

                    # Construir resultado
                    result = {
                        "authenticated": True,
                        "user": user,
                        "confidence": round(confidence, 4),
                        "distance": round(distance, 4),
                        "message": message,
                        "timestamp": self.get_current_timestamp(),
                        "id": user_data.get('id'),
                        "username": user_data.get('username'),
                        "tipo_usuario": tipo_usuario,
                        "nome": user_data.get('nome'),
                        "sobrenome": user_data.get('sobrenome'),
                        "turma": user_data.get('turma'),
                        "user_info": user_data
                    }

                    logger.info(f"RESULTADO ENVIADO AO FRONT-END: {result}")
                    return result
                else:
                    self.metrics.failed_recognitions += 1

            logger.info(f"Tempo de processamento: {processing_time:.3f}s")

            return RecognitionResult(
                authenticated=False,
                confidence=0.0,
                message="Usuário não reconhecido - verifique se está cadastrado no sistema",
                timestamp=self.get_current_timestamp()
            ).__dict__

        except Exception as e:
            with self.metrics._lock:
                self.metrics.failed_recognitions += 1

            logger.error(f"Erro no processamento: {str(e)}", exc_info=True)

            return RecognitionResult(
                authenticated=False,
                confidence=0.0,
                message="Erro interno no processamento",
                timestamp=self.get_current_timestamp()
            ).__dict__
    def _detect_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detecta rostos na imagem"""
        return self.face_recognizer._detect_face(image)

    def get_database_status(self) -> Dict[str, Any]:
        """Retorna status do banco de dados"""
        return self.face_recognizer.get_database_status()

    def get_detailed_database_status(self) -> Dict[str, Any]:
        """Retorna status detalhado do banco de dados"""
        return self.face_recognizer.get_detailed_database_status()

    def reload_database(self) -> Tuple[bool, str]:
        """Recarrega banco de dados manualmente"""
        return self.face_recognizer.reload_database()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de desempenho"""
        with self.metrics._lock:
            times = self.metrics.processing_times
            avg_time = sum(times) / len(times) if times else 0

            success_rate = (
                self.metrics.successful_recognitions / self.metrics.total_attempts
                if self.metrics.total_attempts > 0 else 0
            )

            return {
                "total_attempts": self.metrics.total_attempts,
                "successful_recognitions": self.metrics.successful_recognitions,
                "failed_recognitions": self.metrics.failed_recognitions,
                "no_face_detected": self.metrics.no_face_detected,
                "average_processing_time": round(avg_time, 3),
                "success_rate": round(success_rate, 4)
            }

    @staticmethod
    def get_current_timestamp() -> str:
        """Retorna timestamp atual formatado"""
        return datetime.now().isoformat(sep=' ', timespec='seconds')

    def cleanup(self):
        """Limpeza do serviço"""
        self.face_recognizer.cleanup()
        logger.info("RecognitionService finalizado")