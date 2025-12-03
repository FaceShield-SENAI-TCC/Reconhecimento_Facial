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
        self.security_rejections = 0  # Novas rejeições por segurança
        self.processing_times = []
        self._lock = threading.RLock()

class RecognitionService:
    """Serviço principal de orquestração do reconhecimento facial"""

    def __init__(self):
        self.face_recognizer = FaceRecognizer()
        self.quality_validator = FaceQualityValidator(
            min_face_size=MODEL_CONFIG.MIN_FACE_SIZE,
            min_sharpness=90.0
        )
        self.metrics = RecognitionMetrics()

    def initialize(self) -> bool:
        """Inicializa o serviço"""
        logger.info("Inicializando Serviço de Reconhecimento Facial...")
        logger.info("Configurações de segurança ATIVADAS:")
        logger.info(f"  Confiança mínima: {MODEL_CONFIG.MIN_CONFIDENCE}")
        logger.info(f"  Distância máxima: {MODEL_CONFIG.DISTANCE_THRESHOLD}")
        logger.info(f"  Tamanho mínimo do rosto: {MODEL_CONFIG.MIN_FACE_SIZE}")
        return self.face_recognizer.initialize()

    def _validate_image_for_anti_spoofing(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Validações anti-spoofing para prevenir fotos de celular"""
        try:
            # Verificar se a imagem não está muito escura ou clara (indicativo de foto)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            mean_intensity = np.mean(gray)

            if mean_intensity < 30 or mean_intensity > 220:
                return False, f"Intensidade luminosa fora do padrão: {mean_intensity:.1f}"

            # Verificar variância (fotos de celular tendem a ter variância diferente)
            variance = np.var(gray)

            if variance < 100:
                return False, f"Variância muito baixa (possível foto): {variance:.1f}"

            # Verificar histograma (fotos tendem a ter histogramas mais concentrados)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / hist.sum()  # Normalizar

            entropy = -np.sum(hist * np.log2(hist + 1e-10))

            if entropy < 4.0:  # Entropia baixa pode indicar foto
                return False, f"Entropia muito baixa (possível foto): {entropy:.2f}"

            return True, "Imagem válida para processamento"

        except Exception as e:
            logger.error(f"Erro na validação anti-spoofing: {str(e)}")
            return True, "Validação ignorada devido a erro"  # Permite continuar em caso de erro

    def process_face_login(self, image_data: str) -> Dict[str, Any]:
        """
        Processa o login facial a partir de dados de imagem base64
        """
        start_time = time.time()

        with self.metrics._lock:
            self.metrics.total_attempts += 1

        try:
            # 1. Decodificar imagem usando common.image_utils
            frame = ImageValidator.decode_base64_image(image_data)
            if frame is None:
                raise ImageValidationError("Dados de imagem inválidos")

            # 2. Validação anti-spoofing
            spoof_valid, spoof_msg = self._validate_image_for_anti_spoofing(frame)
            if not spoof_valid:
                with self.metrics._lock:
                    self.metrics.security_rejections += 1
                    self.metrics.failed_recognitions += 1

                logger.warning(f"Validação anti-spoofing falhou: {spoof_msg}")
                return RecognitionResult(
                    authenticated=False,
                    confidence=0.0,
                    message="Imagem não parece ser de uma câmera ao vivo",
                    timestamp=self.get_current_timestamp()
                ).__dict__

            # 3. Pré-processar imagem
            frame = ImageValidator.preprocess_image(frame)

            # 4. Verificar se o banco está vazio
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

            # 5. Detectar rosto com critérios restritivos
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

            # 6. Extrair ROI do rosto
            face_area = face_data["facial_area"]
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']

            # Verificar tamanho mínimo
            min_w, min_h = MODEL_CONFIG.MIN_FACE_SIZE
            if w < min_w or h < min_h:
                with self.metrics._lock:
                    self.metrics.failed_recognitions += 1

                return RecognitionResult(
                    authenticated=False,
                    confidence=0.0,
                    message=f"Rosto muito pequeno - aproxime-se da câmera",
                    timestamp=self.get_current_timestamp()
                ).__dict__

            face_roi = frame[y:y+h, x:x+w]

            # 7. Validar qualidade do rosto com critérios restritivos
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

            # 8. Reconhecer rosto usando IA com critérios de segurança
            user, distance, user_data = self.face_recognizer.recognize_face(face_roi)

            # 9. Processar resultado
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
                    logger.info(f"RECONHECIMENTO BEM SUCEDIDO - Tipo: {tipo_usuario}, "
                              f"Nome: {user_data.get('nome')}, Distância: {distance:.4f}, "
                              f"Confiança: {confidence:.4f}, Tempo: {processing_time:.3f}s")

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

            security_rejection_rate = (
                self.metrics.security_rejections / self.metrics.total_attempts
                if self.metrics.total_attempts > 0 else 0
            )

            return {
                "total_attempts": self.metrics.total_attempts,
                "successful_recognitions": self.metrics.successful_recognitions,
                "failed_recognitions": self.metrics.failed_recognitions,
                "security_rejections": self.metrics.security_rejections,
                "no_face_detected": self.metrics.no_face_detected,
                "average_processing_time": round(avg_time, 3),
                "success_rate": round(success_rate, 4),
                "security_rejection_rate": round(security_rejection_rate, 4)
            }

    @staticmethod
    def get_current_timestamp() -> str:
        """Retorna timestamp atual formatado"""
        return datetime.now().isoformat(sep=' ', timespec='seconds')

    def cleanup(self):
        """Limpeza do serviço"""
        self.face_recognizer.cleanup()
        logger.info("RecognitionService finalizado")