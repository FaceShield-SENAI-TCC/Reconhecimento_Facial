"""
Serviço de Reconhecimento Facial AJUSTADO para câmeras de qualidade inferior
"""

import logging
import time
import threading
import numpy as np
import cv2
import base64
import psycopg2
import psycopg2.extensions
from datetime import datetime
from deepface import DeepFace
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Configurações do banco de dados PostgreSQL"""
    DB_NAME = "faceshild"
    DB_USER = "postgres"
    DB_PASSWORD = "root"
    DB_HOST = "localhost"
    DB_PORT = "5432"

class ModelConfig:
    """Configurações do modelo de reconhecimento - VGG-Face AJUSTADO"""
    MODEL_NAME = "VGG-Face"

    # ✅ THRESHOLD MAIS TOLERANTE
    DISTANCE_THRESHOLD = 0.65  # ✅ AUMENTADO de 0.55 para 0.65 (mais tolerante)

    # ✅ CONFIANÇA MÍNIMA REDUZIDA
    MIN_CONFIDENCE_THRESHOLD = 0.6  # ✅ REDUZIDO de 0.7 para 0.6

    MIN_FACE_SIZE = (80, 80)  # ✅ REDUZIDO tamanho mínimo

    # ✅ DETECTOR MAIS TOLERANTE
    DETECTOR_BACKEND = "opencv"  # ✅ Alterado para opencv (mais tolerante)

    EMBEDDING_DIMENSION = 2622

    # ✅ PARÂMETROS DE QUALIDADE REDUZIDOS
    MIN_SHARPNESS = 40  # ✅ REDUZIDO de 80 para 40
    MIN_BRIGHTNESS = 30  # ✅ REDUZIDO brilho mínimo
    MAX_BRIGHTNESS = 220 # ✅ AUMENTADO brilho máximo

# ... (DatabaseMonitor mantido igual) ...

class FaceRecognitionService:
    """
    Serviço principal de reconhecimento facial AJUSTADO
    """

    def __init__(self):
        self.facial_database = {}
        self.last_update = None
        self._db_config = DatabaseConfig()
        self._model_config = ModelConfig()
        self.database_monitor = DatabaseMonitor(self.load_facial_database)

        # Estatísticas de reconhecimento
        self.recognition_stats = {
            'total_attempts': 0,
            'successful_auth': 0,
            'failed_auth': 0,
            'quality_rejections': 0
        }

    # ... (métodos de banco mantidos) ...

    def _calculate_sharpness(self, image):
        """Calcula nitidez da imagem - VERSÃO SIMPLIFICADA"""
        if image is None or image.size == 0:
            return 0
        try:
            small_img = cv2.resize(image, (100, 100))
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 0

    def _validate_face_quality(self, face_image: np.ndarray) -> Tuple[bool, str]:
        """
        ✅ VALIDAÇÃO SIMPLIFICADA da qualidade da face
        """
        if face_image is None or face_image.size == 0:
            return False, "Imagem vazia"

        height, width = face_image.shape[:2]
        if height < self._model_config.MIN_FACE_SIZE[0] or width < self._model_config.MIN_FACE_SIZE[1]:
            return False, "Rosto muito pequeno"

        # Validar nitidez (mais tolerante)
        sharpness = self._calculate_sharpness(face_image)
        if sharpness < self._model_config.MIN_SHARPNESS:
            return False, f"Imagem um pouco borrada: {sharpness:.1f}"

        # ✅ VALIDAÇÃO DE BRILHO SIMPLIFICADA
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            if brightness < self._model_config.MIN_BRIGHTNESS:
                return False, f"Brilho muito baixo: {brightness:.1f}"
            if brightness > self._model_config.MAX_BRIGHTNESS:
                return False, f"Brilho muito alto: {brightness:.1f}"
        except:
            # Se falhar na análise, continua mesmo assim
            pass

        return True, f"Qualidade aceitável: Sharp={sharpness:.1f}"

    def _extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrai embedding facial com configuração TOLERANTE
        """
        try:
            # ✅ PRÉ-PROCESSAMENTO SIMPLIFICADO
            try:
                # Tenta melhorar o contraste
                lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = cv2.createCLAHE(clipLimit=1.0).apply(lab[:,:,0])  # ✅ Clip reduzido
                processed_face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except:
                processed_face = face_image

            result = DeepFace.represent(
                img_path=processed_face,
                model_name=self._model_config.MODEL_NAME,
                detector_backend=self._model_config.DETECTOR_BACKEND,
                enforce_detection=False,
                align=False  # ✅ Alinhamento desativado para performance
            )

            if result and isinstance(result, list) and "embedding" in result[0]:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)

                if embedding.shape[0] != self._model_config.EMBEDDING_DIMENSION:
                    logger.error(f"Dimensão incorreta: {embedding.shape[0]}")
                    return None

                embedding_norm = np.linalg.norm(embedding)
                return embedding / embedding_norm if embedding_norm > 0 else None

            logger.warning("Nenhum embedding gerado")
            return None

        except Exception as e:
            logger.error(f"Falha na extração do embedding: {str(e)}")
            return None

    def _recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Reconhecimento facial com parâmetros TOLERANTES
        """
        try:
            captured_embedding = self._extract_face_embedding(face_image)
            if captured_embedding is None:
                return None, None, None

            best_match = None
            min_distance = float('inf')
            best_confidence = 0.0

            # Busca no banco de dados
            for user_key, user_data in self.facial_database.items():
                for db_embedding in user_data['embeddings']:
                    # Distância cosseno para VGG-Face
                    distance = 1 - np.dot(captured_embedding, db_embedding)

                    # ✅ VALIDAÇÃO MAIS TOLERANTE
                    confidence = 1 - distance

                    if (distance < min_distance and
                        distance < self._model_config.DISTANCE_THRESHOLD and
                        confidence > self._model_config.MIN_CONFIDENCE_THRESHOLD):

                        min_distance = distance
                        best_match = user_key
                        best_confidence = confidence

            return best_match, min_distance, best_confidence

        except Exception as e:
            logger.error(f"Falha no reconhecimento: {str(e)}")
            return None, None, None

    def _decode_base64_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decodifica imagem base64 para array numpy"""
        try:
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image if image is not None and image.size > 0 else None
        except Exception as e:
            logger.error(f"Falha na decodificação: {str(e)}")
            return None

    def _detect_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detecção de rostos MAIS TOLERANTE
        """
        try:
            detected_faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self._model_config.DETECTOR_BACKEND,
                enforce_detection=False
            )

            if (detected_faces and len(detected_faces) > 0 and
                "facial_area" in detected_faces[0] and
                detected_faces[0].get('confidence', 0) > 0.5):  # ✅ Confiança mínima reduzida
                return detected_faces[0]

            return None

        except Exception as e:
            logger.error(f"Falha na detecção: {str(e)}")
            return None

    def process_face_login(self, image_data: str) -> Dict[str, Any]:
        """
        Processamento de login MAIS TOLERANTE
        """
        self.recognition_stats['total_attempts'] += 1

        # Verificar se há usuários no banco
        if not self.facial_database:
            self.recognition_stats['failed_auth'] += 1
            return {
                "authenticated": False,
                "user": None,
                "confidence": 0.0,
                "message": "⚠️ Nenhum usuário cadastrado no sistema",
                "timestamp": self.get_current_timestamp()
            }

        # Decodificar imagem
        frame = self._decode_base64_image(image_data)
        if frame is None:
            self.recognition_stats['failed_auth'] += 1
            return self._error_response("Dados de imagem inválidos")

        # Detectar rosto
        face_data = self._detect_face(frame)
        if not face_data:
            self.recognition_stats['failed_auth'] += 1
            return self._error_response("Nenhum rosto detectado - aproxime-se da câmera")

        # Extrair região do rosto
        face_area = face_data["facial_area"]
        x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']

        # ✅ VALIDAÇÃO DE TAMANHO MAIS TOLERANTE
        if w < self._model_config.MIN_FACE_SIZE[0] or h < self._model_config.MIN_FACE_SIZE[1]:
            self.recognition_stats['failed_auth'] += 1
            return self._error_response("Rosto muito pequeno - aproxime-se mais da câmera")

        face_roi = frame[y:y+h, x:x+w]

        # ✅ VALIDAÇÃO DE QUALIDADE MAIS TOLERANTE
        is_quality_ok, quality_msg = self._validate_face_quality(face_roi)
        if not is_quality_ok:
            self.recognition_stats['quality_rejections'] += 1
            # ✅ MESMO COM QUALIDADE BAIXA, TENTA RECONHECER
            logger.info(f"⚠️ Qualidade baixa, mas tentando reconhecer: {quality_msg}")

        # Reconhecer rosto
        user, distance, confidence = self._recognize_face(face_roi)

        if user and confidence and confidence > self._model_config.MIN_CONFIDENCE_THRESHOLD:
            self.recognition_stats['successful_auth'] += 1

            logger.info(f"✅ AUTH SUCCESS: {user} - Dist: {distance:.3f} - Conf: {confidence:.3f}")

            return self._success_response(user, confidence, distance)
        else:
            self.recognition_stats['failed_auth'] += 1

            if user:  # Usuário encontrado mas confiança baixa
                logger.info(f"⚠️ AUTH REJECTED: {user} - Confiança baixa: {confidence:.3f}")
            else:
                logger.info(f"❌ AUTH FAILED: Usuário não reconhecido")

            return self._rejection_response()

    def _success_response(self, user: str, confidence: float, distance: float) -> Dict[str, Any]:
        """Resposta para autenticação bem-sucedida"""
        user_data = self.facial_database[user]['info']

        return {
            "authenticated": True,
            "user": user,
            "user_details": user_data,
            "confidence": round(confidence, 4),
            "distance": round(distance, 4),
            "message": f"Bem-vindo(a), {user_data['nome']}!",
            "timestamp": self.get_current_timestamp(),
            "stats": {
                "total_attempts": self.recognition_stats['total_attempts'],
                "success_rate": round(self.recognition_stats['successful_auth'] / self.recognition_stats['total_attempts'] * 100, 1)
            }
        }

    def _rejection_response(self) -> Dict[str, Any]:
        """Resposta para usuário não reconhecido"""
        return {
            "authenticated": False,
            "user": None,
            "confidence": 0.0,
            "message": "Usuário não reconhecido - tente novamente com melhor iluminação",
            "timestamp": self.get_current_timestamp(),
            "stats": {
                "total_attempts": self.recognition_stats['total_attempts'],
                "success_rate": round(self.recognition_stats['successful_auth'] / self.recognition_stats['total_attempts'] * 100, 1)
            }
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Resposta para erro no processamento"""
        return {
            "authenticated": False,
            "user": None,
            "confidence": 0.0,
            "message": message,
            "timestamp": self.get_current_timestamp()
        }

    # ... (restante dos métodos mantidos igual) ...

    def get_database_status(self) -> Dict[str, Any]:
        """Status do banco de dados"""
        user_count = len(self.facial_database)
        total_embeddings = sum(len(user_data['embeddings']) for user_data in self.facial_database.values())

        success_rate = 0
        if self.recognition_stats['total_attempts'] > 0:
            success_rate = round(self.recognition_stats['successful_auth'] / self.recognition_stats['total_attempts'] * 100, 1)

        return {
            "status": "loaded" if self.facial_database else "empty",
            "user_count": user_count,
            "total_embeddings": total_embeddings,
            "avg_embeddings_per_user": round(total_embeddings / user_count, 1) if user_count > 0 else 0,
            "last_update": self.last_update,
            "monitoring_active": self.database_monitor.running if hasattr(self, 'database_monitor') else False,
            "database_type": "PostgreSQL",
            "model": self._model_config.MODEL_NAME,
            "embedding_dimension": self._model_config.EMBEDDING_DIMENSION,
            "threshold": self._model_config.DISTANCE_THRESHOLD,
            "recognition_stats": {
                "total_attempts": self.recognition_stats['total_attempts'],
                "successful_auth": self.recognition_stats['successful_auth'],
                "failed_auth": self.recognition_stats['failed_auth'],
                "quality_rejections": self.recognition_stats['quality_rejections'],
                "success_rate": f"{success_rate}%"
            }
        }

    def reload_database(self) -> Tuple[bool, str]:
        """Recarrega banco de dados"""
        success = self.load_facial_database()
        if success:
            status = self.get_database_status()
            message = (f"Database recarregado - {status['user_count']} usuários, "
                      f"{status['total_embeddings']} embeddings")

            logger.info(f"🔄 {message}")

            return True, message
        else:
            return False, "Falha no recarregamento do banco"

    def initialize(self) -> bool:
        """Inicializa o serviço com configuração TOLERANTE"""
        logger.info("🔧 Inicializando Serviço de Reconhecimento Facial (Modo Tolerante)...")
        logger.info(f"🎯 Modelo: {self._model_config.MODEL_NAME}")
        logger.info(f"📊 Dimensão: {self._model_config.EMBEDDING_DIMENSION}")
        logger.info(f"🎯 Threshold: {self._model_config.DISTANCE_THRESHOLD} (Tolerante)")
        logger.info(f"🔍 Detector: {self._model_config.DETECTOR_BACKEND} (Tolerante)")

        if not self._create_table_if_not_exists():
            logger.error("❌ Falha na criação da tabela")
            return False

        trigger_success = self._setup_database_triggers()
        db_success = self.load_facial_database()
        monitor_success = self.database_monitor.start_monitoring()

        if db_success:
            status = self.get_database_status()
            logger.info(f"✅ Database carregado: {status['user_count']} usuários, {status['total_embeddings']} embeddings")

            if trigger_success and monitor_success:
                logger.info("🎯 Monitoramento em tempo real: ATIVO")
            else:
                logger.warning("⚠️ Monitoramento em tempo real: LIMITADO")

        return db_success

    def cleanup(self):
        """Limpeza do serviço"""
        if hasattr(self, 'database_monitor'):
            self.database_monitor.stop_monitoring()
        logger.info("🧹 Face recognition service cleaned up")

    @staticmethod
    def get_current_timestamp() -> str:
        """Retorna timestamp atual formatado"""
        return datetime.now().isoformat(sep=' ', timespec='seconds')