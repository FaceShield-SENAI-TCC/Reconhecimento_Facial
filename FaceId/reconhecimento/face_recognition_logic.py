"""
Serviço de Reconhecimento Facial - VERSÃO LEVE E COMPATÍVEL
"""
import logging
import time
import threading
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from deepface import DeepFace

# Módulos compartilhados
from common.config import MODEL_CONFIG
from common.database import DatabaseManager
from common.image_utils import ImageValidator

logger = logging.getLogger(__name__)

@dataclass
class RecognitionResult:
    """Resultado do reconhecimento facial"""
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

@dataclass
class DatabaseStatus:
    """Status do banco de dados"""
    status: str
    user_count: int
    total_embeddings: int
    last_update: Optional[float] = None

@dataclass
class SystemMetrics:
    """Métricas do sistema"""
    total_attempts: int
    successful_recognitions: int
    failed_recognitions: int
    no_face_detected: int
    average_processing_time: float
    success_rate: float
    database_reloads: int = 0

class FaceRecognitionService:
    """Serviço principal de reconhecimento facial - VERSÃO LEVE E COMPATÍVEL"""

    def __init__(self):
        self.facial_database = {}
        self.last_update = None
        self.metrics = {
            'total_attempts': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'no_face_detected': 0,
            'processing_times': deque(maxlen=100),
            'database_reloads': 0
        }
        self._metrics_lock = threading.RLock()
        self.db_manager = DatabaseManager()

    def load_facial_database(self) -> bool:
        """Carrega embeddings faciais do PostgreSQL - VERSÃO CORRIGIDA"""
        logger.info("🔄 Carregando banco de dados facial...")

        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, nome, sobrenome, turma, tipo_usuario, username, embeddings
                        FROM usuarios
                        WHERE embeddings IS NOT NULL AND jsonb_array_length(embeddings) > 0
                    """)

                    database = {}
                    for id, nome, sobrenome, turma, tipo_usuario, username, embeddings in cursor.fetchall():
                        display_name = f"{nome} {sobrenome}"

                        user_info = {
                            'id': id,
                            'display_name': display_name,
                            'nome': nome,
                            'sobrenome': sobrenome,
                            'turma': turma,
                            'tipo_usuario': tipo_usuario,
                            'username': username,
                        }

                        valid_embeddings = []
                        for embedding in embeddings:
                            # ✅ CORREÇÃO: Aceita 2622 dimensões
                            if embedding and len(embedding) == 2622:  # ✅ Mudado de 4096 para 2622
                                try:
                                    embedding_array = np.array(embedding, dtype=np.float32)
                                    embedding_norm = np.linalg.norm(embedding_array)
                                    if embedding_norm > 0:
                                        valid_embeddings.append(embedding_array / embedding_norm)
                                except:
                                    continue

                        if valid_embeddings:
                            database[display_name] = {
                                'embeddings': valid_embeddings,
                                'info': user_info
                            }

            self.facial_database = database
            self.last_update = time.time()

            logger.info(
                f"✅ Banco carregado: {len(database)} usuários, {sum(len(data['embeddings']) for data in database.values())} embeddings")
            return True

        except Exception as e:
            logger.error(f"❌ Falha ao carregar banco: {str(e)}")
            return False

    def _extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extrai embedding facial - VERSÃO CORRIGIDA (2622 dimensões)"""
        try:
            result = DeepFace.represent(
                img_path=face_image,
                model_name=MODEL_CONFIG.MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
                align=True
            )

            if result and isinstance(result, list) and "embedding" in result[0]:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)

                # ✅ CORREÇÃO: Aceita 2622 dimensões (valor real do DeepFace)
                if len(embedding) == 2622:
                    embedding_norm = np.linalg.norm(embedding)
                    return embedding / embedding_norm if embedding_norm > 0 else None
                else:
                    logger.warning(f"⚠️ Dimensão inesperada: {len(embedding)} (esperado: 2622)")

            return None

        except Exception as e:
            logger.error(f"Falha na extração de embedding: {str(e)}")
            return None

    def _quick_liveness_check(self, image: np.ndarray) -> bool:
        """Verificação RÁPIDA de vivacidade"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()

            # ✅ Bloqueia imagens muito borradas (fotos de celular)
            if blur_value < 50:
                logger.warning(f"⚠️ Imagem muito borrada: {blur_value:.1f}")
                return False

            return True
        except:
            return True  # Fallback - não bloqueia se der erro

    def _recognize_face_optimized(self, face_image: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """Reconhecimento OTIMIZADO - balance entre confiança e performance"""
        try:
            captured_embedding = self._extract_face_embedding(face_image)
            if captured_embedding is None:
                return None, None

            best_match = None
            min_distance = float('inf')

            # ✅ BUSCA RÁPIDA
            for user_key, user_data in self.facial_database.items():
                for db_embedding in user_data['embeddings']:
                    distance = 1 - np.dot(captured_embedding, db_embedding)

                    if distance < min_distance:
                        min_distance = distance
                        best_match = user_key

            # ✅ CRITÉRIOS SIMPLES E EFICIENTES
            confidence = 1 - min_distance

            if best_match and min_distance < MODEL_CONFIG.DISTANCE_THRESHOLD:
                if confidence >= 0.85:  # ✅ 85% de confiança mínima
                    logger.info(f"✅ RECONHECIDO: {best_match} - Conf: {confidence:.4f}")
                    return best_match, min_distance
                else:
                    logger.info(f"❌ Confiança insuficiente: {confidence:.4f}")
                    return None, None
            else:
                logger.info(f"❌ Distância muito alta: {min_distance:.4f}")
                return None, None

        except Exception as e:
            logger.error(f"❌ Erro no reconhecimento: {str(e)}")
            return None, None

    def _detect_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detecta rostos na imagem"""
        try:
            detected_faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend="opencv",
                enforce_detection=False,
                align=False
            )

            if detected_faces and len(detected_faces) > 0 and "facial_area" in detected_faces[0]:
                face_data = detected_faces[0]
                facial_area = face_data["facial_area"]
                w, h = facial_area['w'], facial_area['h']

                if w >= 80 and h >= 80:
                    return face_data

            return None

        except Exception as e:
            logger.error(f"Falha na detecção facial: {str(e)}")
            return None

    def process_face_login(self, image_data: str) -> Dict[str, Any]:
        """Processamento de login facial - VERSÃO COMPLETA E CORRIGIDA"""
        start_time = time.time()

        try:
            with self._metrics_lock:
                self.metrics['total_attempts'] += 1

            # Verificar se o banco está vazio
            if not self.facial_database:
                self.load_facial_database()
                if not self.facial_database:
                    with self._metrics_lock:
                        self.metrics['failed_recognitions'] += 1

                    return {
                        "authenticated": False,
                        "confidence": 0.0,
                        "message": "⚠️ Nenhum usuário cadastrado no sistema.",
                        "timestamp": self.get_current_timestamp()
                    }

            # Decodificar imagem
            frame = ImageValidator.decode_base64_image(image_data)
            if frame is None:
                with self._metrics_lock:
                    self.metrics['failed_recognitions'] += 1

                return {
                    "authenticated": False,
                    "confidence": 0.0,
                    "message": "Dados de imagem inválidos",
                    "timestamp": self.get_current_timestamp()
                }

            # Pré-processar imagem
            frame = ImageValidator.preprocess_image(frame)

            # Detectar rosto
            face_data = self._detect_face(frame)
            if not face_data:
                with self._metrics_lock:
                    self.metrics['failed_recognitions'] += 1
                    self.metrics['no_face_detected'] += 1

                return {
                    "authenticated": False,
                    "confidence": 0.0,
                    "message": "Nenhum rosto detectado - posicione-se melhor na frente da câmera",
                    "timestamp": self.get_current_timestamp()
                }

            # Extrair ROI do rosto
            face_area = face_data["facial_area"]
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            face_roi = frame[y:y + h, x:x + w]

            # ✅ VERIFICAÇÃO LEVE ANTI-SPOOFING
            if not self._quick_liveness_check(face_roi):
                with self._metrics_lock:
                    self.metrics['failed_recognitions'] += 1

                return {
                    "authenticated": False,
                    "confidence": 0.0,
                    "message": "Qualidade da imagem muito baixa - use a câmera diretamente",
                    "timestamp": self.get_current_timestamp()
                }

            # ✅ USA RECONHECIMENTO OTIMIZADO
            user, distance = self._recognize_face_optimized(face_roi)

            processing_time = time.time() - start_time
            with self._metrics_lock:
                self.metrics['processing_times'].append(processing_time)

                if user:
                    self.metrics['successful_recognitions'] += 1
                    confidence = 1 - distance
                    user_data = self.facial_database[user]['info']

                    # Mensagem personalizada por tipo de usuário
                    if user_data['tipo_usuario'].upper() == "PROFESSOR":
                        if user_data['username']:
                            message = f"Bem-vindo(a), Professor(a) {user_data['nome']} (@{user_data['username']})!"
                        else:
                            message = f"Bem-vindo(a), Professor(a) {user_data['nome']}!"
                    else:
                        message = f"Bem-vindo(a), {user_data['nome']}!"

                    logger.info(f"⏱️ Tempo de processamento: {processing_time:.3f}s")

                    # ✅ ESTRUTURA ORIGINAL COMPLETA
                    result = {
                        "authenticated": True,
                        "user": user,
                        "confidence": round(confidence, 4),
                        "distance": round(distance, 4),
                        "user_info": user_data,
                        "id": user_data['id'],
                        "username": user_data['username'],
                        "tipo_usuario": user_data['tipo_usuario'],
                        "nome": user_data['nome'],
                        "sobrenome": user_data['sobrenome'],
                        "turma": user_data['turma'],
                        "message": message,
                        "timestamp": self.get_current_timestamp()
                    }
                    return result
                else:
                    self.metrics['failed_recognitions'] += 1

            logger.info(f"⏱️ Tempo de processamento: {processing_time:.3f}s")

            return {
                "authenticated": False,
                "confidence": 0.0,
                "message": "Usuário não reconhecido - verifique se está cadastrado no sistema",
                "timestamp": self.get_current_timestamp()
            }

        except Exception as e:
            with self._metrics_lock:
                self.metrics['failed_recognitions'] += 1

            logger.error(f"❌ Erro no processamento: {str(e)}")

            return {
                "authenticated": False,
                "confidence": 0.0,
                "message": "Erro interno no processamento",
                "timestamp": self.get_current_timestamp()
            }

    def get_database_status(self) -> Dict[str, Any]:
        """Status do banco de dados"""
        user_count = len(self.facial_database)
        total_embeddings = sum(
            len(user_data['embeddings'])
            for user_data in self.facial_database.values()
        )

        return DatabaseStatus(
            status="loaded" if self.facial_database else "empty",
            user_count=user_count,
            total_embeddings=total_embeddings,
            last_update=self.last_update
        ).__dict__

    def get_detailed_database_status(self) -> Dict[str, Any]:
        """Status detalhado do banco de dados - COMPATIBILIDADE"""
        user_count = len(self.facial_database)
        total_embeddings = sum(
            len(user_data['embeddings'])
            for user_data in self.facial_database.values()
        )

        # Estatísticas por tipo de usuário
        professores_count = sum(1 for user_data in self.facial_database.values()
                              if user_data['info']['tipo_usuario'].upper() == "PROFESSOR")
        alunos_count = user_count - professores_count

        professores_com_username = sum(1 for user_data in self.facial_database.values()
                                     if user_data['info']['tipo_usuario'].upper() == "PROFESSOR"
                                     and user_data['info']['username'] is not None)

        return {
            "status": "loaded" if self.facial_database else "empty",
            "user_count": user_count,
            "total_embeddings": total_embeddings,
            "professores_count": professores_count,
            "alunos_count": alunos_count,
            "professores_com_username": professores_com_username,
            "last_update": self.last_update,
            "monitoring_active": False,
            "database_type": "PostgreSQL"
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de desempenho - COMPATIBILIDADE"""
        with self._metrics_lock:
            times = list(self.metrics['processing_times'])
            avg_time = sum(times) / len(times) if times else 0

            success_rate = (
                self.metrics['successful_recognitions'] / self.metrics['total_attempts']
                if self.metrics['total_attempts'] > 0 else 0
            )

            return SystemMetrics(
                total_attempts=self.metrics['total_attempts'],
                successful_recognitions=self.metrics['successful_recognitions'],
                failed_recognitions=self.metrics['failed_recognitions'],
                no_face_detected=self.metrics['no_face_detected'],
                average_processing_time=round(avg_time, 3),
                success_rate=round(success_rate, 4),
                database_reloads=self.metrics['database_reloads']
            ).__dict__

    def reload_database(self) -> Tuple[bool, str]:
        """Recarrega banco de dados"""
        logger.info("🔄 Recarregamento manual do banco de dados")
        success = self.load_facial_database()

        with self._metrics_lock:
            if success:
                self.metrics['database_reloads'] += 1
                return True, "Banco recarregado com sucesso"
            else:
                return False, "Falha no recarregamento"

    def initialize(self) -> bool:
        """Inicializa o serviço"""
        logger.info("🔧 Inicializando Serviço de Reconhecimento Facial...")
        logger.info(f"🎯 Configuração: Dimensão={MODEL_CONFIG.EMBEDDING_DIMENSION}, Threshold={MODEL_CONFIG.DISTANCE_THRESHOLD}")
        return self.load_facial_database()

    def cleanup(self):
        """Limpeza do serviço"""
        logger.info("🧹 Serviço finalizado")

    @staticmethod
    def get_current_timestamp() -> str:
        """Retorna timestamp atual"""
        return datetime.now().isoformat(sep=' ', timespec='seconds')