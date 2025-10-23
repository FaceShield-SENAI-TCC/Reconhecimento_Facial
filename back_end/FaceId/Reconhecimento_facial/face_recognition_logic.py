"""
Serviço de Reconhecimento Facial com Monitoramento em Tempo Real
VERSÃO AJUSTADA - Critérios mais tolerantes para aceitar usuários cadastrados
"""

import logging
import time
import threading
import numpy as np
import cv2
import base64
import psycopg2
import psycopg2.extensions
import os
import re
from datetime import datetime
from deepface import DeepFace
from typing import Tuple, Optional, Dict, Any, Generator
from contextlib import contextmanager
from collections import Counter, deque
from dataclasses import dataclass
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Configurações do banco de dados PostgreSQL com variáveis de ambiente"""
    DB_NAME = os.getenv("DB_NAME", "faceshild")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")

    @classmethod
    def validate_config(cls):
        """Valida se as configurações essenciais estão presentes"""
        if not cls.DB_PASSWORD:
            logger.warning("⚠️ DB_PASSWORD não configurada - usando valor vazio")
        if not all([cls.DB_NAME, cls.DB_USER, cls.DB_HOST]):
            raise ValueError("Configurações de banco de dados incompletas")

class ModelConfig:
    """Configurações do modelo de reconhecimento - MAIS TOLERANTE"""
    MODEL_NAME = "VGG-Face"
    DISTANCE_THRESHOLD = 0.68  # ✅ AUMENTADO: era 0.60 - maior distância = mais tolerante
    MIN_FACE_SIZE = (80, 80)   # ✅ REDUZIDO: era 100x100 - aceita rostos menores
    DETECTOR_BACKEND = "skip"  # COMPATÍVEL com cadastro
    EMBEDDING_DIMENSION = 2622
    MIN_CONFIDENCE = 0.55      # ✅ REDUZIDO: era 0.70 - confiança mínima mais baixa
    MARGIN_REQUIREMENT = 0.008 # ✅ REDUZIDO: era 0.015 - margem menor entre 1º e 2º

@dataclass
class RecognitionMetrics:
    """Métricas de desempenho do reconhecimento"""
    total_attempts: int = 0
    successful_recognitions: int = 0
    failed_recognitions: int = 0
    no_face_detected: int = 0
    false_positives: int = 0
    processing_times: deque = None
    user_recognitions: Counter = None

    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = deque(maxlen=100)
        if self.user_recognitions is None:
            self.user_recognitions = Counter()

class ImageValidator:
    """Validador de imagens base64"""

    @staticmethod
    def validate_base64_image(image_data: str) -> Tuple[bool, str]:
        """
        Valida dados de imagem base64

        Returns:
            Tuple[bool, str]: (é_válido, mensagem_erro)
        """
        try:
            if not image_data or not isinstance(image_data, str):
                return False, "Dados de imagem vazios ou inválidos"

            if len(image_data) > 7 * 1024 * 1024:
                return False, "Imagem muito grande (máximo 5MB)"

            if 'data:image' in image_data:
                image_data = image_data.split(',')[1]

            base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
            if not base64_pattern.match(image_data):
                return False, "Formato base64 inválido"

            try:
                decoded = base64.b64decode(image_data)
                if len(decoded) == 0:
                    return False, "Dados base64 vazios após decodificação"
            except Exception:
                return False, "Falha na decodificação base64"

            return True, "Imagem válida"

        except Exception as e:
            return False, f"Erro na validação: {str(e)}"

class FaceRecognitionService:
    """
    Serviço principal de reconhecimento facial - VERSÃO MAIS TOLERANTE
    """

    def __init__(self):
        self.facial_database = {}
        self.last_update = None
        self._db_config = DatabaseConfig()
        self._model_config = ModelConfig()
        self.validator = ImageValidator()
        self.metrics = RecognitionMetrics()
        self._metrics_lock = threading.RLock()

        # Validar configurações na inicialização
        DatabaseConfig.validate_config()

    @contextmanager
    def get_db_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Gerenciador de contexto para conexões de banco de dados"""
        conn = None
        try:
            conn = psycopg2.connect(
                dbname=self._db_config.DB_NAME,
                user=self._db_config.DB_USER,
                password=self._db_config.DB_PASSWORD,
                host=self._db_config.DB_HOST,
                port=self._db_config.DB_PORT,
                connect_timeout=10
            )
            yield conn
        except psycopg2.OperationalError as e:
            logger.error(f"❌ Erro de conexão com o banco: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Erro inesperado na conexão: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def _validate_embedding_dimension(self, embedding) -> bool:
        """Valida se o embedding tem a dimensão correta para VGG-Face"""
        if not embedding or len(embedding) != self._model_config.EMBEDDING_DIMENSION:
            return False
        return True

    def _normalize_embedding(self, embedding) -> np.ndarray:
        """Normaliza o embedding para comparação"""
        embedding_array = np.array(embedding, dtype=np.float32)
        embedding_norm = np.linalg.norm(embedding_array)

        if embedding_norm > 0:
            return embedding_array / embedding_norm
        else:
            raise ValueError("Embedding has zero norm")

    def load_facial_database(self) -> bool:
        """
        Carrega embeddings faciais do PostgreSQL

        Returns:
            bool: True se carregado com sucesso
        """
        logger.info("🔄 Loading facial database from PostgreSQL...")

        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT nome, sobrenome, turma, tipo, embeddings
                        FROM usuarios
                        WHERE embeddings IS NOT NULL AND jsonb_array_length(embeddings) > 0
                    """)

                    database = {}
                    user_count = 0
                    embedding_count = 0
                    invalid_embeddings = 0

                    for nome, sobrenome, turma, tipo, embeddings in cursor.fetchall():
                        user_info = {
                            'display_name': f"{nome} {sobrenome}",
                            'full_info': f"{nome} {sobrenome} - {turma} ({tipo})",
                            'nome': nome,
                            'sobrenome': sobrenome,
                            'turma': turma,
                            'tipo': tipo
                        }

                        valid_embeddings = []
                        for embedding in embeddings:
                            if self._validate_embedding_dimension(embedding):
                                try:
                                    normalized_embedding = self._normalize_embedding(embedding)
                                    valid_embeddings.append(normalized_embedding)
                                except Exception as e:
                                    logger.warning(f"Invalid embedding for {user_info['display_name']}: {str(e)}")
                                    invalid_embeddings += 1
                            else:
                                invalid_embeddings += 1
                                logger.warning(f"Wrong embedding dimension for {user_info['display_name']}: "
                                             f"expected {self._model_config.EMBEDDING_DIMENSION}, "
                                             f"got {len(embedding) if embedding else 'None'}")

                        if valid_embeddings:
                            database[user_info['display_name']] = {
                                'embeddings': valid_embeddings,
                                'info': user_info
                            }
                            user_count += 1
                            embedding_count += len(valid_embeddings)
                            logger.debug(f"Loaded user: {user_info['display_name']} - {len(valid_embeddings)} embeddings")

            self.facial_database = database
            self.last_update = time.time()

            if invalid_embeddings > 0:
                logger.warning(f"⚠️ Found {invalid_embeddings} invalid embeddings (wrong dimension for VGG-Face)")

            if user_count == 0:
                logger.warning("⚠️ Database loaded but no users with valid embeddings found")
                logger.info("💡 Use the registration system to add users with VGG-Face embeddings")
            else:
                logger.info(f"✅ Database loaded: {user_count} users, {embedding_count} embeddings")
            return True

        except Exception as e:
            logger.error(f"❌ Database loading failed: {str(e)}")
            return False

    def _extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrai embedding facial da imagem usando VGG-Face
        ✅ COMPATÍVEL com processo de cadastro (usa 'skip' como detector)
        """
        try:
            # ✅ MESMA CONFIGURAÇÃO DO CADASTRO
            result = DeepFace.represent(
                img_path=face_image,
                model_name=self._model_config.MODEL_NAME,
                detector_backend="skip",  # ✅ COMPATÍVEL: Não detecta, usa ROI direto
                enforce_detection=False,
                align=True  # ✅ COMPATÍVEL: Alinha como no cadastro
            )

            if result and isinstance(result, list) and "embedding" in result[0]:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)

                if embedding.shape[0] != self._model_config.EMBEDDING_DIMENSION:
                    logger.error(f"Wrong embedding dimension from VGG-Face: "
                               f"expected {self._model_config.EMBEDDING_DIMENSION}, "
                               f"got {embedding.shape[0]}")
                    return None

                embedding_norm = np.linalg.norm(embedding)
                return embedding / embedding_norm if embedding_norm > 0 else None

            logger.warning("No face embedding generated")
            return None

        except Exception as e:
            logger.error(f"Face embedding extraction failed: {str(e)}")
            return None

    def _recognize_face_tolerant(self, face_image: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Reconhecimento facial MAIS TOLERANTE para usuários cadastrados
        """
        try:
            captured_embedding = self._extract_face_embedding(face_image)
            if captured_embedding is None:
                return None, None

            best_match = None
            min_distance = float('inf')
            second_best_distance = float('inf')
            best_user_data = None

            # Busca linear no banco de dados
            for user_key, user_data in self.facial_database.items():
                for db_embedding in user_data['embeddings']:
                    # ✅ MESMA MÉTRICA DO CADASTRO: Distância Cosseno
                    distance = 1 - np.dot(captured_embedding, db_embedding)

                    if distance < min_distance:
                        second_best_distance = min_distance
                        min_distance = distance
                        best_match = user_key
                        best_user_data = user_data
                    elif distance < second_best_distance:
                        second_best_distance = distance

            # ✅ CRITÉRIOS MAIS TOLERANTES
            if best_match and min_distance < self._model_config.DISTANCE_THRESHOLD:
                confidence = 1 - min_distance

                # Se há apenas um usuário no banco - CRITÉRIOS MUITO MAIS TOLERANTES
                if len(self.facial_database) == 1:
                    if confidence >= 0.50:  # ✅ MUITO REDUZIDO: era 0.65
                        logger.info(f"✅ Match ÚNICO usuário: {best_match} (dist: {min_distance:.4f}, conf: {confidence:.4f})")
                        return best_match, min_distance
                    else:
                        logger.info(f"❌ Match rejeitado: confiança insuficiente para único usuário ({confidence:.4f} < 0.50)")
                        return None, None

                # Para múltiplos usuários - CRITÉRIOS MAIS TOLERANTES
                margin = second_best_distance - min_distance
                min_margin_required = self._model_config.MARGIN_REQUIREMENT

                # ✅ REGRAS MAIS TOLERANTES:
                high_confidence = confidence >= 0.70  # ✅ REDUZIDO: era 0.80
                very_low_distance = min_distance < 0.45  # ✅ AUMENTADO: era 0.40

                # ✅ ACEITAR COM CRITÉRIOS MAIS BAIXOS:
                if high_confidence or very_low_distance:
                    # Margem mínima muito reduzida para alta confiança
                    if margin >= 0.001:  # ✅ MUITO REDUZIDO: era 0.005
                        logger.info(f"✅ Match ACEITO (critério alto): {best_match} (dist: {min_distance:.4f}, conf: {confidence:.4f}, margem: {margin:.4f})")
                        return best_match, min_distance
                    else:
                        logger.info(f"❌ Match rejeitado: critério alto mas margem insuficiente ({margin:.4f} < 0.001)")
                        return None, None
                elif (confidence >= self._model_config.MIN_CONFIDENCE and
                      margin >= min_margin_required):
                    logger.info(f"✅ Match VÁLIDO: {best_match} (dist: {min_distance:.4f}, conf: {confidence:.4f}, margem: {margin:.4f})")
                    return best_match, min_distance
                else:
                    if confidence < self._model_config.MIN_CONFIDENCE:
                        logger.info(f"❌ Match rejeitado: confiança insuficiente ({confidence:.4f} < {self._model_config.MIN_CONFIDENCE})")
                    else:
                        logger.info(f"❌ Match rejeitado: margem insuficiente ({margin:.4f} < {min_margin_required})")

                    return None, None

            else:
                if best_match:
                    logger.info(f"❌ Match rejeitado: distância alta ({min_distance:.4f} >= {self._model_config.DISTANCE_THRESHOLD})")
                else:
                    logger.info("❌ Nenhum match encontrado")
                return None, None

        except Exception as e:
            logger.error(f"❌ Falha no reconhecimento facial: {str(e)}")
            return None, None

    def _decode_base64_image(self, image_data: str) -> Optional[np.ndarray]:
        """
        Decodifica imagem base64 para array numpy
        """
        try:
            # Validar imagem antes de processar
            is_valid, message = self.validator.validate_base64_image(image_data)
            if not is_valid:
                logger.error(f"❌ Imagem inválida: {message}")
                return None

            # Remover header se presente
            if 'data:image' in image_data:
                image_data = image_data.split(',')[1]

            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None or image.size == 0:
                logger.error("❌ Falha ao decodificar imagem")
                return None

            return image

        except Exception as e:
            logger.error(f"❌ Erro na decodificação de imagem: {str(e)}")
            return None

    def _detect_face_tolerant(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detecta rostos na imagem - MAIS TOLERANTE
        """
        try:
            # ✅ MESMO PROCESSO DO CADASTRO: Detector OpenCV
            detectors_to_try = ["opencv", "ssd"]  # ✅ ADICIONADO ssd como fallback

            for detector in detectors_to_try:
                try:
                    detected_faces = DeepFace.extract_faces(
                        img_path=image,
                        detector_backend=detector,
                        enforce_detection=False,
                        align=False  # ✅ COMPATÍVEL: Não alinha durante detecção
                    )

                    if (detected_faces and len(detected_faces) > 0 and
                        "facial_area" in detected_faces[0]):

                        face_data = detected_faces[0]
                        facial_area = face_data["facial_area"]
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                        # ✅ CRITÉRIOS MAIS TOLERANTES
                        if (w >= self._model_config.MIN_FACE_SIZE[0] and
                            h >= self._model_config.MIN_FACE_SIZE[1] and
                            x >= 0 and y >= 0 and
                            x + w <= image.shape[1] and y + h <= image.shape[0]):

                            # ✅ VALIDAÇÃO DE NITIDEZ MAIS TOLERANTE
                            face_roi = image[y:y+h, x:x+w]

                            # Calcular nitidez (mais tolerante)
                            try:
                                small_img = cv2.resize(face_roi, (100, 100))
                                gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
                                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

                                if sharpness > 30:  # ✅ LIMITE MAIS BAIXO: era 50
                                    logger.info(f"✅ Rosto detectado com {detector} (tamanho: {w}x{h}, nitidez: {sharpness:.1f})")
                                    return face_data
                                else:
                                    logger.warning(f"❌ Rosto muito borrado: {sharpness:.1f}")
                            except:
                                # ✅ ACEITAR MESMO SE NÃO CONSEGUIR CALCULAR NITIDEZ
                                logger.info(f"✅ Rosto detectado com {detector} (tamanho: {w}x{h})")
                                return face_data

                except Exception as e:
                    logger.debug(f"Detector {detector} falhou: {str(e)}")
                    continue

            logger.warning("❌ Nenhum rosto detectado ou rosto de baixa qualidade")
            return None

        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return None

    def process_face_login(self, image_data: str) -> Dict[str, Any]:
        """Processamento MAIS TOLERANTE para usuários cadastrados"""
        start_time = time.time()

        try:
            with self._metrics_lock:
                self.metrics.total_attempts += 1

            # Verificar se há usuários no banco
            if not self.facial_database:
                with self._metrics_lock:
                    self.metrics.failed_recognitions += 1
                return {
                    "authenticated": False,
                    "user": None,
                    "confidence": 0.0,
                    "message": "⚠️ Nenhum usuário cadastrado no sistema. Use o sistema de cadastro primeiro.",
                    "timestamp": self.get_current_timestamp()
                }

            # Validar imagem
            is_valid, message = self.validator.validate_base64_image(image_data)
            if not is_valid:
                with self._metrics_lock:
                    self.metrics.failed_recognitions += 1
                return self._error_response(f"Imagem inválida: {message}")

            # Decodificar imagem
            frame = self._decode_base64_image(image_data)
            if frame is None:
                with self._metrics_lock:
                    self.metrics.failed_recognitions += 1
                return self._error_response("Dados de imagem inválidos")

            # ✅ APLICAR MELHORIAS NA IMAGEM
            try:
                # Redimensionar se necessário
                if frame.shape[0] > 800 or frame.shape[1] > 800:
                    scale = 800 / max(frame.shape[0], frame.shape[1])
                    new_width = int(frame.shape[1] * scale)
                    new_height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Melhorar contraste
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except Exception as e:
                logger.warning(f"⚠️ Melhoria de imagem falhou: {str(e)}")

            # Detectar rosto com validação mais tolerante
            face_data = self._detect_face_tolerant(frame)
            if not face_data:
                with self._metrics_lock:
                    self.metrics.failed_recognitions += 1
                    self.metrics.no_face_detected += 1
                return self._error_response("Nenhum rosto detectado - posicione-se melhor na frente da câmera")

            # Extrair ROI do rosto
            face_area = face_data["facial_area"]
            x, y = face_area['x'], face_area['y']
            w, h = face_area['w'], face_area['h']
            face_roi = frame[y:y+h, x:x+w]

            # ✅ USAR RECONHECIMENTO MAIS TOLERANTE
            user, distance = self._recognize_face_tolerant(face_roi)

            # Coletar métricas
            processing_time = time.time() - start_time
            with self._metrics_lock:
                self.metrics.processing_times.append(processing_time)

                if user:
                    self.metrics.successful_recognitions += 1
                    self.metrics.user_recognitions[user] += 1

                    confidence = 1 - distance
                    user_data = self.facial_database[user]['info']

                    logger.info(f"⏱️ Tempo de processamento: {processing_time:.3f}s")

                    return {
                        "authenticated": True,
                        "user": user,
                        "confidence": round(confidence, 4),
                        "distance": round(distance, 4),
                        "user_info": user_data,
                        "message": f"Bem-vindo(a), {user_data['nome']}!",
                        "timestamp": self.get_current_timestamp()
                    }
                else:
                    self.metrics.failed_recognitions += 1

            logger.info(f"⏱️ Tempo de processamento: {processing_time:.3f}s")
            return self._rejection_response()

        except Exception as e:
            with self._metrics_lock:
                self.metrics.failed_recognitions += 1
            logger.error(f"❌ Erro no processamento: {str(e)}")
            return self._error_response("Erro interno no processamento")

    def _success_response(self, user: str, confidence: float) -> Dict[str, Any]:
        """Resposta para autenticação bem-sucedida"""
        return {
            "authenticated": True,
            "user": user,
            "confidence": round(confidence, 4),
            "message": f"Bem-vindo, {user}!",
            "timestamp": self.get_current_timestamp()
        }

    def _rejection_response(self) -> Dict[str, Any]:
        """Resposta para usuário não reconhecido"""
        return {
            "authenticated": False,
            "user": None,
            "confidence": 0.0,
            "message": "Usuário não reconhecido - verifique se está cadastrado no sistema",
            "timestamp": self.get_current_timestamp()
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

    def get_database_status(self) -> Dict[str, Any]:
        """Retorna status do banco de dados"""
        user_count = len(self.facial_database)
        total_embeddings = sum(len(user_data['embeddings']) for user_data in self.facial_database.values())

        return {
            "status": "loaded" if self.facial_database else "empty",
            "user_count": user_count,
            "total_embeddings": total_embeddings,
            "last_update": self.last_update,
            "database_type": "PostgreSQL",
            "model": self._model_config.MODEL_NAME,
            "embedding_dimension": self._model_config.EMBEDDING_DIMENSION,
            "threshold": self._model_config.DISTANCE_THRESHOLD,
            "min_confidence": self._model_config.MIN_CONFIDENCE,
            "min_face_size": self._model_config.MIN_FACE_SIZE,
            "margin_requirement": self._model_config.MARGIN_REQUIREMENT,
            "tolerance_level": "HIGH"  # ✅ NOVO: Indica tolerância aumentada
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de desempenho"""
        with self._metrics_lock:
            times = list(self.metrics.processing_times)
            avg_time = sum(times) / len(times) if times else 0

            success_rate = (self.metrics.successful_recognitions /
                          self.metrics.total_attempts if self.metrics.total_attempts > 0 else 0)

            return {
                "performance": {
                    "total_attempts": self.metrics.total_attempts,
                    "success_rate": round(success_rate, 4),
                    "successful_recognitions": self.metrics.successful_recognitions,
                    "failed_recognitions": self.metrics.failed_recognitions,
                    "no_face_detected": self.metrics.no_face_detected,
                    "false_positives": self.metrics.false_positives,
                    "average_processing_time": round(avg_time, 3),
                    "recent_processing_times": [round(t, 3) for t in times[-10:]]
                },
                "top_recognized_users": self.metrics.user_recognitions.most_common(5),
                "database_status": self.get_database_status(),
                "compatibility_info": {
                    "cadastro_alignment": "FULL",
                    "embedding_generation": "IDENTICAL",
                    "face_detection": "COMPATIBLE",
                    "recognition_tolerance": "HIGH"  # ✅ NOVO: Tolerância alta
                }
            }

    def reload_database(self) -> Tuple[bool, str]:
        """
        Recarrega banco de dados

        Returns:
            Tuple[bool, str]: (sucesso, mensagem)
        """
        success = self.load_facial_database()
        if success:
            status = self.get_database_status()
            message = f"Database reloaded - {status['user_count']} users, {status['total_embeddings']} embeddings"
            return True, message
        else:
            return False, "Database reload failed"

    def initialize(self) -> bool:
        """Inicializa o serviço"""
        logger.info("🔧 Initializing Face Recognition Service...")
        logger.info(f"🎯 Using model: {self._model_config.MODEL_NAME}")
        logger.info(f"📊 Embedding dimension: {self._model_config.EMBEDDING_DIMENSION}")
        logger.info(f"🎪 MODO TOLERANTE ATIVADO - Critérios relaxados para aceitar usuários cadastrados")
        logger.info(f"   📏 Tamanho mínimo do rosto: {self._model_config.MIN_FACE_SIZE}")
        logger.info(f"   📐 Threshold de distância: {self._model_config.DISTANCE_THRESHOLD}")
        logger.info(f"   🎯 Confiança mínima: {self._model_config.MIN_CONFIDENCE}")
        logger.info(f"   📊 Margem necessária: {self._model_config.MARGIN_REQUIREMENT}")

        # Carregar banco de dados inicial
        db_success = self.load_facial_database()

        if db_success:
            logger.info("✅ Facial recognition service initialized successfully")
        else:
            logger.error("❌ Failed to initialize facial recognition service")

        return db_success

    def cleanup(self):
        """Limpeza do serviço"""
        logger.info("🧹 Face recognition service cleaned up")

    @staticmethod
    def get_current_timestamp() -> str:
        """Retorna timestamp atual formatado"""
        return datetime.now().isoformat(sep=' ', timespec='seconds')