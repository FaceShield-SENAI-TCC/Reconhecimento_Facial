"""
Serviço de Reconhecimento Facial com Monitoramento em Tempo Real
VERSÃO COM ATUALIZAÇÃO AUTOMÁTICA DO BANCO
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
    DB_NAME = os.getenv("DB_NAME", "faceshield")
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
    """Configurações do modelo de reconhecimento - SEGURO E PRECISO"""
    MODEL_NAME = "VGG-Face"
    DISTANCE_THRESHOLD = 0.25
    MIN_FACE_SIZE = (100, 100)
    DETECTOR_BACKEND = "skip"
    EMBEDDING_DIMENSION = 2622
    MIN_CONFIDENCE = 0.75
    MARGIN_REQUIREMENT = 0.03
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    VERY_LOW_DISTANCE = 0.15

@dataclass
class RecognitionMetrics:
    """Métricas de desempenho do reconhecimento"""
    total_attempts: int = 0
    successful_recognitions: int = 0
    failed_recognitions: int = 0
    no_face_detected: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    processing_times: deque = None
    user_recognitions: Counter = None
    database_reloads: int = 0  # ✅ NOVO: Contador de recarregamentos do banco

    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = deque(maxlen=100)
        if self.user_recognitions is None:
            self.user_recognitions = Counter()

class DatabaseMonitor:
    """
    Monitora o banco de dados PostgreSQL em tempo real usando LISTEN/NOTIFY
    para detectar quando novos usuários são cadastrados
    """

    def __init__(self, database_reload_callback):
        self.database_reload_callback = database_reload_callback
        self.connection = None
        self.monitor_thread = None
        self.running = False
        self._db_config = DatabaseConfig()
        self.reconnect_delay = 5

    def start_monitoring(self):
        """Inicia o monitoramento em tempo real"""
        try:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("✅ Database monitoring started - Novos usuários serão detectados automaticamente")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start database monitoring: {str(e)}")
            return False

    def stop_monitoring(self):
        """Para o monitoramento"""
        self.running = False
        if self.connection and not self.connection.closed:
            try:
                self.connection.close()
            except:
                pass
        logger.info("⏹️ Database monitoring stopped")

    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self.running:
            try:
                # Estabelecer conexão para monitoramento
                self.connection = psycopg2.connect(
                    dbname=self._db_config.DB_NAME,
                    user=self._db_config.DB_USER,
                    password=self._db_config.DB_PASSWORD,
                    host=self._db_config.DB_HOST,
                    port=self._db_config.DB_PORT,
                    connect_timeout=10
                )
                self.connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

                cursor = self.connection.cursor()
                cursor.execute("LISTEN usuarios_update;")
                logger.info("👂 Listening for database changes (novos usuários)...")

                # Loop de escuta por notificações
                while self.running and self.connection and not self.connection.closed:
                    try:
                        # Verificar por notificações
                        self.connection.poll()
                        while self.connection.notifies:
                            notify = self.connection.notifies.pop(0)
                            logger.info(f"🔄 Database change detected: {notify.payload}")

                            # Recarregar o banco de dados
                            if self.database_reload_callback:
                                logger.info("🔄 Recarregando banco de dados devido a mudanças...")
                                self.database_reload_callback()

                        # Pequeno delay para evitar uso excessivo de CPU
                        time.sleep(1)

                    except psycopg2.InterfaceError as e:
                        logger.warning(f"Database connection interrupted: {str(e)}")
                        break
                    except Exception as e:
                        logger.error(f"Error in monitor inner loop: {str(e)}")
                        break

            except psycopg2.OperationalError as e:
                logger.warning(f"❌ Database connection failed, retrying in {self.reconnect_delay}s: {str(e)}")
            except Exception as e:
                logger.error(f"❌ Monitor loop error: {str(e)}")

            # Esperar antes de tentar reconectar
            if self.running:
                time.sleep(self.reconnect_delay)

    def trigger_manual_reload(self):
        """Dispara uma atualização manual no banco de dados"""
        try:
            conn = psycopg2.connect(
                dbname=self._db_config.DB_NAME,
                user=self._db_config.DB_USER,
                password=self._db_config.DB_PASSWORD,
                host=self._db_config.DB_HOST,
                port=self._db_config.DB_PORT
            )
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            cursor.execute("NOTIFY usuarios_update, 'manual_reload';")
            conn.close()
            logger.info("🔔 Manual database reload triggered")
        except Exception as e:
            logger.error(f"Failed to trigger manual reload: {str(e)}")

class FaceRecognitionService:
    """
    Serviço principal de reconhecimento facial com atualização automática do banco
    """

    def __init__(self):
        self.facial_database = {}
        self.last_update = None
        self._db_config = DatabaseConfig()
        self._model_config = ModelConfig()
        self.metrics = RecognitionMetrics()
        self._metrics_lock = threading.RLock()

        # ✅ NOVO: Monitor de banco de dados em tempo real
        self.database_monitor = DatabaseMonitor(self.load_facial_database)

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
        except Exception as e:
            logger.error(f"❌ Erro de conexão com o banco: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def _setup_database_triggers(self):
        """Configura triggers no PostgreSQL para notificar quando novos usuários são cadastrados"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Criar função de trigger se não existir
                    cursor.execute("""
                        CREATE OR REPLACE FUNCTION notify_usuarios_update()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            PERFORM pg_notify('usuarios_update',
                                CASE
                                    WHEN TG_OP = 'INSERT' THEN 'user_added'
                                    WHEN TG_OP = 'UPDATE' THEN 'user_updated' 
                                    WHEN TG_OP = 'DELETE' THEN 'user_deleted'
                                END
                            );
                            RETURN NEW;
                        END;
                        $$ LANGUAGE plpgsql;
                    """)

                    # Criar trigger para INSERT, UPDATE, DELETE
                    cursor.execute("""
                        DROP TRIGGER IF EXISTS usuarios_notify_trigger ON usuarios;
                    """)

                    cursor.execute("""
                        CREATE TRIGGER usuarios_notify_trigger
                        AFTER INSERT OR UPDATE OR DELETE ON usuarios
                        FOR EACH ROW EXECUTE FUNCTION notify_usuarios_update();
                    """)

                    conn.commit()
                    logger.info("✅ Database triggers configured for real-time updates")
                    return True

        except Exception as e:
            logger.warning(f"⚠️ Could not setup database triggers: {str(e)}")
            logger.info("💡 Sistema funcionará com atualização manual do banco")
            return False

    def load_facial_database(self) -> bool:
        """
        Carrega embeddings faciais do PostgreSQL com verificação de atualização
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

                    for nome, sobrenome, turma, tipo, embeddings in cursor.fetchall():
                        user_info = {
                            'display_name': f"{nome} {sobrenome}",
                            'nome': nome,
                            'sobrenome': sobrenome,
                            'turma': turma,
                            'tipo': tipo
                        }

                        valid_embeddings = []
                        for embedding in embeddings:
                            if embedding and len(embedding) == self._model_config.EMBEDDING_DIMENSION:
                                embedding_array = np.array(embedding, dtype=np.float32)
                                embedding_norm = np.linalg.norm(embedding_array)
                                if embedding_norm > 0:
                                    valid_embeddings.append(embedding_array / embedding_norm)

                        if valid_embeddings:
                            database[user_info['display_name']] = {
                                'embeddings': valid_embeddings,
                                'info': user_info
                            }
                            user_count += 1
                            embedding_count += len(valid_embeddings)

            # ✅ ATUALIZAR BANCO EM MEMÓRIA
            old_user_count = len(self.facial_database)
            self.facial_database = database
            self.last_update = time.time()

            with self._metrics_lock:
                self.metrics.database_reloads += 1

            logger.info(f"✅ Database loaded: {user_count} users, {embedding_count} embeddings")

            if user_count > old_user_count:
                logger.info(f"🎉 NOVOS USUÁRIOS DETECTADOS! Antes: {old_user_count}, Agora: {user_count}")
            elif user_count < old_user_count:
                logger.warning(f"⚠️ Usuários removidos: Antes: {old_user_count}, Agora: {user_count}")

            return True

        except Exception as e:
            logger.error(f"❌ Database loading failed: {str(e)}")
            return False

    def _extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extrai embedding facial da imagem"""
        try:
            result = DeepFace.represent(
                img_path=face_image,
                model_name=self._model_config.MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
                align=True
            )

            if result and isinstance(result, list) and "embedding" in result[0]:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)

                if embedding.shape[0] == self._model_config.EMBEDDING_DIMENSION:
                    embedding_norm = np.linalg.norm(embedding)
                    return embedding / embedding_norm if embedding_norm > 0 else None

            return None
        except Exception as e:
            logger.error(f"Face embedding extraction failed: {str(e)}")
            return None

    def _recognize_face_secure(self, face_image: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Reconhecimento facial SEGURO
        """
        try:
            captured_embedding = self._extract_face_embedding(face_image)
            if captured_embedding is None:
                return None, None

            best_match = None
            min_distance = float('inf')
            second_best_distance = float('inf')

            # Busca no banco de dados
            for user_key, user_data in self.facial_database.items():
                for db_embedding in user_data['embeddings']:
                    distance = 1 - np.dot(captured_embedding, db_embedding)

                    if distance < min_distance:
                        second_best_distance = min_distance
                        min_distance = distance
                        best_match = user_key
                    elif distance < second_best_distance:
                        second_best_distance = distance

            # ✅ CRITÉRIOS SEGUROS
            if best_match and min_distance < self._model_config.DISTANCE_THRESHOLD:
                confidence = 1 - min_distance
                margin = second_best_distance - min_distance

                logger.info(f"🔍 Match encontrado: {best_match} - Distância: {min_distance:.4f}, Confiança: {confidence:.4f}, Margem: {margin:.4f}")

                # Critérios de aceitação
                very_high_confidence = confidence >= self._model_config.HIGH_CONFIDENCE_THRESHOLD
                very_low_distance = min_distance <= self._model_config.VERY_LOW_DISTANCE
                good_confidence_with_margin = (confidence >= self._model_config.MIN_CONFIDENCE and
                                            margin >= self._model_config.MARGIN_REQUIREMENT)

                if very_high_confidence or very_low_distance:
                    logger.info(f"✅ ACEITO - Critério alto: {best_match}")
                    return best_match, min_distance

                elif good_confidence_with_margin:
                    logger.info(f"✅ ACEITO - Boa confiança com margem: {best_match}")
                    return best_match, min_distance

                else:
                    logger.info(f"❌ REJEITADO - Confiança insuficiente ou margem pequena")
                    return None, None

            else:
                if best_match:
                    logger.info(f"❌ REJEITADO - Distância acima do threshold: {min_distance:.4f}")
                else:
                    logger.info("❌ Nenhum match encontrado")
                return None, None

        except Exception as e:
            logger.error(f"❌ Falha no reconhecimento facial: {str(e)}")
            return None, None

    def _decode_base64_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decodifica imagem base64 para array numpy"""
        try:
            if 'data:image' in image_data:
                image_data = image_data.split(',')[1]

            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image if image is not None and image.size > 0 else None

        except Exception as e:
            logger.error(f"❌ Erro na decodificação de imagem: {str(e)}")
            return None

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

                if w >= self._model_config.MIN_FACE_SIZE[0] and h >= self._model_config.MIN_FACE_SIZE[1]:
                    return face_data

            return None
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return None

    def process_face_login(self, image_data: str) -> Dict[str, Any]:
        """Processamento de login facial com verificação de banco atualizado"""
        start_time = time.time()

        try:
            with self._metrics_lock:
                self.metrics.total_attempts += 1

            # ✅ VERIFICAR SE O BANCO ESTÁ VAZIO
            if not self.facial_database:
                logger.warning("⚠️ Banco de dados vazio - tentando recarregar...")
                self.load_facial_database()

                if not self.facial_database:
                    with self._metrics_lock:
                        self.metrics.failed_recognitions += 1
                    return {
                        "authenticated": False,
                        "user": None,
                        "confidence": 0.0,
                        "message": "⚠️ Nenhum usuário cadastrado no sistema.",
                        "timestamp": self.get_current_timestamp()
                    }

            logger.info(f"📊 Banco atual: {len(self.facial_database)} usuários cadastrados")

            # Decodificar imagem
            frame = self._decode_base64_image(image_data)
            if frame is None:
                with self._metrics_lock:
                    self.metrics.failed_recognitions += 1
                return self._error_response("Dados de imagem inválidos")

            # Detectar rosto
            face_data = self._detect_face(frame)
            if not face_data:
                with self._metrics_lock:
                    self.metrics.failed_recognitions += 1
                    self.metrics.no_face_detected += 1
                return self._error_response("Nenhum rosto detectado")

            # Extrair ROI do rosto
            face_area = face_data["facial_area"]
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            face_roi = frame[y:y+h, x:x+w]

            # Reconhecer face
            user, distance = self._recognize_face_secure(face_roi)

            # Coletar métricas
            processing_time = time.time() - start_time
            with self._metrics_lock:
                self.metrics.processing_times.append(processing_time)

                if user:
                    self.metrics.successful_recognitions += 1
                    self.metrics.user_recognitions[user] += 1

                    confidence = 1 - distance
                    user_data = self.facial_database[user]['info']

                    return {
                        "authenticated": True,
                        "user": user,
                        "confidence": round(confidence, 4),
                        "distance": round(distance, 4),
                        "user_info": user_data,
                        "message": f"Bem-vindo(a), {user_data['nome']}!",
                        "timestamp": self.get_current_timestamp(),
                        "database_info": f"{len(self.facial_database)} usuários cadastrados"  # ✅ INFO EXTRA
                    }
                else:
                    self.metrics.failed_recognitions += 1

            return self._rejection_response()

        except Exception as e:
            with self._metrics_lock:
                self.metrics.failed_recognitions += 1
            logger.error(f"❌ Erro no processamento: {str(e)}")
            return self._error_response("Erro interno no processamento")

    def _rejection_response(self) -> Dict[str, Any]:
        """Resposta para usuário não reconhecido"""
        return {
            "authenticated": False,
            "user": None,
            "confidence": 0.0,
            "message": "Usuário não reconhecido - verifique se está cadastrado no sistema",
            "timestamp": self.get_current_timestamp(),
            "database_info": f"{len(self.facial_database)} usuários cadastrados"  # ✅ INFO EXTRA
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
            "monitoring_active": self.database_monitor.running,
            "database_reloads": self.metrics.database_reloads,
            "security_level": "HIGH",
            "threshold": self._model_config.DISTANCE_THRESHOLD,
            "min_confidence": self._model_config.MIN_CONFIDENCE
        }

    def reload_database(self) -> Tuple[bool, str]:
        """Recarrega banco de dados manualmente"""
        logger.info("🔄 Recarregamento manual do banco de dados solicitado")
        success = self.load_facial_database()
        if success:
            status = self.get_database_status()
            message = f"Database reloaded - {status['user_count']} users, {status['total_embeddings']} embeddings"
            return True, message
        else:
            return False, "Database reload failed"

    def initialize(self) -> bool:
        """Inicializa o serviço com monitoramento em tempo real"""
        logger.info("🔧 Initializing Face Recognition Service with Real-Time Database Monitoring...")
        logger.info(f"🎯 Using model: {self._model_config.MODEL_NAME}")
        logger.info("🔄 SISTEMA DE ATUALIZAÇÃO AUTOMÁTICA ATIVADO")

        # 1. Configurar triggers no banco de dados
        trigger_success = self._setup_database_triggers()

        # 2. Carregar banco de dados inicial
        db_success = self.load_facial_database()

        # 3. Iniciar monitoramento em tempo real
        monitor_success = self.database_monitor.start_monitoring()

        if db_success:
            if trigger_success and monitor_success:
                logger.info("🎯 Real-time database monitoring: ACTIVE - Novos usuários serão detectados automaticamente")
            else:
                logger.warning("⚠️ Real-time database monitoring: LIMITED")
                logger.info("💡 Manual reload available via /reload-database endpoint")

        return db_success

    def cleanup(self):
        """Limpeza do serviço"""
        if hasattr(self, 'database_monitor'):
            self.database_monitor.stop_monitoring()
        logger.info("🧹 Serviço de reconhecimento facial finalizado")

    @staticmethod
    def get_current_timestamp() -> str:
        """Retorna timestamp atual formatado"""
        return datetime.now().isoformat(sep=' ', timespec='seconds')