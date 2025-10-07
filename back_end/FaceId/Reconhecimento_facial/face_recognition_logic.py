"""
Serviço de Reconhecimento Facial com Monitoramento em Tempo Real
Gerencia banco de dados PostgreSQL e processamento de imagens
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
    """Configurações do modelo de reconhecimento"""
    MODEL_NAME = "Facenet"
    DISTANCE_THRESHOLD = 0.40
    MIN_FACE_SIZE = (100, 100)
    DETECTOR_BACKEND = "opencv"

class DatabaseMonitor:
    """
    Monitora o banco de dados PostgreSQL em tempo real usando LISTEN/NOTIFY
    """

    def __init__(self, callback_function):
        self.callback = callback_function
        self.connection = None
        self.monitor_thread = None
        self.running = False
        self._db_config = DatabaseConfig()

    def start_monitoring(self):
        """Inicia o monitoramento em tempo real"""
        try:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("✅ Database monitoring started")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start database monitoring: {str(e)}")
            return False

    def stop_monitoring(self):
        """Para o monitoramento"""
        self.running = False
        if self.connection and not self.connection.closed:
            self.connection.close()
        logger.info("⏹️ Database monitoring stopped")

    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self.running:
            try:
                self.connection = psycopg2.connect(
                    dbname=self._db_config.DB_NAME,
                    user=self._db_config.DB_USER,
                    password=self._db_config.DB_PASSWORD,
                    host=self._db_config.DB_HOST,
                    port=self._db_config.DB_PORT
                )
                self.connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

                cursor = self.connection.cursor()
                cursor.execute("LISTEN usuarios_update;")
                logger.info("👂 Listening for database changes...")

                while self.running:
                    self.connection.poll()
                    while self.connection.notifies:
                        notify = self.connection.notifies.pop(0)
                        logger.info(f"🔄 Database change detected: {notify.payload}")
                        self.callback()  # Recarrega o banco de dados

                    time.sleep(1)  # Pequeno delay para evitar CPU alto

            except psycopg2.OperationalError as e:
                logger.warning(f"Database connection lost, reconnecting...: {str(e)}")
                time.sleep(5)  # Espera antes de reconectar
            except Exception as e:
                logger.error(f"Monitor loop error: {str(e)}")
                time.sleep(5)  # Espera antes de tentar novamente
            finally:
                if self.connection and not self.connection.closed:
                    self.connection.close()

    def trigger_manual_update(self):
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
            cursor.execute("NOTIFY usuarios_update, 'manual_update';")
            conn.close()
            logger.info("🔔 Manual database update triggered")
        except Exception as e:
            logger.error(f"Failed to trigger manual update: {str(e)}")

class FaceRecognitionService:
    """
    Serviço principal de reconhecimento facial com monitoramento em tempo real
    """

    def __init__(self):
        self.facial_database = {}
        self.last_update = None
        self._db_config = DatabaseConfig()
        self._model_config = ModelConfig()
        self.database_monitor = DatabaseMonitor(self.load_facial_database)

    def _get_db_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Estabelece conexão com PostgreSQL"""
        try:
            return psycopg2.connect(
                dbname=self._db_config.DB_NAME,
                user=self._db_config.DB_USER,
                password=self._db_config.DB_PASSWORD,
                host=self._db_config.DB_HOST,
                port=self._db_config.DB_PORT
            )
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return None

    def _create_table_if_not_exists(self):
        """Cria a tabela usuarios se não existir"""
        try:
            conn = self._get_db_connection()
            if not conn:
                return False

            cursor = conn.cursor()

            # Verificar se a tabela já existe
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'usuarios'
                );
            """)
            table_exists = cursor.fetchone()[0]

            if not table_exists:
                logger.info("📦 Creating 'usuarios' table...")
                cursor.execute("""
                    CREATE TABLE usuarios(
                        id SERIAL PRIMARY KEY,
                        nome VARCHAR(100) NOT NULL,
                        sobrenome VARCHAR(100) NOT NULL,
                        turma VARCHAR(50) NOT NULL,
                        tipo VARCHAR(20) NOT NULL DEFAULT 'aluno',
                        embeddings JSONB,
                        foto_perfil BYTEA,
                        data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(nome, sobrenome, turma)
                    )
                """)

                # Criar índices para melhor performance
                cursor.execute("""
                    CREATE INDEX idx_usuarios_nome_sobrenome 
                    ON usuarios(nome, sobrenome, turma)
                """)

                conn.commit()
                logger.info("✅ 'usuarios' table created successfully")
            else:
                logger.info("✅ 'usuarios' table already exists")

            conn.close()
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create table: {str(e)}")
            return False

    def _setup_database_triggers(self):
        """Configura triggers no PostgreSQL para notificar mudanças"""
        try:
            conn = self._get_db_connection()
            if not conn:
                return False

            cursor = conn.cursor()

            # Criar função de trigger
            cursor.execute("""
                CREATE OR REPLACE FUNCTION notify_usuarios_update()
                RETURNS TRIGGER AS $$
                BEGIN
                    PERFORM pg_notify('usuarios_update', 'table_updated');
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)

            # Criar trigger para INSERT, UPDATE, DELETE
            cursor.execute("""
                DROP TRIGGER IF EXISTS usuarios_notify_trigger ON usuarios;
                CREATE TRIGGER usuarios_notify_trigger
                AFTER INSERT OR UPDATE OR DELETE ON usuarios
                FOR EACH ROW EXECUTE FUNCTION notify_usuarios_update();
            """)

            conn.commit()
            conn.close()
            logger.info("✅ Database triggers configured successfully")
            return True

        except Exception as e:
            logger.warning(f"Could not setup database triggers: {str(e)}")
            return False

    def load_facial_database(self) -> bool:
        """
        Carrega embeddings faciais do PostgreSQL

        Returns:
            bool: True se carregado com sucesso
        """
        logger.info("🔄 Loading facial database from PostgreSQL...")

        conn = self._get_db_connection()
        if not conn:
            return False

        try:
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
                    # Armazenar informações separadas para formatação flexível
                    user_info = {
                        'display_name': f"{nome} {sobrenome}",  # Apenas nome e sobrenome
                        'full_info': f"{nome} {sobrenome} - {turma} ({tipo})",
                        'nome': nome,
                        'sobrenome': sobrenome,
                        'turma': turma,
                        'tipo': tipo
                    }

                    # Processar embeddings
                    valid_embeddings = []
                    for embedding in embeddings:
                        if len(embedding) == 128:  # Dimensão do Facenet
                            embedding_array = np.array(embedding, dtype=np.float32)
                            embedding_norm = np.linalg.norm(embedding_array)

                            if embedding_norm > 0:
                                embedding_array /= embedding_norm
                                valid_embeddings.append(embedding_array)

                    if valid_embeddings:
                        # Usar display_name como chave para mostrar apenas nome
                        database[user_info['display_name']] = {
                            'embeddings': valid_embeddings,
                            'info': user_info
                        }
                        user_count += 1
                        embedding_count += len(valid_embeddings)
                        logger.debug(f"Loaded user: {user_info['display_name']} - {len(valid_embeddings)} embeddings")

            self.facial_database = database
            self.last_update = time.time()

            if user_count == 0:
                logger.warning("⚠️ Database loaded but no users with embeddings found")
                logger.info("💡 Use the registration system to add users first")
            else:
                logger.info(f"✅ Database loaded: {user_count} users, {embedding_count} embeddings")
            return True

        except Exception as e:
            logger.error(f"❌ Database loading failed: {str(e)}")
            return False
        finally:
            conn.close()

    def _extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrai embedding facial da imagem

        Args:
            face_image: Imagem do rosto (BGR format)

        Returns:
            Optional[np.ndarray]: Embedding normalizado ou None
        """
        try:
            result = DeepFace.represent(
                img_path=face_image,
                model_name=self._model_config.MODEL_NAME,
                detector_backend=self._model_config.DETECTOR_BACKEND,
                enforce_detection=False
            )

            if result and isinstance(result, list) and "embedding" in result[0]:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)
                embedding_norm = np.linalg.norm(embedding)
                return embedding / embedding_norm if embedding_norm > 0 else None

            logger.warning("No face embedding generated")
            return None

        except Exception as e:
            logger.error(f"Face embedding extraction failed: {str(e)}")
            return None

    def _recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Reconhece face comparando com banco de dados

        Args:
            face_image: Imagem do rosto para reconhecer

        Returns:
            Tuple[Optional[str], Optional[float]]: (usuário, distância) ou (None, None)
        """
        try:
            captured_embedding = self._extract_face_embedding(face_image)
            if captured_embedding is None:
                return None, None

            best_match = None
            min_distance = float('inf')

            # Busca linear no banco de dados
            for user_key, user_data in self.facial_database.items():
                for db_embedding in user_data['embeddings']:
                    distance = 1 - np.dot(captured_embedding, db_embedding)

                    if distance < min_distance and distance < self._model_config.DISTANCE_THRESHOLD:
                        min_distance = distance
                        best_match = user_key  # Já é o display_name (apenas nome)

            return best_match, min_distance if best_match else None

        except Exception as e:
            logger.error(f"Face recognition failed: {str(e)}")
            return None, None

    def _decode_base64_image(self, image_data: str) -> Optional[np.ndarray]:
        """
        Decodifica imagem base64 para array numpy

        Args:
            image_data: String base64 da imagem

        Returns:
            Optional[np.ndarray]: Imagem decodificada ou None
        """
        try:
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image if image is not None and image.size > 0 else None
        except Exception as e:
            logger.error(f"Image decoding failed: {str(e)}")
            return None

    def _detect_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detecta rostos na imagem

        Args:
            image: Imagem para detecção

        Returns:
            Optional[Dict]: Informações do rosto detectado ou None
        """
        try:
            detected_faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self._model_config.DETECTOR_BACKEND,
                enforce_detection=False
            )

            if (detected_faces and len(detected_faces) > 0 and
                "facial_area" in detected_faces[0]):
                return detected_faces[0]

            return None

        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return None

    def process_face_login(self, image_data: str) -> Dict[str, Any]:
        """
        Processa tentativa de login por reconhecimento facial

        Args:
            image_data: Imagem em base64 (sem prefixo)

        Returns:
            Dict: Resultado do processamento
        """
        # Verificar se há usuários no banco
        if not self.facial_database:
            return {
                "authenticated": False,
                "user": None,
                "confidence": 0.0,
                "message": "⚠️ Nenhum usuário cadastrado no sistema. Use o sistema de cadastro primeiro.",
                "timestamp": self.get_current_timestamp()
            }

        # Decodificar imagem
        frame = self._decode_base64_image(image_data)
        if frame is None:
            return self._error_response("Invalid image data")

        # Detectar rosto
        face_data = self._detect_face(frame)
        if not face_data:
            return self._error_response("No face detected")

        # Validar tamanho do rosto
        face_area = face_data["facial_area"]
        w, h = face_area['w'], face_area['h']

        if w < self._model_config.MIN_FACE_SIZE[0] or h < self._model_config.MIN_FACE_SIZE[1]:
            return self._error_response("Face too small - move closer to camera")

        # Extrair e reconhecer rosto
        x, y = face_area['x'], face_area['y']
        face_roi = frame[y:y+h, x:x+w]

        user, distance = self._recognize_face(face_roi)

        if user:
            confidence = 1 - distance
            return self._success_response(user, confidence)
        else:
            return self._rejection_response()

    def _success_response(self, user: str, confidence: float) -> Dict[str, Any]:
        """Resposta para autenticação bem-sucedida"""
        return {
            "authenticated": True,
            "user": user,  # Apenas nome e sobrenome
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
            "message": "Usuário não reconhecido",
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
            "monitoring_active": self.database_monitor.running if hasattr(self, 'database_monitor') else False,
            "database_type": "PostgreSQL",
            "model": self._model_config.MODEL_NAME,
            "threshold": self._model_config.DISTANCE_THRESHOLD
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
        """Inicializa o serviço com monitoramento em tempo real"""
        logger.info("🔧 Initializing Face Recognition Service...")

        # 1. Criar tabela se não existir
        if not self._create_table_if_not_exists():
            logger.error("❌ Failed to create database table")
            return False

        # 2. Configurar triggers no banco de dados
        self._setup_database_triggers()

        # 3. Carregar banco de dados inicial
        success = self.load_facial_database()

        # 4. Iniciar monitoramento em tempo real
        if success:
            self.database_monitor.start_monitoring()
            logger.info("🎯 Real-time database monitoring activated")

        return success

    def cleanup(self):
        """Limpeza do serviço"""
        if hasattr(self, 'database_monitor'):
            self.database_monitor.stop_monitoring()
        logger.info("🧹 Face recognition service cleaned up")

    @staticmethod
    def get_current_timestamp() -> str:
        """Retorna timestamp atual formatado"""
        return datetime.now().isoformat(sep=' ', timespec='seconds')