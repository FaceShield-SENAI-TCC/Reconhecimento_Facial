"""
Serviço de Reconhecimento Facial MELHORADO
Com validações rigorosas e tratamento para óculos
Versão: 2.0 - Com 8 embeddings e reconhecimento mais preciso
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
    """Configurações do modelo de reconhecimento - VGG-Face OTIMIZADO"""
    MODEL_NAME = "VGG-Face"

    # ✅ THRESHOLD MAIS RIGOROSO para evitar falsos positivos
    DISTANCE_THRESHOLD = 0.55  # ✅ Ajustado de 0.6 para 0.55

    # ✅ NOVO: Threshold de confiança mínimo
    MIN_CONFIDENCE_THRESHOLD = 0.7

    MIN_FACE_SIZE = (120, 120)  # ✅ Aumentado tamanho mínimo

    # ✅ DETECTOR OTIMIZADO PARA ÓCULOS
    DETECTOR_BACKEND = "ssd"  # ✅ Alterado para SSD (melhor com óculos)

    EMBEDDING_DIMENSION = 2622

    # ✅ NOVOS PARÂMETROS DE QUALIDADE
    MIN_SHARPNESS = 80  # ✅ Nitidez mínima para reconhecimento
    MIN_BRIGHTNESS = 60  # ✅ Brilho mínimo
    MAX_BRIGHTNESS = 190 # ✅ Brilho máximo

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
        self.reconnect_delay = 5  # Delay entre tentativas de reconexão

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
                logger.info("👂 Listening for database changes...")

                # Loop de escuta por notificações
                while self.running and self.connection and not self.connection.closed:
                    try:
                        # Verificar por notificações
                        self.connection.poll()
                        while self.connection.notifies:
                            notify = self.connection.notifies.pop(0)
                            logger.info(f"🔄 Database change detected: {notify.payload}")
                            self.callback()  # Recarrega o banco de dados

                        # Pequeno delay para evitar uso excessivo de CPU
                        time.sleep(0.5)

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
    Serviço principal de reconhecimento facial MELHORADO
    Com 8 embeddings por usuário e validações rigorosas
    """

    def __init__(self):
        self.facial_database = {}
        self.last_update = None
        self._db_config = DatabaseConfig()
        self._model_config = ModelConfig()
        self.database_monitor = DatabaseMonitor(self.load_facial_database)

        # ✅ NOVO: Estatísticas de reconhecimento
        self.recognition_stats = {
            'total_attempts': 0,
            'successful_auth': 0,
            'failed_auth': 0,
            'quality_rejections': 0
        }

    def _get_db_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Estabelece conexão com PostgreSQL"""
        try:
            return psycopg2.connect(
                dbname=self._db_config.DB_NAME,
                user=self._db_config.DB_USER,
                password=self._db_config.DB_PASSWORD,
                host=self._db_config.DB_HOST,
                port=self._db_config.DB_PORT,
                connect_timeout=10
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
            conn.close()
            logger.info("✅ Database triggers configured successfully")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Could not setup database triggers: {str(e)}")
            logger.info("💡 System will work without real-time monitoring")
            return False

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

    def _calculate_sharpness(self, image):
        """Calcula nitidez da imagem (melhorado)"""
        if image is None or image.size == 0:
            return 0
        try:
            small_img = cv2.resize(image, (100, 100))
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)  # ✅ Reduzir ruído
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 0

    def _validate_face_quality(self, face_image: np.ndarray) -> Tuple[bool, str]:
        """
        ✅ VALIDAÇÃO RIGOROSA da qualidade da face para reconhecimento
        """
        if face_image is None or face_image.size == 0:
            return False, "Imagem vazia"

        height, width = face_image.shape[:2]
        if height < self._model_config.MIN_FACE_SIZE[0] or width < self._model_config.MIN_FACE_SIZE[1]:
            return False, "Rosto muito pequeno"

        # Validar nitidez
        sharpness = self._calculate_sharpness(face_image)
        if sharpness < self._model_config.MIN_SHARPNESS:
            return False, f"Imagem muito borrada: {sharpness:.1f}"

        # Validar brilho
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < self._model_config.MIN_BRIGHTNESS:
            return False, f"Brilho muito baixo: {brightness:.1f}"
        if brightness > self._model_config.MAX_BRIGHTNESS:
            return False, f"Brilho muito alto: {brightness:.1f}"

        # Validar contraste
        contrast = np.std(gray)
        if contrast < 35:
            return False, f"Contraste insuficiente: {contrast:.1f}"

        return True, f"Qualidade OK: Sharp={sharpness:.1f}, Bright={brightness:.1f}"

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
                invalid_embeddings = 0

                for nome, sobrenome, turma, tipo, embeddings in cursor.fetchall():
                    # Armazenar informações separadas para formatação flexível
                    user_info = {
                        'display_name': f"{nome} {sobrenome}",
                        'full_info': f"{nome} {sobrenome} - {turma} ({tipo})",
                        'nome': nome,
                        'sobrenome': sobrenome,
                        'turma': turma,
                        'tipo': tipo
                    }

                    # Processar embeddings - ✅ ATUALIZADO para VGG-Face
                    valid_embeddings = []
                    for embedding in embeddings:
                        # ✅ Verificar dimensão para VGG-Face (2622)
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
                logger.info(f"📊 Average embeddings per user: {embedding_count/user_count:.1f}")
            return True

        except Exception as e:
            logger.error(f"❌ Database loading failed: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()

    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        ✅ NOVO: Pré-processamento da imagem para melhor reconhecimento
        Especialmente útil para usuários com óculos
        """
        try:
            # Equalização de histograma para melhorar contraste
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Suavização leve para reduzir ruído
            processed = cv2.GaussianBlur(processed, (1, 1), 0)

            return processed
        except:
            return face_image

    def _extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrai embedding facial com validação MELHORADA
        """
        try:
            # ✅ PRÉ-PROCESSAMENTO para melhorar reconhecimento com óculos
            processed_face = self._preprocess_face(face_image)

            result = DeepFace.represent(
                img_path=processed_face,
                model_name=self._model_config.MODEL_NAME,
                detector_backend=self._model_config.DETECTOR_BACKEND,
                enforce_detection=False,
                align=True  # ✅ Alinhamento ativado
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
        ✅ RECONHECIMENTO MELHORADO com múltiplas validações
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

                    # ✅ VALIDAÇÃO DUPLA: distância E confiança
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
        Detecção de rostos MELHORADA
        """
        try:
            detected_faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self._model_config.DETECTOR_BACKEND,  # ✅ SSD
                enforce_detection=False
            )

            if (detected_faces and len(detected_faces) > 0 and
                "facial_area" in detected_faces[0] and
                detected_faces[0].get('confidence', 0) > 0.7):  # ✅ Confiança mínima
                return detected_faces[0]

            return None

        except Exception as e:
            logger.error(f"Falha na detecção: {str(e)}")
            return None

    def process_face_login(self, image_data: str) -> Dict[str, Any]:
        """
        ✅ PROCESSAMENTO DE LOGIN MELHORADO com validações rigorosas
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
            return self._error_response("Nenhum rosto detectado - posicione-se melhor")

        # Extrair região do rosto
        face_area = face_data["facial_area"]
        x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']

        # ✅ VALIDAR TAMANHO DO ROSTO
        if w < self._model_config.MIN_FACE_SIZE[0] or h < self._model_config.MIN_FACE_SIZE[1]:
            self.recognition_stats['failed_auth'] += 1
            return self._error_response("Rosto muito pequeno - aproxime-se da câmera")

        face_roi = frame[y:y+h, x:x+w]

        # ✅ VALIDAÇÃO RIGOROSA DE QUALIDADE
        is_quality_ok, quality_msg = self._validate_face_quality(face_roi)
        if not is_quality_ok:
            self.recognition_stats['quality_rejections'] += 1
            return self._error_response(f"Qualidade insuficiente: {quality_msg}")

        # Reconhecer rosto
        user, distance, confidence = self._recognize_face(face_roi)

        if user and confidence and confidence > self._model_config.MIN_CONFIDENCE_THRESHOLD:
            self.recognition_stats['successful_auth'] += 1

            # ✅ LOG DETALHADO para análise
            logger.info(f"✅ AUTH SUCCESS: {user} - Dist: {distance:.3f} - Conf: {confidence:.3f}")

            return self._success_response(user, confidence, distance)
        else:
            self.recognition_stats['failed_auth'] += 1

            # ✅ LOG PARA ANÁLISE DE FALHAS
            if user:  # Usuário encontrado mas confiança baixa
                logger.warning(f"⚠️ AUTH REJECTED: {user} - Confiança muito baixa: {confidence:.3f}")
            else:
                logger.info(f"❌ AUTH FAILED: Usuário não reconhecido")

            return self._rejection_response()

    def _success_response(self, user: str, confidence: float, distance: float) -> Dict[str, Any]:
        """Resposta para autenticação bem-sucedida"""
        user_data = self.facial_database[user]['info']

        return {
            "authenticated": True,
            "user": user,
            "user_details": user_data,  # ✅ MAIS INFORMAÇÕES
            "confidence": round(confidence, 4),
            "distance": round(distance, 4),  # ✅ NOVO: incluir distância
            "message": f"Bem-vindo(a), {user_data['nome']}!",
            "timestamp": self.get_current_timestamp(),
            "stats": {  # ✅ NOVO: estatísticas
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
            "message": "Usuário não reconhecido - verifique posicionamento e iluminação",
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

    def get_database_status(self) -> Dict[str, Any]:
        """Status do banco de dados com estatísticas MELHORADAS"""
        user_count = len(self.facial_database)
        total_embeddings = sum(len(user_data['embeddings']) for user_data in self.facial_database.values())

        # ✅ ESTATÍSTICAS DE RECONHECIMENTO
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
            "recognition_stats": {  # ✅ NOVO: estatísticas
                "total_attempts": self.recognition_stats['total_attempts'],
                "successful_auth": self.recognition_stats['successful_auth'],
                "failed_auth": self.recognition_stats['failed_auth'],
                "quality_rejections": self.recognition_stats['quality_rejections'],
                "success_rate": f"{success_rate}%"
            }
        }

    def reload_database(self) -> Tuple[bool, str]:
        """
        Recarrega banco de dados com logging MELHORADO

        Returns:
            Tuple[bool, str]: (sucesso, mensagem)
        """
        success = self.load_facial_database()
        if success:
            status = self.get_database_status()
            message = (f"Database recarregado - {status['user_count']} usuários, "
                      f"{status['total_embeddings']} embeddings "
                      f"(média: {status['avg_embeddings_per_user']} por usuário)")

            # ✅ LOG DETALHADO
            logger.info(f"🔄 {message}")
            logger.info(f"📊 Estatísticas: {status['recognition_stats']}")

            return True, message
        else:
            return False, "Falha no recarregamento do banco"

    def initialize(self) -> bool:
        """Inicializa o serviço com logging MELHORADO"""
        logger.info("🔧 Inicializando Serviço de Reconhecimento Facial MELHORADO...")
        logger.info(f"🎯 Modelo: {self._model_config.MODEL_NAME}")
        logger.info(f"📊 Dimensão: {self._model_config.EMBEDDING_DIMENSION}")
        logger.info(f"🎯 Threshold: {self._model_config.DISTANCE_THRESHOLD}")
        logger.info(f"🔍 Detector: {self._model_config.DETECTOR_BACKEND} (otimizado para óculos)")

        # 1. Criar tabela se não existir
        if not self._create_table_if_not_exists():
            logger.error("❌ Falha na criação da tabela")
            return False

        # 2. Configurar triggers no banco de dados
        trigger_success = self._setup_database_triggers()

        # 3. Carregar banco de dados inicial
        db_success = self.load_facial_database()

        # 4. Iniciar monitoramento em tempo real (mesmo se triggers falharem)
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