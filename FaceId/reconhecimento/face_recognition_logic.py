"""
Serviço de Reconhecimento Facial Refatorado
Usa módulos compartilhados e estrutura profissional
"""
import logging
import time
import threading
import numpy as np
import cv2
import psycopg2
import psycopg2.extensions
from typing import Tuple, Optional, Dict, Any
from contextlib import contextmanager
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from deepface import DeepFace

# Módulos compartilhados
from common.config import DATABASE_CONFIG, MODEL_CONFIG
from common.database import DatabaseManager
from common.image_utils import ImageValidator, FaceQualityValidator
from common.exceptions import (
    DatabaseError, DatabaseConnectionError, ImageValidationError,
    FaceDetectionError, FaceRecognitionServiceError
)

logger = logging.getLogger(__name__)

# Classes de dados locais (substituindo shared_models.models)
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

@dataclass
class DatabaseStatus:
    """Status do banco de dados"""
    status: str
    user_count: int
    total_embeddings: int
    last_update: Optional[float]
    monitoring_active: bool
    database_type: str

@dataclass
class SystemMetrics:
    """Métricas do sistema"""
    total_attempts: int
    successful_recognitions: int
    failed_recognitions: int
    no_face_detected: int
    average_processing_time: float
    success_rate: float
    database_reloads: int

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
    database_reloads: int = 0

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
        self.reconnect_delay = 5

    def start_monitoring(self):
        """Inicia o monitoramento em tempo real"""
        try:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="DatabaseMonitor"
            )
            self.monitor_thread.start()
            logger.info("✅ Monitoramento do banco de dados iniciado - Novos usuários serão detectados automaticamente")
            return True
        except Exception as e:
            logger.error(f"❌ Falha ao iniciar monitoramento do banco: {str(e)}")
            return False

    def stop_monitoring(self):
        """Para o monitoramento"""
        self.running = False
        if self.connection and not self.connection.closed:
            try:
                self.connection.close()
            except:
                pass
        logger.info("⏹️ Monitoramento do banco de dados parado")

    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self.running:
            try:
                # Estabelecer conexão para monitoramento
                self.connection = psycopg2.connect(
                    **DATABASE_CONFIG.to_dict(),
                    connect_timeout=10
                )
                self.connection.set_isolation_level(
                    psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT
                )

                cursor = self.connection.cursor()
                cursor.execute("LISTEN usuarios_update;")
                logger.info("👂 Escutando por mudanças no banco de dados...")

                # Loop de escuta por notificações
                while self.running and self.connection and not self.connection.closed:
                    try:
                        # Verificar por notificações
                        self.connection.poll()
                        while self.connection.notifies:
                            notify = self.connection.notifies.pop(0)
                            logger.info(f"🔄 Mudança no banco detectada: {notify.payload}")

                            # Recarregar o banco de dados
                            if self.database_reload_callback:
                                logger.info("🔄 Recarregando banco de dados devido a mudanças...")
                                self.database_reload_callback()

                        # Pequeno delay para evitar uso excessivo de CPU
                        time.sleep(1)

                    except psycopg2.InterfaceError as e:
                        logger.warning(f"Conexão com banco interrompida: {str(e)}")
                        break
                    except Exception as e:
                        logger.error(f"Erro no loop interno de monitoramento: {str(e)}")
                        break

            except psycopg2.OperationalError as e:
                logger.warning(f"❌ Conexão com banco falhou, tentando novamente em {self.reconnect_delay}s: {str(e)}")
            except Exception as e:
                logger.error(f"❌ Erro no loop de monitoramento: {str(e)}")

            # Esperar antes de tentar reconectar
            if self.running:
                time.sleep(self.reconnect_delay)

class FaceRecognitionService:
    """
    Serviço principal de reconhecimento facial com atualização automática do banco
    """

    def __init__(self):
        self.facial_database = {}
        self.last_update = None
        self.metrics = RecognitionMetrics()
        self._metrics_lock = threading.RLock()
        self.db_manager = DatabaseManager()
        self.quality_validator = FaceQualityValidator()

        # Monitor de banco de dados em tempo real
        self.database_monitor = DatabaseMonitor(self.load_facial_database)

    def load_facial_database(self) -> bool:
        """
        Carrega embeddings faciais do PostgreSQL com verificação de atualização

        Returns:
            bool: True se carregado com sucesso
        """
        logger.info("🔄 Carregando banco de dados facial do PostgreSQL...")

        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT nome, sobrenome, turma, tipo_usuario, username, embeddings
                        FROM usuarios
                        WHERE embeddings IS NOT NULL AND jsonb_array_length(embeddings) > 0
                    """)

                    database = {}
                    user_count = 0
                    embedding_count = 0
                    invalid_embeddings = 0

                    for nome, sobrenome, turma, tipo_usuario, username, embeddings in cursor.fetchall():
                        # Criar display name com informações completas
                        if tipo_usuario.upper() == "PROFESSOR" and username:
                            display_name = f"{nome} {sobrenome} (@{username})"
                        else:
                            display_name = f"{nome} {sobrenome}"

                        user_info = {
                            'display_name': display_name,
                            'nome': nome,
                            'sobrenome': sobrenome,
                            'turma': turma,
                            'tipo_usuario': tipo_usuario,
                            'username': username,
                            'is_professor': tipo_usuario.upper() == "PROFESSOR"
                        }

                        valid_embeddings = []
                        for embedding in embeddings:
                            if embedding and len(embedding) == MODEL_CONFIG.EMBEDDING_DIMENSION:
                                try:
                                    embedding_array = np.array(embedding, dtype=np.float32)
                                    embedding_norm = np.linalg.norm(embedding_array)
                                    if embedding_norm > 0:
                                        valid_embeddings.append(embedding_array / embedding_norm)
                                except Exception as e:
                                    logger.warning(f"Embedding inválido para {user_info['display_name']}: {str(e)}")
                                    invalid_embeddings += 1
                            else:
                                invalid_embeddings += 1

                        if valid_embeddings:
                            database[display_name] = {
                                'embeddings': valid_embeddings,
                                'info': user_info
                            }
                            user_count += 1
                            embedding_count += len(valid_embeddings)

            # Atualizar banco em memória
            old_user_count = len(self.facial_database)
            self.facial_database = database
            self.last_update = time.time()

            with self._metrics_lock:
                self.metrics.database_reloads += 1

            # Contar estatísticas por tipo de usuário
            professores_count = sum(1 for user_data in database.values()
                                  if user_data['info']['tipo_usuario'].upper() == "PROFESSOR")
            alunos_count = user_count - professores_count

            logger.info(f"✅ Banco carregado: {user_count} usuários ({professores_count} professores, {alunos_count} alunos), {embedding_count} embeddings")

            if invalid_embeddings > 0:
                logger.warning(f"⚠️ Encontrados {invalid_embeddings} embeddings inválidos")

            if user_count > old_user_count:
                logger.info(f"🎉 NOVOS USUÁRIOS DETECTADOS! Antes: {old_user_count}, Agora: {user_count}")
            elif user_count < old_user_count:
                logger.warning(f"⚠️ Usuários removidos: Antes: {old_user_count}, Agora: {user_count}")

            return True

        except DatabaseConnectionError as e:
            logger.error(f"❌ Falha de conexão ao carregar banco: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Falha ao carregar banco de dados: {str(e)}")
            return False

    def _extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrai embedding facial da imagem usando modelo configurado

        Args:
            face_image: Imagem do rosto

        Returns:
            Optional[np.ndarray]: Embedding normalizado ou None
        """
        try:
            result = DeepFace.represent(
                img_path=face_image,
                model_name=MODEL_CONFIG.MODEL_NAME,
                detector_backend=MODEL_CONFIG.DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )

            if result and isinstance(result, list) and "embedding" in result[0]:
                embedding = np.array(result[0]["embedding"], dtype=np.float32)

                if embedding.shape[0] == MODEL_CONFIG.EMBEDDING_DIMENSION:
                    embedding_norm = np.linalg.norm(embedding)
                    return embedding / embedding_norm if embedding_norm > 0 else None

            logger.warning("Nenhum embedding facial gerado")
            return None

        except Exception as e:
            logger.error(f"Falha na extração de embedding facial: {str(e)}")
            return None

    def _recognize_face_secure(self, face_image: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Reconhecimento facial SEGURO com critérios balanceados

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
            second_best_distance = float('inf')
            best_user_data = None

            # Busca no banco de dados
            for user_key, user_data in self.facial_database.items():
                for db_embedding in user_data['embeddings']:
                    # Distância cosseno (1 - similaridade cosseno)
                    distance = 1 - np.dot(captured_embedding, db_embedding)

                    if distance < min_distance:
                        second_best_distance = min_distance
                        min_distance = distance
                        best_match = user_key
                        best_user_data = user_data
                    elif distance < second_best_distance:
                        second_best_distance = distance

            # NOVOS CRITÉRIOS MAIS RIGOROSOS
            if best_match and min_distance < MODEL_CONFIG.DISTANCE_THRESHOLD:
                confidence = 1 - min_distance
                margin = second_best_distance - min_distance

                logger.info(
                    f"🔍 Match encontrado: {best_match} - Dist: {min_distance:.4f}, Conf: {confidence:.4f}, Margem: {margin:.4f}")

                # CRITÉRIOS HIERÁRQUICOS MAIS RIGOROSOS
                extremely_high_confidence = confidence >= 0.90
                very_high_confidence = confidence >= 0.85
                good_margin = margin >= 0.05  # Margem aumentada significativamente
                acceptable_margin = margin >= 0.02

                # Apenas aceita com confiança muito alta
                if extremely_high_confidence:
                    logger.info(f"✅ ACEITO - Confiança extremamente alta: {best_match}")
                    return best_match, min_distance

                elif very_high_confidence and good_margin:
                    logger.info(f"✅ ACEITO - Confiança muito alta com boa margem: {best_match}")
                    return best_match, min_distance

                elif very_high_confidence and acceptable_margin:
                    logger.info(f"✅ ACEITO - Confiança muito alta com margem aceitável: {best_match}")
                    return best_match, min_distance

                else:
                    logger.info(
                        f"❌ REJEITADO - Confiança insuficiente ({confidence:.4f}) ou margem pequena ({margin:.4f})")
                    return None, None

            else:
                if best_match:
                    logger.info(
                        f"❌ REJEITADO - Distância acima do threshold: {min_distance:.4f} > {MODEL_CONFIG.DISTANCE_THRESHOLD}")
                else:
                    logger.info("❌ Nenhum match encontrado")
                return None, None

        except Exception as e:
            logger.error(f"❌ Falha no reconhecimento facial: {str(e)}")
            return None, None

    def _detect_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detecta rostos na imagem com validação de qualidade

        Args:
            image: Imagem para detecção

        Returns:
            Optional[Dict]: Dados do rosto detectado ou None
        """
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

                # Validar tamanho mínimo
                if w >= MODEL_CONFIG.MIN_FACE_SIZE[0] and h >= MODEL_CONFIG.MIN_FACE_SIZE[1]:
                    return face_data

            return None

        except Exception as e:
            logger.error(f"Falha na detecção facial: {str(e)}")
            return None

    def process_face_login(self, image_data: str) -> Dict[str, Any]:
        """
        Processamento de login facial com verificação completa

        Args:
            image_data: Imagem em base64

        Returns:
            Dict: Resultado do reconhecimento
        """
        start_time = time.time()

        try:
            with self._metrics_lock:
                self.metrics.total_attempts += 1

            # Verificar se o banco está vazio
            if not self.facial_database:
                logger.warning("⚠️ Banco de dados vazio - tentando recarregar...")
                self.load_facial_database()

                if not self.facial_database:
                    with self._metrics_lock:
                        self.metrics.failed_recognitions += 1

                    return RecognitionResult(
                        authenticated=False,
                        confidence=0.0,
                        message="⚠️ Nenhum usuário cadastrado no sistema.",
                        timestamp=self.get_current_timestamp()
                    ).__dict__

            # Estatísticas do banco atual
            professores_count = sum(1 for user_data in self.facial_database.values()
                                  if user_data['info']['tipo_usuario'].upper() == "PROFESSOR")
            alunos_count = len(self.facial_database) - professores_count

            logger.info(f"📊 Banco atual: {len(self.facial_database)} usuários ({professores_count} professores, {alunos_count} alunos)")

            # Decodificar imagem
            frame = ImageValidator.decode_base64_image(image_data)
            if frame is None:
                with self._metrics_lock:
                    self.metrics.failed_recognitions += 1

                return RecognitionResult(
                    authenticated=False,
                    confidence=0.0,
                    message="Dados de imagem inválidos",
                    timestamp=self.get_current_timestamp()
                ).__dict__

            # Pré-processar imagem
            frame = ImageValidator.preprocess_image(frame)

            # Detectar rosto
            face_data = self._detect_face(frame)
            if not face_data:
                with self._metrics_lock:
                    self.metrics.failed_recognitions += 1
                    self.metrics.no_face_detected += 1

                return RecognitionResult(
                    authenticated=False,
                    confidence=0.0,
                    message="Nenhum rosto detectado - posicione-se melhor na frente da câmera",
                    timestamp=self.get_current_timestamp()
                ).__dict__

            # Extrair ROI do rosto
            face_area = face_data["facial_area"]
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            face_roi = frame[y:y+h, x:x+w]

            # Validar qualidade do rosto
            is_valid, validation_msg = self.quality_validator.validate_face_image(face_roi)
            if not is_valid:
                with self._metrics_lock:
                    self.metrics.failed_recognitions += 1

                return RecognitionResult(
                    authenticated=False,
                    confidence=0.0,
                    message=f"Qualidade do rosto insuficiente: {validation_msg}",
                    timestamp=self.get_current_timestamp()
                ).__dict__

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

                    # Mensagem personalizada por tipo de usuário
                    if user_data['tipo_usuario'].upper() == "PROFESSOR":
                        if user_data['username']:
                            message = f"Bem-vindo(a), Professor(a) {user_data['nome']} (@{user_data['username']})!"
                        else:
                            message = f"Bem-vindo(a), Professor(a) {user_data['nome']}!"
                    else:
                        message = f"Bem-vindo(a), {user_data['nome']}!"

                    logger.info(f"⏱️ Tempo de processamento: {processing_time:.3f}s")

                    return RecognitionResult(
                        authenticated=True,
                        user=user,
                        confidence=round(confidence, 4),
                        distance=round(distance, 4),
                        user_info=user_data,
                        message=message,
                        timestamp=self.get_current_timestamp()
                    ).__dict__
                else:
                    self.metrics.failed_recognitions += 1

            logger.info(f"⏱️ Tempo de processamento: {processing_time:.3f}s")

            return RecognitionResult(
                authenticated=False,
                confidence=0.0,
                message="Usuário não reconhecido - verifique se está cadastrado no sistema",
                timestamp=self.get_current_timestamp()
            ).__dict__

        except Exception as e:
            with self._metrics_lock:
                self.metrics.failed_recognitions += 1

            logger.error(f"❌ Erro no processamento: {str(e)}")

            return RecognitionResult(
                authenticated=False,
                confidence=0.0,
                message="Erro interno no processamento",
                timestamp=self.get_current_timestamp()
            ).__dict__

    def get_database_status(self) -> Dict[str, Any]:
        """Retorna status do banco de dados"""
        user_count = len(self.facial_database)
        total_embeddings = sum(
            len(user_data['embeddings'])
            for user_data in self.facial_database.values()
        )

        # Estatísticas por tipo de usuário
        professores_count = sum(1 for user_data in self.facial_database.values()
                              if user_data['info']['tipo_usuario'].upper() == "PROFESSOR")
        alunos_count = user_count - professores_count

        return DatabaseStatus(
            status="loaded" if self.facial_database else "empty",
            user_count=user_count,
            total_embeddings=total_embeddings,
            last_update=self.last_update,
            monitoring_active=self.database_monitor.running,
            database_type="PostgreSQL"
        ).__dict__

    def get_detailed_database_status(self) -> Dict[str, Any]:
        """Retorna status detalhado do banco de dados com estatísticas por tipo"""
        user_count = len(self.facial_database)
        total_embeddings = sum(
            len(user_data['embeddings'])
            for user_data in self.facial_database.values()
        )

        # Estatísticas detalhadas
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
            "monitoring_active": self.database_monitor.running,
            "database_type": "PostgreSQL"
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de desempenho"""
        with self._metrics_lock:
            times = list(self.metrics.processing_times)
            avg_time = sum(times) / len(times) if times else 0

            success_rate = (
                self.metrics.successful_recognitions / self.metrics.total_attempts
                if self.metrics.total_attempts > 0 else 0
            )

            return SystemMetrics(
                total_attempts=self.metrics.total_attempts,
                successful_recognitions=self.metrics.successful_recognitions,
                failed_recognitions=self.metrics.failed_recognitions,
                no_face_detected=self.metrics.no_face_detected,
                average_processing_time=round(avg_time, 3),
                success_rate=round(success_rate, 4),
                database_reloads=self.metrics.database_reloads
            ).__dict__

    def reload_database(self) -> Tuple[bool, str]:
        """Recarrega banco de dados manualmente"""
        logger.info("🔄 Recarregamento manual do banco de dados solicitado")
        success = self.load_facial_database()
        if success:
            status = self.get_detailed_database_status()
            message = f"Banco recarregado - {status['user_count']} usuários ({status['professores_count']} professores, {status['alunos_count']} alunos), {status['total_embeddings']} embeddings"
            return True, message
        else:
            return False, "Falha no recarregamento do banco"

    def _setup_database_triggers(self):
        """Configura triggers no PostgreSQL para notificar mudanças"""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
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

                    cursor.execute("""
                        DROP TRIGGER IF EXISTS usuarios_notify_trigger ON usuarios;
                    """)

                    cursor.execute("""
                        CREATE TRIGGER usuarios_notify_trigger
                        AFTER INSERT OR UPDATE OR DELETE ON usuarios
                        FOR EACH ROW EXECUTE FUNCTION notify_usuarios_update();
                    """)

                    conn.commit()
                    logger.info("✅ Triggers do banco de dados configuradas com sucesso")
                    return True

        except Exception as e:
            logger.warning(f"⚠️ Não foi possível configurar triggers: {str(e)}")
            logger.info("💡 Sistema funcionará sem monitoramento em tempo real")
            return False

    def initialize(self) -> bool:
        """Inicializa o serviço com monitoramento em tempo real"""
        logger.info("🔧 Inicializando Serviço de Reconhecimento Facial...")
        logger.info(f"🎯 Usando modelo: {MODEL_CONFIG.MODEL_NAME}")
        logger.info(f"📊 Dimensão do embedding: {MODEL_CONFIG.EMBEDDING_DIMENSION}")
        logger.info("🔄 SISTEMA DE ATUALIZAÇÃO AUTOMÁTICA ATIVADO")

        # 1. Configurar triggers no banco de dados
        trigger_success = self._setup_database_triggers()

        # 2. Carregar banco de dados inicial
        db_success = self.load_facial_database()

        # 3. Iniciar monitoramento em tempo real
        monitor_success = self.database_monitor.start_monitoring()

        if db_success:
            if trigger_success and monitor_success:
                logger.info("🎯 Monitoramento em tempo real do banco: ATIVO - Novos usuários detectados automaticamente")
            else:
                logger.warning("⚠️ Monitoramento em tempo real do banco: LIMITADO")
                logger.info("💡 Recarregamento manual disponível via /api/database/reload")

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