"""
Lógica de IA para Reconhecimento Facial
Contém a classe FaceRecognizer (núcleo da IA: DeepFace)
"""
import logging
import time
import threading
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import Counter, deque
import psycopg2
import psycopg2.extensions

from deepface import DeepFace
from common.config import DATABASE_CONFIG, MODEL_CONFIG
from common.database import DatabaseManager
from common.exceptions import DatabaseError, DatabaseConnectionError

logger = logging.getLogger(__name__)

class DatabaseMonitor:
    """Monitora o banco de dados PostgreSQL em tempo real"""
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
            logger.info("Monitoramento do banco de dados iniciado")
            return True
        except Exception as e:
            logger.error(f"Falha ao iniciar monitoramento do banco: {str(e)}")
            return False

    def stop_monitoring(self):
        """Para o monitoramento"""
        self.running = False
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
        logger.info("Monitoramento do banco de dados parado")

    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self.running:
            try:
                self.connection = psycopg2.connect(
                    **DATABASE_CONFIG.to_dict(),
                    connect_timeout=10
                )
                self.connection.set_isolation_level(
                    psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT
                )

                cursor = self.connection.cursor()
                cursor.execute("LISTEN usuarios_update;")
                logger.info("Escutando por mudanças no banco de dados...")

                while self.running and self.connection and not self.connection.closed:
                    try:
                        self.connection.poll()
                        while self.connection.notifies:
                            notify = self.connection.notifies.pop(0)
                            logger.info(f"Mudança no banco detectada: {notify.payload}")

                            if self.database_reload_callback:
                                logger.info("Recarregando banco de dados devido a mudanças...")
                                self.database_reload_callback()

                        time.sleep(1)

                    except psycopg2.InterfaceError as e:
                        logger.warning(f"Conexão com banco interrompida: {str(e)}")
                        break
                    except Exception as e:
                        logger.error(f"Erro no loop interno de monitoramento: {str(e)}")
                        break

            except psycopg2.OperationalError as e:
                logger.warning(f"Conexão com banco falhou, tentando novamente em {self.reconnect_delay}s: {str(e)}")
            except Exception as e:
                logger.error(f"Erro no loop de monitoramento: {str(e)}")

            if self.running:
                time.sleep(self.reconnect_delay)

class FaceRecognizer:
    """
    Núcleo da IA para reconhecimento facial
    Usa DeepFace para extração e comparação de embeddings
    """

    def __init__(self):
        self.facial_database = {}
        self.last_update = None
        self.metrics_lock = threading.RLock()
        self.db_manager = DatabaseManager()
        self.database_monitor = DatabaseMonitor(self.load_facial_database)

    def load_facial_database(self) -> bool:
        """Carrega embeddings faciais do PostgreSQL incluindo ID"""
        logger.info("Carregando banco de dados facial do PostgreSQL...")

        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, nome, sobrenome, turma, tipo_usuario, username, embeddings
                        FROM usuarios
                        WHERE embeddings IS NOT NULL AND jsonb_array_length(embeddings) > 0
                    """)

                    database = {}
                    user_count = 0
                    embedding_count = 0

                    rows = cursor.fetchall()
                    logger.info(f"Encontrados {len(rows)} usuários no banco de dados")

                    for id, nome, sobrenome, turma, tipo_usuario, username, embeddings in rows:
                        # IMPORTANTE: Converter tipo_usuario para maiúsculas
                        tipo_usuario_normalized = tipo_usuario.upper() if tipo_usuario else "ALUNO"

                        # Formato do display name
                        if tipo_usuario_normalized == "PROFESSOR" and username:
                            display_name = f"{nome} {sobrenome} (@{username})"
                        else:
                            display_name = f"{nome} {sobrenome}"

                        user_info = {
                            'id': id,
                            'display_name': display_name,
                            'nome': nome,
                            'sobrenome': sobrenome,
                            'turma': turma,
                            'tipo_usuario': tipo_usuario_normalized,  # Garantido em maiúsculas
                            'username': username,
                            'is_professor': tipo_usuario_normalized == "PROFESSOR"
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

                        if valid_embeddings:
                            database[display_name] = {
                                'embeddings': valid_embeddings,
                                'info': user_info
                            }
                            user_count += 1
                            embedding_count += len(valid_embeddings)
                            logger.debug(f"Carregado: {display_name} - Tipo: {tipo_usuario_normalized}")

            old_user_count = len(self.facial_database)
            self.facial_database = database
            self.last_update = time.time()

            professores_count = sum(1 for user_data in database.values()
                                  if user_data['info']['tipo_usuario'] == "PROFESSOR")
            alunos_count = user_count - professores_count

            logger.info(f"Banco carregado: {user_count} usuários ({professores_count} professores, {alunos_count} alunos), {embedding_count} embeddings")

            if user_count > old_user_count:
                logger.info(f"NOVOS USUÁRIOS DETECTADOS! Antes: {old_user_count}, Agora: {user_count}")
            elif user_count < old_user_count:
                logger.warning(f"Usuários removidos: Antes: {old_user_count}, Agora: {user_count}")

            return True

        except DatabaseConnectionError as e:
            logger.error(f"Falha de conexão ao carregar banco: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Falha ao carregar banco de dados: {str(e)}")
            return False

    def _extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extrai embedding facial da imagem usando DeepFace"""
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

    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], Optional[float], Optional[Dict]]:
        """
        Reconhecimento facial usando embeddings
        """
        try:
            captured_embedding = self._extract_face_embedding(face_image)
            if captured_embedding is None:
                logger.warning("Não foi possível extrair embedding")
                return None, None, None

            best_match = None
            min_distance = float('inf')
            second_best_distance = float('inf')
            best_user_data = None

            # Busca no banco de dados
            for user_key, user_data in self.facial_database.items():
                for db_embedding in user_data['embeddings']:
                    distance = 1 - np.dot(captured_embedding, db_embedding)

                    if distance < min_distance:
                        second_best_distance = min_distance
                        min_distance = distance
                        best_match = user_key
                        best_user_data = user_data['info']
                    elif distance < second_best_distance:
                        second_best_distance = distance

            # Parâmetros ajustados
            if best_match and min_distance <= 0.65:
                margin = second_best_distance - min_distance if second_best_distance != float('inf') else 0
                confidence = 1 - min_distance
                logger.info(f"RECONHECIDO: {best_match} - Tipo: {best_user_data['tipo_usuario']}, Dist: {min_distance:.4f}, Conf: {confidence:.4f}")
                return best_match, min_distance, best_user_data
            else:
                if best_match:
                    logger.warning(f"REJEITADO: {best_match} - Dist: {min_distance:.4f} > 0.65, Tipo: {best_user_data['tipo_usuario']}")
                else:
                    logger.info("Nenhum match encontrado")
                return None, None, None

        except Exception as e:
            logger.error(f"Falha no reconhecimento facial: {str(e)}")
            return None, None, None

    def _detect_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detecta rostos na imagem usando DeepFace"""
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

                if w >= 40 and h >= 40:
                    return face_data

            logger.warning("Nenhum rosto detectado ou rosto muito pequeno")
            return None

        except Exception as e:
            logger.error(f"Falha na detecção facial: {str(e)}")
            return None

    def get_database_status(self) -> Dict[str, Any]:
        """Retorna status do banco de dados"""
        user_count = len(self.facial_database)
        total_embeddings = sum(
            len(user_data['embeddings'])
            for user_data in self.facial_database.values()
        )

        professores_count = sum(1 for user_data in self.facial_database.values()
                              if user_data['info']['tipo_usuario'] == "PROFESSOR")
        alunos_count = user_count - professores_count

        return {
            "status": "loaded" if self.facial_database else "empty",
            "user_count": user_count,
            "total_embeddings": total_embeddings,
            "professores_count": professores_count,
            "alunos_count": alunos_count,
            "last_update": self.last_update,
            "monitoring_active": self.database_monitor.running,
            "database_type": "PostgreSQL"
        }

    def get_detailed_database_status(self) -> Dict[str, Any]:
        """Retorna status detalhado do banco de dados"""
        status = self.get_database_status()

        professores_com_username = sum(1 for user_data in self.facial_database.values()
                                     if user_data['info']['tipo_usuario'] == "PROFESSOR"
                                     and user_data['info']['username'] is not None)

        status["professores_com_username"] = professores_com_username
        return status

    def reload_database(self) -> Tuple[bool, str]:
        """Recarrega banco de dados manualmente"""
        logger.info("Recarregamento manual do banco de dados solicitado")
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

                    cursor.execute("DROP TRIGGER IF EXISTS usuarios_notify_trigger ON usuarios;")
                    cursor.execute("""
                        CREATE TRIGGER usuarios_notify_trigger
                        AFTER INSERT OR UPDATE OR DELETE ON usuarios
                        FOR EACH ROW EXECUTE FUNCTION notify_usuarios_update();
                    """)

                    conn.commit()
                    logger.info("Triggers do banco de dados configuradas com sucesso")
                    return True

        except Exception as e:
            logger.warning(f"Não foi possível configurar triggers: {str(e)}")
            logger.info("Sistema funcionará sem monitoramento em tempo real")
            return False

    def initialize(self) -> bool:
        """Inicializa o serviço com monitoramento em tempo real"""
        logger.info("Inicializando FaceRecognizer...")
        logger.info(f"Usando modelo: {MODEL_CONFIG.MODEL_NAME}")
        logger.info(f"Dimensão do embedding: {MODEL_CONFIG.EMBEDDING_DIMENSION}")
        logger.info(f"Limite de distância: 0.65")

        # Configurar triggers no banco de dados
        trigger_success = self._setup_database_triggers()

        # Carregar banco de dados inicial
        db_success = self.load_facial_database()

        # Iniciar monitoramento em tempo real
        monitor_success = self.database_monitor.start_monitoring()

        if db_success:
            if trigger_success and monitor_success:
                logger.info("Monitoramento em tempo real do banco: ATIVO")
            else:
                logger.warning("Monitoramento em tempo real do banco: LIMITADO")

        return db_success

    def cleanup(self):
        """Limpeza do serviço"""
        if hasattr(self, 'database_monitor'):
            self.database_monitor.stop_monitoring()
        logger.info("FaceRecognizer finalizado")