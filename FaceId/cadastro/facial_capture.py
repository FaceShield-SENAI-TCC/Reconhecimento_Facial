"""
Módulo de Captura Facial Refatorado - VERSÃO CORRIGIDA
"""
import logging
import time
import cv2
import numpy as np
import base64
import os
from datetime import datetime
from deepface import DeepFace

from common.config import MODEL_CONFIG, APP_CONFIG
from common.database import db_manager
from common.image_utils import ImageValidator, FaceQualityValidator
from common.exceptions import DatabaseError

from face_detector import FaceDetector

logger = logging.getLogger(__name__)

class FluidFaceCapture:
    """Capturador facial fluido com reutilização de recursos"""

    def __init__(self, nome: str, sobrenome: str, turma: str, tipo_usuario: str,
                 progress_callback=None, frame_callback=None):
        self.nome = nome
        self.sobrenome = sobrenome
        self.turma = turma
        self.tipo_usuario = tipo_usuario.upper()

        self.progress_callback = progress_callback
        self.frame_callback = frame_callback

        self.captured_faces = []
        self.captured_count = 0
        self.running = False

        self.detector = FaceDetector()
        self.last_face_time = 0
        self.face_capture_interval = 0.5
        self.last_face_detected_time = 0
        self.consecutive_no_face_count = 0

        self.model_name = MODEL_CONFIG.MODEL_NAME
        self.embedding_dimension = MODEL_CONFIG.EMBEDDING_DIMENSION
        self.quality_validator = FaceQualityValidator()

    def update_progress(self, message=None):
        """Atualiza progresso via callback"""
        if self.progress_callback:
            self.progress_callback({
                "captured": self.captured_count,
                "total": APP_CONFIG.MIN_PHOTOS_REQUIRED,
                "message": message
            })

    def send_frame(self, frame):
        """Envia frame para o cliente via callback"""
        if self.frame_callback:
            try:
                small_frame = cv2.resize(frame, (426, 320))
                _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                self.frame_callback(jpg_as_text)
            except Exception as e:
                logger.error(f"Erro ao enviar frame: {e}")

    def generate_embeddings(self):
        """Gera embeddings das faces capturadas usando modelo configurado"""
        try:
            # VERIFICAÇÃO CRÍTICA: garantir que temos rostos válidos
            if not self.captured_faces or len(self.captured_faces) == 0:
                return False, "Nenhuma face capturada válida"

            embeddings = []
            successful = 0

            for i, face_img in enumerate(self.captured_faces):
                try:
                    # Validar qualidade da imagem do rosto
                    is_valid, validation_msg = self.quality_validator.validate_face_image(face_img)
                    if not is_valid:
                        logger.debug(f"Face {i + 1} inválida: {validation_msg}")
                        continue

                    # Verificar se a imagem do rosto não está vazia
                    if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                        logger.debug(f"Face {i + 1} vazia ou inválida")
                        continue

                    # Garantir que estamos usando apenas o rosto recortado
                    temp_path = f"temp_face_{int(time.time())}_{i}.jpg"

                    # Redimensionar rosto para tamanho mínimo adequado
                    # Mínimo recomendado: 160x160 pixels para modelos faciais
                    min_size = 160
                    h, w = face_img.shape[:2]

                    # Se o rosto for muito pequeno, redimensionar mantendo proporção
                    if h < min_size or w < min_size:
                        scale = min_size / min(h, w)
                        new_h = int(h * scale)
                        new_w = int(w * scale)
                        face_img_resized = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        face_img_resized = face_img

                    # Salvar APENAS o rosto recortado e redimensionado
                    cv2.imwrite(temp_path, face_img_resized)

                    # Processar APENAS o rosto (sem nova detecção)
                    embedding_obj = DeepFace.represent(
                        img_path=temp_path,
                        model_name=self.model_name,
                        enforce_detection=False,  # IMPORTANTE: False porque já temos apenas o rosto
                        detector_backend="skip"   # IMPORTANTE: "skip" para não tentar detectar novamente
                    )

                    embedding = np.array(embedding_obj[0]["embedding"])

                    if len(embedding) != self.embedding_dimension:
                        logger.warning(f"Embedding com dimensão incorreta: {len(embedding)}")
                        continue

                    embeddings.append(embedding)
                    successful += 1
                    logger.debug(f"Embedding {i + 1} gerado com sucesso (tamanho: {face_img_resized.shape})")

                    # Limpar arquivo temporário
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    logger.debug(f"Erro no embedding {i + 1}: {e}")
                    continue

            if successful >= APP_CONFIG.MIN_PHOTOS_REQUIRED:
                # Selecionar melhor imagem para foto de perfil
                best_face_idx = self._select_best_profile_image()
                profile_image = self.captured_faces[best_face_idx]

                # Converter imagem de perfil para bytes
                _, buffer = cv2.imencode('.jpg', profile_image)
                image_bytes = buffer.tobytes()

                # Salvar usuário no banco de dados
                success, result = db_manager.save_user(
                    nome=self.nome,
                    sobrenome=self.sobrenome,
                    turma=self.turma,
                    tipo_usuario=self.tipo_usuario,
                    embeddings=[emb.tolist() for emb in embeddings],
                    profile_image=image_bytes
                )

                if success:
                    user_id = result
                    logger.info(f"Usuário salvo: ID {user_id}")
                    return True, user_id
                else:
                    logger.error(f"Falha ao salvar usuário: {result}")
                    return False, result

            else:
                return False, f"Embeddings insuficientes: {successful}/{APP_CONFIG.MIN_PHOTOS_REQUIRED}"

        except DatabaseError as e:
            logger.error(f"Erro ao salvar no banco: {str(e)}")
            return False, f"Erro ao salvar: {str(e)}"
        except Exception as e:
            logger.error(f"Erro crítico na geração de embeddings: {str(e)}")
            return False, f"Erro crítico: {str(e)}"

    def _select_best_profile_image(self):
        """Seleciona a melhor imagem para foto de perfil baseado na nitidez"""
        best_score = -1
        best_idx = 0

        for i, face_img in enumerate(self.captured_faces):
            score = ImageValidator.calculate_sharpness(face_img)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _setup_camera(self):
        """Configura a câmera com fallbacks robustos"""
        max_attempts = 5
        camera_indexes = [0, 1, -1]  # Tenta diferentes índices

        for camera_index in camera_indexes:
            for attempt in range(max_attempts):
                try:
                    # Limpar câmeras anteriores
                    self._cleanup_camera()
                    time.sleep(0.5)

                    cap = cv2.VideoCapture(camera_index)

                    # Configurações alternativas se a câmera não abrir
                    if not cap.isOpened():
                        # Tentar abrir sem parâmetros específicos
                        cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)

                    if cap.isOpened():
                        # Testar leitura básica
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            logger.debug(f"Câmera {camera_index} funcionando na tentativa {attempt + 1}")

                            # Configurações otimizadas
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            cap.set(cv2.CAP_PROP_FPS, 20)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                            return cap
                        else:
                            cap.release()
                    else:
                        logger.debug(f"Câmera {camera_index} não abriu na tentativa {attempt + 1}")

                except Exception as e:
                    logger.debug(f"Erro na câmera {camera_index}, tentativa {attempt + 1}: {e}")
                    if 'cap' in locals() and cap.isOpened():
                        cap.release()
                    time.sleep(1)

        logger.error("Todas as tentativas de câmera falharam")
        return None

    def _cleanup_camera(self):
        """Limpa recursos da câmera para reutilização"""
        try:
            # Tentar liberar todas as câmeras possíveis
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cap.release()
                time.sleep(0.1)
            cv2.destroyAllWindows()
        except Exception as e:
            logger.debug(f"Erro durante cleanup: {e}")

    def capture(self):
        """Método principal de captura - reutilizável"""
        self.running = True
        self.captured_faces = []
        self.captured_count = 0
        self.last_face_detected_time = time.time()
        self.consecutive_no_face_count = 0

        cap = None
        try:
            cap = self._setup_camera()
            if not cap:
                return False, "Não foi possível acessar a câmera"

            self.update_progress("Preparando câmera...")
            start_time = time.time()
            no_face_timeout = 10

            while (self.running and
                   self.captured_count < APP_CONFIG.MIN_PHOTOS_REQUIRED and
                   (time.time() - start_time) < 120):  # Timeout de 2 minutos

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                # Espelhar frame para visualização
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                # Detectar rostos
                faces = self.detector.detect_faces(frame)
                face_detected = len(faces) == 1

                if face_detected:
                    self.last_face_detected_time = time.time()
                    self.consecutive_no_face_count = 0

                    x, y, w, h = faces[0]

                    # Verificar tamanho mínimo do rosto
                    if h < 100 or w < 100:  # Mínimo 100x100 pixels
                        logger.debug(f"Rosto muito pequeno: {w}x{h} pixels")
                        continue

                    cropped_face = frame[y:y + h, x:x + w]

                    # VERIFICAÇÃO CRÍTICA: garantir que o recorte não está vazio
                    if cropped_face.size == 0 or cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
                        logger.warning(f"Recorte de rosto vazio ou inválido: {cropped_face.shape}")
                        continue

                    # Verificar se deve capturar
                    current_time = time.time()
                    if (current_time - self.last_face_time) > self.face_capture_interval:
                        is_valid, validation_msg = self.quality_validator.validate_face_image(cropped_face)
                        if is_valid:
                            self.captured_faces.append(cropped_face.copy())
                            self.captured_count += 1
                            self.last_face_time = current_time

                            logger.debug(f"Face {self.captured_count} capturada (tamanho: {cropped_face.shape})")
                            self.update_progress(f"Capturado: {self.captured_count}/{APP_CONFIG.MIN_PHOTOS_REQUIRED}")

                else:
                    # Contagem de frames sem rosto
                    self.consecutive_no_face_count += 1
                    elapsed_no_face = time.time() - self.last_face_detected_time

                    if elapsed_no_face > no_face_timeout:
                        return False, " Nenhum rosto detectado por 10 segundos. Posicione seu rosto na câmera."

                self.send_frame(display_frame)
                time.sleep(0.03)

            # Finalizar captura
            if self.captured_count >= APP_CONFIG.MIN_PHOTOS_REQUIRED:
                self.update_progress("Processando embeddings...")
                logger.info(f"Captura concluída: {self.captured_count} faces")
                success, message = self.generate_embeddings()
                return success, message
            else:
                elapsed_no_face = time.time() - self.last_face_detected_time
                if elapsed_no_face > no_face_timeout:
                    return False, " Nenhum rosto detectado por 10 segundos."
                else:
                    return False, f"Captura incompleta: {self.captured_count}/{APP_CONFIG.MIN_PHOTOS_REQUIRED}"

        except Exception as e:
            logger.error(f"Erro na captura: {str(e)}")
            return False, f"Erro na captura: {str(e)}"
        finally:
            # Sempre liberar recursos da câmera
            if cap and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            # Limpar cache do detector para próxima captura
            self.detector.cached_faces = []
            self.detector.last_detection_time = 0
            logger.debug("Recursos da câmera liberados")

    def stop(self):
        """Para a captura de forma segura"""
        self.running = False
        logger.debug("Captura interrompida")