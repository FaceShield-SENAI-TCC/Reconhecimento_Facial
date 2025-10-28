"""
Módulo de Captura Facial Refatorado
Usa utilitários compartilhados e configuração centralizada
"""
import logging
import time
import threading
import cv2
import numpy as np
import base64
import os
from datetime import datetime
from deepface import DeepFace

# Módulos compartilhados
from common.config import MODEL_CONFIG, APP_CONFIG
from common.database import db_manager
from common.image_utils import ImageValidator, FaceQualityValidator
from common.exceptions import DatabaseError, ImageValidationError, FaceDetectionError

logger = logging.getLogger(__name__)

class FaceDetector:
    """Detector facial otimizado com cache"""

    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 0.2  # 5 FPS para detecção
        self.cached_faces = []

    def detect_faces(self, frame):
        """Detecta rostos no frame com otimização"""
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.cached_faces

        try:
            small_frame = cv2.resize(frame, (320, 240))
            detected_faces = DeepFace.extract_faces(
                img_path=small_frame,
                detector_backend="opencv",
                enforce_detection=False,
                align=False
            )

            faces = []
            quality_validator = FaceQualityValidator()

            for face in detected_faces:
                if 'facial_area' in face:
                    x = int(face['facial_area']['x'] * frame.shape[1] / 320)
                    y = int(face['facial_area']['y'] * frame.shape[0] / 240)
                    w = int(face['facial_area']['w'] * frame.shape[1] / 320)
                    h = int(face['facial_area']['h'] * frame.shape[0] / 240)

                    # Validar qualidade do rosto
                    face_roi = frame[y:y + h, x:x + w]
                    is_valid, validation_msg = quality_validator.validate_face_image(face_roi)

                    if is_valid:
                        faces.append((x, y, w, h))

            self.cached_faces = faces
            self.last_detection_time = current_time
            return faces

        except Exception as e:
            logger.warning(f"Erro na detecção: {e}")
            return []

class FluidFaceCapture:
    """Capturador facial fluido com reutilização de recursos"""

    def __init__(self, nome, sobrenome, turma, tipo, progress_callback=None, frame_callback=None):
        self.nome = nome
        self.sobrenome = sobrenome
        self.turma = turma
        self.tipo = tipo
        self.progress_callback = progress_callback
        self.frame_callback = frame_callback

        self.captured_faces = []
        self.captured_count = 0
        self.running = False

        self.detector = FaceDetector()
        self.last_face_time = 0
        self.face_capture_interval = 0.5  # 2 faces por segundo máximo
        self.last_face_detected_time = 0
        self.consecutive_no_face_count = 0

        # Configuração do modelo
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
                # Reduzir qualidade para transmissão
                small_frame = cv2.resize(frame, (426, 320))
                _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                self.frame_callback(jpg_as_text)
            except Exception as e:
                logger.error(f"Erro ao enviar frame: {e}")

    def generate_embeddings(self):
        """Gera embeddings das faces capturadas usando modelo configurado"""
        try:
            embeddings = []
            successful = 0

            for i, face_img in enumerate(self.captured_faces):
                try:
                    # Validar qualidade da face
                    is_valid, validation_msg = self.quality_validator.validate_face_image(face_img)
                    if not is_valid:
                        logger.warning(f"Face {i + 1} inválida: {validation_msg}")
                        continue

                    # Salvar temporariamente para DeepFace
                    temp_path = f"temp_face_{int(time.time())}_{i}.jpg"
                    cv2.imwrite(temp_path, face_img)

                    # Gerar embedding usando modelo configurado
                    embedding_obj = DeepFace.represent(
                        img_path=temp_path,
                        model_name=self.model_name,
                        enforce_detection=False,
                        detector_backend="skip"
                    )

                    embedding = np.array(embedding_obj[0]["embedding"])

                    # Validar dimensão do embedding
                    if len(embedding) != self.embedding_dimension:
                        logger.warning(f"Embedding com dimensão incorreta: {len(embedding)}")
                        continue

                    embeddings.append(embedding)
                    successful += 1
                    logger.info(f"✅ Embedding {i + 1} gerado com sucesso")

                    # Limpar arquivo temporário
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    logger.warning(f"Erro no embedding {i + 1}: {e}")
                    continue

            if successful >= APP_CONFIG.MIN_PHOTOS_REQUIRED:
                # Usar a melhor imagem como perfil
                best_face_idx = self._select_best_profile_image()
                profile_image = self.captured_faces[best_face_idx]

                # Converter para bytes
                _, buffer = cv2.imencode('.jpg', profile_image)
                image_bytes = buffer.tobytes()

                # Salvar no banco
                return db_manager.save_user(
                    nome=self.nome,
                    sobrenome=self.sobrenome,
                    turma=self.turma,
                    tipo=self.tipo,
                    embeddings=[emb.tolist() for emb in embeddings],
                    profile_image=image_bytes
                )
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
        """Configura a câmera de forma reutilizável"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Fechar qualquer câmera aberta antes
                self._cleanup_camera()

                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.warning(f"Tentativa {attempt + 1}: Câmera não abriu")
                    time.sleep(1)
                    continue

                # Configurações de performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, APP_CONFIG.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, APP_CONFIG.FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Testar se a câmera funciona
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    logger.info(f"✅ Câmera configurada na tentativa {attempt + 1}")
                    return cap
                else:
                    cap.release()

            except Exception as e:
                logger.warning(f"Erro na tentativa {attempt + 1}: {e}")
                if 'cap' in locals():
                    cap.release()
                time.sleep(1)

        logger.error("❌ Não foi possível configurar a câmera após várias tentativas")
        return None

    def _cleanup_camera(self):
        """Limpa recursos da câmera para reutilização"""
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
            time.sleep(0.5)  # Dar tempo para o sistema liberar
        except:
            pass

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
                    logger.warning("Frame vazio da câmera")
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
                    cropped_face = frame[y:y + h, x:x + w]

                    # Verificar se deve capturar
                    current_time = time.time()
                    if (current_time - self.last_face_time) > self.face_capture_interval:
                        is_valid, validation_msg = self.quality_validator.validate_face_image(cropped_face)
                        if is_valid:
                            self.captured_faces.append(cropped_face.copy())
                            self.captured_count += 1
                            self.last_face_time = current_time

                            # Feedback visual
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                            cv2.putText(display_frame, f"CAPTURADO: {self.captured_count}/{APP_CONFIG.MIN_PHOTOS_REQUIRED}",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            self.update_progress(f"Capturado: {self.captured_count}/{APP_CONFIG.MIN_PHOTOS_REQUIRED}")
                            logger.info(f"📸 Face {self.captured_count} capturada com sucesso")
                        else:
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(display_frame, "QUALIDADE BAIXA",
                                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    else:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                        cv2.putText(display_frame, "AGUARDANDO...",
                                    (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                else:
                    # Contagem de frames sem rosto
                    self.consecutive_no_face_count += 1
                    elapsed_no_face = time.time() - self.last_face_detected_time

                    if elapsed_no_face > no_face_timeout:
                        return False, "❌ Nenhum rosto detectado por 10 segundos. Posicione seu rosto na câmera."

                    # Feedback visual
                    if elapsed_no_face > 3:
                        remaining_time = no_face_timeout - elapsed_no_face
                        if remaining_time <= 5:
                            cv2.putText(display_frame, f"PROCURE A CÂMERA! {int(remaining_time)}s",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(display_frame, "PROCURANDO ROSTO...",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # Informações na tela
                cv2.putText(display_frame, f"Capturadas: {self.captured_count}/{APP_CONFIG.MIN_PHOTOS_REQUIRED}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Usuário: {self.nome} {self.sobrenome}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                self.send_frame(display_frame)
                time.sleep(0.03)

            # Finalizar captura
            if self.captured_count >= APP_CONFIG.MIN_PHOTOS_REQUIRED:
                self.update_progress("Processando embeddings...")
                success, message = self.generate_embeddings()
                return success, message
            else:
                elapsed_no_face = time.time() - self.last_face_detected_time
                if elapsed_no_face > no_face_timeout:
                    return False, "❌ Nenhum rosto detectado por 10 segundos."
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
            logger.info("✅ Recursos da câmera liberados para próxima captura")

    def stop(self):
        """Para a captura de forma segura"""
        self.running = False
        logger.info("⏹️ Captura interrompida")