"""
Módulo de Captura Facial Refatorado - VERSÃO LEVE
"""
import logging
import time
import threading
import cv2
import numpy as np
import base64
import os
from datetime import datetime
from typing import Optional, List, Tuple
from deepface import DeepFace

# Módulos compartilhados
from common.config import MODEL_CONFIG, APP_CONFIG
from common.database import db_manager
from common.image_utils import ImageValidator, FaceQualityValidator

logger = logging.getLogger(__name__)

class FaceDetector:
    """Detector facial otimizado com cache"""

    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 0.2
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
            for face in detected_faces:
                if 'facial_area' in face:
                    x = int(face['facial_area']['x'] * frame.shape[1] / 320)
                    y = int(face['facial_area']['y'] * frame.shape[0] / 240)
                    w = int(face['facial_area']['w'] * frame.shape[1] / 320)
                    h = int(face['facial_area']['h'] * frame.shape[0] / 240)

                    # Validar tamanho mínimo
                    if w >= 80 and h >= 80:
                        faces.append((x, y, w, h))

            self.cached_faces = faces
            self.last_detection_time = current_time
            return faces

        except Exception as e:
            logger.warning(f"Erro na detecção: {e}")
            return []

class FluidFaceCapture:
    """Capturador facial fluido - VERSÃO LEVE"""

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

    def _select_best_faces(self, faces: List[np.ndarray], max_faces: int = 4) -> List[np.ndarray]:
        """Seleciona as melhores faces baseado em nitidez"""
        face_scores = []

        for face in faces:
            score = ImageValidator.calculate_sharpness(face)
            face_scores.append((face, score))

        # Ordena por nitidez e pega as melhores
        face_scores.sort(key=lambda x: x[1], reverse=True)
        return [face for face, score in face_scores[:max_faces]]

    def _check_embedding_consistency(self, embeddings: List[np.ndarray]) -> bool:
        """Verifica se os embeddings são consistentes entre si"""
        if len(embeddings) < 2:
            return True

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j])
                similarities.append(similarity)

        avg_similarity = np.mean(similarities)
        logger.info(f"📊 Similaridade média entre embeddings: {avg_similarity:.4f}")

        return avg_similarity > 0.85

    def _get_best_profile_image(self, faces: List[np.ndarray]) -> bytes:
        """Seleciona a melhor imagem para foto de perfil"""
        best_score = -1
        best_face = faces[0] if faces else None

        for face in faces:
            score = ImageValidator.calculate_sharpness(face)
            if score > best_score:
                best_score = score
                best_face = face

        if best_face is not None:
            _, buffer = cv2.imencode('.jpg', best_face)
            return buffer.tobytes()
        return faces[0].tobytes() if faces else b''

    def generate_embeddings(self) -> Tuple[bool, object]:
        """Gera embeddings - MANTÉM ESTRUTURA ORIGINAL DE RESPOSTA"""
        try:
            embeddings = []

            # ✅ SELECIONA APENAS AS MELHORES 4 FOTOS
            best_faces = self._select_best_faces(self.captured_faces, max_faces=4)

            for i, face_img in enumerate(best_faces):
                try:
                    # ✅ Gera embedding DIRETO
                    embedding_obj = DeepFace.represent(
                        img_path=face_img,
                        model_name=MODEL_CONFIG.MODEL_NAME,
                        detector_backend="skip",
                        enforce_detection=False,
                        align=True
                    )

                    if embedding_obj:
                        embedding = np.array(embedding_obj[0]["embedding"], dtype=np.float32)

                        # ✅ CORREÇÃO: Aceita 2622 dimensões
                        if len(embedding) == 2622:
                            embedding_norm = np.linalg.norm(embedding)
                            if embedding_norm > 0:
                                embeddings.append(embedding / embedding_norm)
                                logger.info(f"✅ Embedding {i + 1} gerado: {len(embedding)} dimensões")
                        else:
                            logger.warning(f"⚠️ Dimensão inesperada: {len(embedding)}")

                except Exception as e:
                    logger.warning(f"Embedding {i + 1} falhou: {e}")
                    continue

            # ✅ VERIFICA CONSISTÊNCIA (MANTIDO)
            if len(embeddings) >= 3:
                is_consistent = self._check_embedding_consistency(embeddings)
                if not is_consistent:
                    return False, "Fotos muito diferentes - tente novamente com iluminação consistente"

            if len(embeddings) >= 3:
                success, result = db_manager.save_user(
                    nome=self.nome,
                    sobrenome=self.sobrenome,
                    turma=self.turma,
                    tipo_usuario=self.tipo_usuario,
                    embeddings=[emb.tolist() for emb in embeddings],
                    profile_image=self._get_best_profile_image(best_faces)
                )

                # ✅ ESTRUTURA ORIGINAL RESTAURADA - Retorna ID como string
                if success:
                    user_id = result['id']  # result já é o dicionário com id
                    logger.info(f"✅ Usuário salvo com ID: {user_id}")
                    return True, str(user_id)  # ✅ RETORNA COMO STRING (igual antes)
                else:
                    return False, result  # result já é a string de erro

            else:
                return False, f"Embeddings insuficientes: {len(embeddings)}/3"

        except Exception as e:
            logger.error(f"Erro na geração de embeddings: {str(e)}")
            return False, f"Erro: {str(e)}"
    def _setup_camera(self):
        """Configura a câmera com fallbacks"""
        for camera_index in [0, 1, -1]:
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"✅ Câmera {camera_index} funcionando")
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return cap
                    cap.release()
            except:
                continue
        return None

    def capture(self) -> Tuple[bool, object]:
        """Método principal de captura"""
        self.running = True
        self.captured_faces = []
        self.captured_count = 0

        cap = self._setup_camera()
        if not cap:
            return False, "Não foi possível acessar a câmera"

        try:
            start_time = time.time()

            while (self.running and
                   self.captured_count < APP_CONFIG.MIN_PHOTOS_REQUIRED and
                   (time.time() - start_time) < 120):

                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                # Detectar rostos
                faces = self.detector.detect_faces(frame)

                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    cropped_face = frame[y:y + h, x:x + w]

                    # Verificar se deve capturar
                    current_time = time.time()
                    if (current_time - self.last_face_time) > self.face_capture_interval:
                        self.captured_faces.append(cropped_face.copy())
                        self.captured_count += 1
                        self.last_face_time = current_time

                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        self.update_progress(f"Capturado: {self.captured_count}/{APP_CONFIG.MIN_PHOTOS_REQUIRED}")

                # Informações na tela
                cv2.putText(display_frame, f"Capturadas: {self.captured_count}/{APP_CONFIG.MIN_PHOTOS_REQUIRED}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                self.send_frame(display_frame)
                time.sleep(0.03)

            # Finalizar captura
            if self.captured_count >= APP_CONFIG.MIN_PHOTOS_REQUIRED:
                self.update_progress("Processando embeddings...")
                success, message = self.generate_embeddings()
                return success, message
            else:
                return False, f"Captura incompleta: {self.captured_count}/{APP_CONFIG.MIN_PHOTOS_REQUIRED}"

        except Exception as e:
            logger.error(f"Erro na captura: {str(e)}")
            return False, f"Erro na captura: {str(e)}"
        finally:
            if cap and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        """Para a captura"""
        self.running = False