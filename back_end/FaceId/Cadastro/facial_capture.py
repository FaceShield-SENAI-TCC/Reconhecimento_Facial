import os
import cv2
import time
import logging
import numpy as np
import base64
import re
import psycopg2
from psycopg2.extras import Json
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ====================== CONFIGURAÇÕES ======================
DB_CONFIG = {
    "dbname": "faceshild",
    "user": "postgres",
    "password": "root",
    "host": "localhost",
    "port": "5432"
}

MIN_PHOTOS_REQUIRED = 8
MIN_FACE_SIZE = (80, 80)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_SHARPNESS = 50
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220
FACE_CONFIDENCE_THRESHOLD = 0.6


@contextmanager
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def init_face_database():
    """Inicializa tabela de usuários no PostgreSQL"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS usuarios(
                    id SERIAL PRIMARY KEY,
                    nome VARCHAR(100) NOT NULL,
                    sobrenome VARCHAR(100) NOT NULL,
                    turma VARCHAR(50) NOT NULL,
                    tipo VARCHAR(20) DEFAULT 'aluno',
                    embeddings JSONB NOT NULL,
                    foto_perfil BYTEA,
                    data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(nome, sobrenome, turma)
                )
            """)
            conn.commit()
            logger.info("✅ Banco facial inicializado")
            return True
    except Exception as e:
        logger.error(f"❌ Erro no banco facial: {e}")
        return False


def check_user_exists(nome, sobrenome, turma):
    """Verifica se usuário já existe"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM usuarios WHERE nome = %s AND sobrenome = %s AND turma = %s",
                (nome, sobrenome, turma)
            )
            count = cur.fetchone()[0]
            return count
    except Exception as e:
        logger.error(f"Erro ao verificar usuário: {e}")
        return 0


# ====================== DETECTOR FACIAL SIMPLIFICADO ======================
class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_detection_time = 0
        self.detection_interval = 0.4
        self.cached_faces = []

    def detect_faces(self, frame):
        """Detecta rostos usando OpenCV Haar Cascades (mais leve)"""
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.cached_faces

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            self.cached_faces = [(x, y, w, h, 0.8) for (x, y, w, h) in faces]  # confidence fixo
            self.last_detection_time = current_time
            return self.cached_faces

        except Exception as e:
            logger.warning(f"Erro na detecção: {e}")
            return []


# ====================== CAPTURA PRINCIPAL (SEM DEEPFACE) ======================
class FluidFaceCapture:
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
        self.face_capture_interval = 0.8
        self.last_face_detected_time = 0
        self.consecutive_no_face_count = 0

    def update_progress(self, message=None):
        if self.progress_callback:
            self.progress_callback({
                "captured": self.captured_count,
                "total": MIN_PHOTOS_REQUIRED,
                "message": message
            })

    def send_frame(self, frame):
        if self.frame_callback:
            try:
                small_frame = cv2.resize(frame, (320, 240))
                _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                self.frame_callback(jpg_as_text)
            except Exception as e:
                logger.error(f"Erro ao enviar frame: {e}")

    def calculate_sharpness(self, image):
        """Calcula nitidez da imagem"""
        if image is None or image.size == 0:
            return 0
        try:
            small_img = cv2.resize(image, (100, 100))
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 0

    def validate_face_quality(self, face_img):
        """Validação simplificada da qualidade"""
        if face_img is None or face_img.size == 0:
            return False, "Imagem vazia"

        height, width = face_img.shape[:2]
        if height < MIN_FACE_SIZE[0] or width < MIN_FACE_SIZE[1]:
            return False, "Rosto muito pequeno"

        sharpness = self.calculate_sharpness(face_img)
        if sharpness < MIN_SHARPNESS:
            return False, f"Imagem um pouco borrada: {sharpness:.1f}"

        return True, f"Qualidade aceitável: Sharp={sharpness:.1f}"

    def save_to_database(self, profile_image):
        """Salva apenas a foto de perfil (sem embeddings por enquanto)"""
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()

                # Converter imagem para bytes
                _, buffer = cv2.imencode('.jpg', profile_image)
                image_bytes = buffer.tobytes()

                # ✅ SALVAR APENAS FOTO POR ENQUANTO (sem embeddings)
                cur.execute("""
                    INSERT INTO usuarios (nome, sobrenome, turma, tipo, embeddings, foto_perfil)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (nome, sobrenome, turma)
                    DO UPDATE SET foto_perfil = EXCLUDED.foto_perfil,
                                  tipo = EXCLUDED.tipo,
                                  data_cadastro = CURRENT_TIMESTAMP
                """, (self.nome, self.sobrenome, self.turma, self.tipo, Json([]), image_bytes))

                conn.commit()
                return True, f"Usuário salvo com {self.captured_count} fotos"

        except Exception as e:
            logger.error(f"Erro ao salvar no banco: {str(e)}")
            return False, f"Erro ao salvar: {str(e)}"

    def _select_best_profile_image(self):
        """Seleciona a melhor imagem para foto de perfil"""
        best_score = -1
        best_idx = 0

        for i, face_img in enumerate(self.captured_faces):
            score = self.calculate_sharpness(face_img)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _setup_camera(self):
        """Configura a câmera"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.warning(f"Tentativa {attempt + 1}: Câmera não abriu")
                    time.sleep(2)
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 15)

                # Aguardar estabilização
                time.sleep(1)

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
                time.sleep(2)

        logger.error("❌ Não foi possível configurar a câmera")
        return None

    def capture(self):
        """Método principal de captura"""
        self.running = True
        self.captured_faces = []
        self.captured_count = 0
        self.last_face_detected_time = time.time()

        cap = None
        try:
            cap = self._setup_camera()
            if not cap:
                return False, "Não foi possível acessar a câmera"

            self.update_progress("Preparando câmera...")
            start_time = time.time()
            no_face_timeout = 20

            while (self.running and
                   self.captured_count < MIN_PHOTOS_REQUIRED and
                   (time.time() - start_time) < 240):

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.2)
                    continue

                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                faces = self.detector.detect_faces(frame)
                face_detected = len(faces) == 1

                if face_detected:
                    self.last_face_detected_time = time.time()
                    x, y, w, h, confidence = faces[0]
                    cropped_face = frame[y:y + h, x:x + w]

                    current_time = time.time()
                    if (current_time - self.last_face_time) > self.face_capture_interval:
                        is_valid, validation_msg = self.validate_face_quality(cropped_face)

                        if is_valid:
                            self.captured_faces.append(cropped_face.copy())
                            self.captured_count += 1
                            self.last_face_time = current_time

                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                            cv2.putText(display_frame, f"CAPTURADO: {self.captured_count}/{MIN_PHOTOS_REQUIRED}",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            self.update_progress(f"Capturado: {self.captured_count}/{MIN_PHOTOS_REQUIRED}")
                            logger.info(f"📸 Face {self.captured_count} capturada")

                    else:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                else:
                    elapsed_no_face = time.time() - self.last_face_detected_time
                    if elapsed_no_face > no_face_timeout:
                        return False, "❌ Nenhum rosto detectado por 20 segundos"

                cv2.putText(display_frame, f"Capturadas: {self.captured_count}/{MIN_PHOTOS_REQUIRED}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Usuário: {self.nome} {self.sobrenome}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                self.send_frame(display_frame)
                time.sleep(0.05)

            if self.captured_count >= MIN_PHOTOS_REQUIRED:
                self.update_progress("Salvando no banco...")
                best_face_idx = self._select_best_profile_image()
                profile_image = self.captured_faces[best_face_idx]
                success, message = self.save_to_database(profile_image)
                return success, message
            else:
                return False, f"Captura incompleta: {self.captured_count}/{MIN_PHOTOS_REQUIRED}"

        except Exception as e:
            logger.error(f"Erro na captura: {str(e)}")
            return False, f"Erro na captura: {str(e)}"
        finally:
            if cap and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            logger.info("✅ Captura finalizada")

    def stop(self):
        self.running = False
        logger.info("⏹️ Captura interrompida")