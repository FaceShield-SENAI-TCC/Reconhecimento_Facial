import cv2
import os
import time
import logging
import numpy as np
import base64
import re
import psycopg2
from psycopg2.extras import Json
from deepface import DeepFace
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Configurações
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "faceshild"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

MIN_PHOTOS_REQUIRED = 10
MIN_FACE_SIZE = (100, 100)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ====================== BANCO DE DADOS ======================
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


# ====================== UTILITÁRIOS ======================
def sanitize_name(name):
    """Cria nome seguro para arquivos"""
    if not name:
        return "unknown"
    name = str(name).lower().strip()
    return re.sub(r'[^a-z0-9_]', '', name.replace(' ', '_')) or "unknown"


def calculate_sharpness(image):
    """Calcula nitidez da imagem"""
    if image is None or image.size == 0:
        return 0
    try:
        small_img = cv2.resize(image, (100, 100))
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        return 0


def validate_face_image(face_img):
    """Valida qualidade da imagem facial de forma mais rigorosa"""
    if face_img is None or face_img.size == 0:
        return False, "Imagem vazia"

    height, width = face_img.shape[:2]
    if height < MIN_FACE_SIZE[0] or width < MIN_FACE_SIZE[1]:
        return False, "Rosto muito pequeno"

    # ✅ VERIFICAÇÃO MAIS RIGOROSA: Proporção do rosto
    aspect_ratio = width / height
    if aspect_ratio < 0.6 or aspect_ratio > 1.4:  # Rostos humanos tem proporção ~0.7-1.3
        return False, f"Proporção inválida: {aspect_ratio:.2f}"

    sharpness = calculate_sharpness(face_img)
    if sharpness < 100:  # ✅ AUMENTADO o limite de nitidez
        return False, f"Imagem muito borrada: {sharpness:.1f}"

    # ✅ NOVA VERIFICAÇÃO: Verificar se há características faciais
    try:
        # Tentar detectar olhos para confirmar que é um rosto
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

        if len(eyes) < 1:  # Se não detectar pelo menos 1 olho
            return False, "Não detectou características faciais"
    except Exception as e:
        logger.warning(f"Erro na detecção de olhos: {e}")

    return True, f"Qualidade: {sharpness:.1f}"


# ====================== DETECTOR FACIAL MELHORADO ======================
class FaceDetector:
    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 0.3  # ✅ AUMENTADO o intervalo para mais precisão
        self.cached_faces = []
        self.consecutive_failures = 0
        self.max_failures = 3

    def detect_faces(self, frame):
        """Detecta rostos no frame com verificações mais rigorosas"""
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.cached_faces

        try:
            # ✅ REDUZIR MAIS a resolução para melhor performance e precisão
            small_frame = cv2.resize(frame, (240, 180))

            detected_faces = DeepFace.extract_faces(
                img_path=small_frame,
                detector_backend="opencv",
                enforce_detection=False,
                align=False
            )

            faces = []
            valid_faces_count = 0

            for face in detected_faces:
                if 'facial_area' in face and face['confidence'] > 0.8:  # ✅ FILTRAR por confiança
                    x = int(face['facial_area']['x'] * frame.shape[1] / 240)
                    y = int(face['facial_area']['y'] * frame.shape[0] / 180)
                    w = int(face['facial_area']['w'] * frame.shape[1] / 240)
                    h = int(face['facial_area']['h'] * frame.shape[0] / 180)

                    # ✅ VALIDAÇÃO MAIS RIGOROSA
                    if (w >= MIN_FACE_SIZE[0] and h >= MIN_FACE_SIZE[1] and
                            w <= 400 and h <= 400):  # ✅ LIMITE MÁXIMO para evitar falsos positivos

                        face_roi = frame[y:y + h, x:x + w]
                        if face_roi.size > 0:
                            # ✅ VERIFICAÇÃO DE QUALIDADE ANTES de considerar como rosto
                            is_valid, validation_msg = validate_face_image(face_roi)
                            if is_valid:
                                faces.append((x, y, w, h))
                                valid_faces_count += 1
                                logger.info(f"✅ Rosto válido detectado: {validation_msg}")
                            else:
                                logger.debug(f"❌ Rosto inválido: {validation_msg}")

            self.cached_faces = faces
            self.last_detection_time = current_time
            self.consecutive_failures = 0

            logger.debug(f"👁️  Rostos detectados: {valid_faces_count}")
            return faces

        except Exception as e:
            self.consecutive_failures += 1
            logger.warning(f"Erro na detecção (tentativa {self.consecutive_failures}): {e}")
            if self.consecutive_failures >= self.max_failures:
                logger.error("Muitas falhas consecutivas na detecção")
                self.consecutive_failures = 0
            return []


# ====================== CAPTURA PRINCIPAL CORRIGIDA ======================
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
        self.face_capture_interval = 0.8  # ✅ AUMENTADO intervalo entre capturas
        self.last_face_detected_time = 0
        self.consecutive_no_face_count = 0

    def update_progress(self, message=None):
        """Atualiza progresso"""
        if self.progress_callback:
            self.progress_callback({
                "captured": self.captured_count,
                "total": MIN_PHOTOS_REQUIRED,
                "message": message
            })

    def send_frame(self, frame):
        """Envia frame para o cliente"""
        if self.frame_callback:
            try:
                small_frame = cv2.resize(frame, (426, 320))
                _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                self.frame_callback(jpg_as_text)
            except Exception as e:
                logger.error(f"Erro ao enviar frame: {e}")

    def save_to_database(self, embeddings, profile_image):
        """Salva usuário no PostgreSQL"""
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()

                _, buffer = cv2.imencode('.jpg', profile_image)
                image_bytes = buffer.tobytes()

                embeddings_list = [embedding.tolist() for embedding in embeddings]

                cur.execute("""
                    INSERT INTO usuarios (nome, sobrenome, turma, tipo, embeddings, foto_perfil)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (nome, sobrenome, turma) 
                    DO UPDATE SET embeddings = EXCLUDED.embeddings, 
                                  foto_perfil = EXCLUDED.foto_perfil,
                                  tipo = EXCLUDED.tipo,
                                  data_cadastro = CURRENT_TIMESTAMP
                """, (self.nome, self.sobrenome, self.turma, self.tipo, Json(embeddings_list), image_bytes))

                conn.commit()
                return True, "Usuário salvo com sucesso"

        except Exception as e:
            return False, f"Erro ao salvar: {str(e)}"

    def generate_embeddings(self):
        """Gera embeddings das faces capturadas"""
        try:
            embeddings = []
            successful = 0

            # ✅ VERIFICAÇÃO FINAL: Garantir que todas as faces são válidas
            valid_faces = []
            for i, face_img in enumerate(self.captured_faces):
                is_valid, validation_msg = validate_face_image(face_img)
                if is_valid:
                    valid_faces.append(face_img)
                    logger.info(f"✅ Face {i + 1} válida para embedding")
                else:
                    logger.warning(f"❌ Face {i + 1} descartada: {validation_msg}")

            if len(valid_faces) < MIN_PHOTOS_REQUIRED:
                return False, f"Faces válidas insuficientes: {len(valid_faces)}/{MIN_PHOTOS_REQUIRED}"

            for i, face_img in enumerate(valid_faces):
                try:
                    temp_path = f"temp_face_{i}.jpg"
                    cv2.imwrite(temp_path, face_img)

                    embedding_obj = DeepFace.represent(
                        img_path=temp_path,
                        model_name="Facenet",
                        enforce_detection=False,
                        detector_backend="skip"
                    )

                    embedding = np.array(embedding_obj[0]["embedding"])
                    embeddings.append(embedding)
                    successful += 1

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    logger.warning(f"Erro no embedding {i}: {e}")
                    continue

            if successful >= MIN_PHOTOS_REQUIRED:
                best_face = max(valid_faces, key=lambda img: calculate_sharpness(img))
                return self.save_to_database(embeddings, best_face)
            else:
                return False, f"Embeddings insuficientes: {successful}/{MIN_PHOTOS_REQUIRED}"

        except Exception as e:
            return False, f"Erro crítico: {str(e)}"

    def capture(self):
        """Método principal de captura com verificações rigorosas"""
        self.running = True
        self.captured_faces = []
        self.captured_count = 0
        self.last_face_detected_time = time.time()
        self.consecutive_no_face_count = 0

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Não foi possível acessar a câmera"

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            self.update_progress("Preparando câmera...")
            start_time = time.time()
            no_face_timeout = 10

            while (self.running and
                   self.captured_count < MIN_PHOTOS_REQUIRED and
                   (time.time() - start_time) < 120):

                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                # ✅ DETECÇÃO COM VERIFICAÇÃO RIGOROSA
                faces = self.detector.detect_faces(frame)
                face_detected = len(faces) == 1

                if face_detected:
                    self.last_face_detected_time = time.time()
                    self.consecutive_no_face_count = 0

                    x, y, w, h = faces[0]
                    cropped_face = frame[y:y + h, x:x + w]

                    # ✅ VERIFICAÇÃO DUPLA antes de capturar
                    current_time = time.time()
                    if (current_time - self.last_face_time) > self.face_capture_interval:
                        is_valid, validation_msg = validate_face_image(cropped_face)
                        if is_valid:
                            self.captured_faces.append(cropped_face.copy())
                            self.captured_count += 1
                            self.last_face_time = current_time

                            # Feedback visual
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                            cv2.putText(display_frame, f"✅ CAPTURADO: {self.captured_count}/{MIN_PHOTOS_REQUIRED}",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            self.update_progress(f"Capturado: {self.captured_count}/{MIN_PHOTOS_REQUIRED}")
                            logger.info(f"📸 Face {self.captured_count} capturada com sucesso")
                        else:
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(display_frame, "❌ QUALIDADE BAIXA",
                                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    else:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                        cv2.putText(display_frame, "⏳ AGUARDANDO...",
                                    (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                else:
                    # ✅ CONTAGEM DE FRAMES SEM ROSTO
                    self.consecutive_no_face_count += 1
                    elapsed_no_face = time.time() - self.last_face_detected_time

                    if elapsed_no_face > no_face_timeout:
                        return False, "❌ Nenhum rosto detectado por 10 segundos. Posicione seu rosto na câmera e tente novamente."

                    # Feedback visual
                    if elapsed_no_face > 3:  # Só mostra alerta após 3 segundos
                        remaining_time = no_face_timeout - elapsed_no_face
                        if remaining_time <= 5:
                            cv2.putText(display_frame, f"🚨 PROCURE A CÂMERA! {int(remaining_time)}s",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(display_frame, "🔍 PROCURANDO ROSTO...",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # Informações na tela
                cv2.putText(display_frame, f"Capturadas: {self.captured_count}/{MIN_PHOTOS_REQUIRED}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Mostrar estatísticas de detecção
                if self.consecutive_no_face_count > 0:
                    cv2.putText(display_frame, f"Sem rosto: {self.consecutive_no_face_count} frames",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                self.send_frame(display_frame)
                time.sleep(0.03)

            # ✅ VERIFICAÇÃO FINAL ANTES DE SALVAR
            if self.captured_count >= MIN_PHOTOS_REQUIRED:
                self.update_progress("Validando imagens...")
                logger.info(f"🔍 Validando {len(self.captured_faces)} faces capturadas...")

                # Verificação final de qualidade
                valid_faces_count = 0
                for i, face in enumerate(self.captured_faces):
                    is_valid, msg = validate_face_image(face)
                    if is_valid:
                        valid_faces_count += 1
                    else:
                        logger.warning(f"Face {i + 1} inválida na validação final: {msg}")

                if valid_faces_count >= MIN_PHOTOS_REQUIRED:
                    self.update_progress("Salvando dados...")
                    success, message = self.generate_embeddings()
                    return success, message
                else:
                    return False, f"❌ Validação final falhou: {valid_faces_count}/{MIN_PHOTOS_REQUIRED} faces válidas"
            else:
                elapsed_no_face = time.time() - self.last_face_detected_time
                if elapsed_no_face > no_face_timeout:
                    return False, "❌ Nenhum rosto detectado por 10 segundos. Posicione seu rosto na câmera e tente novamente."
                else:
                    return False, f"Captura incompleta: {self.captured_count}/{MIN_PHOTOS_REQUIRED}"

        except Exception as e:
            logger.error(f"Erro na captura: {e}")
            return False, f"Erro na captura: {str(e)}"
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("🎬 Captura finalizada")