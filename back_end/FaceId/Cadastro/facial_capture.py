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

# ====================== CONFIGURAÇÕES ATUALIZADAS ======================
DB_CONFIG = {
    "dbname": "faceshild",
    "user": "postgres",
    "password": "root",
    "host": "localhost",
    "port": "5432"
}

# ✅ AUMENTADO DE 3 PARA 8 EMBEDDINGS
MIN_PHOTOS_REQUIRED = 8
MIN_FACE_SIZE = (120, 120)  # ✅ Aumentado tamanho mínimo
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ✅ NOVOS PARÂMETROS DE QUALIDADE
MIN_SHARPNESS = 100  # ✅ Aumentado de 70 para 100
FACE_CONFIDENCE_THRESHOLD = 0.8  # ✅ Nova validação de confiança


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


# ====================== UTILITÁRIOS MELHORADOS ======================
def calculate_sharpness(image):
    """Calcula nitidez da imagem com método melhorado"""
    if image is None or image.size == 0:
        return 0
    try:
        # Reduzir ruído antes do cálculo
        small_img = cv2.resize(image, (100, 100))
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

        # Aplicar filtro Gaussiano para reduzir ruído
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        return 0


def validate_face_quality(face_img):
    """Validação MELHORADA da qualidade da imagem facial"""
    if face_img is None or face_img.size == 0:
        return False, "Imagem vazia"

    height, width = face_img.shape[:2]
    if height < MIN_FACE_SIZE[0] or width < MIN_FACE_SIZE[1]:
        return False, "Rosto muito pequeno"

    sharpness = calculate_sharpness(face_img)
    if sharpness < MIN_SHARPNESS:
        return False, f"Imagem muito borrada: {sharpness:.1f}"

    # ✅ NOVA VALIDAÇÃO: Brilho adequado
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 50 or brightness > 200:
        return False, f"Brilho inadequado: {brightness:.1f}"

    # ✅ NOVA VALIDAÇÃO: Contraste
    contrast = np.std(gray)
    if contrast < 40:
        return False, f"Contraste baixo: {contrast:.1f}"

    return True, f"Qualidade: Sharp={sharpness:.1f}, Bright={brightness:.1f}, Contrast={contrast:.1f}"


# ====================== DETECTOR FACIAL MELHORADO ======================
class FaceDetector:
    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 0.3  # ✅ Reduzido FPS para mais precisão
        self.cached_faces = []

    def detect_faces(self, frame):
        """Detecta rostos no frame com validação MELHORADA"""
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.cached_faces

        try:
            small_frame = cv2.resize(frame, (320, 240))

            # ✅ USAR DETECTOR MAIS ROBUSTO PARA ÓCULOS
            detected_faces = DeepFace.extract_faces(
                img_path=small_frame,
                detector_backend="ssd",  # ✅ Alterado para SSD (melhor com óculos)
                enforce_detection=False,
                align=True  # ✅ Alinhamento para melhor reconhecimento
            )

            faces = []
            for face in detected_faces:
                if 'facial_area' in face and face['confidence'] > FACE_CONFIDENCE_THRESHOLD:
                    x = int(face['facial_area']['x'] * frame.shape[1] / 320)
                    y = int(face['facial_area']['y'] * frame.shape[0] / 240)
                    w = int(face['facial_area']['w'] * frame.shape[1] / 320)
                    h = int(face['facial_area']['h'] * frame.shape[0] / 240)

                    # Validar tamanho mínimo e qualidade
                    face_roi = frame[y:y + h, x:x + w]
                    if (w >= MIN_FACE_SIZE[0] and h >= MIN_FACE_SIZE[1] and
                            face_roi.size > 0):
                        faces.append((x, y, w, h, face['confidence']))

            self.cached_faces = faces
            self.last_detection_time = current_time
            return faces

        except Exception as e:
            logger.warning(f"Erro na detecção: {e}")
            return []


# ====================== CAPTURA PRINCIPAL ATUALIZADA ======================
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
        self.face_capture_interval = 1.0  # ✅ Aumentado intervalo para capturas mais variadas
        self.last_face_detected_time = 0
        self.consecutive_no_face_count = 0

        # ✅ CONFIGURAÇÃO VGG-FACE MANTIDA
        self.model_name = "VGG-Face"
        self.embedding_dimension = 2622

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

                # Converter imagem para bytes
                _, buffer = cv2.imencode('.jpg', profile_image)
                image_bytes = buffer.tobytes()

                # ✅ VALIDAR DIMENSÕES VGG-FACE
                valid_embeddings = []
                for embedding in embeddings:
                    if len(embedding) == self.embedding_dimension:
                        valid_embeddings.append(embedding.tolist())
                    else:
                        logger.warning(
                            f"Embedding com dimensão incorreta: {len(embedding)} (esperado: {self.embedding_dimension})")

                if len(valid_embeddings) < MIN_PHOTOS_REQUIRED:
                    return False, f"Embeddings válidos insuficientes: {len(valid_embeddings)}/{MIN_PHOTOS_REQUIRED}"

                # Inserir ou atualizar
                cur.execute("""
                    INSERT INTO usuarios (nome, sobrenome, turma, tipo, embeddings, foto_perfil)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (nome, sobrenome, turma)
                    DO UPDATE SET embeddings = EXCLUDED.embeddings,
                                  foto_perfil = EXCLUDED.foto_perfil,
                                  tipo = EXCLUDED.tipo,
                                  data_cadastro = CURRENT_TIMESTAMP
                """, (self.nome, self.sobrenome, self.turma, self.tipo, Json(valid_embeddings), image_bytes))

                conn.commit()
                return True, f"Usuário salvo com {len(valid_embeddings)} embeddings"

        except Exception as e:
            logger.error(f"Erro ao salvar no banco: {str(e)}")
            return False, f"Erro ao salvar: {str(e)}"

    def generate_embeddings(self):
        """Gera embeddings das faces capturadas usando VGG-Face"""
        try:
            embeddings = []
            successful = 0
            quality_scores = []

            for i, face_img in enumerate(self.captured_faces):
                try:
                    # ✅ VALIDAÇÃO MAIS RIGOROSA
                    is_valid, validation_msg = validate_face_quality(face_img)
                    if not is_valid:
                        logger.warning(f"Face {i + 1} rejeitada: {validation_msg}")
                        continue

                    # Calcular score de qualidade
                    sharpness = calculate_sharpness(face_img)
                    quality_scores.append((i, sharpness))

                    # Salvar temporariamente para DeepFace
                    temp_path = f"temp_face_{int(time.time())}_{i}.jpg"
                    cv2.imwrite(temp_path, face_img)

                    # ✅ USAR VGG-FACE COM CONFIGURAÇÃO OTIMIZADA
                    embedding_obj = DeepFace.represent(
                        img_path=temp_path,
                        model_name=self.model_name,
                        enforce_detection=False,
                        detector_backend="skip",
                        align=True  # ✅ Alinhamento para melhor precisão
                    )

                    embedding = np.array(embedding_obj[0]["embedding"])

                    # ✅ VALIDAR DIMENSÃO VGG-FACE
                    if len(embedding) != self.embedding_dimension:
                        logger.warning(
                            f"Embedding com dimensão incorreta: {len(embedding)} (esperado: {self.embedding_dimension})")
                        continue

                    embeddings.append(embedding)
                    successful += 1
                    logger.info(f"✅ Embedding {i + 1} gerado - {validation_msg}")

                    # Limpar arquivo temporário
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    logger.warning(f"Erro no embedding {i + 1}: {e}")
                    continue

            # ✅ SELECIONAR AS MELHORES 8 IMAGENS (se tiver mais)
            if successful > MIN_PHOTOS_REQUIRED:
                # Ordenar por qualidade e pegar as melhores
                quality_scores.sort(key=lambda x: x[1], reverse=True)
                best_indices = [idx for idx, _ in quality_scores[:MIN_PHOTOS_REQUIRED]]
                embeddings = [embeddings[i] for i in best_indices]
                successful = MIN_PHOTOS_REQUIRED
                logger.info(f"📊 Selecionadas as {MIN_PHOTOS_REQUIRED} melhores imagens de {successful}")

            if successful >= MIN_PHOTOS_REQUIRED:
                # Usar a melhor imagem como perfil
                best_face_idx = self._select_best_profile_image()
                profile_image = self.captured_faces[best_face_idx]

                return self.save_to_database(embeddings, profile_image)
            else:
                return False, f"Embeddings insuficientes: {successful}/{MIN_PHOTOS_REQUIRED}"

        except Exception as e:
            logger.error(f"Erro crítico na geração de embeddings: {str(e)}")
            return False, f"Erro crítico: {str(e)}"

    def _select_best_profile_image(self):
        """Seleciona a melhor imagem para foto de perfil"""
        best_score = -1
        best_idx = 0

        for i, face_img in enumerate(self.captured_faces):
            score = calculate_sharpness(face_img)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _setup_camera(self):
        """Configura a câmera - AGORA REUTILIZÁVEL"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self._cleanup_camera()

                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.warning(f"Tentativa {attempt + 1}: Câmera não abriu")
                    time.sleep(1)
                    continue

                # Configurações de performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
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
        """Limpa recursos da câmera"""
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
            time.sleep(0.5)
        except:
            pass

    def capture(self):
        """Método principal de captura - ATUALIZADO PARA 8 FOTOS"""
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
            no_face_timeout = 15  # ✅ Aumentado timeout

            while (self.running and
                   self.captured_count < MIN_PHOTOS_REQUIRED and
                   (time.time() - start_time) < 180):  # ✅ Timeout de 3 minutos

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

                    x, y, w, h, confidence = faces[0]
                    cropped_face = frame[y:y + h, x:x + w]

                    # Verificar se deve capturar
                    current_time = time.time()
                    if (current_time - self.last_face_time) > self.face_capture_interval:
                        is_valid, validation_msg = validate_face_quality(cropped_face)
                        if is_valid:
                            self.captured_faces.append(cropped_face.copy())
                            self.captured_count += 1
                            self.last_face_time = current_time

                            # Feedback visual
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                            cv2.putText(display_frame, f"CAPTURADO: {self.captured_count}/{MIN_PHOTOS_REQUIRED}",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            self.update_progress(f"Capturado: {self.captured_count}/{MIN_PHOTOS_REQUIRED}")
                            logger.info(f"📸 Face {self.captured_count} capturada - Confiança: {confidence:.2f}")
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
                        return False, "❌ Nenhum rosto detectado por 15 segundos. Posicione seu rosto na câmera e tente novamente."

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
                cv2.putText(display_frame, f"Capturadas: {self.captured_count}/{MIN_PHOTOS_REQUIRED}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Usuário: {self.nome} {self.sobrenome}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                self.send_frame(display_frame)
                time.sleep(0.03)

            # Finalizar captura
            if self.captured_count >= MIN_PHOTOS_REQUIRED:
                self.update_progress("Processando embeddings...")
                success, message = self.generate_embeddings()
                return success, message
            else:
                elapsed_no_face = time.time() - self.last_face_detected_time
                if elapsed_no_face > no_face_timeout:
                    return False, "❌ Nenhum rosto detectado por 15 segundos. Posicione seu rosto na câmera e tente novamente."
                else:
                    return False, f"Captura incompleta: {self.captured_count}/{MIN_PHOTOS_REQUIRED}"

        except Exception as e:
            logger.error(f"Erro na captura: {str(e)}")
            return False, f"Erro na captura: {str(e)}"
        finally:
            if cap and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            self.detector.cached_faces = []
            self.detector.last_detection_time = 0
            logger.info("✅ Recursos da câmera liberados para próxima captura")

    def stop(self):
        """Para a captura de forma segura"""
        self.running = False
        logger.info("⏹️ Captura interrompida")