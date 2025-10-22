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
    "dbname": "faceshild",
    "user": "postgres",
    "password": "root",
    "host": "localhost",
    "port": "5432"
}

MIN_PHOTOS_REQUIRED = 3
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
    """Valida qualidade da imagem facial"""
    if face_img is None or face_img.size == 0:
        return False, "Imagem vazia"

    height, width = face_img.shape[:2]
    if height < MIN_FACE_SIZE[0] or width < MIN_FACE_SIZE[1]:
        return False, "Rosto muito pequeno"

    sharpness = calculate_sharpness(face_img)
    if sharpness < 20:
        return False, "Imagem muito borrada"

    return True, f"Qualidade: {sharpness:.1f}"


# ====================== DETECTOR FACIAL SIMPLIFICADO ======================
class FaceDetector:
    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 0.1
        self.cached_faces = []

        # Carregar classificador Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if self.face_cascade.empty():
            logger.error("❌ Não foi possível carregar o classificador de faces")
        else:
            logger.info("✅ Classificador Haar Cascade carregado")

    def detect_faces_simple(self, frame):
        """Detecção simples e direta usando OpenCV"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar rostos
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            valid_faces = []
            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                if face_roi.size > 0:
                    valid_faces.append((x, y, w, h))
                    logger.info(f"✅ Rosto detectado: {w}x{h}")

            return valid_faces

        except Exception as e:
            logger.error(f"Erro na detecção simples: {e}")
            return []

    def detect_faces(self, frame):
        """Combina métodos de detecção"""
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.cached_faces

        try:
            # Usar apenas o método simples (OpenCV)
            faces = self.detect_faces_simple(frame)

            self.cached_faces = faces
            self.last_detection_time = current_time

            logger.info(f"🔍 Total de rostos detectados: {len(faces)}")
            return faces

        except Exception as e:
            logger.error(f"❌ Erro na detecção: {str(e)}")
            return []


# ====================== CAPTURA PRINCIPAL ======================
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
        self.face_capture_interval = 1.0
        self.last_face_detected_time = 0
        self.consecutive_no_face_count = 0

        # ✅ CONFIGURAÇÃO VGG-FACE
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
                return True, "Usuário salvo com sucesso"

        except Exception as e:
            logger.error(f"Erro ao salvar no banco: {str(e)}")
            return False, f"Erro ao salvar: {str(e)}"

    def generate_embeddings(self):
        """Gera embeddings das faces capturadas - COMPATÍVEL COM DEEPFACE 0.0.75"""
        try:
            embeddings = []
            successful = 0

            for i, face_img in enumerate(self.captured_faces):
                try:
                    # Validar face
                    is_valid, validation_msg = validate_face_image(face_img)
                    if not is_valid:
                        logger.warning(f"Face {i + 1} inválida: {validation_msg}")
                        continue

                    # ✅ CORREÇÃO ESPECÍFICA PARA DEEPFACE 0.0.75
                    # Na versão 0.0.75, o DeepFace.represent tem comportamento diferente
                    temp_path = f"temp_face_{int(time.time())}_{i}.jpg"
                    cv2.imwrite(temp_path, face_img)

                    try:
                        # Na versão 0.0.75, DeepFace.represent pode retornar diferentes estruturas
                        result = DeepFace.represent(
                            img_path=temp_path,
                            model_name=self.model_name,
                            enforce_detection=False,
                            detector_backend="skip"
                        )

                        logger.info(f"🔍 Estrutura do retorno DeepFace (tipo): {type(result)}")
                        if isinstance(result, list):
                            logger.info(f"🔍 Tamanho da lista: {len(result)}")
                            if len(result) > 0:
                                logger.info(f"🔍 Tipo do primeiro elemento: {type(result[0])}")

                        embedding = None

                        # ✅ CASO 1: Retorno é uma lista de dicionários (estrutura comum)
                        if isinstance(result, list) and len(result) > 0:
                            first_item = result[0]
                            if isinstance(first_item, dict) and 'embedding' in first_item:
                                embedding = np.array(first_item['embedding'])
                            elif isinstance(first_item, (list, np.ndarray)):
                                embedding = np.array(first_item)
                        # ✅ CASO 2: Retorno é um array numpy direto
                        elif isinstance(result, np.ndarray):
                            embedding = result
                        # ✅ CASO 3: Retorno é uma lista simples de números
                        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], (int, float)):
                            embedding = np.array(result)
                        # ✅ CASO 4: Retorno é um único número (float) - tratamento especial
                        elif isinstance(result, (int, float)):
                            logger.warning("DeepFace retornou um único número, não um embedding")
                            continue

                        # Validar o embedding obtido
                        if embedding is not None and len(embedding) == self.embedding_dimension:
                            embeddings.append(embedding)
                            successful += 1
                            logger.info(f"✅ Embedding {i + 1} gerado com sucesso")
                        else:
                            logger.warning(
                                f"❌ Embedding inválido: tipo={type(embedding)}, dimensão={len(embedding) if embedding is not None else 'None'}")

                    except Exception as e:
                        logger.warning(f"❌ Erro no DeepFace.represent: {str(e)}")

                        # ✅ TENTATIVA ALTERNATIVA: Usar verificação facial primeiro
                        try:
                            logger.info("🔄 Tentando método alternativo com verificação facial...")

                            # Criar uma imagem de referência temporária
                            reference_path = "reference_face.jpg"
                            cv2.imwrite(reference_path, face_img)

                            # Usar verify para obter embeddings
                            verification = DeepFace.verify(
                                img1_path=reference_path,
                                img2_path=reference_path,  # Comparar consigo mesma
                                model_name=self.model_name,
                                enforce_detection=False,
                                detector_backend="skip"
                            )

                            logger.info(f"🔍 Estrutura do verify: {type(verification)}")
                            if isinstance(verification, dict):
                                logger.info(f"🔍 Chaves do verify: {verification.keys()}")

                            # Em algumas versões, verify retorna embeddings
                            if isinstance(verification, dict) and 'embedding' in verification:
                                embedding = np.array(verification['embedding'])
                                if len(embedding) == self.embedding_dimension:
                                    embeddings.append(embedding)
                                    successful += 1
                                    logger.info(f"✅ Embedding {i + 1} gerado via verify")
                            else:
                                logger.warning("Verify não retornou embedding")

                            # Limpar arquivo de referência
                            if os.path.exists(reference_path):
                                os.remove(reference_path)

                        except Exception as alt_e:
                            logger.warning(f"❌ Método alternativo também falhou: {str(alt_e)}")

                    finally:
                        # Limpar arquivo temporário
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                except Exception as e:
                    logger.warning(f"❌ Erro no processamento da face {i + 1}: {str(e)}")
                    continue

            if successful >= MIN_PHOTOS_REQUIRED:
                # Usar a melhor imagem como perfil
                best_face_idx = self._select_best_profile_image()
                profile_image = self.captured_faces[best_face_idx]

                return self.save_to_database(embeddings, profile_image)
            else:
                return False, f"Embeddings insuficientes: {successful}/{MIN_PHOTOS_REQUIRED}"

        except Exception as e:
            logger.error(f"❌ Erro crítico na geração de embeddings: {str(e)}")
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
        """Configura a câmera"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self._cleanup_camera()

                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.warning(f"Tentativa {attempt + 1}: Câmera não abriu")
                    time.sleep(1)
                    continue

                # Configurações básicas
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 15)
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

        logger.error("❌ Não foi possível configurar a câmera")
        return None

    def _cleanup_camera(self):
        """Limpa recursos da câmera"""
        try:
            for i in range(3):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cap.release()
                time.sleep(0.3)
        except:
            pass

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
            logger.info("🎬 Iniciando processo de captura facial")
            start_time = time.time()
            no_face_timeout = 20

            while (self.running and
                   self.captured_count < MIN_PHOTOS_REQUIRED and
                   (time.time() - start_time) < 60):

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame vazio da câmera")
                    time.sleep(0.05)
                    continue

                # Espelhar frame para visualização
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                # Detectar rostos
                faces = self.detector.detect_faces(frame)
                face_detected = len(faces) == 1

                if face_detected:
                    self.last_face_detected_time = time.time()

                    x, y, w, h = faces[0]
                    cropped_face = frame[y:y + h, x:x + w]

                    # Verificar se deve capturar
                    current_time = time.time()
                    if (current_time - self.last_face_time) > self.face_capture_interval:
                        is_valid, validation_msg = validate_face_image(cropped_face)
                        if is_valid:
                            self.captured_faces.append(cropped_face.copy())
                            self.captured_count += 1
                            self.last_face_time = current_time

                            # Feedback visual
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                            cv2.putText(display_frame, f"CAPTURADO: {self.captured_count}/{MIN_PHOTOS_REQUIRED}",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            self.update_progress(f"Capturado: {self.captured_count}/{MIN_PHOTOS_REQUIRED}")
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
                    elapsed_no_face = time.time() - self.last_face_detected_time

                    # Feedback visual
                    cv2.putText(display_frame, "POSICIONE SEU ROSTO NA CAMERA",
                                (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if elapsed_no_face > no_face_timeout:
                        return False, "❌ Nenhum rosto detectado. Verifique a iluminação e posicione seu rosto na câmera."

                # Informações na tela
                cv2.putText(display_frame, f"Capturadas: {self.captured_count}/{MIN_PHOTOS_REQUIRED}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Usuário: {self.nome} {self.sobrenome}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                self.send_frame(display_frame)
                time.sleep(0.03)

            # Finalizar captura
            if self.captured_count >= MIN_PHOTOS_REQUIRED:
                self.update_progress("Processando embeddings...")
                success, message = self.generate_embeddings()
                return success, message
            else:
                return False, f"Captura incompleta: {self.captured_count}/{MIN_PHOTOS_REQUIRED}"

        except Exception as e:
            logger.error(f"Erro na captura: {str(e)}", exc_info=True)
            return False, f"Erro na captura: {str(e)}"
        finally:
            if cap and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            self.detector.cached_faces = []
            logger.info("✅ Recursos da câmera liberados")

    def stop(self):
        """Para a captura de forma segura"""
        self.running = False
        logger.info("⏹️ Captura interrompida")