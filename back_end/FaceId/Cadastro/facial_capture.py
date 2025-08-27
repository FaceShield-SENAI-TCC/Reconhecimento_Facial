import cv2
import os
import time
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
import urllib.request
import shutil
import base64

# ====================== CONFIGURAÇÕES AVANÇADAS ======================
DATABASE_DIR = "facial_database"
CAPTURE_DURATION = 15  # Segundos
TARGET_FACES = 50
MIN_FACE_SIZE = (120, 120)
FACE_DETECTOR = "dnn"
MODELS_DIR = "models"

# Parâmetros de qualidade
MIN_SHARPNESS = 100
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 200
MIN_FACE_CONFIDENCE = 0.9

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Verificar disponibilidade do MTCNN
try:
    from mtcnn import MTCNN

    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


# ====================== FUNÇÕES AUXILIARES ======================
def sanitize_name(name):
    """Remove caracteres especiais do nome para criar um nome de diretório seguro"""
    safe_name = re.sub(r'[^\w\s-]', '', name).strip()
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return safe_name


def calculate_sharpness(image):
    """Calcula a nitidez da imagem usando o operador Laplaciano"""
    if image.size == 0 or len(image.shape) < 2:
        return 0
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        return 0


def calculate_brightness(image):
    """Calcula o brilho médio da imagem no espaço de cores HSV"""
    if image.size == 0 or len(image.shape) < 3:
        return 0
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 2])
    except Exception:
        return 0


def enhance_face_image(face_img):
    """Melhora a qualidade da imagem facial usando CLAHE e denoising"""
    if face_img is None or face_img.size == 0:
        return np.zeros((100, 100, 3), np.uint8)
    try:
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        return enhanced
    except Exception as e:
        logging.error(f"Erro em enhance_face_image: {e}")
        return face_img


def face_quality_score(face_img):
    """Calcula uma pontuação de qualidade para a imagem facial"""
    if face_img is None or face_img.size == 0:
        return 0
    try:
        sharpness = calculate_sharpness(face_img)
        sharp_score = min(1, sharpness / 200) * 50
        brightness = calculate_brightness(face_img)
        bright_score = 0
        if MIN_BRIGHTNESS < brightness < MAX_BRIGHTNESS:
            bright_score = (1 - abs(brightness - 120) / 80) * 30
        return sharp_score + bright_score
    except Exception:
        return 0


# ====================== DETECÇÃO FACIAL ======================
def detect_faces(frame, detector):
    """Detecta rostos no frame usando o detector selecionado"""
    faces = []
    if detector["type"] == "dnn":
        try:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            detector["net"].setInput(blob)
            detections = detector["net"].forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > MIN_FACE_CONFIDENCE:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    faces.append([max(0, startX), max(0, startY), min(w, endX), min(h, endY)])
        except Exception as e:
            logging.error(f"Erro na detecção DNN: {str(e)}")
    # Outros detectores (mtcnn e haar) permanecem os mesmos
    # ...
    return faces


# ====================== DOWNLOAD DE MODELOS ======================
def download_dnn_model():
    """Baixa os modelos DNN necessários se não existirem localmente"""
    logging.info("Verificando modelos DNN...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_file = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    config_file = os.path.join(MODELS_DIR, "deploy.prototxt")

    files = {
        config_file: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        model_file: "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }

    for file_path, url in files.items():
        if not os.path.exists(file_path):
            logging.info(f"Baixando modelo: {os.path.basename(file_path)}")
            try:
                with urllib.request.urlopen(url) as response, open(file_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                logging.info(f"Download concluído: {os.path.basename(file_path)}")
            except Exception as e:
                logging.error(f"Erro ao baixar modelo {os.path.basename(file_path)}: {str(e)}")
                raise FileNotFoundError(f"Não foi possível baixar o modelo DNN: {file_path}")


def initialize_detector(detector_type):
    """Inicializa o detector facial escolhido"""
    if detector_type == "dnn":
        try:
            download_dnn_model()
            model_file = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
            config_file = os.path.join(MODELS_DIR, "deploy.prototxt")
            net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            return {"type": "dnn", "net": net}
        except Exception as e:
            logging.error(f"Erro ao carregar modelo DNN: {str(e)}. Tentando fallback para Haar.")
            detector_type = "haar"

    # Haar Cascade padrão (fallback)
    if detector_type == "haar":
        try:
            cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_file):
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                os.makedirs(MODELS_DIR, exist_ok=True)
                cascade_file_local = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
                logging.info("Baixando modelo Haar Cascade...")
                urllib.request.urlretrieve(url, cascade_file_local)
                cascade_file = cascade_file_local

            detector = cv2.CascadeClassifier(cascade_file)
            if detector.empty():
                raise RuntimeError("Falha ao carregar Haar Cascade")
            return {"type": "haar", "detector": detector}
        except Exception as e:
            logging.error(f"Erro fatal ao inicializar detector Haar: {str(e)}")
            raise RuntimeError("Nenhum detector facial disponível")

    raise ValueError("Tipo de detector não suportado.")


# ====================== CLASSE DE CAPTURA FACIAL ATUALIZADA ======================
class FaceCapture:
    def __init__(self, user_name, progress_callback=None, frame_callback=None):
        self.user_name = user_name
        self.safe_name = sanitize_name(user_name)
        self.user_dir = os.path.join(DATABASE_DIR, self.safe_name)
        self.progress_callback = progress_callback
        self.frame_callback = frame_callback
        self.captured_count = 0
        self.running = False
        self.start_time = None
        self.last_frame_sent = time.time()
        self.frame_interval = 0.1  # Envia frames a cada 0.1s (10 FPS)
        self.processing_interval = 0.2  # Processa detecção a cada 0.2s (5 FPS)

    def update_progress(self, message, count=None):
        if self.progress_callback:
            progress = {
                "message": message,
                "captured": count if count is not None else self.captured_count,
                "total": TARGET_FACES,
            }
            self.progress_callback(progress)

    def stop(self):
        self.running = False

    def capture(self):
        self.running = True
        self.start_time = time.time()
        cap = None
        try:
            os.makedirs(self.user_dir, exist_ok=True)
            detector = initialize_detector(FACE_DETECTOR)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.update_progress("Erro: Câmera não disponível")
                return False, "Câmera não disponível"

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.update_progress("Preparando câmera...")
            time.sleep(2)

            executor = ThreadPoolExecutor(max_workers=4)
            faces = []

            self.update_progress("Capturando rostos...")

            while self.running and self.captured_count < TARGET_FACES:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                # Processar detecção periodicamente
                current_time = time.time()
                if current_time - self.last_frame_sent >= self.processing_interval:
                    faces = detect_faces(frame, detector)
                    self.last_frame_sent = current_time

                # Desenhar e salvar faces detectadas
                for (x1, y1, x2, y2) in faces:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face_img = frame[y1:y2, x1:x2]

                    if face_img.size > 0:
                        quality = face_quality_score(face_img)

                        if quality > 40:
                            executor.submit(self.save_face_image, face_img, self.captured_count, quality)
                            self.captured_count += 1
                            self.update_progress(f"Face {self.captured_count} capturada", self.captured_count)

                # Adicionar contador e tempo ao frame de exibição
                cv2.putText(display_frame, f"Capturadas: {self.captured_count}/{TARGET_FACES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Enviar frame para o front-end
                _, buffer = cv2.imencode('.jpg', display_frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                self.frame_callback(jpg_as_text)

                # Controle de tempo para evitar uso excessivo da CPU
                if time.time() - self.start_time > CAPTURE_DURATION:
                    break

            executor.shutdown(wait=True)

            success = self.captured_count >= TARGET_FACES
            message = f"{self.captured_count} faces capturadas com sucesso!" if success else "Falha na captura"
            self.update_progress(message)
            return success, message

        except Exception as e:
            logging.error(f"Erro na captura: {str(e)}")
            self.update_progress(f"Erro: {str(e)}")
            return False, str(e)
        finally:
            if cap and cap.isOpened():
                cap.release()
            self.running = False

    def save_face_image(self, face_img, index, quality):
        """Salva a imagem facial aprimorada no diretório do usuário"""
        try:
            if face_img is None or face_img.size == 0:
                return

            enhanced = enhance_face_image(face_img)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            filename = f"{self.safe_name}_{index:03d}_{int(quality)}.jpg"
            filepath = os.path.join(self.user_dir, filename)

            # Tenta salvar a imagem, garantindo que o diretório existe
            os.makedirs(self.user_dir, exist_ok=True)
            cv2.imwrite(filepath, enhanced)
            logging.info(f"Face salva: {filename}")
        except Exception as e:
            logging.error(f"Erro ao salvar face: {str(e)}")