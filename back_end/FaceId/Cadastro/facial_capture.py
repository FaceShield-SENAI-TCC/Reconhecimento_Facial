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
CAPTURE_DURATION = 15
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


# ====================== FUNÇÕES AUXILIARES CORRIGIDAS ======================
def sanitize_name(name):
    """Remove caracteres especiais do nome para criar um nome de diretório seguro"""
    # Substitui espaços por underscores e remove caracteres inválidos
    safe_name = re.sub(r'[^\w\s-]', '', name).strip()
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return safe_name


def calculate_sharpness(image):
    """Calcula a nitidez da imagem usando o operador Laplaciano"""
    if image.size == 0:
        return 0

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        return 0


def calculate_brightness(image):
    """Calcula o brilho médio da imagem no espaço de cores HSV"""
    if image.size == 0:
        return 0

    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 2])
    except:
        return 0


def enhance_face_image(face_img):
    """Melhora a qualidade da imagem facial usando CLAHE e denoising"""
    if face_img.size == 0:
        return face_img

    try:
        # Aplicar CLAHE para melhorar o contraste
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Reduzir ruído
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        return enhanced
    except:
        return face_img


def face_quality_score(face_img):
    """Calcula uma pontuação de qualidade para a imagem facial"""
    if face_img.size == 0:
        return 0

    try:
        sharpness = calculate_sharpness(face_img)
        sharp_score = min(1, sharpness / 200) * 50

        brightness = calculate_brightness(face_img)
        if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
            bright_score = 0
        else:
            bright_score = (1 - abs(brightness - 120) / 80) * 30

        return sharp_score + bright_score
    except:
        return 0


# ====================== DETECÇÃO FACIAL CORRIGIDA ======================
def detect_faces(frame, detector):
    """Detecta rostos no frame usando o detector selecionado"""
    faces = []  # Inicializa lista vazia

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

                    # Garantir que as coordenadas estão dentro dos limites da imagem
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    # Verificar se a região é válida
                    if endX - startX > 0 and endY - startY > 0:
                        faces.append([startX, startY, endX, endY])
        except Exception as e:
            logging.error(f"Erro na detecção DNN: {str(e)}")

    elif detector["type"] == "mtcnn":
        if not MTCNN_AVAILABLE:
            logging.error("MTCNN não está disponível. Use outro detector.")
            return faces
        try:
            results = detector["detector"].detect_faces(frame)
            for res in results:
                if res['confidence'] > MIN_FACE_CONFIDENCE:
                    x, y, w, h = res['box']
                    # Garantir que as coordenadas estão dentro dos limites
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    faces.append([x, y, x + w, y + h])
        except Exception as e:
            logging.error(f"Erro na detecção MTCNN: {str(e)}")

    else:  # Haar Cascade
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_faces = detector["detector"].detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=MIN_FACE_SIZE
            )
            for (x, y, w, h) in haar_faces:
                # Garantir que as coordenadas estão dentro dos limites
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                faces.append([x, y, x + w, y + h])
        except Exception as e:
            logging.error(f"Erro na detecção Haar: {str(e)}")

    return faces


# ====================== DOWNLOAD DE MODELOS CORRIGIDO ======================
def download_dnn_model():
    """Baixa os modelos DNN necessários se não existirem localmente"""
    logging.info("Verificando modelos DNN...")

    # Criar diretório de modelos se não existir
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
                logging.error(f"Erro ao baixar modelo: {str(e)}")
                # Tentar fallback para modelo Haar se DNN falhar
                global FACE_DETECTOR
                FACE_DETECTOR = "haar"


# ====================== INICIALIZAÇÃO DE MODELOS CORRIGIDA ======================
def initialize_detector(detector_type):
    """Inicializa o detector facial escolhido"""
    if detector_type == "dnn":
        # Verificar e baixar modelos se necessário
        download_dnn_model()

        model_file = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        config_file = os.path.join(MODELS_DIR, "deploy.prototxt")

        if os.path.exists(model_file) and os.path.exists(config_file):
            try:
                net = cv2.dnn.readNetFromCaffe(config_file, model_file)
                return {"type": "dnn", "net": net}
            except Exception as e:
                logging.error(f"Erro ao carregar modelo DNN: {str(e)}")
                # Fallback para Haar se DNN falhar
                detector_type = "haar"

    if detector_type == "mtcnn":
        if not MTCNN_AVAILABLE:
            logging.warning("MTCNN não disponível. Usando DNN como fallback.")
            return initialize_detector("dnn")
        try:
            return {"type": "mtcnn", "detector": MTCNN()}
        except Exception as e:
            logging.error(f"Erro ao inicializar MTCNN: {str(e)}")
            return initialize_detector("dnn")

    # Haar Cascade padrão (fallback)
    try:
        cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_file):
            # Tentar fallback local se o arquivo padrão não existir
            cascade_file = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
            if not os.path.exists(cascade_file):
                # Baixar o Haar Cascade se não estiver disponível
                logging.info("Baixando modelo Haar Cascade...")
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                urllib.request.urlretrieve(url, cascade_file)

        detector = cv2.CascadeClassifier(cascade_file)
        if detector.empty():
            raise RuntimeError("Falha ao carregar Haar Cascade")

        return {"type": "haar", "detector": detector}
    except Exception as e:
        logging.error(f"Erro fatal ao inicializar detector facial: {str(e)}")
        raise RuntimeError("Nenhum detector facial disponível")


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

    def update_progress(self, message, count=None):
        if self.progress_callback:
            progress = {
                "message": message,
                "captured": count if count is not None else self.captured_count,
                "total": TARGET_FACES,
                "time_elapsed": time.time() - self.start_time if self.start_time else 0
            }
            self.progress_callback(progress)

    def capture(self):
        try:
            self.running = True
            self.start_time = time.time()
            os.makedirs(self.user_dir, exist_ok=True)

            detector = initialize_detector(FACE_DETECTOR)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduzido para melhor performance
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduzido para melhor performance

            if not cap.isOpened():
                self.update_progress("Erro: Câmera não disponível")
                return False, "Câmera não disponível"

            self.update_progress("Preparando câmera...")
            time.sleep(2)

            executor = ThreadPoolExecutor(max_workers=4)
            futures = []
            frame_count = 0
            faces = []

            self.update_progress("Capturando rostos...")

            while self.running and (
                    time.time() - self.start_time) < CAPTURE_DURATION and self.captured_count < TARGET_FACES:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame_count += 1
                    frame = cv2.flip(frame, 1)

                    # Criar uma cópia do frame para exibição
                    display_frame = frame.copy()

                    # Detectar rostos periodicamente (frequência reduzida)
                    if frame_count % 5 == 0 or not faces:
                        try:
                            faces = detect_faces(frame, detector)
                            logging.info(f"Faces detectadas: {len(faces)}")
                        except Exception as e:
                            logging.error(f"Erro na detecção facial: {str(e)}")
                            faces = []

                    # Processar cada rosto detectado
                    for (x1, y1, x2, y2) in faces:
                        # Calcular margem de segurança ao redor do rosto
                        margin = int((x2 - x1) * 0.15)
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(frame.shape[1], x2 + margin)
                        y2 = min(frame.shape[0], y2 + margin)

                        # Verificar se a região é válida
                        if x2 <= x1 or y2 <= y1:
                            continue

                        face_img = frame[y1:y2, x1:x2]

                        if face_img.size > 0:
                            # Desenhar retângulo no frame de exibição
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            quality = face_quality_score(face_img)
                            logging.info(f"Qualidade calculada: {quality}")

                            if quality > 40 and self.captured_count < TARGET_FACES:  # Limite reduzido
                                future = executor.submit(
                                    self.save_face_image,
                                    face_img,
                                    self.captured_count,
                                    quality
                                )
                                futures.append(future)
                                self.captured_count += 1
                                self.update_progress(
                                    f"Face {self.captured_count} capturada (Qualidade: {int(quality)}%)")
                                logging.info(f"Faces capturadas: {self.captured_count}/{TARGET_FACES}")

                    # Adicionar contador ao frame de exibição
                    cv2.putText(
                        display_frame,
                        f"Capturadas: {self.captured_count}/{TARGET_FACES}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    # Enviar frame para o front-end, se houver callback
                    if self.frame_callback:
                        # Reduzir a resolução para melhor desempenho
                        small_frame = cv2.resize(display_frame, (640, 360))
                        _, buffer = cv2.imencode('.jpg', small_frame)
                        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                        self.frame_callback(jpg_as_text)

                    # Controle de taxa de envio de frames (reduzido)
                    time.sleep(0.1)  # ~10 FPS

                except Exception as e:
                    logging.error(f"Erro no loop de captura: {str(e)}")
                    continue

            cap.release()
            executor.shutdown(wait=True)
            self.running = False

            success = self.captured_count >= 30  # Pelo menos 30 faces para considerar sucesso
            message = f"{self.captured_count} faces capturadas com sucesso!" if success else "Falha na captura"
            self.update_progress(message)
            return success, message

        except Exception as e:
            logging.error(f"Erro na captura: {str(e)}")
            self.update_progress(f"Erro: {str(e)}")
            return False, str(e)

    def save_face_image(self, face_img, index, quality):
        """Salva a imagem facial aprimorada no diretório do usuário"""
        try:
            # Verifique se a imagem é válida
            if face_img is None or face_img.size == 0:
                logging.warning("Tentativa de salvar imagem facial vazia")
                return

            enhanced = enhance_face_image(face_img)
            timestamp = datetime.now().strftime("%H%M%S%f")
            filename = f"{self.safe_name}_{index:03d}_{timestamp}_{int(quality)}.jpg"
            filepath = os.path.join(self.user_dir, filename)

            # Tente salvar a imagem com múltiplas tentativas
            for attempt in range(3):
                try:
                    cv2.imwrite(filepath, enhanced)
                    logging.info(f"Face salva: {filename}")
                    break
                except Exception as e:
                    logging.warning(f"Tentativa {attempt + 1} falhou ao salvar face: {str(e)}")
                    time.sleep(0.1)

        except Exception as e:
            logging.error(f"Erro ao salvar face: {str(e)}")

    def stop(self):
        self.running = False


# ====================== FUNÇÃO PRINCIPAL (PARA EXECUÇÃO DIRETA) ======================
def main():
    print("======== SISTEMA DE CADASTRO FACIAL AVANÇADO ========")
    print(f"Configuração: {TARGET_FACES} faces em {CAPTURE_DURATION} segundos")
    print(f"Detector: {FACE_DETECTOR.upper()}\n")

    user_name = input("Digite o nome completo do usuário: ").strip()
    if not user_name:
        print("Nome inválido!")
        return

    safe_name = sanitize_name(user_name)
    user_dir = os.path.join(DATABASE_DIR, safe_name)
    os.makedirs(user_dir, exist_ok=True)

    start_time = time.time()

    # Criar instância de captura
    capture = FaceCapture(user_name)
    success, message = capture.capture()

    capture_time = time.time() - start_time

    print("\n" + "=" * 50)
    if success:
        print(f"CADASTRO CONCLUÍDO PARA: {user_name}")
        print(f"Faces capturadas: {capture.captured_count}")
        print(f"Tempo total: {capture_time:.1f} segundos")
        print(f"Taxa de captura: {capture.captured_count / capture_time:.1f} faces/segundo")
        print(f"Armazenado em: {user_dir}")
    else:
        print(f"FALHA NO CADASTRO: {message}")
    print("=" * 50)


if __name__ == "__main__":
    main()