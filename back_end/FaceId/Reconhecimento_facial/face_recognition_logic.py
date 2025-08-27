import os
import numpy as np
import cv2
from deepface import DeepFace
import pickle
import time
import logging
import psutil
from datetime import datetime
import uuid
import traceback
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import sys
import io
import base64

# Configurar stdout e stderr para UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ====================== CONFIGURAÇÕES ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "../Cadastro/facial_database")
MODEL_NAME = "VGG-Face"
DISTANCE_THRESHOLD = 0.40
MIN_FACE_SIZE = (100, 100)
MAX_IMAGE_SIZE = (1280, 720)
EMBEDDINGS_CACHE = os.path.join(BASE_DIR, "embeddings_cache.pkl")
CACHE_VALIDITY_HOURS = 24  # Cache válido por 24 horas
DB_CHECK_INTERVAL = 30  # Verificar mudanças a cada 30 segundos

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("faceid.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Variável global para armazenar o banco de dados facial
facial_db = None
last_db_update = None
db_observer = None
last_db_reload_time = 0
RELOAD_COOLDOWN = 60  # Tempo mínimo entre recarregamentos (segundos)

# Criar diretório de banco de dados se não existir
os.makedirs(DATABASE_DIR, exist_ok=True)


# ====================== MONITORAMENTO DO BANCO DE DADOS ======================
class DatabaseChangeHandler(FileSystemEventHandler):
    """Monitora mudanças no diretório do banco de dados"""

    def __init__(self):
        self.last_checked = time.time()
        self.ignore_events_until = 0

    def on_any_event(self, event):
        # Ignorar eventos temporariamente após um recarregamento
        if time.time() < self.ignore_events_until:
            return

        # Só processa eventos a cada intervalo para evitar processamento excessivo
        current_time = time.time()
        if current_time - self.last_checked >= DB_CHECK_INTERVAL:
            self.last_checked = current_time

            # Processar apenas eventos relevantes (imagens e diretórios)
            if (event.is_directory or
                    (hasattr(event, 'src_path') and event.src_path.endswith(('.jpg', '.jpeg', '.png')))):

                # Ignorar eventos relacionados a arquivos temporários
                if (hasattr(event, 'src_path') and
                        ('temp_face_' in event.src_path or 'embeddings_cache.pkl' in event.src_path)):
                    return

                logger.info(f"Mudanca detectada no banco de dados: {getattr(event, 'src_path', 'Unknown')}")
                self.schedule_reload()

    def schedule_reload(self):
        """Agenda o recarregamento do banco de dados"""
        global last_db_reload_time

        # Verificar se já se passou tempo suficiente desde o último recarregamento
        current_time = time.time()
        if current_time - last_db_reload_time < RELOAD_COOLDOWN:
            logger.info(f"Recarregamento adiado. Aguardando cooldown de {RELOAD_COOLDOWN} segundos.")
            return

        # Usar um thread separado para o recarregamento
        reload_thread = threading.Thread(target=self.reload_database)
        reload_thread.daemon = True
        reload_thread.start()

        # Ignorar eventos temporariamente
        self.ignore_events_until = time.time() + 10  # Ignorar eventos por 10 segundos

    def reload_database(self):
        """Recarrega o banco de dados quando mudanças são detectadas"""
        global last_db_reload_time

        logger.info("Iniciando recarregamento do banco de dados devido a mudancas detectadas...")
        last_db_reload_time = time.time()

        try:
            # Criar uma cópia local do banco de dados atual para evitar bloqueios
            old_db = facial_db

            # Recarregar o banco de dados
            clear_embeddings_cache()
            new_db = load_facial_database()

            if new_db is not None:
                logger.info(f"Banco de dados recarregado com {len(new_db)} usuarios")
            else:
                logger.error("Falha ao recarregar banco de dados. Mantendo versao anterior.")

        except Exception as e:
            logger.error(f"Erro ao recarregar banco de dados: {str(e)}")
            logger.error(traceback.format_exc())


def start_database_monitor():
    """Inicia o monitoramento do diretório do banco de dados"""
    global db_observer
    try:
        event_handler = DatabaseChangeHandler()
        db_observer = Observer()
        db_observer.schedule(event_handler, DATABASE_DIR, recursive=True)
        db_observer.start()
        logger.info(f"Monitoramento do banco de dados iniciado em: {DATABASE_DIR}")
        return True
    except Exception as e:
        logger.error(f"Falha ao iniciar monitoramento do banco de dados: {str(e)}")
        return False


def stop_database_monitor():
    """Para o monitoramento do banco de dados"""
    global db_observer
    if db_observer and db_observer.is_alive():
        db_observer.stop()
        db_observer.join()
        logger.info("Monitoramento do banco de dados parado")


# ====================== FUNÇÕES AUXILIARES ======================
def log_memory_usage():
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2  # Em MB
        logger.info(f"Uso de memoria: {mem:.2f} MB")
    except ImportError:
        logger.warning("psutil nao instalado, nao e possivel monitorar memoria")
    except Exception as e:
        logger.error(f"Erro ao monitorar memoria: {str(e)}")


def resize_image(image, max_size):
    """Redimensiona a imagem mantendo a proporção"""
    height, width = image.shape[:2]

    if width > max_size[0] or height > max_size[1]:
        scale = min(max_size[0] / width, max_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image


def get_embedding(image_path):
    """Obtém o embedding de uma imagem de rosto de forma robusta"""
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=False
        )

        # Handle different return formats
        if isinstance(result, list) and len(result) > 0:
            embedding_data = result[0]
            if "embedding" in embedding_data:
                return embedding_data["embedding"]
        elif isinstance(result, dict) and "embedding" in result:
            return result["embedding"]
        logger.warning(f"Embedding nao encontrado para {image_path}")
        return None
    except Exception as e:
        logger.error(f"Erro ao obter embedding: {str(e)}")
        return None


def is_cache_valid():
    """Verifica se o cache ainda é válido baseado no tempo"""
    if not os.path.exists(EMBEDDINGS_CACHE):
        return False

    cache_time = datetime.fromtimestamp(os.path.getmtime(EMBEDDINGS_CACHE))
    current_time = datetime.now()
    time_diff = (current_time - cache_time).total_seconds() / 3600  # horas

    return time_diff < CACHE_VALIDITY_HOURS


def clear_embeddings_cache():
    """Limpa o cache de embeddings"""
    global facial_db
    facial_db = None
    if os.path.exists(EMBEDDINGS_CACHE):
        os.remove(EMBEDDINGS_CACHE)
        logger.info("Cache de embeddings limpo")


# ====================== CARREGAR BANCO DE DADOS FACIAL ======================
def load_facial_database():
    global facial_db, last_db_update
    start_time = time.time()
    logger.info("Iniciando carregamento do banco de dados facial...")
    logger.info(f"Caminho absoluto do banco: {os.path.abspath(DATABASE_DIR)}")

    database = {}
    user_count = 0
    embedding_count = 0

    # Verificar se o diretório existe
    if not os.path.exists(DATABASE_DIR):
        logger.error(f"Diretorio do banco de dados nao existe: {DATABASE_DIR}")
        return None

    # Verificar se o diretório está vazio
    if not os.listdir(DATABASE_DIR):
        logger.error("Diretorio do banco de dados esta vazio!")
        return None

    for user_folder in os.listdir(DATABASE_DIR):
        user_path = os.path.join(DATABASE_DIR, user_folder)

        if os.path.isdir(user_path):
            user_name = " ".join([part.capitalize() for part in user_folder.split('_')])
            embeddings = []

            logger.info(f"Processando usuario: {user_name}")

            for face_file in os.listdir(user_path):
                if face_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    face_path = os.path.join(user_path, face_file)

                    try:
                        file_start_time = time.time()
                        embedding = get_embedding(face_path)
                        if embedding is not None:
                            embedding = np.array(embedding)
                            norm = np.linalg.norm(embedding)

                            if norm > 1e-8:  # Evita divisão por zero
                                normalized_embedding = embedding / norm
                                embeddings.append(normalized_embedding)
                                embedding_count += 1
                            else:
                                logger.warning(f"AVISO: Vetor de embedding invalido (norma zero) em: {face_path}")
                        else:
                            logger.warning(f"AVISO: Nenhum embedding retornado para: {face_path}")

                        logger.info(f"Tempo de processamento para {face_file}: {time.time() - file_start_time:.2f}s")

                    except Exception as e:
                        logger.error(f"ERRO: Erro ao processar {face_path}: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue

            if embeddings:
                database[user_name] = embeddings
                user_count += 1
                logger.info(f"Usuario {user_name}: {len(embeddings)} amostras carregadas")
            else:
                logger.warning(f"AVISO: Nenhum embedding valido para: {user_name}")

    if not database:
        logger.error("Banco de dados vazio ou nenhum rosto valido encontrado!")
        facial_db = None
        return None

    logger.info(f"SUCESSO: Banco de dados carregado com {user_count} usuarios e {embedding_count} embeddings")
    logger.info(f"Tempo total para carregar banco: {time.time() - start_time:.2f}s")
    last_db_update = time.time()
    log_memory_usage()
    return database


def load_or_create_cache():
    """Carrega o cache de embeddings se existir e estiver válido, senão cria"""
    global facial_db

    # Verificar se o cache existe e é válido
    if os.path.exists(EMBEDDINGS_CACHE) and is_cache_valid():
        try:
            with open(EMBEDDINGS_CACHE, 'rb') as f:
                facial_db = pickle.load(f)
            logger.info("Cache de embeddings carregado com sucesso.")
            return facial_db
        except Exception as e:
            logger.error(f"Erro ao carregar cache: {str(e)}")
            # Recriar cache em caso de erro

    # Cache não existe ou não é válido - recriar
    facial_db = load_facial_database()
    if facial_db is not None:
        try:
            with open(EMBEDDINGS_CACHE, 'wb') as f:
                pickle.dump(facial_db, f)
            logger.info("Cache de embeddings atualizado.")
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {str(e)}")

    return facial_db


# ====================== FUNÇÃO DE RECONHECIMENTO ======================
def recognize_face_from_image(face_img_path):
    global facial_db
    start_time = time.time()
    try:
        logger.info("Iniciando reconhecimento facial...")

        embedding = get_embedding(face_img_path)
        if embedding is None:
            logger.warning("Nenhum embedding retornado pelo DeepFace")
            return None, None

        captured_embedding = np.array(embedding)
        norm = np.linalg.norm(captured_embedding)

        if norm < 1e-8:
            logger.warning("Embedding capturado tem norma zero")
            return None, None

        captured_embedding_normalized = captured_embedding / norm

        best_match = None
        min_distance = float('inf')

        logger.info(f"Comparando com {len(facial_db)} usuarios no banco...")
        compare_start = time.time()

        for user_name, embeddings in facial_db.items():
            for db_embedding in embeddings:
                cosine_similarity = np.dot(captured_embedding_normalized, db_embedding)
                cosine_distance = 1 - cosine_similarity

                if cosine_distance < min_distance and cosine_distance < DISTANCE_THRESHOLD:
                    min_distance = cosine_distance
                    best_match = user_name

        logger.info(f"Tempo de comparacao: {time.time() - compare_start:.2f}s")

        if best_match:
            logger.info(f"Usuario reconhecido: {best_match} com distancia {min_distance:.4f}")
        else:
            logger.info("Nenhum usuario reconhecido")

        logger.info(f"Tempo total de reconhecimento: {time.time() - start_time:.2f}s")
        return best_match, min_distance

    except Exception as e:
        logger.error(f"ERRO: Erro no reconhecimento: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None


def process_face_login(image_data):
    """Processa uma solicitação de login facial a partir de dados de imagem"""
    global facial_db

    # Carregar banco de dados se necessário
    if facial_db is None:
        logger.info("Carregando banco de dados facial...")
        facial_db = load_or_create_cache()
        if facial_db is None:
            return {"error": "Facial database not loaded"}, 500

    try:
        # Verificar se a string base64 está vazia
        if not image_data:
            logger.error("String base64 vazia recebida")
            return {"error": "Empty image data"}, 400

        # Remover header se presente
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Verificar comprimento mínimo
        if len(image_data) < 100:
            logger.error(f"String base64 muito curta: {len(image_data)} caracteres")
            return {"error": "Invalid image data"}, 400

        try:
            img_bytes = base64.b64decode(image_data)
            logger.info(f"Tamanho dos bytes decodificados: {len(img_bytes)} bytes")
        except Exception as e:
            logger.error(f"Erro ao decodificar base64: {str(e)}")
            return {"error": "Base64 decoding failed"}, 400

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Falha ao decodificar imagem - frame e None")
            return {"error": "Failed to decode image"}, 400

        # Redimensionar imagem grande
        frame = resize_image(frame, MAX_IMAGE_SIZE)

        # Verificar se a imagem está vazia
        if frame.size == 0:
            logger.error("Imagem decodificada esta vazia")
            return {"error": "Empty image after decoding"}, 400

        # Detectar rosto usando DeepFace diretamente
        logger.info("Detectando rostos com DeepFace...")
        faces = []

        try:
            detected_faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False
            )

            for face in detected_faces:
                if 'facial_area' in face:
                    faces.append(face)
        except Exception as e:
            logger.warning(f"DeepFace detection failed: {str(e)}")
            logger.warning("Usando fallback para Haar Cascade")

            # Fallback: Haar Cascade tradicional
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)

            if face_cascade.empty():
                logger.error(f"Haar cascade nao encontrado em: {cascade_path}")
                return {"error": "Face detector not available"}, 500

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE
            )

            # Converter para formato padronizado
            for (x, y, w, h) in detected_faces:
                faces.append({
                    'facial_area': {
                        'x': int(x),
                        'y': int(y),
                        'w': int(w),
                        'h': int(h)
                    }
                })

        if len(faces) == 0:
            logger.info("Nenhum rosto detectado na imagem")
            return {
                "authenticated": False,
                "user": None,
                "message": "Nenhum rosto detectado. Posicione seu rosto na câmera."
            }, 200

        # Usar o rosto com maior área
        if 'facial_area' in faces[0]:
            # Formato do DeepFace
            face_area = faces[0]['facial_area']
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
        else:
            # Formato do Haar Cascade
            face_area = faces[0]
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']

        # Verificar tamanho mínimo do rosto
        if w < MIN_FACE_SIZE[0] or h < MIN_FACE_SIZE[1]:
            logger.warning(f"Rosto muito pequeno: {w}x{h}")
            return {
                "authenticated": False,
                "user": None,
                "message": "Rosto muito pequeno. Aproxime-se da câmera."
            }, 200

        face_img = frame[y:y + h, x:x + w]

        # Salvar temporariamente com nome único
        temp_filename = f"temp_face_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(BASE_DIR, temp_filename)
        cv2.imwrite(temp_path, face_img)
        logger.info(f"Rosto detectado salvo temporariamente em {temp_path} (tamanho: {w}x{h})")

        # Reconhecer rosto
        user, distance = recognize_face_from_image(temp_path)

        # Limpar arquivo temporário
        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Erro ao remover arquivo temporario: {str(e)}")

        if user:
            confidence = 1 - distance
            logger.info(f"Autenticacao bem-sucedida para {user} com confianca {confidence:.2f}")
            return {
                "authenticated": True,
                "user": user,
                "confidence": float(confidence)
            }, 200
        else:
            logger.info("Usuario nao reconhecido")
            return {
                "authenticated": False,
                "user": None,
                "message": "Usuario nao reconhecido. Tente novamente ou faca cadastro."
            }, 200

    except Exception as e:
        logger.error(f"ERRO: Erro no processamento facial: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": "Internal server error"}, 500


def get_database_status():
    """Retorna o status atual do banco de dados"""
    global facial_db
    return {
        "loaded": facial_db is not None,
        "user_count": len(facial_db) if facial_db else 0,
        "total_embeddings": sum(len(emb) for emb in facial_db.values()) if facial_db else 0,
        "users": list(facial_db.keys()) if facial_db else []
    }


def reload_database():
    """Recarrega o banco de dados manualmente"""
    global facial_db
    try:
        # Limpar cache existente
        clear_embeddings_cache()

        # Recarregar banco de dados
        facial_db = load_facial_database()

        if facial_db is None:
            return False, "Failed to load database"

        status = get_database_status()
        return True, f"Database reloaded with {status['user_count']} users"
    except Exception as e:
        logger.error(f"Erro ao recarregar banco de dados: {str(e)}")
        return False, str(e)


def initialize():
    """Inicializa o sistema de reconhecimento facial"""
    load_or_create_cache()
    return start_database_monitor()


def cleanup():
    """Limpeza do sistema"""
    stop_database_monitor()

    # Remover todos os arquivos temporários
    for filename in os.listdir(BASE_DIR):
        if filename.startswith("temp_face_") and filename.endswith(".jpg"):
            try:
                os.remove(os.path.join(BASE_DIR, filename))
                logger.info(f"Arquivo temporario {filename} removido")
            except Exception as e:
                logger.error(f"Erro ao remover {filename}: {str(e)}")

    logger.info("Sistema de reconhecimento facial finalizado")