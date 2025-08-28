import os
import numpy as np
import cv2
from deepface import DeepFace
import pickle
import time
import logging
import psutil
from datetime import datetime
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
# AJUSTE O CAMINHO PARA A PASTA DE FOTOS DO SISTEMA DE CADASTRO.
DATABASE_DIR = "C:/Users/Aluno/Desktop/Arduino/back_end/FaceId/Cadastro/Faces"

MODEL_NAME = "VGG-Face"
DISTANCE_THRESHOLD = 0.40
MIN_FACE_SIZE = (100, 100)
EMBEDDINGS_CACHE = os.path.join(BASE_DIR, "embeddings_cache.pkl")
CACHE_VALIDITY_HOURS = 24  # Cache válido por 24 horas
RELOAD_COOLDOWN = 60  # Tempo mínimo entre recarregamentos (segundos)

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


# ====================== MONITORAMENTO DO BANCO DE DADOS ======================
class DatabaseChangeHandler(FileSystemEventHandler):
    """Monitora mudanças no diretório do banco de dados"""

    def on_any_event(self, event):
        global last_db_reload_time
        current_time = time.time()
        # Verificar se já se passou tempo suficiente desde o último recarregamento
        if current_time - last_db_reload_time >= RELOAD_COOLDOWN:
            logger.info("Mudanca detectada no banco de dados. Recarregando...")
            reload_thread = threading.Thread(target=reload_database)
            reload_thread.daemon = True
            reload_thread.start()
            last_db_reload_time = current_time


def start_database_monitor():
    """Inicia o monitoramento do diretório do banco de dados"""
    global db_observer
    try:
        if not os.path.exists(DATABASE_DIR):
            logger.error(f"Diretorio de monitoramento nao existe: {DATABASE_DIR}")
            return False

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


def get_embedding_from_file(image_path):
    """Obtém o embedding de uma imagem de rosto a partir de um arquivo"""
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=True
        )
        if isinstance(result, list) and len(result) > 0 and "embedding" in result[0]:
            return result[0]["embedding"]

        logger.warning(f"Nenhum rosto detectado para gerar embedding em {image_path}")
        return None
    except Exception as e:
        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=MODEL_NAME,
                detector_backend="opencv",
                enforce_detection=False
            )
            if isinstance(result, list) and len(result) > 0 and "embedding" in result[0]:
                return result[0]["embedding"]
        except Exception as e_fallback:
            logger.error(f"Erro ao obter embedding para {image_path}: {str(e_fallback)}")
            return None
    return None


def is_cache_valid():
    """Verifica se o cache ainda é válido baseado no tempo"""
    if not os.path.exists(EMBEDDINGS_CACHE):
        return False
    cache_time = datetime.fromtimestamp(os.path.getmtime(EMBEDDINGS_CACHE))
    current_time = datetime.now()
    time_diff = (current_time - cache_time).total_seconds() / 3600
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

    database = {}
    user_count = 0
    embedding_count = 0

    # Verificar se o diretório existe e se não está vazio
    if not os.path.exists(DATABASE_DIR):
        logger.error(f"Diretorio do banco de dados nao existe: {DATABASE_DIR}")
        return None

    turma_folders = [f for f in os.listdir(DATABASE_DIR) if os.path.isdir(os.path.join(DATABASE_DIR, f))]
    if not turma_folders:
        logger.warning(f"Diretorio do banco de dados vazio: {DATABASE_DIR}")
        return {}

    # Iterar por cada turma
    for turma_folder in turma_folders:
        turma_path = os.path.join(DATABASE_DIR, turma_folder)
        logger.info(f"Processando turma: {turma_folder}")

        # Iterar por cada usuário dentro da turma
        for user_folder in os.listdir(turma_path):
            user_path = os.path.join(turma_path, user_folder)

            if os.path.isdir(user_path):
                user_name = user_folder.replace('_', ' ').title()
                embeddings = []

                logger.info(f"Processando usuario: {user_name} em {turma_folder}")

                for face_file in os.listdir(user_path):
                    if face_file.lower().endswith((".jpg", ".png", ".jpeg")):
                        face_path = os.path.join(user_path, face_file)
                        try:
                            embedding = get_embedding_from_file(face_path)
                            if embedding is not None:
                                normalized_embedding = np.array(embedding) / np.linalg.norm(embedding)
                                embeddings.append(normalized_embedding)
                                embedding_count += 1
                                logger.debug(f"Embedding de {face_file} gerado com sucesso.")
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
    if os.path.exists(EMBEDDINGS_CACHE) and is_cache_valid():
        try:
            with open(EMBEDDINGS_CACHE, 'rb') as f:
                facial_db = pickle.load(f)
            logger.info("Cache de embeddings carregado com sucesso.")
            return facial_db
        except Exception as e:
            logger.error(f"Erro ao carregar cache: {str(e)}")

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
def recognize_face_from_array(face_img_array):
    global facial_db
    start_time = time.time()
    try:
        logger.info("Iniciando reconhecimento facial...")

        result = DeepFace.represent(
            img_path=face_img_array,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=True
        )

        if not isinstance(result, list) or not result or "embedding" not in result[0]:
            logger.warning("Nenhum embedding retornado pelo DeepFace")
            return None, None

        captured_embedding = np.array(result[0]["embedding"])
        captured_embedding_normalized = captured_embedding / np.linalg.norm(captured_embedding)

        best_match = None
        min_distance = float('inf')

        if not facial_db:
            logger.warning("Banco de dados facial esta vazio. Nenhum reconhecimento sera feito.")
            return None, None

        logger.info(f"Comparando com {len(facial_db)} usuarios no banco...")
        compare_start = time.time()

        for user_name, embeddings in facial_db.items():
            for db_embedding in embeddings:
                cosine_distance = 1 - np.dot(captured_embedding_normalized, db_embedding)
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
    global facial_db

    if facial_db is None or (last_db_update and (time.time() - last_db_update) > CACHE_VALIDITY_HOURS * 3600):
        logger.info("Recarregando banco de dados facial...")
        facial_db = load_or_create_cache()
        if facial_db is None:
            return {"error": "Facial database not loaded"}, 500

    try:
        # A nova lógica no appFaceId.py garante que image_data não tenha o prefixo,
        # mas vamos manter a verificação aqui para segurança.
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return {"authenticated": False, "message": "Falha ao decodificar a imagem. Imagem vazia ou inválida."}, 400

        try:
            detected_face = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=True
            )
            if not detected_face or "facial_area" not in detected_face[0]:
                return {
                    "authenticated": False,
                    "user": None,
                    "message": "Nenhum rosto detectado na imagem."
                }, 200

            face_area = detected_face[0]["facial_area"]
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            face_img_array = frame[y:y + h, x:x + w]

            if w < MIN_FACE_SIZE[0] or h < MIN_FACE_SIZE[1]:
                return {
                    "authenticated": False,
                    "user": None,
                    "message": "Rosto muito pequeno. Aproxime-se."
                }, 200

        except Exception as e:
            logger.error(f"Erro na deteccao de rosto: {str(e)}")
            return {
                "authenticated": False,
                "user": None,
                "message": "Erro ao processar imagem."
            }, 500

        user, distance = recognize_face_from_array(face_img_array)

        if user:
            confidence = 1 - distance
            return {
                "authenticated": True,
                "user": user,
                "confidence": float(confidence),
                "message": f"Bem-vindo, {user}!"
            }, 200
        else:
            return {
                "authenticated": False,
                "user": None,
                "message": "Usuario nao reconhecido. Faca o cadastro."
            }, 200

    except Exception as e:
        logger.error(f"ERRO: Erro no processamento facial: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": "Internal server error"}, 500


def get_database_status():
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
        clear_embeddings_cache()
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
    logger.info("Sistema de reconhecimento facial finalizado")