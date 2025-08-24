import os
import numpy as np
import cv2
from deepface import DeepFace
from flask import Flask, request, jsonify, send_from_directory
import base64
import logging
import atexit
import time
import traceback
from flask_cors import CORS
import pickle
import uuid
import psutil

# ====================== CONFIGURAÇÕES ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "../Cadastro/facial_database")
MODEL_NAME = "VGG-Face"
DISTANCE_THRESHOLD = 0.40
MIN_FACE_SIZE = (100, 100)
MAX_IMAGE_SIZE = (1280, 720)
EMBEDDINGS_CACHE = os.path.join(BASE_DIR, "embeddings_cache.pkl")

app = Flask(__name__)
# Configuração CORS para permitir todas as origens (apenas para desenvolvimento)
CORS(app, origins="*")  # Permite todas as origens

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("faceid.log")
    ]
)
logger = logging.getLogger(__name__)

# Variável global para armazenar o banco de dados facial
facial_db = None

# Criar diretório de banco de dados se não existir
os.makedirs(DATABASE_DIR, exist_ok=True)


# ====================== FUNÇÕES AUXILIARES ======================
def log_memory_usage():
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2  # Em MB
        logger.info(f"Uso de memória: {mem:.2f} MB")
    except ImportError:
        logger.warning("psutil não instalado, não é possível monitorar memória")
    except Exception as e:
        logger.error(f"Erro ao monitorar memória: {str(e)}")


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
        logger.warning(f"Embedding não encontrado para {image_path}")
        return None
    except Exception as e:
        logger.error(f"Erro ao obter embedding: {str(e)}")
        return None


# ====================== CARREGAR BANCO DE DADOS FACIAL ======================
def load_facial_database():
    global facial_db
    logger.info("Iniciando carregamento do banco de dados facial...")
    logger.info(f"Caminho absoluto do banco: {os.path.abspath(DATABASE_DIR)}")

    database = {}
    user_count = 0
    embedding_count = 0

    # Verificar se o diretório existe
    if not os.path.exists(DATABASE_DIR):
        logger.error(f"Diretório do banco de dados não existe: {DATABASE_DIR}")
        return None

    for user_folder in os.listdir(DATABASE_DIR):
        user_path = os.path.join(DATABASE_DIR, user_folder)

        if os.path.isdir(user_path):
            user_name = " ".join([part.capitalize() for part in user_folder.split('_')])
            embeddings = []

            logger.info(f"Processando usuário: {user_name}")

            for face_file in os.listdir(user_path):
                if face_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    face_path = os.path.join(user_path, face_file)

                    try:
                        start_time = time.time()
                        embedding = get_embedding(face_path)
                        if embedding is not None:
                            embedding = np.array(embedding)
                            norm = np.linalg.norm(embedding)

                            if norm > 1e-8:  # Evita divisão por zero
                                normalized_embedding = embedding / norm
                                embeddings.append(normalized_embedding)
                                embedding_count += 1
                            else:
                                logger.warning(f"⚠️ Vetor de embedding inválido (norma zero) em: {face_path}")
                        else:
                            logger.warning(f"⚠️ Nenhum embedding retornado para: {face_path}")

                        logger.info(f"Tempo de processamento para {face_file}: {time.time() - start_time:.2f}s")

                    except Exception as e:
                        logger.error(f"🚨 Erro ao processar {face_path}: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue

            if embeddings:
                database[user_name] = embeddings
                user_count += 1
                logger.info(f"» {user_name}: {len(embeddings)} amostras carregadas")
            else:
                logger.warning(f"⚠️ Nenhum embedding válido para: {user_name}")

    if not database:
        logger.error("❌ Banco de dados vazio ou nenhum rosto válido encontrado!")
        facial_db = None
        return None

    logger.info(f"✅ Banco de dados carregado com {user_count} usuários e {embedding_count} embeddings")
    log_memory_usage()
    return database


def load_or_create_cache():
    """Carrega o cache de embeddings se existir, senão cria"""
    global facial_db
    if os.path.exists(EMBEDDINGS_CACHE):
        try:
            with open(EMBEDDINGS_CACHE, 'rb') as f:
                facial_db = pickle.load(f)
            logger.info("Cache de embeddings carregado com sucesso.")
            return facial_db
        except Exception as e:
            logger.error(f"Erro ao carregar cache: {str(e)}")
            # Recriar cache
            facial_db = load_facial_database()
            if facial_db is not None:
                with open(EMBEDDINGS_CACHE, 'wb') as f:
                    pickle.dump(facial_db, f)
            return facial_db
    else:
        facial_db = load_facial_database()
        if facial_db is not None:
            with open(EMBEDDINGS_CACHE, 'wb') as f:
                pickle.dump(facial_db, f)
        return facial_db


# ====================== FUNÇÃO DE RECONHECIMENTO ======================
def recognize_face_from_image(face_img_path):
    global facial_db
    try:
        logger.info("Iniciando reconhecimento facial...")
        start_time = time.time()

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

        logger.info(f"Comparando com {len(facial_db)} usuários no banco...")
        compare_start = time.time()

        for user_name, embeddings in facial_db.items():
            for db_embedding in embeddings:
                cosine_similarity = np.dot(captured_embedding_normalized, db_embedding)
                cosine_distance = 1 - cosine_similarity

                if cosine_distance < min_distance and cosine_distance < DISTANCE_THRESHOLD:
                    min_distance = cosine_distance
                    best_match = user_name

        logger.info(f"Tempo de comparação: {time.time() - compare_start:.2f}s")
        logger.info(f"Tempo total de reconhecimento: {time.time() - start_time:.2f}s")

        if best_match:
            logger.info(f"Usuário reconhecido: {best_match} com distância {min_distance:.4f}")
        else:
            logger.info("Nenhum usuário reconhecido")

        return best_match, min_distance

    except Exception as e:
        logger.error(f"🚨 Erro no reconhecimento: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None


# ====================== ROTAS DA API ======================
@app.route('/face-login', methods=['POST', 'OPTIONS'])
def face_login():
    if request.method == 'OPTIONS':
        # Resposta para pré-voo CORS
        response = jsonify()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    global facial_db
    logger.info("Recebendo solicitação de login facial")

    # Carregar banco de dados se necessário
    if facial_db is None:
        logger.info("Carregando banco de dados facial...")
        facial_db = load_or_create_cache()
        if facial_db is None:
            return jsonify({"error": "Facial database not loaded"}), 500

    # Obter imagem do request
    data = request.json
    if not data or 'imagem' not in data:
        logger.warning("Nenhuma imagem fornecida na solicitação")
        return jsonify({"error": "No image provided"}), 400

    try:
        img_b64 = data['imagem']

        # Verificar se a string base64 está vazia
        if not img_b64:
            logger.error("String base64 vazia recebida")
            return jsonify({"error": "Empty image data"}), 400

        # Remover header se presente
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]

        # Verificar comprimento mínimo
        if len(img_b64) < 100:
            logger.error(f"String base64 muito curta: {len(img_b64)} caracteres")
            return jsonify({"error": "Invalid image data"}), 400

        try:
            img_bytes = base64.b64decode(img_b64)
            logger.info(f"Tamanho dos bytes decodificados: {len(img_bytes)} bytes")
        except Exception as e:
            logger.error(f"Erro ao decodificar base64: {str(e)}")
            return jsonify({"error": "Base64 decoding failed"}), 400

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Falha ao decodificar imagem - frame é None")
            return jsonify({"error": "Failed to decode image"}), 400

        # Redimensionar imagem grande
        frame = resize_image(frame, MAX_IMAGE_SIZE)

        # Verificar se a imagem está vazia
        if frame.size == 0:
            logger.error("Imagem decodificada está vazia")
            return jsonify({"error": "Empty image after decoding"}), 400

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
                logger.error(f"Haar cascade não encontrado em: {cascade_path}")
                return jsonify({"error": "Face detector not available"}), 500

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
            return jsonify({
                "authenticated": False,
                "user": None,
                "message": "Nenhum rosto detectado. Posicione seu rosto na câmera."
            })

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
            return jsonify({
                "authenticated": False,
                "user": None,
                "message": "Rosto muito pequeno. Aproxime-se da câmera."
            })

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
            logger.warning(f"Erro ao remover arquivo temporário: {str(e)}")

        if user:
            confidence = 1 - distance
            logger.info(f"Autenticação bem-sucedida para {user} com confiança {confidence:.2f}")
            response = jsonify({
                "authenticated": True,
                "user": user,
                "confidence": float(confidence)
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        else:
            logger.info("Usuário não reconhecido")
            response = jsonify({
                "authenticated": False,
                "user": None,
                "message": "Usuário não reconhecido. Tente novamente ou faça cadastro."
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

    except Exception as e:
        logger.error(f"🚨 Erro no endpoint /face-login: {str(e)}")
        logger.error(traceback.format_exc())
        response = jsonify({"error": "Internal server error"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500


@app.route('/test-db', methods=['GET'])
def test_db():
    global facial_db
    if facial_db is None:
        response = jsonify({"status": "Database not loaded"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    response = jsonify({
        "status": "Database loaded",
        "users": list(facial_db.keys()),
        "user_count": len(facial_db),
        "total_embeddings": sum(len(emb) for emb in facial_db.values())
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/reload-db', methods=['POST'])
def reload_db():
    """Rota para recarregar o banco de dados manualmente"""
    global facial_db
    try:
        # Remover cache existente
        if os.path.exists(EMBEDDINGS_CACHE):
            os.remove(EMBEDDINGS_CACHE)

        # Recarregar banco de dados
        facial_db = load_facial_database()

        if facial_db is None:
            response = jsonify({"success": False, "error": "Failed to load database"})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500

        response = jsonify({
            "success": True,
            "message": f"Database reloaded with {len(facial_db)} users"
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        logger.error(f"Erro ao recarregar banco de dados: {str(e)}")
        response = jsonify({"success": False, "error": str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500


@app.route('/')
def home():
    return send_from_directory('templates', 'login.html')


# ====================== LIMPEZA E INICIALIZAÇÃO ======================
@atexit.register
def cleanup():
    # Remover todos os arquivos temporários
    for filename in os.listdir(BASE_DIR):
        if filename.startswith("temp_face_") and filename.endswith(".jpg"):
            try:
                os.remove(os.path.join(BASE_DIR, filename))
                logger.info(f"Arquivo temporário {filename} removido")
            except Exception as e:
                logger.error(f"Erro ao remover {filename}: {str(e)}")

    logger.info("Servidor encerrado")


if __name__ == '__main__':
    # Verificar caminho do banco de dados
    logger.info(f"Caminho absoluto do banco: {os.path.abspath(DATABASE_DIR)}")

    # Carregar banco de dados ao iniciar
    load_or_create_cache()

    # Iniciar servidor
    app.run(host='0.0.0.0', port=5005, debug=True)