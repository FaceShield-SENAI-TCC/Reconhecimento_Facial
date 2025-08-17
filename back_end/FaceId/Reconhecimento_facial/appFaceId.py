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

# ====================== CONFIGURAÇÕES ======================
DATABASE_DIR = "../Cadastro/facial_database"
MODEL_NAME = "VGG-Face"
DISTANCE_THRESHOLD = 0.40
MIN_FACE_SIZE = (100, 100)
MAX_IMAGE_SIZE = (1280, 720)  # Tamanho máximo para redimensionamento

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas as rotas

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


# ====================== FUNÇÕES AUXILIARES ======================
def log_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2  # Em MB
        logger.info(f"Uso de memória: {mem:.2f} MB")
    except ImportError:
        pass


def resize_image(image, max_size):
    """Redimensiona a imagem mantendo a proporção"""
    height, width = image.shape[:2]

    if width > max_size[0] or height > max_size[1]:
        scale = min(max_size[0] / width, max_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image


# ====================== CARREGAR BANCO DE DADOS FACIAL ======================
def load_facial_database():
    global facial_db
    logger.info("Iniciando carregamento do banco de dados facial...")
    logger.info(f"Caminho absoluto do banco: {os.path.abspath(DATABASE_DIR)}")

    database = {}
    user_count = 0
    embedding_count = 0

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
                        embedding_obj = DeepFace.represent(
                            img_path=face_path,
                            model_name=MODEL_NAME,
                            detector_backend="opencv",
                            enforce_detection=False
                        )

                        # Verificação robusta da estrutura de retorno
                        if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                            embedding_data = embedding_obj[0]
                            if "embedding" in embedding_data:
                                embedding = np.array(embedding_data["embedding"])
                                norm = np.linalg.norm(embedding)

                                if norm > 1e-8:  # Evita divisão por zero
                                    normalized_embedding = embedding / norm
                                    embeddings.append(normalized_embedding)
                                    embedding_count += 1
                                else:
                                    logger.warning(f"⚠️ Vetor de embedding inválido (norma zero) em: {face_path}")
                            else:
                                logger.warning(f"⚠️ Chave 'embedding' ausente em: {face_path}")
                        else:
                            logger.warning(f"⚠️ Nenhum rosto detectado em: {face_path}")

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
    facial_db = database
    return database


# ====================== FUNÇÃO DE RECONHECIMENTO ======================
def recognize_face_from_image(face_img):
    global facial_db
    try:
        logger.info("Iniciando reconhecimento facial...")
        start_time = time.time()

        embedding_obj = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=False
        )

        if not isinstance(embedding_obj, list) or len(embedding_obj) == 0:
            logger.warning("Nenhum embedding retornado pelo DeepFace")
            return None, None

        embedding_data = embedding_obj[0]
        if "embedding" not in embedding_data:
            logger.warning("Chave 'embedding' não encontrada no objeto retornado")
            return None, None

        captured_embedding = np.array(embedding_data["embedding"])
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
@app.route('/face-login', methods=['POST'])
def face_login():
    global facial_db
    logger.info("Recebendo solicitação de login facial")

    # Carregar banco de dados se necessário
    if facial_db is None:
        logger.info("Carregando banco de dados facial...")
        if load_facial_database() is None:
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

        # Detectar rosto na imagem (abordagem compatível com versões do DeepFace)
        logger.info("Detectando rostos...")
        faces = []

        try:
            # Tentativa 1: Usar o detector do DeepFace
            from deepface.detectors import DetectorWrapper
            logger.info("Usando detector do DeepFace (OpenCV backend)...")
            detected_faces = DetectorWrapper.detect_faces('opencv', frame)

            # Converter para formato padronizado
            for face in detected_faces:
                x, y, w, h = face
                faces.append({
                    'facial_area': {
                        'x': int(x),
                        'y': int(y),
                        'w': int(w),
                        'h': int(h)
                    }
                })

        except Exception as e:
            logger.warning(f"Detector do DeepFace falhou: {str(e)}")
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
        faces = sorted(faces, key=lambda f: f['facial_area']['w'] * f['facial_area']['h'], reverse=True)
        face = faces[0]
        x = face['facial_area']['x']
        y = face['facial_area']['y']
        w = face['facial_area']['w']
        h = face['facial_area']['h']

        # Verificar tamanho mínimo do rosto
        if w < MIN_FACE_SIZE[0] or h < MIN_FACE_SIZE[1]:
            logger.warning(f"Rosto muito pequeno: {w}x{h}")
            return jsonify({
                "authenticated": False,
                "user": None,
                "message": "Rosto muito pequeno. Aproxime-se da câmera."
            })

        face_img = frame[y:y + h, x:x + w]

        # Salvar temporariamente para processamento
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_img)
        logger.info(f"Rosto detectado salvo temporariamente em {temp_path} (tamanho: {w}x{h})")

        # Reconhecer rosto
        user, distance = recognize_face_from_image(temp_path)
        os.remove(temp_path)  # Limpar arquivo temporário

        if user:
            confidence = 1 - distance
            logger.info(f"Autenticação bem-sucedida para {user} com confiança {confidence:.2f}")
            return jsonify({
                "authenticated": True,
                "user": user,
                "confidence": float(confidence)
            })
        else:
            logger.info("Usuário não reconhecido")
            return jsonify({
                "authenticated": False,
                "user": None,
                "message": "Usuário não reconhecido. Tente novamente ou faça cadastro."
            })

    except Exception as e:
        logger.error(f"🚨 Erro no endpoint /face-login: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


@app.route('/test-db', methods=['GET'])
def test_db():
    global facial_db
    if facial_db is None:
        return jsonify({"status": "Database not loaded"})

    return jsonify({
        "status": "Database loaded",
        "users": list(facial_db.keys()),
        "user_count": len(facial_db),
        "total_embeddings": sum(len(emb) for emb in facial_db.values())
    })


@app.route('/')
def home():
    return send_from_directory('templates', 'login.html')


# ====================== LIMPEZA E INICIALIZAÇÃO ======================
@atexit.register
def cleanup():
    temp_path = "temp_face.jpg"
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            logger.info(f"Arquivo temporário {temp_path} removido")
        except Exception as e:
            logger.error(f"Erro ao remover {temp_path}: {str(e)}")

    logger.info("Servidor encerrado")


if __name__ == '__main__':
    # Verificar caminho do banco de dados
    logger.info(f"Caminho absoluto do banco: {os.path.abspath(DATABASE_DIR)}")

    # Carregar banco de dados ao iniciar
    load_facial_database()

    # Iniciar servidor
    app.run(host='0.0.0.0', port=5000, debug=True)