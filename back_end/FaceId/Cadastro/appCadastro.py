import os
import cv2
from deepface import DeepFace
import numpy as np
import base64
import time
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
import threading
import re
import json
from functools import wraps
import jwt
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

# ====================== CONFIGURAÇÕES ======================
CAPTURE_BASE_DIR = "../Cadastro/Faces"
os.makedirs(CAPTURE_BASE_DIR, exist_ok=True)

MIN_PHOTOS_REQUIRED = 10
face_capture_state = {}
connected_clients = {}  # Para controlar clientes conectados

# Configurações de autenticação
SECRET_KEY = "sua_chave_secreta_super_segura_aqui_altere_para_uma_chave_real"
TOKEN_EXPIRATION_HOURS = 24
DATABASE_PATH = "facial_auth.db"

# Modelo para embeddings
EMBEDDING_MODEL = "Facenet"

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

# Configuração CORS mais permissiva para desenvolvimento
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuração do Socket.IO com CORS mais permissivo
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    async_mode='threading'
)


# ====================== FUNÇÕES DE BANCO DE DADOS ======================
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_db_connection()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            nome TEXT NOT NULL,
            sobrenome TEXT NOT NULL,
            turma TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if c.fetchone()[0] == 0:
        password_hash = generate_password_hash("admin123")
        c.execute(
            "INSERT INTO users (username, password_hash, nome, sobrenome, turma) VALUES (?, ?, ?, ?, ?)",
            ('admin', password_hash, 'Administrador', 'Sistema', 'admin')
        )

    conn.commit()
    conn.close()


# Inicializar banco de dados
init_database()


# ====================== FUNÇÕES DE AUTENTICAÇÃO ======================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token está faltando!'}), 401

        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user_id = data['sub']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expirado!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token inválido!'}), 401

        return f(*args, **kwargs)

    return decorated


def generate_token(user_id):
    payload = {
        'exp': datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS),
        'iat': datetime.utcnow(),
        'sub': user_id
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')


# ====================== FUNÇÕES AUXILIARES ======================
def sanitize_name(name):
    name = str(name).lower().strip()
    return re.sub(r'[^a-z0-9_]', '', name.replace(' ', '_'))


def generate_and_store_embedding(face_image, user_dir, user_id, image_index):
    """Gera e armazena embedding facial"""
    try:
        # Gerar embedding
        embedding_obj = DeepFace.represent(
            img_path=face_image,
            model_name=EMBEDDING_MODEL,
            enforce_detection=False,
            detector_backend="skip"
        )

        embedding = np.array(embedding_obj[0]["embedding"])

        # Salvar embedding em arquivo
        embedding_path = os.path.join(user_dir, f"{user_id}_embedding_{image_index}.npy")
        np.save(embedding_path, embedding)

        return embedding
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return None


# ====================== ROTAS DE AUTENTICAÇÃO ======================
@app.route('/api/login', methods=['POST'])
def login():
    auth_data = request.get_json()
    username = auth_data.get('username')
    password = auth_data.get('password')

    if not username or not password:
        return jsonify({'message': 'Usuário e senha são obrigatórios!'}), 400

    conn = get_db_connection()
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        token = generate_token(user['id'])
        return jsonify({
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'nome': user['nome'],
                'sobrenome': user['sobrenome'],
                'turma': user['turma']
            }
        })

    return jsonify({'message': 'Credenciais inválidas!'}), 401


# ====================== ROTAS PROTEGIDAS ======================
@app.route('/api/users', methods=['GET'])
@token_required
def get_users():
    conn = get_db_connection()
    c = conn.cursor()

    c.execute("SELECT id, username, nome, sobrenome, turma FROM users")
    users = [dict(row) for row in c.fetchall()]

    conn.close()
    return jsonify(users)


# ====================== CAPTURA DE FACES OTIMIZADA ======================
def capture_frames(nome, sobrenome, turma, session_id):
    global face_capture_state

    face_capture_state[session_id] = {
        "thread_running": True,
        "captured_faces": [],
        "captured_count": 0,
        "success": False,
        "message": ""
    }

    cap = None
    for i in range(3):
        print(f"[{session_id}] Tentando abrir a câmera com índice: {i}")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[{session_id}] Câmera aberta com sucesso no índice: {i}")
            # Configurar resolução reduzida para melhor performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Reduzir FPS para melhor performance
            break
        else:
            print(f"[{session_id}] Falha ao abrir a câmera no índice: {i}")
            if i == 2:
                face_capture_state[session_id][
                    "message"] = "Não foi possível acessar a câmera em nenhum índice (0, 1, 2). Verifique se ela está conectada e disponível."
                print(f"[{session_id}] ERRO CRÍTICO: {face_capture_state[session_id]['message']}")
                socketio.emit("capture_complete", {
                    "success": False,
                    "message": face_capture_state[session_id]["message"],
                    "captured_count": 0,
                    "total_to_capture": MIN_PHOTOS_REQUIRED
                }, room=session_id)
                return

    try:
        if not cap or not cap.isOpened():
            face_capture_state[session_id]["message"] = "Não foi possível acessar a câmera após várias tentativas."
            print(f"[{session_id}] ERRO: {face_capture_state[session_id]['message']}")
            return

        sanitized_nome = sanitize_name(nome)
        sanitized_sobrenome = sanitize_name(sobrenome)
        sanitized_turma = sanitize_name(turma)
        user_id = f"{sanitized_nome}_{sanitized_sobrenome}_{sanitized_turma}"

        turma_dir = os.path.join(CAPTURE_BASE_DIR, sanitized_turma)
        user_dir = os.path.join(turma_dir, f"{sanitized_nome}_{sanitized_sobrenome}")
        os.makedirs(user_dir, exist_ok=True)

        # Verificar se usuário já existe e contar fotos existentes
        existing_photos = [f for f in os.listdir(user_dir) if f.endswith('.jpg')]
        existing_embeddings = [f for f in os.listdir(user_dir) if f.endswith('.npy')]
        current_photos = len(existing_photos)

        # Se já tem fotos suficientes, não permitir novo cadastro
        if current_photos >= MIN_PHOTOS_REQUIRED:
            face_capture_state[session_id]["success"] = False
            face_capture_state[session_id][
                "message"] = f"Usuário já cadastrado com {current_photos} fotos. Não é possível adicionar mais."
            print(f"[{session_id}] {face_capture_state[session_id]['message']}")

            response_data = {
                "success": face_capture_state[session_id]["success"],
                "message": face_capture_state[session_id]["message"],
                "captured_count": current_photos,
                "total_to_capture": MIN_PHOTOS_REQUIRED,
                "user_id": user_id
            }
            socketio.emit("capture_complete", response_data, room=session_id)
            return

        print(f"[{session_id}] Diretório de usuário para fotos: {user_dir}")
        print(
            f"[{session_id}] Fotos existentes: {current_photos}, Fotos necessárias: {MIN_PHOTOS_REQUIRED - current_photos}")

        start_time = time.time()
        last_face_time = 0
        face_capture_interval = 0.5
        QUALITY_THRESHOLD = 100.0
        frame_skip = 2  # Processar apenas 1 a cada 3 frames para melhor performance
        frame_counter = 0

        # Pré-carregar o detector facial para melhor performance
        face_detector = DeepFace.build_model("OpenFace")

        while (face_capture_state[session_id]["captured_count"] < (MIN_PHOTOS_REQUIRED - current_photos) and
               (time.time() - start_time) < 60 and
               session_id in connected_clients):

            if not face_capture_state[session_id]["thread_running"]:
                print(f"[{session_id}] Thread de captura interrompida por solicitação externa.")
                break

            ret, frame = cap.read()
            if not ret:
                face_capture_state[session_id]["message"] = "Câmera parou de enviar frames inesperadamente."
                print(f"[{session_id}] ERRO: {face_capture_state[session_id]['message']}")
                break

            # Corrigir orientação da câmera (flip vertical)
            frame = cv2.flip(frame, 1)
            frame_display = frame.copy()

            # Pular frames para melhorar performance
            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            try:
                # Usar detector mais rápido (SSD) para melhor performance
                detected_faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="ssd",  # Mais rápido que opencv
                    enforce_detection=False
                )

                if len(detected_faces) == 1:
                    face = detected_faces[0]
                    if 'facial_area' in face:
                        x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], \
                            face['facial_area']['h']
                        cropped_face = frame[y:y + h, x:x + w]

                        if cropped_face.size > 0:
                            gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                            quality_score = cv2.Laplacian(gray, cv2.CV_64F).var()

                            if quality_score > QUALITY_THRESHOLD and (
                                    time.time() - last_face_time) > face_capture_interval:
                                face_capture_state[session_id]["captured_faces"].append(cropped_face)
                                face_capture_state[session_id]["captured_count"] += 1
                                last_face_time = time.time()

                                # Gerar embedding em uma thread separada para não bloquear a captura
                                threading.Thread(
                                    target=generate_and_store_embedding,
                                    args=(cropped_face, user_dir, user_id,
                                          current_photos + face_capture_state[session_id]["captured_count"])
                                ).start()

                                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                print(
                                    f"[{session_id}] Rosto e qualidade OK! Foto capturada: {face_capture_state[session_id]['captured_count'] + current_photos} de {MIN_PHOTOS_REQUIRED}")
                            else:
                                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 0, 255), 3)

            except Exception as e:
                print(f"Erro na detecção facial: {e}")
                pass

            # Otimização: reduzir qualidade do frame para transmissão mais rápida
            frame_resized = cv2.resize(frame_display, (320, 240))  # Reduzir ainda mais a resolução
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 30])  # Reduzir qualidade
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Emitir frame apenas se o cliente ainda estiver conectado
            if session_id in connected_clients:
                socketio.emit('capture_frame', {'frame': jpg_as_text}, room=session_id)

            socketio.emit('capture_progress', {
                'captured': face_capture_state[session_id]["captured_count"] + current_photos,
                'total': MIN_PHOTOS_REQUIRED,
                'session_id': session_id
            }, room=session_id)

            time.sleep(0.03)  # Reduzir sleep para melhor responsividade

        total_captured = current_photos + face_capture_state[session_id]["captured_count"]

        if total_captured >= MIN_PHOTOS_REQUIRED:
            for i, face_img in enumerate(face_capture_state[session_id]["captured_faces"]):
                filename = f"{sanitized_nome}_{sanitized_sobrenome}_{current_photos + i + 1}.jpg"
                filepath = os.path.join(user_dir, filename)
                cv2.imwrite(filepath, face_img)

            face_capture_state[session_id]["success"] = True
            face_capture_state[session_id][
                "message"] = f"Captura completa e salva com sucesso. {total_captured} fotos salvas."
            print(f"[{session_id}] {face_capture_state[session_id]['message']}")
        else:
            face_capture_state[session_id]["success"] = False
            if not face_capture_state[session_id]["message"]:
                face_capture_state[session_id][
                    "message"] = f"Captura falhou. Capturadas apenas {face_capture_state[session_id]['captured_count']} novas fotos. Total: {total_captured}/{MIN_PHOTOS_REQUIRED}."
            print(f"[{session_id}] ERRO: {face_capture_state[session_id]['message']}")

    except Exception as e:
        face_capture_state[session_id]["success"] = False
        face_capture_state[session_id]["message"] = f"Erro inesperado durante a captura: {e}"
        print(f"[{session_id}] ERRO CRÍTICO INESPERADO: {face_capture_state[session_id]['message']}")

    finally:
        if cap and cap.isOpened():
            print(f"[{session_id}] Liberando recursos da câmera.")
            cap.release()
            cv2.destroyAllWindows()

        response_data = {
            "success": face_capture_state[session_id]["success"],
            "message": face_capture_state[session_id]["message"],
            "captured_count": total_captured,
            "total_to_capture": MIN_PHOTOS_REQUIRED,
            "user_id": user_id
        }
        socketio.emit("capture_complete", response_data, room=session_id)
        face_capture_state[session_id]["thread_running"] = False
        print(f"[{session_id}] Fim da thread de captura. Resultado: {response_data['success']}")


# ====================== WEBSOCKETS ======================
@socketio.on('start_camera')
def on_start_camera(data):
    nome = data.get("nome")
    sobrenome = data.get("sobrenome")
    turma = data.get("turma")
    session_id = data.get("session_id")

    if not all([nome, sobrenome, turma, session_id]):
        print("Erro: Dados de início de captura ausentes.")
        socketio.emit("capture_complete",
                      {"success": False, "message": "Dados do formulário incompletos para iniciar a captura."},
                      room=request.sid)
        return

    join_room(request.sid)
    connected_clients[request.sid] = True

    print(f"Recebido pedido para iniciar a câmera. Session_ID da Captura: {session_id}, SID da Conexão: {request.sid}")

    thread = threading.Thread(target=capture_frames, args=(nome, sobrenome, turma, request.sid))
    thread.daemon = True
    thread.start()


@socketio.on("connect")
def on_connect(auth):
    session_id = request.args.get("session_id")
    connected_clients[request.sid] = True
    if session_id:
        print(f"Cliente conectado com session_id: {session_id}, SID da Conexão: {request.sid}")
    else:
        print(f"Cliente conectado sem session_id. SID da Conexão: {request.sid}")


@socketio.on("disconnect")
def on_disconnect():
    session_id = request.args.get("session_id")
    if request.sid in connected_clients:
        del connected_clients[request.sid]
    if session_id in face_capture_state:
        face_capture_state[session_id]["thread_running"] = False
        print(f"Cliente com session_id {session_id} desconectado. SID da Conexão: {request.sid}")
    else:
        print(f"Cliente desconectado: {request.sid}")


# ====================== ROTA DE VERIFICAÇÃO ======================
@app.route('/api/verify', methods=['GET'])
def verify_system():
    """Rota para verificar se o sistema está funcionando"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == "__main__":
    print("Iniciando servidor de cadastro facial...")
    print("Servidor Socket.IO rodando na porta 7001")
    print("Permitindo conexões de qualquer origem (CORS)")
    socketio.run(app, host='0.0.0.0', port=7001, allow_unsafe_werkzeug=True, debug=False)