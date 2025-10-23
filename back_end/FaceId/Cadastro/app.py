import os
import logging
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
import threading
from datetime import datetime
import cv2
import time
import numpy as np
import base64
import re
import psycopg2
from psycopg2.extras import Json
from deepface import DeepFace
from contextlib import contextmanager
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurações
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
MAX_FILE_SIZE = 16 * 1024 * 1024
SERVER_PORT = int(os.getenv("SERVER_PORT", "7001"))

# Configurações do banco via environment variables
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "faceshild"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

MIN_PHOTOS_REQUIRED = 8
MIN_FACE_SIZE = (100, 100)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=MAX_FILE_SIZE,
    ping_timeout=60,
    ping_interval=25,
    logger=True,
    engineio_logger=True
)

# Estado da aplicação
connected_clients = {}
active_captures = {}  # Controle de capturas ativas

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

def count_users():
    """Conta o número total de usuários cadastrados"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM usuarios")
            count = cur.fetchone()[0]
            return count
    except Exception as e:
        logger.error(f"Erro ao contar usuários: {e}")
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
    if sharpness < 70:  # Limite de nitidez
        return False, "Imagem muito borrada"

    return True, f"Qualidade: {sharpness:.1f}"

# ====================== DETECTOR FACIAL ======================
class FaceDetector:
    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 0.2  # 5 FPS para detecção
        self.cached_faces = []

    def detect_faces(self, frame):
        """Detecta rostos no frame com otimização"""
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.cached_faces

        try:
            small_frame = cv2.resize(frame, (320, 240))
            detected_faces = DeepFace.extract_faces(
                img_path=small_frame,
                detector_backend="opencv",
                enforce_detection=False,
                align=False
            )

            faces = []
            for face in detected_faces:
                if 'facial_area' in face:
                    x = int(face['facial_area']['x'] * frame.shape[1] / 320)
                    y = int(face['facial_area']['y'] * frame.shape[0] / 240)
                    w = int(face['facial_area']['w'] * frame.shape[1] / 320)
                    h = int(face['facial_area']['h'] * frame.shape[0] / 240)

                    # Validar tamanho mínimo e qualidade básica
                    face_roi = frame[y:y + h, x:x + w]
                    if (w >= MIN_FACE_SIZE[0] and h >= MIN_FACE_SIZE[1] and
                            face_roi.size > 0 and calculate_sharpness(face_roi) > 50):
                        faces.append((x, y, w, h))

            self.cached_faces = faces
            self.last_detection_time = current_time
            return faces

        except Exception as e:
            logger.warning(f"Erro na detecção: {e}")
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
        self.face_capture_interval = 0.5  # 2 faces por segundo no máximo
        self.last_face_detected_time = 0
        self.consecutive_no_face_count = 0

        # ✅ CONFIGURAÇÃO VGG-FACE
        self.model_name = "VGG-Face"
        self.embedding_dimension = 2622  # Dimensão do VGG-Face

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
                # Reduzir qualidade para transmissão
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
        """Gera embeddings das faces capturadas usando VGG-Face"""
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

                    # Salvar temporariamente para DeepFace
                    temp_path = f"temp_face_{int(time.time())}_{i}.jpg"
                    cv2.imwrite(temp_path, face_img)

                    # ✅ USAR VGG-FACE
                    embedding_obj = DeepFace.represent(
                        img_path=temp_path,
                        model_name=self.model_name,  # VGG-Face
                        enforce_detection=False,
                        detector_backend="skip"
                    )

                    embedding = np.array(embedding_obj[0]["embedding"])

                    # ✅ VALIDAR DIMENSÃO VGG-FACE
                    if len(embedding) != self.embedding_dimension:
                        logger.warning(
                            f"Embedding com dimensão incorreta: {len(embedding)} (esperado: {self.embedding_dimension})")
                        continue

                    embeddings.append(embedding)
                    successful += 1
                    logger.info(f"✅ Embedding {i + 1} gerado com sucesso")

                    # Limpar arquivo temporário
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    logger.warning(f"Erro no embedding {i + 1}: {e}")
                    continue

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
                # ✅ FECHAR QUALQUER CÂMERA ABERTA ANTES
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
        """Limpa recursos da câmera - IMPORTANTE PARA REUTILIZAÇÃO"""
        try:
            # Tentar liberar qualquer câmera que possa estar aberta
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
            time.sleep(0.5)  # Dar tempo para o sistema liberar
        except:
            pass

    def capture(self):
        """Método principal de captura - AGORA REUTILIZÁVEL"""
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
            no_face_timeout = 10

            while (self.running and
                   self.captured_count < MIN_PHOTOS_REQUIRED and
                   (time.time() - start_time) < 120):  # Timeout de 2 minutos

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
                    # Contagem de frames sem rosto
                    self.consecutive_no_face_count += 1
                    elapsed_no_face = time.time() - self.last_face_detected_time

                    if elapsed_no_face > no_face_timeout:
                        return False, "❌ Nenhum rosto detectado por 10 segundos. Posicione seu rosto na câmera e tente novamente."

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
                    return False, "❌ Nenhum rosto detectado por 10 segundos. Posicione seu rosto na câmera e tente novamente."
                else:
                    return False, f"Captura incompleta: {self.captured_count}/{MIN_PHOTOS_REQUIRED}"

        except Exception as e:
            logger.error(f"Erro na captura: {str(e)}")
            return False, f"Erro na captura: {str(e)}"
        finally:
            # ✅ SEMPRE LIBERAR RECURSOS DA CÂMERA
            if cap and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            # ✅ LIMPAR CACHE DO DETECTOR PARA PRÓXIMA CAPTURA
            self.detector.cached_faces = []
            self.detector.last_detection_time = 0
            logger.info("✅ Recursos da câmera liberados para próxima captura")

    def stop(self):
        """Para a captura de forma segura"""
        self.running = False
        logger.info("⏹️ Captura interrompida")

# ====================== ROTAS HTTP ======================
@app.route('/api/login', methods=['POST'])
def login():
    """Endpoint de login para administração do sistema"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Dados de login necessários'}), 400

        username = data.get('username', '').strip()
        password = data.get('password', '')

        # Credenciais fixas para demonstração
        if username == 'admin' and password == 'admin123':
            return jsonify({
                'success': True,
                'message': 'Login realizado com sucesso',
                'user': {'username': 'admin', 'role': 'admin'}
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Credenciais inválidas'
            }), 401

    except Exception as e:
        logger.error(f"Erro no login: {str(e)}")
        return jsonify({'error': 'Erro interno no servidor'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check do servidor"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_clients': len(connected_clients),
        'active_captures': len(active_captures),
        'service': 'facial_capture_api',
        'port': SERVER_PORT
    }), 200

@app.route('/api/database/status', methods=['GET'])
def database_status():
    """Status do banco de dados"""
    try:
        user_count = count_users()
        return jsonify({
            'status': 'connected',
            'user_count': user_count,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Erro ao verificar status do banco: {str(e)}")
        return jsonify({'error': 'Erro ao conectar com o banco'}), 500

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Informações do sistema"""
    return jsonify({
        'service': 'facial_capture_system',
        'version': '2.0.0',
        'status': 'operational',
        'model': 'VGG-Face',
        'embedding_dimension': 2622,
        'min_photos_required': 8,
        'circuito_continuo': True,
        'timestamp': datetime.now().isoformat(),
        'recognition_compatibility': 'FULL'
    }), 200

@app.route('/', methods=['GET'])
def index():
    """Página inicial"""
    return jsonify({
        'message': 'Servidor de Captura Facial - Sistema de Cadastro Biométrico',
        'status': 'online',
        'version': '2.0.0',
        'circuito_continuo': True,
        'port': SERVER_PORT,
        'endpoints': {
            'websocket': '/socket.io/',
            'health': '/api/health',
            'system_info': '/api/system/info',
            'database_status': '/api/database/status'
        }
    })

# ====================== WEBSOCKETS ======================
@socketio.on('connect')
def on_connect():
    """Cliente conectado via WebSocket"""
    logger.info(f"✅ Cliente conectado: {request.sid}")
    connected_clients[request.sid] = {
        'connect_time': datetime.now(),
        'status': 'connected',
        'ip_address': request.environ.get('REMOTE_ADDR', 'unknown')
    }
    emit("connected", {
        "status": "connected",
        "sid": request.sid,
        "message": "Conectado ao servidor de captura facial",
        "compatibility": "FULL"
    })

@socketio.on('disconnect')
def on_disconnect():
    """Cliente desconectado"""
    if request.sid in active_captures:
        capture = active_captures[request.sid]
        if capture:
            capture.stop()
            logger.info(f"⏹️ Captura interrompida para cliente desconectado: {request.sid}")
        del active_captures[request.sid]

    if request.sid in connected_clients:
        del connected_clients[request.sid]

    logger.info(f"❌ Cliente desconectado: {request.sid}")

@socketio.on('start_camera')
def on_start_camera(data):
    """Inicia captura facial para cadastro biométrico"""
    try:
        logger.info(f"🎬 Iniciando captura para SID: {request.sid}")
        logger.info(f"📋 Dados recebidos: {data}")

        # Validação dos dados
        nome = data.get("nome", "").strip()
        sobrenome = data.get("sobrenome", "").strip()
        turma = data.get("turma", "").strip()

        if not nome or not sobrenome or not turma:
            error_msg = "Nome, sobrenome e turma são obrigatórios"
            logger.warning(f"❌ {error_msg} para SID: {request.sid}")
            emit("capture_complete", {
                "success": False,
                "message": error_msg
            })
            return

        # Verificar se já existe captura ativa para este cliente
        if request.sid in active_captures:
            error_msg = "Já existe uma captura em andamento para este cliente"
            logger.warning(f"❌ {error_msg}")
            emit("capture_complete", {
                "success": False,
                "message": error_msg
            })
            return

        # Verificar se usuário já está cadastrado
        existing_count = check_user_exists(nome, sobrenome, turma)
        if existing_count > 0:
            error_msg = f"Usuário {nome} {sobrenome} já está cadastrado na turma {turma}"
            logger.warning(f"❌ {error_msg}")
            emit("capture_complete", {
                "success": False,
                "message": error_msg,
                "user_exists": True
            })
            return

        # Determinar tipo de usuário
        tipo_usuario = data.get("tipoUsuario", "1")
        tipo = "professor" if str(tipo_usuario) == "2" else "aluno"

        # Registrar cliente na sala
        join_room(request.sid)

        # Atualizar informações do cliente
        connected_clients[request.sid].update({
            'status': 'capturing',
            'start_time': datetime.now(),
            'user_data': {
                'nome': nome,
                'sobrenome': sobrenome,
                'turma': turma,
                'tipo': tipo
            }
        })

        # Registrar captura ativa (inicialmente None)
        active_captures[request.sid] = None

        logger.info(f"🚀 Iniciando captura para: {nome} {sobrenome} - {turma} ({tipo})")

        # Iniciar captura em thread separada
        thread = threading.Thread(
            target=run_face_capture,
            args=(nome, sobrenome, turma, tipo, request.sid),
            daemon=True
        )
        thread.start()

    except Exception as e:
        logger.error(f"❌ Erro ao iniciar captura: {str(e)}", exc_info=True)

        if request.sid in active_captures:
            del active_captures[request.sid]

        emit("capture_complete", {
            "success": False,
            "message": f"Erro interno ao iniciar captura: {str(e)}"
        })

@socketio.on('stop_camera')
def on_stop_camera():
    """Para a captura ativa do cliente"""
    try:
        if request.sid in active_captures:
            capture = active_captures[request.sid]
            if capture:
                capture.stop()
                logger.info(f"⏹️ Captura interrompida manualmente para SID: {request.sid}")
            del active_captures[request.sid]
            emit("capture_stopped", {
                "success": True,
                "message": "Captura interrompida com sucesso"
            })
        else:
            emit("capture_stopped", {
                "success": False,
                "message": "Nenhuma captura ativa para interromper"
            })
    except Exception as e:
        logger.error(f"❌ Erro ao parar captura: {str(e)}")
        emit("capture_stopped", {
            "success": False,
            "message": f"Erro ao interromper captura: {str(e)}"
        })

def run_face_capture(nome, sobrenome, turma, tipo, session_id):
    """Executa o processo de captura facial em thread separada"""
    try:
        logger.info(f"📷 Iniciando thread de captura para sessão: {session_id}")

        def progress_callback(progress):
            """Callback para atualizações de progresso"""
            try:
                socketio.emit('capture_progress', {
                    'captured': progress['captured'],
                    'total': progress['total'],
                    'message': progress.get('message', ''),
                    'session_id': session_id
                }, room=session_id)
            except Exception as e:
                logger.error(f"Erro no callback de progresso: {str(e)}")

        def frame_callback(frame_data):
            """Callback para envio de frames"""
            try:
                socketio.emit('capture_frame', {
                    'frame': frame_data,
                    'session_id': session_id
                }, room=session_id)
            except Exception as e:
                logger.error(f"Erro no callback de frame: {str(e)}")

        # Criar instância do capturador facial
        capture = FluidFaceCapture(
            nome=nome,
            sobrenome=sobrenome,
            turma=turma,
            tipo=tipo,
            progress_callback=progress_callback,
            frame_callback=frame_callback
        )

        # Registrar instância ativa
        active_captures[session_id] = capture

        # Executar captura
        success, message = capture.capture()

        # Enviar resultado final
        result_data = {
            "success": success,
            "message": message,
            "captured_count": capture.captured_count,
            "user": f"{nome} {sobrenome}",
            "turma": turma,
            "tipo": tipo,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "compatibility": "FULL"
        }

        socketio.emit("capture_complete", result_data, room=session_id)

        if success:
            logger.info(f"✅ Captura concluída com sucesso: {nome} {sobrenome}")
        else:
            logger.warning(f"⚠️ Captura falhou: {message}")

    except Exception as e:
        logger.error(f"❌ Erro na thread de captura: {str(e)}", exc_info=True)

        socketio.emit("capture_complete", {
            "success": False,
            "message": f"Erro durante a captura: {str(e)}",
            "captured_count": 0,
            "session_id": session_id
        }, room=session_id)

    finally:
        if session_id in active_captures:
            del active_captures[session_id]
            logger.info(f"🧹 Captura finalizada e limpa para sessão: {session_id}")

# ====================== INICIALIZAÇÃO ======================
def initialize_application():
    """Inicializa a aplicação e serviços"""
    logger.info("🚀 Inicializando Servidor de Captura Facial...")

    try:
        if init_face_database():
            logger.info("✅ Banco de dados inicializado com sucesso")
        else:
            logger.error("❌ Falha na inicialização do banco de dados")
            return False

        logger.info("🎯 Sistema configurado com as seguintes características:")
        logger.info(f"   📍 Porta: {SERVER_PORT}")
        logger.info("   🔧 Modelo: VGG-Face (2622 dimensões)")
        logger.info("   📸 Fotos necessárias: 8 por usuário")
        logger.info("   🌐 WebSocket: Ativo")
        logger.info("   💾 Banco: PostgreSQL")
        logger.info("   🔄 Compatibilidade: FULL com reconhecimento facial")

        return True

    except Exception as e:
        logger.error(f"❌ Falha crítica na inicialização: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🟢 INICIANDO SISTEMA DE CAPTURA FACIAL")
    logger.info("=" * 60)

    if initialize_application():
        try:
            socketio.run(
                app,
                host='0.0.0.0',
                port=SERVER_PORT,
                debug=False,
                allow_unsafe_werkzeug=True
            )
        except KeyboardInterrupt:
            logger.info("🛑 Servidor interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro durante execução do servidor: {str(e)}")
    else:
        logger.critical("💥 Falha na inicialização - Encerrando aplicação")
        exit(1)