import os
import logging
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
import threading
from datetime import datetime
from auth import token_required, login, init_auth_database
from facial_capture import FluidFaceCapture, check_user_exists, init_face_database

# Configurações
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
MAX_FILE_SIZE = 16 * 1024 * 1024

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

# ====================== ROTAS PRINCIPAIS ======================
app.route('/api/login', methods=['POST'])(login)


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_clients': len(connected_clients)
    })


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Servidor de Captura Facial',
        'status': 'online'
    })


# ====================== WEBSOCKETS ======================
@socketio.on('connect')
def on_connect():
    logger.info(f"Cliente conectado: {request.sid}")
    connected_clients[request.sid] = {
        'connect_time': datetime.now(),
        'status': 'connected'
    }
    emit("connected", {"status": "connected", "sid": request.sid})


@socketio.on('disconnect')
def on_disconnect():
    if request.sid in connected_clients:
        del connected_clients[request.sid]
    logger.info(f"Cliente desconectado: {request.sid}")


@socketio.on('start_camera')
def on_start_camera(data):
    """Inicia captura facial"""
    try:
        logger.info(f"Iniciando captura para SID: {request.sid}")

        # Validação básica
        nome = data.get("nome", "").strip()
        sobrenome = data.get("sobrenome", "").strip()
        turma = data.get("turma", "").strip()

        if not nome or not sobrenome or not turma:
            emit("capture_complete", {
                "success": False,
                "message": "Nome, sobrenome e turma são obrigatórios"
            })
            return

        # Verificar se usuário já existe
        if check_user_exists(nome, sobrenome, turma) > 0:
            emit("capture_complete", {
                "success": False,
                "message": "Usuário já cadastrado"
            })
            return

        join_room(request.sid)

        tipo = "professor" if str(data.get("tipoUsuario", "1")) == "2" else "aluno"

        connected_clients[request.sid] = {
            'start_time': datetime.now(),
            'status': 'capturing',
            'tipo': tipo
        }

        # Iniciar captura em thread
        thread = threading.Thread(
            target=run_face_capture,
            args=(nome, sobrenome, turma, tipo, request.sid),
            daemon=True
        )
        thread.start()

    except Exception as e:
        logger.error(f"Erro ao iniciar captura: {e}")
        emit("capture_complete", {
            "success": False,
            "message": f"Erro interno: {str(e)}"
        })


def run_face_capture(nome, sobrenome, turma, tipo, session_id):
    """Executa captura em thread separada"""
    try:
        def progress_callback(progress):
            socketio.emit('capture_progress', {
                'captured': progress['captured'],
                'total': progress['total'],
                'message': progress.get('message')
            }, room=session_id)

        def frame_callback(frame_data):
            socketio.emit('capture_frame', {'frame': frame_data}, room=session_id)

        capture = FluidFaceCapture(
            nome=nome,
            sobrenome=sobrenome,
            turma=turma,
            tipo=tipo,
            progress_callback=progress_callback,
            frame_callback=frame_callback
        )

        success, message = capture.capture()

        socketio.emit("capture_complete", {
            "success": success,
            "message": message,
            "captured_count": capture.captured_count
        }, room=session_id)

    except Exception as e:
        logger.error(f"Erro na captura: {e}")
        socketio.emit("capture_complete", {
            "success": False,
            "message": f"Erro na captura: {str(e)}"
        }, room=session_id)
    finally:
        if session_id in connected_clients:
            del connected_clients[session_id]


# ====================== INICIALIZAÇÃO ======================
def initialize_app():
    """Inicializa a aplicação"""
    try:
        init_auth_database()
        init_face_database()
        logger.info("✅ Aplicação inicializada com sucesso")
        return True
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")
        return False


if __name__ == "__main__":
    logger.info("🚀 Iniciando servidor de captura facial...")

    if initialize_app():
        socketio.run(
            app,
            host='0.0.0.0',
            port=7001,
            debug=True,
            allow_unsafe_werkzeug=True  # ✅ CORREÇÃO ADICIONADA AQUI
        )