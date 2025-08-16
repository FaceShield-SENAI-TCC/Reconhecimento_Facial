import os
import sys
import threading
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
from facial_capture import FaceCapture

# Configurar para evitar o erro do recarregamento
os.environ['WERKZEUG_RUN_MAIN'] = 'true'

app = Flask(__name__)
CORS(app)

# Solução para Windows: Usar async_mode 'threading'
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Dicionário para armazenar processos ativos
active_captures = {}


def check_camera():
    """Verifica se a câmera está acessível com múltiplas tentativas e índices"""
    import cv2
    logging.info("Verificando disponibilidade da câmera...")

    # Tenta diferentes índices de câmera
    for camera_index in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                logging.info(f"Câmera encontrada no índice {camera_index}")
                cap.release()
                return True
            else:
                logging.warning(f"Câmera no índice {camera_index} não abriu")
        except Exception as e:
            logging.warning(f"Erro ao acessar câmera {camera_index}: {str(e)}")

    # Se nenhuma câmera foi encontrada, tenta sem backend específico
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            logging.info("Câmera encontrada com backend padrão")
            cap.release()
            return True
        else:
            logging.warning("Câmera não abriu com backend padrão")
    except Exception as e:
        logging.error(f"Erro grave ao acessar câmera: {str(e)}")

    logging.error("Nenhuma câmera detectada após todas as tentativas")
    return False


@app.route('/start_capture', methods=['POST'])
def handle_start_capture():
    data = request.json
    name = data.get('name')
    session_id = data.get('session_id')

    if not name:
        return jsonify({"success": False, "error": "Nome é obrigatório"}), 400

    if session_id in active_captures:
        return jsonify({"success": False, "error": "Sessão já em andamento"}), 400

    # Função de callback para enviar progresso
    def progress_callback(progress):
        captured = progress.get('captured', 0)
        total = progress.get('total', 60)
        progress_percent = int((captured / total) * 100)
        socketio.emit('capture_progress', {
            'session_id': session_id,
            'progress': progress_percent,
            'message': progress.get('message', '')
        }, room=session_id)

    # Função de callback para enviar frames
    def frame_callback(frame_base64):
        socketio.emit('capture_frame', {
            'session_id': session_id,
            'frame': frame_base64
        }, room=session_id)

    # Criar instância de captura
    capture = FaceCapture(name, progress_callback, frame_callback)
    active_captures[session_id] = capture

    # Iniciar captura em thread separada
    def capture_thread():
        try:
            success, message = capture.capture()
            socketio.emit('capture_complete', {
                'session_id': session_id,
                'success': success,
                'message': message,
                'captured_count': capture.captured_count
            }, room=session_id)
        except Exception as e:
            logging.error(f"Erro na thread de captura: {str(e)}")
            socketio.emit('capture_complete', {
                'session_id': session_id,
                'success': False,
                'message': f"Erro: {str(e)}",
                'captured_count': 0
            }, room=session_id)
        finally:
            # Limpar após conclusão
            if session_id in active_captures:
                del active_captures[session_id]

    thread = threading.Thread(target=capture_thread)
    thread.daemon = True
    thread.start()

    return jsonify({
        "success": True,
        "message": "Captura iniciada",
        "session_id": session_id
    })


@app.route('/stop_capture', methods=['POST'])
def handle_stop_capture():
    data = request.json
    session_id = data.get('session_id')

    if session_id in active_captures:
        active_captures[session_id].stop()
        del active_captures[session_id]
        return jsonify({"success": True, "message": "Captura interrompida"})

    return jsonify({"success": False, "error": "Sessão não encontrada"}), 404


@socketio.on('connect')
def handle_connect(auth):
    session_id = request.args.get('session_id')
    if session_id:
        join_room(session_id)
        emit('connection_ack', {'status': 'connected', 'session_id': session_id})
    else:
        emit('connection_ack', {'status': 'error', 'message': 'session_id missing'})


@socketio.on('join')
def handle_join(data):
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        emit('connection_ack', {'status': 'connected', 'session_id': session_id})


def run_server():
    """Executa o servidor de forma compatível com Windows"""
    try:
        from waitress import serve
        logging.info("Iniciando servidor com Waitress (compatível com Windows)")
        serve(app, host="0.0.0.0", port=5000)
    except ImportError:
        logging.info("Waitress não instalado, usando servidor de desenvolvimento")
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True,
            use_reloader=False
        )


if __name__ == '__main__':
    camera_ok = check_camera()

    if not camera_ok:
        logging.error("Servidor não iniciado devido a problemas com a câmera")

        # Tentativa adicional como último recurso
        import platform

        if platform.system() == 'Windows':
            logging.info("Tentando solução alternativa para Windows...")
            os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
            camera_ok = check_camera()

        if not camera_ok:
            sys.exit(1)

    run_server()