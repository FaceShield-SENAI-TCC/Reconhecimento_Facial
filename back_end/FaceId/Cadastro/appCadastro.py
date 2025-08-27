import os
import cv2
from deepface import DeepFace
import numpy as np
import base64
import time
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading

# Configuração da captura facial
# AJUSTE O CAMINHO PARA O SEU USUÁRIO
CAPTURE_DIR = "C:/Users/Aluno/Desktop/Arduino/back_end/FaceId/Cadastro/Faces"
os.makedirs(CAPTURE_DIR, exist_ok=True)

TOTAL_FRAMES = 100
MAX_FACES = 25
last_frame_time = 0
FRAME_DELAY = 0.1  # 100ms
face_capture_state = {}

app = Flask(__name__)

# Configuração CORS para permitir o Live Server e o seu servidor local
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:7001",
    "http://127.0.0.1:7001",
    "http://127.0.0.1:5500",
    "http://localhost:8001",
    "http://127.0.0.1:8001"
]}})

# Configuração do Socket.IO com CORS
socketio = SocketIO(
    app,
    cors_allowed_origins=[
        "http://localhost:7001",
        "http://127.0.0.1:7001",
        "http://127.0.0.1:5500",
        "http://localhost:8001",
        "http://127.0.0.1:8001"
    ],
)


def capture_frames(name, session_id):
    """
    Captura frames da câmera, detecta rostos e envia o progresso via Socket.IO.
    """
    global face_capture_state

    face_capture_state[session_id] = {
        "thread_running": True,
        "captured_frames": [],
        "captured_count": 0,
        "total_to_capture": TOTAL_FRAMES,
        "success": False,
        "message": ""
    }

    # Simplifica a inicialização da câmera para o método padrão
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        face_capture_state[session_id]["success"] = False
        face_capture_state[session_id]["message"] = "Não foi possível acessar a câmera."
        socketio.emit("capture_complete", {"success": False, "message": "Não foi possível acessar a câmera."},
                      room=session_id)
        return

    face_count = 0
    start_time = time.time()

    while face_count < TOTAL_FRAMES and (time.time() - start_time) < 30:
        if not face_capture_state[session_id]["thread_running"]:
            break

        ret, frame = cap.read()
        if not ret:
            break

        faces_found = []
        try:
            detected_faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False
            )
            for face in detected_faces:
                if 'facial_area' in face:
                    faces_found.append(face['facial_area'])
        except Exception as e:
            print(f"Erro na detecção de rosto com DeepFace: {e}")
            faces_found = []

        frame_copy = frame.copy()
        for face_area in faces_found:
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('capture_frame', {'frame': jpg_as_text, 'session_id': session_id}, room=session_id)

        if len(faces_found) == 1:
            face_count += 1
            face_capture_state[session_id]["captured_frames"].append(frame)
            face_capture_state[session_id]["captured_count"] = face_count
            socketio.emit('capture_progress', {
                'captured': face_count,
                'total': TOTAL_FRAMES,
                'session_id': session_id
            }, room=session_id)
        else:
            face_count = max(0, face_count - 1)
            face_capture_state[session_id]["captured_count"] = face_count
            face_capture_state[session_id]["captured_frames"] = []  # Reinicia a contagem
            socketio.emit('capture_progress', {
                'captured': face_count,
                'total': TOTAL_FRAMES,
                'session_id': session_id
            }, room=session_id)

        time.sleep(FRAME_DELAY)

    cap.release()
    cv2.destroyAllWindows()

    if len(face_capture_state[session_id]["captured_frames"]) >= TOTAL_FRAMES:
        user_dir = os.path.join(CAPTURE_DIR, name)
        os.makedirs(user_dir, exist_ok=True)
        for i, frame in enumerate(face_capture_state[session_id]["captured_frames"]):
            file_path = os.path.join(user_dir, f"frame_{i}.jpg")
            cv2.imwrite(file_path, frame)
        face_capture_state[session_id]["success"] = True
        face_capture_state[session_id]["message"] = "Captura completa e salva com sucesso."
    else:
        face_capture_state[session_id]["success"] = False
        face_capture_state[session_id][
            "message"] = "Captura falhou. Não foi possível detectar um rosto único e consistente."

    # Crie um dicionário para enviar, sem o ndarray
    response_data = {
        "success": face_capture_state[session_id]["success"],
        "message": face_capture_state[session_id]["message"],
        "captured_count": face_capture_state[session_id]["captured_count"],
        "total_to_capture": face_capture_state[session_id]["total_to_capture"]
    }

    socketio.emit("capture_complete", response_data, room=session_id)
    face_capture_state[session_id]["thread_running"] = False


@app.route("/start_capture", methods=["POST"])
def start_capture():
    data = request.json
    name = data.get("name")
    session_id = data.get("session_id")

    if not name or not session_id:
        return jsonify({"success": False, "error": "Nome e session_id são obrigatórios"}), 400

    thread = threading.Thread(target=capture_frames, args=(name, session_id))
    thread.start()
    return jsonify({"success": True}), 200


@socketio.on("connect")
def on_connect():
    session_id = request.args.get("session_id")
    if session_id:
        print(f"Cliente conectado com session_id: {session_id}")
        socketio.join_room(session_id)


@socketio.on("disconnect")
def on_disconnect():
    session_id = request.args.get("session_id")
    if session_id in face_capture_state:
        face_capture_state[session_id]["thread_running"] = False
    print(f"Cliente desconectado: {request.sid}")


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=7001, allow_unsafe_werkzeug=True)