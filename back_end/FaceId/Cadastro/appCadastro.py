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

# AJUSTE O CAMINHO PARA SEU USUÁRIO
CAPTURE_BASE_DIR = "C:/Users/Aluno/Desktop/Arduino/back_end/FaceId/Cadastro/Faces"
os.makedirs(CAPTURE_BASE_DIR, exist_ok=True)

MIN_PHOTOS_REQUIRED = 10
face_capture_state = {}

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": [
    "http://localhost:7001",
    "http://127.0.0.1:7001",
    "http://127.0.0.1:5500",
    "http://localhost:8001",
    "http://127.0.0.1:8001"
]}})

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


def sanitize_name(name):
    name = str(name).lower().strip()
    return re.sub(r'[^a-z0-9_]', '', name.replace(' ', '_'))


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

        turma_dir = os.path.join(CAPTURE_BASE_DIR, sanitized_turma)
        user_dir = os.path.join(turma_dir, f"{sanitized_nome}_{sanitized_sobrenome}")
        os.makedirs(user_dir, exist_ok=True)
        print(f"[{session_id}] Diretório de usuário para fotos: {user_dir}")

        start_time = time.time()
        last_face_time = 0
        face_capture_interval = 0.5
        QUALITY_THRESHOLD = 100.0

        while face_capture_state[session_id]["captured_count"] < MIN_PHOTOS_REQUIRED and (
                time.time() - start_time) < 60:
            if not face_capture_state[session_id]["thread_running"]:
                print(f"[{session_id}] Thread de captura interrompida por solicitação externa.")
                break

            ret, frame = cap.read()
            if not ret:
                face_capture_state[session_id]["message"] = "Câmera parou de enviar frames inesperadamente."
                print(f"[{session_id}] ERRO: {face_capture_state[session_id]['message']}")
                break

            frame_display = frame.copy()

            try:
                detected_faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="opencv",
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
                                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                print(
                                    f"[{session_id}] Rosto e qualidade OK! Foto capturada: {face_capture_state[session_id]['captured_count']} de {MIN_PHOTOS_REQUIRED}")
                            else:
                                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 0, 255), 3)

            except Exception as e:
                pass

            _, buffer = cv2.imencode('.jpg', frame_display)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('capture_frame', {'frame': jpg_as_text}, room=session_id)

            socketio.emit('capture_progress', {
                'captured': face_capture_state[session_id]["captured_count"],
                'total': MIN_PHOTOS_REQUIRED,
                'session_id': session_id
            }, room=session_id)

            time.sleep(0.05)

        if face_capture_state[session_id]["captured_count"] >= MIN_PHOTOS_REQUIRED:
            for i, face_img in enumerate(face_capture_state[session_id]["captured_faces"]):
                filename = f"{sanitized_nome}_{sanitized_sobrenome}_{i + 1}.jpg"
                filepath = os.path.join(user_dir, filename)
                cv2.imwrite(filepath, face_img)

            face_capture_state[session_id]["success"] = True
            face_capture_state[session_id][
                "message"] = f"Captura completa e salva com sucesso. {MIN_PHOTOS_REQUIRED} fotos salvas."
            print(f"[{session_id}] {face_capture_state[session_id]['message']}")
        else:
            face_capture_state[session_id]["success"] = False
            if not face_capture_state[session_id]["message"]:
                face_capture_state[session_id][
                    "message"] = "Captura falhou. Não foi possível capturar o número necessário de fotos ou tempo esgotado."
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
            "captured_count": face_capture_state[session_id]["captured_count"],
            "total_to_capture": MIN_PHOTOS_REQUIRED
        }
        socketio.emit("capture_complete", response_data, room=session_id)
        face_capture_state[session_id]["thread_running"] = False
        print(f"[{session_id}] Fim da thread de captura. Resultado: {response_data['success']}")


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

    print(f"Recebido pedido para iniciar a câmera. Session_ID da Captura: {session_id}, SID da Conexão: {request.sid}")

    thread = threading.Thread(target=capture_frames, args=(nome, sobrenome, turma, request.sid))
    thread.start()


@socketio.on("connect")
def on_connect(auth):
    session_id = request.args.get("session_id")
    if session_id:
        print(f"Cliente conectado com session_id: {session_id}, SID da Conexão: {request.sid}")
    else:
        print(f"Cliente conectado sem session_id. SID da Conexão: {request.sid}")


@socketio.on("disconnect")
def on_disconnect():
    session_id = request.args.get("session_id")
    if session_id in face_capture_state:
        face_capture_state[session_id]["thread_running"] = False
        print(f"Cliente com session_id {session_id} desconectado. SID da Conexão: {request.sid}")
    else:
        print(f"Cliente desconectado: {request.sid}")


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=7001, allow_unsafe_werkzeug=True)