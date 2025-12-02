"""
Handlers para eventos WebSocket
"""
import logging
from datetime import datetime
import threading
from flask import request
from flask_socketio import emit, join_room

from common.database import db_manager
from capture_service import run_face_capture

logger = logging.getLogger('cadastro')

def register_websocket_handlers(socketio, connected_clients, active_captures):
    """Registra todos os handlers WebSocket"""

    @socketio.on('connect')
    def on_connect():
        """Cliente conectado via WebSocket"""
        connected_clients[request.sid] = {
            'connect_time': datetime.now(),
            'status': 'connected',
            'ip_address': request.environ.get('REMOTE_ADDR', 'unknown')
        }
        logger.info(f"Cliente conectado: {request.sid}")

        emit("connected", {
            "status": "connected",
            "sid": request.sid,
            "message": "Conectado ao servidor de captura facial",
            "timestamp": datetime.now().isoformat()
        })

    @socketio.on('disconnect')
    def on_disconnect():
        """Cliente desconectado"""
        client_sid = request.sid

        if client_sid in active_captures:
            capture = active_captures[client_sid]
            if capture:
                capture.stop()
                logger.info(f"Captura parada para sessão: {client_sid}")
            del active_captures[client_sid]

        if client_sid in connected_clients:
            del connected_clients[client_sid]

        logger.info(f"Cliente desconectado: {client_sid}")

    @socketio.on('start_camera')
    def on_start_camera(data):
        """Inicia captura facial para cadastro biométrico"""
        try:
            nome = data.get("nome", "").strip()
            sobrenome = data.get("sobrenome", "").strip()
            turma = data.get("turma", "").strip()
            tipo_usuario_num = data.get("tipoUsuario", "1")

            # Determinar tipo de usuário
            if str(tipo_usuario_num) == "2":
                tipo_usuario = "PROFESSOR"
            else:
                tipo_usuario = "ALUNO"

            if not nome or not sobrenome or not turma:
                error_msg = "Nome, sobrenome e turma são obrigatórios"
                logger.warning(f"Validação falhou: {error_msg}")
                emit("capture_complete", {"success": False, "message": error_msg})
                return

            # Verificar se já existe captura ativa
            if request.sid in active_captures:
                error_msg = "Já existe uma captura em andamento"
                logger.warning(f"Captura já ativa: {request.sid}")
                emit("capture_complete", {"success": False, "message": error_msg})
                return

            # Verificar se usuário já está cadastrado
            existing_count = db_manager.check_user_exists(nome, sobrenome, turma)
            if existing_count > 0:
                error_msg = f"Usuário {nome} {sobrenome} já está cadastrado na turma {turma}"
                logger.warning(f"Usuário existente: {nome} {sobrenome}")
                emit("capture_complete", {
                    "success": False,
                    "message": error_msg,
                    "user_exists": True
                })
                return

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
                    'tipo_usuario': tipo_usuario,
                }
            })

            # Registrar captura ativa
            active_captures[request.sid] = None

            user_type_display = "aluno" if tipo_usuario == "ALUNO" else "professor"
            logger.info(f"Iniciando captura: {nome} {sobrenome} - {turma} ({user_type_display})")

            # Enviar confirmação de início
            emit("capture_started", {
                "message": "Captura iniciada com sucesso",
                "session_id": request.sid,
                "user": f"{nome} {sobrenome}",
                "tipo_usuario": tipo_usuario,
                "timestamp": datetime.now().isoformat()
            })

            # Iniciar captura em thread separada
            thread = threading.Thread(
                target=run_face_capture,
                args=(socketio, nome, sobrenome, turma, tipo_usuario, request.sid, active_captures),
                daemon=True
            )
            thread.start()

        except Exception as e:
            logger.error(f"Erro ao iniciar captura: {str(e)}", exc_info=True)

            if request.sid in active_captures:
                del active_captures[request.sid]

            emit("capture_complete", {
                "success": False,
                "message": f"Erro interno ao iniciar captura: {str(e)}"
            })

    logger.info("Handlers WebSocket registrados")