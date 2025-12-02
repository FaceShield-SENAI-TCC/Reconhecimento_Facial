"""
Serviço de Captura Facial - Lógica de negócio principal
"""
import logging
import time
import threading
from datetime import datetime

# Módulos compartilhados
from common.config import APP_CONFIG
from common.database import db_manager

from face_detector import FaceDetector

logger = logging.getLogger(__name__)

def run_face_capture(socketio, nome, sobrenome, turma, tipo_usuario, session_id, active_captures):
    """Executa o processo de captura facial em thread separada"""
    try:
        def progress_callback(progress):
            """Callback para atualizações de progresso"""
            try:
                socketio.emit('capture_progress', {
                    'captured': progress['captured'],
                    'total': progress['total'],
                    'message': progress.get('message', ''),
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)
            except Exception as e:
                logger.error(f"Erro no callback de progresso: {str(e)}")

        def frame_callback(frame_data):
            """Callback para envio de frames"""
            try:
                socketio.emit('capture_frame', {
                    'frame': frame_data,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)
            except Exception as e:
                logger.error(f"Erro no callback de frame: {str(e)}")

        # Criar instância do capturador facial
        from facial_capture import FluidFaceCapture
        capture = FluidFaceCapture(
            nome=nome,
            sobrenome=sobrenome,
            turma=turma,
            tipo_usuario=tipo_usuario,
            progress_callback=progress_callback,
            frame_callback=frame_callback
        )

        # Registrar instância ativa
        active_captures[session_id] = capture

        # Executar captura
        logger.info(f"Executando captura: Sessão {session_id}")
        success, message = capture.capture()

        # Enviar resultado final
        result_data = {
            "success": success,
            "message": message,
            "captured_count": capture.captured_count,
            "user": f"{nome} {sobrenome}",
            "tipo_usuario": tipo_usuario,
            "turma": turma,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

        if success:
            user_id = message
            result_data['id'] = user_id
            result_data['message'] = f"Usuário {nome} {sobrenome} salvo com ID: {user_id}"

            user_type = "aluno" if tipo_usuario == "ALUNO" else "professor"
            logger.info(f"CAPTURA CONCLUÍDA: {nome} {sobrenome} ({user_type}) - ID: {user_id}")

        else:
            logger.warning(f"CAPTURA FALHOU: {message}")

        # Emitir o resultado
        socketio.emit("capture_complete", result_data, room=session_id)

    except Exception as e:
        logger.error(f"ERRO NA THREAD DE CAPTURA: {str(e)}", exc_info=True)
        socketio.emit("capture_complete", {
            "success": False,
            "message": f"Erro durante a captura: {str(e)}",
            "session_id": session_id
        }, room=session_id)
    finally:
        if session_id in active_captures:
            del active_captures[session_id]
            logger.debug(f"Captura finalizada: Sessão {session_id}")

def initialize_application():
    """Inicializa a aplicação e serviços"""
    logger.info("INICIALIZAÇÃO: Inicializando Servidor de Captura Facial...")

    try:
        if db_manager.init_database():
            logger.info("SUCESSO: Banco de dados inicializado")
        else:
            logger.error("FALHA: Erro na inicialização do banco de dados")
            return False

        logger.info("CONFIGURAÇÃO DO SISTEMA:")
        logger.info(f"  Porta: {APP_CONFIG.SERVER_PORT_CADASTRO}")
        logger.info(f"  Fotos necessárias: {APP_CONFIG.MIN_PHOTOS_REQUIRED}")
        logger.info(f"  WebSocket: Ativo com Eventlet")
        logger.info(f"  Banco: PostgreSQL")
        logger.info(f"  Compatibilidade: FULL")

        return True

    except Exception as e:
        logger.error(f"FALHA CRÍTICA: {str(e)}")
        return False