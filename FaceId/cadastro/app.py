"""
Servidor de Cadastro Facial Refatorado
Usa módulos compartilhados e estrutura profissional
"""
import eventlet
eventlet.monkey_patch()  # IMPORTANTE: Deve ser a primeira linha

import os
import logging
import signal
import sys
import threading
from datetime import datetime
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS

# Módulos compartilhados
from common.config import APP_CONFIG, SECURITY_CONFIG
from common.database import db_manager
from common.auth import token_required
from common.exceptions import DatabaseError, ImageValidationError
from facial_capture import FluidFaceCapture

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuração da aplicação Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = SECURITY_CONFIG.SECRET_KEY
app.config['JSON_SORT_KEYS'] = False

# Configuração CORS mais permissiva para WebSocket
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
}})

# SocketIO com configurações otimizadas PARA PRODUÇÃO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=SECURITY_CONFIG.MAX_FILE_SIZE,
    ping_timeout=120,  # Aumentado
    ping_interval=60,   # Aumentado
    logger=True,
    engineio_logger=True,
    async_mode='eventlet',  # Forçar eventlet
    always_connect=True,
    allow_upgrades=True,
    http_compression=True,
    compression_threshold=1024
)

# Estado da aplicação
connected_clients = {}
active_captures = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check do servidor"""
    try:
        user_count = db_manager.count_users()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'active_clients': len(connected_clients),
            'active_captures': len(active_captures),
            'registered_users': user_count,
            'service': 'facial_capture_api',
            'port': APP_CONFIG.SERVER_PORT_CADASTRO
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'degraded'}), 500

@app.route('/api/database/status', methods=['GET'])
def database_status():
    """Status do banco de dados"""
    try:
        user_count = db_manager.count_users()
        return jsonify({
            'status': 'connected',
            'user_count': user_count,
            'timestamp': datetime.now().isoformat()
        }), 200
    except DatabaseError as e:
        logger.error(f"Database status error: {str(e)}")
        return jsonify({'error': 'Erro ao conectar com o banco'}), 500

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Informações do sistema"""
    return jsonify({
        'service': 'facial_capture_system',
        'version': '2.0.0',
        'status': 'operational',
        'min_photos_required': APP_CONFIG.MIN_PHOTOS_REQUIRED,
        'timestamp': datetime.now().isoformat(),
        'compatibility': 'FULL'
    }), 200

@app.route('/', methods=['GET'])
def index():
    """Página inicial"""
    return jsonify({
        'message': 'Servidor de Captura Facial - Sistema de Cadastro Biométrico',
        'status': 'online',
        'version': '2.0.0',
        'port': APP_CONFIG.SERVER_PORT_CADASTRO
    })

# WebSocket Events
@socketio.on('connect')
def on_connect():
    """Cliente conectado via WebSocket"""
    logger.info(f"✅ Cliente conectado: {request.sid}")
    connected_clients[request.sid] = {
        'connect_time': datetime.now(),
        'status': 'connected',
        'ip_address': request.environ.get('REMOTE_ADDR', 'unknown')
    }

    # Enviar confirmação de conexão
    emit("connected", {
        "status": "connected",
        "sid": request.sid,
        "message": "Conectado ao servidor de captura facial",
        "timestamp": datetime.now().isoformat()
    })

    logger.info(f"📡 Cliente {request.sid} registrado com sucesso")

@socketio.on('disconnect')
def on_disconnect():
    """Cliente desconectado"""
    client_sid = request.sid
    logger.info(f"🔌 Iniciando desconexão para: {client_sid}")

    if client_sid in active_captures:
        capture = active_captures[client_sid]
        if capture:
            logger.info(f"⏹️ Parando captura ativa para: {client_sid}")
            capture.stop()
        del active_captures[client_sid]

    if client_sid in connected_clients:
        del connected_clients[client_sid]

    logger.info(f"❌ Cliente desconectado: {client_sid}")

@socketio.on('start_camera')
def on_start_camera(data):
    """Inicia captura facial para cadastro biométrico"""
    try:
        logger.info(f"🎬 Iniciando captura para SID: {request.sid}")
        logger.info(f"📦 Dados recebidos: {data}")

        # Validação dos dados
        nome = data.get("nome", "").strip()
        sobrenome = data.get("sobrenome", "").strip()
        turma = data.get("turma", "").strip()
        tipo_usuario_num = data.get("tipoUsuario", "1")  # 1=ALUNO, 2=PROFESSOR

        # --- CORREÇÃO: Username não é mais pego ou validado aqui ---
        # (O Java vai cuidar disso)

        # Determinar tipo de usuário
        if str(tipo_usuario_num) == "2":
            tipo_usuario = "PROFESSOR"
        else:
            tipo_usuario = "ALUNO"

        if not nome or not sobrenome or not turma:
            error_msg = "Nome, sobrenome e turma são obrigatórios"
            logger.warning(f"❌ {error_msg}")
            emit("capture_complete", {"success": False, "message": error_msg})
            return

        # Verificar se já existe captura ativa
        if request.sid in active_captures:
            error_msg = "Já existe uma captura em andamento"
            logger.warning(f"❌ {error_msg}")
            emit("capture_complete", {"success": False, "message": error_msg})
            return

        # Verificar se usuário já está cadastrado (por nome, sobrenome e turma)
        existing_count = db_manager.check_user_exists(nome, sobrenome, turma)
        if existing_count > 0:
            error_msg = f"Usuário {nome} {sobrenome} já está cadastrado na turma {turma}"
            logger.warning(f"❌ {error_msg}")
            emit("capture_complete", {
                "success": False,
                "message": error_msg,
                "user_exists": True
            })
            return

        # Registrar cliente na sala
        join_room(request.sid)
        logger.info(f"🏠 Cliente {request.sid} entrou na sala")

        # Atualizar informações do cliente
        connected_clients[request.sid].update({
            'status': 'capturing',
            'start_time': datetime.now(),
            'user_data': {
                'nome': nome,
                'sobrenome': sobrenome,
                'turma': turma,
                'tipo_usuario': tipo_usuario,
                # 'username' removido
            }
        })

        # Registrar captura ativa
        active_captures[request.sid] = None

        user_type_display = "aluno" if tipo_usuario == "ALUNO" else "professor"
        logger.info(f"🚀 Iniciando captura para: {nome} {sobrenome} - {turma} ({user_type_display})")

        # Enviar confirmação de início
        emit("capture_started", {
            "message": "Captura iniciada com sucesso",
            "session_id": request.sid,
            "user": f"{nome} {sobrenome}",
            "tipo_usuario": tipo_usuario,
            "timestamp": datetime.now().isoformat()
        })

        # Iniciar captura em thread separada
        # --- CORREÇÃO: Removido 'username' dos argumentos da thread ---
        thread = threading.Thread(
            target=run_face_capture,
            args=(nome, sobrenome, turma, tipo_usuario, request.sid),
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

# --- CORREÇÃO: Removido 'username' dos parâmetros da função ---
def run_face_capture(nome, sobrenome, turma, tipo_usuario, session_id):
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
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)
                logger.debug(f"📊 Progresso enviado para {session_id}: {progress['captured']}/{progress['total']}")
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
        # --- CORREÇÃO: Removido 'username' do construtor ---
        capture = FluidFaceCapture(
            nome=nome,
            sobrenome=sobrenome,
            turma=turma,
            tipo_usuario=tipo_usuario,
            # username=username, (removido)
            progress_callback=progress_callback,
            frame_callback=frame_callback
        )

        # Registrar instância ativa
        active_captures[session_id] = capture

        # Executar captura
        logger.info(f"🎯 Executando captura para sessão: {session_id}")
        success, message = capture.capture()

        # Enviar resultado final
        result_data = {
            "success": success,
            "message": message, # 'message' aqui é o ID (se sucesso) ou o Erro (se falha)
            "captured_count": capture.captured_count,
            "user": f"{nome} {sobrenome}",
            "tipo_usuario": tipo_usuario,
            # "username": username, (removido)
            "turma": turma,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

        # (Esta parte já estava correta, pois foi corrigida antes)
        if success:
            # Se deu certo, 'message' (vindo do facial_capture) contém o ID do usuário.
            user_id = message  # 'message' é o ID (NÚMERO)
            result_data['id'] = user_id # Adiciona a chave 'id'

            # Atualiza a mensagem para ser algo mais claro
            result_data['message'] = f"Usuário {nome} {sobrenome} salvo com ID: {user_id}"

            user_type = "aluno" if tipo_usuario == "ALUNO" else "professor"
            logger.info(f"✅ Captura concluída com sucesso: {nome} {sobrenome} ({user_type}) - ID: {user_id}")

        else:
            # Se deu errado, 'message' já é a mensagem de erro correta.
            logger.warning(f"⚠️ Captura falhou: {message}")

        # Emitir o resultado DEPOIS de ter formatado os dados corretos
        socketio.emit("capture_complete", result_data, room=session_id)

    except Exception as e:
        logger.error(f"❌ Erro na thread de captura: {str(e)}", exc_info=True)
        socketio.emit("capture_complete", {
            "success": False,
            "message": f"Erro durante a captura: {str(e)}",
            "session_id": session_id
        }, room=session_id)
    finally:
        if session_id in active_captures:
            del active_captures[session_id]
            logger.info(f"🧹 Captura finalizada para sessão: {session_id}")

def initialize_application():
    """Inicializa a aplicação e serviços"""
    logger.info("🚀 Inicializando Servidor de Captura Facial...")

    try:
        if db_manager.init_database():
            logger.info("✅ Banco de dados inicializado com sucesso")
        else:
            logger.error("❌ Falha na inicialização do banco de dados")
            return False

        logger.info("🎯 Sistema configurado com as seguintes características:")
        logger.info(f"   📍 Porta: {APP_CONFIG.SERVER_PORT_CADASTRO}")
        logger.info(f"   📸 Fotos necessárias: {APP_CONFIG.MIN_PHOTOS_REQUIRED}")
        logger.info("   🌐 WebSocket: Ativo com Eventlet")
        logger.info("   💾 Banco: PostgreSQL")
        # --- CORREÇÃO no Log (só para ficar certo) ---
        logger.info("   👤 Estrutura: Pré-cadastro (Python) -> Registro (Java c/ username)")
        logger.info("   🔄 Compatibilidade: FULL")

        return True

    except Exception as e:
        logger.error(f"❌ Falha crítica na inicialização: {str(e)}")
        return False

# Handler para shutdown
def signal_handler(sig, frame):
    """Manipula sinais de desligamento"""
    logger.info("🛑 Recebido sinal de desligamento...")
    logger.info("🧹 Limpando recursos...")

    # Parar todas as capturas ativas
    for session_id, capture in list(active_captures.items()):
        if capture:
            capture.stop()
            logger.info(f"⏹️ Captura parada para sessão: {session_id}")

    logger.info("👋 Servidor finalizado graciosamente")
    sys.exit(0)

# Registrar handlers de sinal
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🟢 INICIANDO SISTEMA DE CAPTURA FACIAL")
    logger.info("=" * 60)

    if initialize_application():
        try:
            logger.info(f"🌐 Servidor WebSocket iniciando na porta {APP_CONFIG.SERVER_PORT_CADASTRO}")
            logger.info("📡 Aguardando conexões WebSocket...")

            socketio.run(
                app,
                host='0.0.0.0',
                port=APP_CONFIG.SERVER_PORT_CADASTRO,
                debug=False,
                allow_unsafe_werkzeug=True,
                use_reloader=False  # Importante para evitar dupla inicialização
            )
        except KeyboardInterrupt:
            logger.info("🛑 Servidor interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro durante execução do servidor: {str(e)}")
    else:
        logger.critical("💥 Falha na inicialização - Encerrando aplicação")
        exit(1)