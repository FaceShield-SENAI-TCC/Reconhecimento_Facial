import os
import logging
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
import threading
from datetime import datetime
from facial_capture import FluidFaceCapture, check_user_exists, init_face_database

# Configurações
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
MAX_FILE_SIZE = 16 * 1024 * 1024

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
        # Em produção, usar banco de dados para usuários
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
        'service': 'facial_capture_api'
    }), 200


@app.route('/api/database/status', methods=['GET'])
def database_status():
    """Status do banco de dados"""
    try:
        # Contar usuários no banco
        user_count = check_user_exists("", "", "")  # Chamada vazia para contar todos
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
        'min_photos_required': 10,
        'circuito_continuo': True,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Página inicial"""
    return jsonify({
        'message': 'Servidor de Captura Facial - Sistema de Cadastro Biométrico',
        'status': 'online',
        'version': '2.0.0',
        'circuito_continuo': True,
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
        "message": "Conectado ao servidor de captura facial"
    })


@socketio.on('disconnect')
def on_disconnect():
    """Cliente desconectado"""
    # Parar captura ativa se existir
    if request.sid in active_captures:
        capture = active_captures[request.sid]
        if capture:
            capture.stop()
            logger.info(f"⏹️ Captura interrompida para cliente desconectado: {request.sid}")
        del active_captures[request.sid]

    # Remover cliente da lista de conectados
    if request.sid in connected_clients:
        del connected_clients[request.sid]

    logger.info(f"❌ Cliente desconectado: {request.sid}")


@socketio.on('start_camera')
def on_start_camera(data):
    """
    Inicia captura facial para cadastro biométrico
    Payload esperado:
    {
        "nome": "Nome do usuário",
        "sobrenome": "Sobrenome do usuário",
        "turma": "Turma do usuário",
        "tipoUsuario": "1" ou "2" (1=aluno, 2=professor)
    }
    """
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

        # Limpar captura ativa em caso de erro
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


@socketio.on('get_capture_status')
def on_get_capture_status():
    """Retorna o status atual da captura do cliente"""
    try:
        if request.sid in active_captures:
            status = "active"
            if active_captures[request.sid] is not None:
                capture = active_captures[request.sid]
                status = f"capturing_{capture.captured_count}_of_10"
        else:
            status = "inactive"

        emit("capture_status", {
            "status": status,
            "session_id": request.sid
        })
    except Exception as e:
        logger.error(f"Erro ao obter status da captura: {str(e)}")


def run_face_capture(nome, sobrenome, turma, tipo, session_id):
    """
    Executa o processo de captura facial em thread separada
    """
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
            "timestamp": datetime.now().isoformat()
        }

        socketio.emit("capture_complete", result_data, room=session_id)

        if success:
            logger.info(f"✅ Captura concluída com sucesso: {nome} {sobrenome}")
        else:
            logger.warning(f"⚠️ Captura falhou: {message}")

    except Exception as e:
        logger.error(f"❌ Erro na thread de captura: {str(e)}", exc_info=True)

        # Enviar erro para o cliente
        socketio.emit("capture_complete", {
            "success": False,
            "message": f"Erro durante a captura: {str(e)}",
            "captured_count": 0,
            "session_id": session_id
        }, room=session_id)

    finally:
        # Sempre limpar captura ativa ao finalizar
        if session_id in active_captures:
            del active_captures[session_id]
            logger.info(f"🧹 Captura finalizada e limpa para sessão: {session_id}")


# ====================== MANIPULADORES DE ERRO ======================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint não encontrado"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Método não permitido"}), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Erro interno do servidor"}), 500


# ====================== INICIALIZAÇÃO ======================

def initialize_application():
    """Inicializa a aplicação e serviços"""
    logger.info("🚀 Inicializando Servidor de Captura Facial...")

    try:
        # Inicializar banco de dados
        if init_face_database():
            logger.info("✅ Banco de dados inicializado com sucesso")
        else:
            logger.error("❌ Falha na inicialização do banco de dados")
            return False

        logger.info("🎯 Sistema configurado com as seguintes características:")
        logger.info("   📍 Modo: Circuito Contínuo - Múltiplos Usuários")
        logger.info("   🔧 Modelo: VGG-Face (2622 dimensões)")
        logger.info("   📸 Fotos necessárias: 3 por usuário")
        logger.info("   🌐 WebSocket: Ativo na porta 7001")
        logger.info("   💾 Banco: PostgreSQL com monitoramento em tempo real")

        logger.info("📡 Endpoints disponíveis:")
        logger.info("   WebSocket  /socket.io/    - Captura facial em tempo real")
        logger.info("   GET        /api/health    - Status do servidor")
        logger.info("   GET        /api/system/info - Informações do sistema")
        logger.info("   GET        /api/database/status - Status do banco")
        logger.info("   POST       /api/login     - Autenticação de administração")

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
                port=7001,
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