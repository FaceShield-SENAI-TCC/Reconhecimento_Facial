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
import socket
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

# ========== NOVOS ENDPOINTS HTTP ==========

@app.route('/api/face-login', methods=['POST'])
def face_login():
    """Endpoint para login via reconhecimento facial"""
    try:
        data = request.get_json()

        if not data or 'imagem' not in data:
            return jsonify({
                'authenticated': False,
                'message': 'Imagem não fornecida'
            }), 400

        logger.info("🔐 Tentativa de login facial recebida")

        # Buscar usuário por reconhecimento facial
        user_data = db_manager.get_user_by_facial_data(data['imagem'])

        if user_data:
            return jsonify({
                'authenticated': True,
                'user': f"{user_data['nome']} {user_data['sobrenome']}",
                'userType': user_data['tpousuario'],  # ✅ RETORNA O TIPO DE USUÁRIO
                'username': user_data.get('username'),
                'confidence': 0.95,
                'message': 'Login realizado com sucesso'
            }), 200
        else:
            return jsonify({
                'authenticated': False,
                'message': 'Usuário não reconhecido'
            }), 401

    except Exception as e:
        logger.error(f"❌ Erro no face-login: {str(e)}")
        return jsonify({
            'authenticated': False,
            'message': 'Erro interno no servidor'
        }), 500

@app.route('/api/usuarios', methods=['GET'])
def listar_usuarios():
    """Endpoint para listar todos os usuários cadastrados"""
    try:
        usuarios = db_manager.get_all_users()

        usuarios_list = []
        for user in usuarios:
            usuarios_list.append({
                'id': user['id'],
                'nome': user['nome'],
                'sobrenome': user['sobrenome'],
                'tipo_usuario': user['tpousuario'],  # ✅ INCLUI O TIPO DE USUÁRIO
                'username': user.get('username'),
                'turma': user['turma'],
                'data_cadastro': user.get('data_cadastro', 'N/A')
            })

        return jsonify({
            'success': True,
            'usuarios': usuarios_list,
            'total': len(usuarios_list)
        }), 200

    except Exception as e:
        logger.error(f"❌ Erro ao listar usuários: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Erro interno no servidor'
        }), 500

@app.route('/api/verificar-usuario', methods=['POST'])
def verificar_usuario():
    """Endpoint para verificar se um usuário está cadastrado corretamente"""
    try:
        data = request.get_json()
        nome = data.get('nome', '').strip()
        sobrenome = data.get('sobrenome', '').strip()
        turma = data.get('turma', '').strip()

        if not nome or not sobrenome:
            return jsonify({
                'success': False,
                'message': 'Nome e sobrenome são obrigatórios'
            }), 400

        # Buscar usuário no banco
        usuario = db_manager.get_user_by_name(nome, sobrenome, turma)

        if usuario:
            return jsonify({
                'success': True,
                'usuario': {
                    'id': usuario['id'],
                    'nome': usuario['nome'],
                    'sobrenome': usuario['sobrenome'],
                    'tpousuario': usuario['tpousuario'],
                    'turma': usuario['turma'],
                    'username': usuario.get('username')
                },
                'message': 'Usuário encontrado'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Usuário não encontrado'
            }), 404

    except Exception as e:
        logger.error(f"❌ Erro ao verificar usuário: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Erro interno no servidor'
        }), 500

# ========== ENDPOINTS EXISTENTES ATUALIZADOS ==========

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check do servidor"""
    try:
        user_count = db_manager.count_users()
        user_stats = db_manager.get_user_type_stats()  # ✅ NOVA ESTATÍSTICA

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'active_clients': len(connected_clients),
            'active_captures': len(active_captures),
            'registered_users': user_count,
            'user_types': user_stats,  # ✅ INCLUI ESTATÍSTICAS POR TIPO
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
        user_stats = db_manager.get_user_type_stats()  # ✅ NOVA ESTATÍSTICA

        return jsonify({
            'status': 'connected',
            'user_count': user_count,
            'user_stats': user_stats,  # ✅ INCLUI ESTATÍSTICAS POR TIPO
            'timestamp': datetime.now().isoformat()
        }), 200
    except DatabaseError as e:
        logger.error(f"Database status error: {str(e)}")
        return jsonify({'error': 'Erro ao conectar com o banco'}), 500

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Informações do sistema"""
    user_stats = db_manager.get_user_type_stats()  # ✅ NOVA ESTATÍSTICA

    return jsonify({
        'service': 'facial_capture_system',
        'version': '2.0.0',
        'status': 'operational',
        'min_photos_required': APP_CONFIG.MIN_PHOTOS_REQUIRED,
        'user_statistics': user_stats,  # ✅ INCLUI ESTATÍSTICAS DE USUÁRIOS
        'timestamp': datetime.now().isoformat(),
        'compatibility': 'FULL'
    }), 200

@app.route('/', methods=['GET'])
def index():
    """Página inicial"""
    user_stats = db_manager.get_user_type_stats()  # ✅ NOVA ESTATÍSTICA

    return jsonify({
        'message': 'Servidor de Captura Facial - Sistema de Cadastro Biométrico',
        'status': 'online',
        'version': '2.0.0',
        'port': APP_CONFIG.SERVER_PORT_CADASTRO,
        'user_statistics': user_stats  # ✅ INCLUI ESTATÍSTICAS
    })

# ========== WEBSOCKET EVENTS ATUALIZADOS ==========

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
    """Inicia captura facial para cadastro biométrico - CORRIGIDO"""
    try:
        logger.info(f"🎬 Iniciando captura para SID: {request.sid}")
        logger.info(f"📦 Dados recebidos: {data}")

        # Validação dos dados
        nome = data.get("nome", "").strip()
        sobrenome = data.get("sobrenome", "").strip()
        turma = data.get("turma", "").strip()
        tipo_usuario_num = data.get("tipoUsuario", "1")  # 1=ALUNO, 2=PROFESSOR
        username = data.get("username", "").strip()

        # ✅ CORREÇÃO: Lógica aprimorada para determinar tipo de usuário
        tipo_usuario = "ALUNO"  # padrão
        username_final = None

        # Verificar se é professor
        if str(tipo_usuario_num) == "2":
            tipo_usuario = "PROFESSOR"
            logger.info(f"👨‍🏫 Cadastrando como PROFESSOR: {nome} {sobrenome}")

            # ✅ CORREÇÃO: Validação mais flexível do username
            if not username:
                # Gerar username automaticamente se não fornecido
                username = f"{nome.lower()}.{sobrenome.lower()}"
                logger.info(f"🔧 Username gerado automaticamente: {username}")

            # Verificar se username já existe
            existing_username = db_manager.check_username_exists(username)
            if existing_username > 0:
                error_msg = f"Username '{username}' já está em uso"
                logger.warning(f"❌ {error_msg}")
                emit("capture_complete", {
                    "success": False,
                    "message": error_msg,
                    "userType": tipo_usuario  # ✅ SEMPRE RETORNA O TIPO
                })
                return

            username_final = username
        else:
            tipo_usuario = "ALUNO"
            username_final = None
            logger.info(f"👨‍🎓 Cadastrando como ALUNO: {nome} {sobrenome}")

        # ✅ CORREÇÃO: Validação de campos obrigatórios
        campos_obrigatorios = [
            (nome, "Nome"),
            (sobrenome, "Sobrenome"),
            (turma, "Turma")
        ]

        campos_faltantes = [campo[1] for campo in campos_obrigatorios if not campo[0]]
        if campos_faltantes:
            error_msg = f"Campos obrigatórios faltando: {', '.join(campos_faltantes)}"
            logger.warning(f"❌ {error_msg}")
            emit("capture_complete", {
                "success": False,
                "message": error_msg,
                "userType": tipo_usuario  # ✅ SEMPRE RETORNA O TIPO
            })
            return

        # Verificar se já existe captura ativa
        if request.sid in active_captures:
            error_msg = "Já existe uma captura em andamento"
            logger.warning(f"❌ {error_msg}")
            emit("capture_complete", {
                "success": False,
                "message": error_msg,
                "userType": tipo_usuario  # ✅ SEMPRE RETORNA O TIPO
            })
            return

        # Verificar se usuário já está cadastrado (por nome, sobrenome e turma)
        existing_count = db_manager.check_user_exists(nome, sobrenome, turma)
        if existing_count > 0:
            error_msg = f"Usuário {nome} {sobrenome} já está cadastrado na turma {turma}"
            logger.warning(f"❌ {error_msg}")
            emit("capture_complete", {
                "success": False,
                "message": error_msg,
                "user_exists": True,
                "userType": tipo_usuario  # ✅ SEMPRE RETORNA O TIPO
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
                'username': username_final
            }
        })

        # Registrar captura ativa
        active_captures[request.sid] = None

        user_type_display = "aluno" if tipo_usuario == "ALUNO" else "professor"
        logger.info(f"🚀 Iniciando captura para: {nome} {sobrenome} - {turma} ({user_type_display})")

        # ✅ CORREÇÃO: Enviar confirmação com todos os dados
        emit("capture_started", {
            "message": "Captura iniciada com sucesso",
            "session_id": request.sid,
            "user": f"{nome} {sobrenome}",
            "tipo_usuario": tipo_usuario,  # ✅ RETORNA O TIPO DE USUÁRIO
            "userType": tipo_usuario,  # ✅ MANTÉM COMPATIBILIDADE
            "timestamp": datetime.now().isoformat()
        })

        # Iniciar captura em thread separada
        thread = threading.Thread(
            target=run_face_capture,
            args=(nome, sobrenome, turma, tipo_usuario, username_final, request.sid),
            daemon=True
        )
        thread.start()

    except Exception as e:
        logger.error(f"❌ Erro ao iniciar captura: {str(e)}", exc_info=True)

        if request.sid in active_captures:
            del active_captures[request.sid]

        # ✅ CORREÇÃO: Mensagem de erro mais informativa
        emit("capture_complete", {
            "success": False,
            "message": f"Erro interno ao iniciar captura: {str(e)}",
            "userType": tipo_usuario if 'tipo_usuario' in locals() else 'ALUNO'  # ✅ SEMPRE RETORNA O TIPO
        })

def run_face_capture(nome, sobrenome, turma, tipo_usuario, username, session_id):
    """Executa o processo de captura facial em thread separada - CORRIGIDO"""
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
        capture = FluidFaceCapture(
            nome=nome,
            sobrenome=sobrenome,
            turma=turma,
            tipo_usuario=tipo_usuario,
            username=username,
            progress_callback=progress_callback,
            frame_callback=frame_callback
        )

        # Registrar instância ativa
        active_captures[session_id] = capture

        # Executar captura
        logger.info(f"🎯 Executando captura para sessão: {session_id}")
        success, message = capture.capture()

        # ✅ CORREÇÃO: Resultado detalhado
        result_data = {
            "success": success,
            "message": message,
            "captured_count": capture.captured_count,
            "user": f"{nome} {sobrenome}",
            "tipo_usuario": tipo_usuario,  # ✅ RETORNA O TIPO DE USUÁRIO
            "userType": tipo_usuario,  # ✅ MANTÉM COMPATIBILIDADE
            "username": username,
            "turma": turma,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

        socketio.emit("capture_complete", result_data, room=session_id)

        if success:
            user_type = "aluno" if tipo_usuario == "ALUNO" else "professor"
            logger.info(f"✅ Captura concluída com sucesso: {nome} {sobrenome} ({user_type})")

            # ✅ CORREÇÃO: Log adicional para professor
            if tipo_usuario == "PROFESSOR" and username:
                logger.info(f"👨‍🏫 Professor cadastrado com username: {username}")
        else:
            logger.warning(f"⚠️ Captura falhou: {message}")

    except Exception as e:
        logger.error(f"❌ Erro na thread de captura: {str(e)}", exc_info=True)

        # ✅ CORREÇÃO: Mensagem de erro detalhada
        socketio.emit("capture_complete", {
            "success": False,
            "message": f"Erro durante a captura: {str(e)}",
            "session_id": session_id,
            "tipo_usuario": tipo_usuario,  # ✅ SEMPRE RETORNA O TIPO
            "userType": tipo_usuario  # ✅ MANTÉM COMPATIBILIDADE
        }, room=session_id)
    finally:
        if session_id in active_captures:
            del active_captures[session_id]
            logger.info(f"🧹 Captura finalizada para sessão: {session_id}")

# ========== FUNÇÕES AUXILIARES PARA PORTAS ==========

def check_port_available(port):
    """Verifica se uma porta está disponível"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_available_port():
    """Encontra uma porta disponível"""
    ports_to_try = [5005, 5006, 5007, 5008, 5009]
    for port in ports_to_try:
        if check_port_available(port):
            logger.info(f"✅ Porta {port} disponível")
            return port
    return None

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
        logger.info("   👤 Estrutura: ALUNO (sem username) / PROFESSOR (com username)")
        logger.info("   🔄 Compatibilidade: FULL")
        logger.info("   ✅ CORREÇÕES: Cadastro de PROFESSOR funcionando")
        logger.info("   ✅ NOVO: Endpoints de reconhecimento facial adicionados")

        return True

    except Exception as e:
        logger.error(f"❌ Falha crítica na inicialização: {str(e)}")
        return False

# Handler para graceful shutdown
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
    logger.info("🟢 INICIANDO SISTEMA DE CAPTURA FACIAL - ATUALIZADO")
    logger.info("=" * 60)

    # ✅ CORREÇÃO: Verificar se a porta está disponível
    selected_port = find_available_port()
    if not selected_port:
        logger.error("❌ Nenhuma porta disponível. Tentando usar a porta configurada...")
        selected_port = APP_CONFIG.SERVER_PORT_CADASTRO
    else:
        # Atualizar a porta configurada
        APP_CONFIG.SERVER_PORT_CADASTRO = selected_port

    if initialize_application():
        try:
            logger.info(f"🌐 Servidor WebSocket iniciando na porta {APP_CONFIG.SERVER_PORT_CADASTRO}")
            logger.info("📡 Aguardando conexões WebSocket...")
            logger.info("🔗 Endpoints disponíveis:")
            logger.info(f"   - http://localhost:{APP_CONFIG.SERVER_PORT_CADASTRO}/api/face-login")
            logger.info(f"   - http://localhost:{APP_CONFIG.SERVER_PORT_CADASTRO}/api/usuarios")
            logger.info(f"   - http://localhost:{APP_CONFIG.SERVER_PORT_CADASTRO}/api/verificar-usuario")
            logger.info(f"   - http://localhost:{APP_CONFIG.SERVER_PORT_CADASTRO}/api/health")

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