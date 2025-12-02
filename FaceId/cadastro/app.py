# CORREÇÃO PARA IMPORTAÇÕES - DEVE SER AS PRIMEIRAS LINHAS DO ARQUIVO
import sys
import os
import logging

# Adicionar o diretório raiz do projeto ao Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configuração de logging ANTES de outras importações para capturar tudo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Reduzir verbosidade de bibliotecas específicas
logging.getLogger('socketio.server').setLevel(logging.WARNING)
logging.getLogger('engineio.server').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

logger = logging.getLogger('cadastro')

# Verificar se as importações funcionam
try:
    # Testar importação do common
    from common.config import APP_CONFIG, SECURITY_CONFIG
    print("SUCESSO: Módulos common importados")
except ImportError as e:
    print(f"ERRO: Falha na importação do common: {e}")
    print(f"DEBUG: Python path: {sys.path}")
    sys.exit(1)

"""
Servidor de Cadastro Facial Refatorado
Usa módulos compartilhados e estrutura profissional
"""
import eventlet
eventlet.monkey_patch()

import signal
from datetime import datetime
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

# Módulos compartilhados
from common.config import APP_CONFIG, SECURITY_CONFIG
from common.database import db_manager

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

# SocketIO com logs REDUZIDOS - somente erros importantes
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=SECURITY_CONFIG.MAX_FILE_SIZE,
    ping_timeout=120,
    ping_interval=60,
    logger=False,                     # DESATIVADO para reduzir logs
    engineio_logger=False,            # DESATIVADO para reduzir logs
    async_mode='eventlet',
    always_connect=True,
    allow_upgrades=True,
    http_compression=True,
    compression_threshold=1024
)

# Importar rotas e handlers
from routes import register_routes
from websocket_handlers import register_websocket_handlers
from capture_service import initialize_application

# Estado da aplicação (compartilhado entre módulos)
connected_clients = {}
active_captures = {}

# Registrar rotas e handlers
register_routes(app, connected_clients, active_captures)
register_websocket_handlers(socketio, connected_clients, active_captures)

def signal_handler(sig, frame):
    """Manipula sinais de desligamento"""
    logger.info("SINAL: Recebido sinal de desligamento...")
    logger.info("LIMPANDO: Limpando recursos...")

    # Parar todas as capturas ativas
    for session_id, capture in list(active_captures.items()):
        if capture:
            capture.stop()
            logger.info(f"CAPTURA PARADA: Sessão {session_id}")

    logger.info("SERVIDOR FINALIZADO")
    sys.exit(0)

# Registrar handlers de sinal
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("INICIANDO SISTEMA DE CAPTURA FACIAL")
    logger.info("=" * 60)

    if initialize_application():
        try:
            logger.info(f"SERVIDOR INICIANDO: Porta {APP_CONFIG.SERVER_PORT_CADASTRO}")
            logger.info("AGUARDANDO: Aguardando conexões...")

            socketio.run(
                app,
                host='0.0.0.0',
                port=APP_CONFIG.SERVER_PORT_CADASTRO,
                debug=False,
                allow_unsafe_werkzeug=False,  # Alterado para False em produção
                use_reloader=False
            )
        except KeyboardInterrupt:
            logger.info("INTERRUPÇÃO: Servidor interrompido pelo usuário")
        except Exception as e:
            logger.error(f"ERRO DURANTE EXECUÇÃO: {str(e)}")
    else:
        logger.critical("FALHA: Erro na inicialização - Encerrando aplicação")
        exit(1)