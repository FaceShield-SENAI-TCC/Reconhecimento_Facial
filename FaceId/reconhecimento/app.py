"""
Servidor Principal de Reconhecimento Facial Refatorado - VERSÃO CORRIGIDA
"""
import eventlet
eventlet.monkey_patch()

import os
import logging
import signal
import sys
from flask import Flask, jsonify
from flask_cors import CORS

# Módulos compartilhados
from common.config import APP_CONFIG
from common.event_loop_manager import EventLoopManager
from recognition_routes import recognition_bp, recognition_service

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('reconhecimento')

# Configuração da aplicação Flask
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Configuração CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
        "supports_credentials": True
    }
})

@app.after_request
def after_request(response):
    """Handler global para CORS"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Registrar Blueprint de reconhecimento
app.register_blueprint(recognition_bp)

# Handlers de erro globais
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint não encontrado",
        "message": "Verifique a URL e tente novamente"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Método não permitido",
        "message": "Este endpoint não suporta o método HTTP utilizado"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"ERRO INTERNO DO SERVIDOR: {error}")
    return jsonify({
        "error": "Erro interno do servidor",
        "message": "Ocorreu um erro inesperado. Tente novamente mais tarde."
    }), 500

def signal_handler(sig, frame):
    """Handler para sinais de desligamento"""
    logger.info("SINAL: Recebido sinal de desligamento, limpando recursos...")
    recognition_service.cleanup()
    sys.exit(0)

def initialize_application():
    """Inicializa a aplicação"""
    logger.info("Inicializando aplicação...")

    # Inicializar serviço de reconhecimento
    if not recognition_service.initialize():
        logger.error("Falha na inicialização do serviço de reconhecimento")
        return False

    # Inicializar controlador da trava usando EventLoopManager
    from common.locker_controller import LockerController
    locker_controller = LockerController()

    try:
        locker_success = EventLoopManager.run_async(locker_controller.iniciar())

        if locker_success:
            logger.info("Controlador da trava inicializado com sucesso")
        else:
            logger.warning("Controlador da trava não pode se conectar a ESP32")
    except Exception as e:
        logger.warning(f"Falha na inicialização do controlador da trava: {e}")

    # Obter status detalhado do banco
    db_status = recognition_service.get_detailed_database_status()

    logger.info("ESTATÍSTICAS DO BANCO:")
    logger.info(f"   Total de usuários: {db_status['user_count']}")
    logger.info(f"   Professores: {db_status['professores_count']}")
    logger.info(f"   Alunos: {db_status['alunos_count']}")
    logger.info(f"   Total de embeddings: {db_status['total_embeddings']}")

    logger.info("CONFIGURAÇÃO DO MODELO:")
    logger.info(f"   Distância máxima: {0.60}")
    logger.info(f"   Confiança mínima: {0.80}")
    logger.info(f"   Margem mínima: {0.001}")

    logger.info("MONITORAMENTO: Monitoramento em tempo real do banco: ATIVO")
    return True

if __name__ == "__main__":
    # Registrar handlers para sinais de desligamento
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info("INICIANDO SISTEMA DE RECONHECIMENTO FACIAL")
    logger.info("=" * 60)

    if initialize_application():
        try:
            app.run(
                host='0.0.0.0',
                port=APP_CONFIG.SERVER_PORT_RECONHECIMENTO,
                debug=False,
                threaded=True
            )
        except KeyboardInterrupt:
            logger.info("INTERRUPÇÃO: Servidor interrompido pelo usuário")
        except Exception as e:
            logger.error(f"ERRO DURANTE EXECUÇÃO: {str(e)}")
        finally:
            recognition_service.cleanup()
    else:
        logger.critical("FALHA: Aplicação falhou ao iniciar - encerrando")
        exit(1)