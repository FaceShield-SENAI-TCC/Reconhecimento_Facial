"""
Servidor QR Code - VERSÃO FINAL CORRIGIDA
"""
import os
import logging
from flask import Flask, jsonify, request  # Adicione request aqui
from flask_cors import CORS

# Importar e inicializar EventLoopManager ANTES do Flask
try:
    from common.event_loop_manager import EventLoopManager
    EventLoopManager.initialize()
    logger = logging.getLogger(__name__)
    logger.info("Event Loop Manager inicializado para IoT")
except ImportError as e:
    logging.warning(f"Não foi possível importar EventLoopManager: {e}")
except Exception as e:
    logging.error(f"Falha ao inicializar Event Loop Manager: {e}")

# Configurar logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuração CORS completa
CORS(app, resources={r"/*": {
    "origins": ["http://127.0.0.1:5500", "http://localhost:5500",
                "http://localhost:8080", "http://localhost:5000",
                "https://faceshield.onrender.com", "https://*.onrender.com"],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

# Importar e registrar rotas
from qr_routes import qr_bp
from iot_routes import iot_bp

app.register_blueprint(qr_bp)
app.register_blueprint(iot_bp)

@app.route('/test', methods=['GET'])
def test():
    """Endpoint de teste"""
    logger.info("✅ Endpoint /test chamado com sucesso")
    return jsonify({
        'message': 'Backend Python funcionando!',
        'status': 'ok',
        'service': 'qr_code_server',
        'endpoints': {
            'qr_scan': '/read-qrcode (POST)',
            'iot_control': '/iot/control (POST)',
            'test': '/test (GET)',
            'health': '/health (GET)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    logger.info("Health check chamado")
    return jsonify({
        'status': 'healthy',
        'service': 'qr_code_server',
        'timestamp': '2024-12-02 21:59:53'
    })

@app.errorhandler(404)
def not_found(e):
    """Handler para 404 - CORRIGIDO: request já foi importado"""
    logger.warning(f"Endpoint não encontrado: {request.path}")
    return jsonify({
        'error': 'Endpoint não encontrado',
        'path': request.path,
        'available_endpoints': [
            '/test',
            '/health',
            '/read-qrcode',
            '/iot/control'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handler para 500"""
    logger.error(f"Erro interno do servidor: {str(e)}")
    return jsonify({
        'error': 'Erro interno do servidor',
        'message': 'Ocorreu um erro inesperado. Tente novamente mais tarde.'
    }), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("INICIANDO SERVIDOR QR CODE COM SUPORTE IoT")
    logger.info("=" * 60)
    logger.info("ENDPOINTS DISPONÍVEIS:")
    logger.info("  • GET  /test           - Teste de conexão")
    logger.info("  • GET  /health         - Health check")
    logger.info("  • POST /read-qrcode    - Escanear QR Code")
    logger.info("  • POST /iot/control    - Controlar trava IoT")
    logger.info("=" * 60)

    # Listar rotas registradas
    logger.info("Rotas registradas:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {rule.endpoint}: {rule.rule}")

    logger.info("=" * 60)
    logger.info("Servidor iniciando na porta 5000...")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Mude para False em produção
        threaded=True
    )