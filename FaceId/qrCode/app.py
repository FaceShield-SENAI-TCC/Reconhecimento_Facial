"""
Servidor QR Code - VERSÃO CORRIGIDA
"""
import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

# Configurar logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuração CORS completa
CORS(app, origins=[
    'http://127.0.0.1:5500',
    'http://localhost:5500',
    'http://localhost:8080',
    'https://faceshield.onrender.com',
    'https://*.onrender.com'
])

# Ou para permitir TODOS (desenvolvimento)
CORS(app)

# Importar e registrar rotas
from qr_routes import qr_bp
from iot_routes import iot_bp

app.register_blueprint(qr_bp)
app.register_blueprint(iot_bp)


@app.route('/test', methods=['GET'])
def test():
    """Endpoint de teste"""
    logger.info("Test endpoint chamado")
    return jsonify({
        'message': 'Backend Python funcionando!',
        'status': 'ok',
        'endpoints': {
            'qr_scan': '/read-qrcode (POST)',
            'iot_control': '/iot/control (POST)',
            'test': '/test (GET)'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'service': 'qr_code_server'})


@app.errorhandler(404)
def not_found(e):
    """Handler para 404"""
    logger.warning(f"Endpoint não encontrado: {request.path}")
    return jsonify({'error': 'Endpoint não encontrado'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handler para 500"""
    logger.error(f"Erro interno: {e}")
    return jsonify({'error': 'Erro interno do servidor'}), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("INICIANDO SERVIDOR QR CODE")
    logger.info("=" * 60)
    logger.info("ENDPOINTS DISPONÍVEIS:")
    logger.info("  • POST /read-qrcode    - Escanear QR Code")
    logger.info("  • POST /iot/control    - Controlar trava IoT")
    logger.info("  • GET  /test           - Teste de conexão")
    logger.info("  • GET  /health         - Health check")
    logger.info("=" * 60)
    logger.info("Servidor iniciando na porta 5000...")

    app.run(host='0.0.0.0', port=5000, debug=True)