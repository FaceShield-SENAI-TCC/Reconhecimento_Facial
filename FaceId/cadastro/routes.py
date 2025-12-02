"""
Rotas HTTP para o servidor de captura facial
"""
import logging
from datetime import datetime
from flask import jsonify

from common.config import APP_CONFIG
from common.database import db_manager
from common.exceptions import DatabaseError

logger = logging.getLogger('cadastro')


def register_routes(app, connected_clients, active_captures):
    """Registra todas as rotas HTTP na aplicação Flask"""

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
            logger.error(f"ERRO Health check: {str(e)}")
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
            logger.error(f"ERRO Database status: {str(e)}")
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

    logger.info("Rotas HTTP registradas com sucesso")