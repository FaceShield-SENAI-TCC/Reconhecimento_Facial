"""
Servidor Principal de Reconhecimento Facial - VERSÃO COMPATÍVEL
"""
import eventlet
eventlet.monkey_patch()

import os
import logging
import signal
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS

# Módulos compartilhados
from common.config import APP_CONFIG, MODEL_CONFIG
from face_recognition_logic import FaceRecognitionService

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Inicialização do serviço
face_service = FaceRecognitionService()

# Configuração da aplicação Flask
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configuração CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"]
    }
})

@app.after_request
def after_request(response):
    """Handler global para CORS"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check do servidor"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/health', methods=['GET'])
def health_check_new():
    """Health check do servidor"""
    return jsonify({
        "status": "operational",
        "service": "face_recognition_api",
        "timestamp": face_service.get_current_timestamp(),
        "port": APP_CONFIG.SERVER_PORT_RECONHECIMENTO
    }), 200

@app.route('/face-login', methods=['POST'])
def face_login_legacy():
    """Endpoint LEGACY para autenticação facial"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type deve ser application/json"}), 400

        data = request.get_json()
        if not data or 'imagem' not in data:
            return jsonify({"error": "Campo obrigatório faltando: 'imagem'"}), 400

        image_data = data['imagem']
        result = face_service.process_face_login(image_data)

        # Garantir que todos os campos estejam presentes
        if result.get('authenticated'):
            required_fields = ['id', 'username', 'tipo_usuario', 'nome', 'sobrenome', 'turma']
            for field in required_fields:
                if field not in result:
                    result[field] = None

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Erro interno no face login: {str(e)}")
        return jsonify({
            "error": "Erro interno do servidor",
            "message": "Serviço de autenticação temporariamente indisponível"
        }), 500

@app.route('/api/face-login', methods=['POST'])
def face_login():
    """Endpoint NOVO para autenticação facial"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type deve ser application/json"}), 400

        data = request.get_json()
        if not data or 'imagem' not in data:
            return jsonify({"error": "Campo obrigatório faltando: 'imagem'"}), 400

        image_data = data['imagem']
        result = face_service.process_face_login(image_data)

        if result.get('authenticated'):
            logger.info(f"✅ Login bem-sucedido para usuário ID: {result.get('id')}")

            required_fields = ['id', 'username', 'tipo_usuario', 'nome', 'sobrenome', 'turma']
            for field in required_fields:
                if field not in result:
                    result[field] = None

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Erro interno no face login: {str(e)}")
        return jsonify({
            "authenticated": False,
            "error": "Erro interno do servidor",
            "message": "Falha temporária no serviço de autenticação"
        }), 500

@app.route('/database-status', methods=['GET'])
def database_status_legacy():
    """Status do banco de dados facial"""
    try:
        status = face_service.get_database_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Erro ao obter status do banco: {str(e)}")
        return jsonify({"error": "Erro ao acessar banco de dados"}), 500

@app.route('/api/database/status', methods=['GET'])
def database_status():
    """Status do banco de dados facial"""
    try:
        status = face_service.get_database_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Erro ao obter status do banco: {str(e)}")
        return jsonify({
            "error": "Erro ao acessar banco de dados",
            "message": "Não foi possível conectar ao banco de dados"
        }), 500

@app.route('/api/database/detailed-status', methods=['GET'])
def detailed_database_status():
    """Status detalhado do banco de dados"""
    try:
        status = face_service.get_detailed_database_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Erro ao obter status detalhado do banco: {str(e)}")
        return jsonify({
            "error": "Erro ao obter status detalhado do banco"
        }), 500

@app.route('/reload-database', methods=['POST'])
def reload_database_legacy():
    """Recarregamento manual do banco de dados"""
    try:
        success, message = face_service.reload_database()
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 500
    except Exception as e:
        logger.error(f"Erro no recarregamento do banco: {str(e)}")
        return jsonify({"error": "Falha no recarregamento do banco"}), 500

@app.route('/api/database/reload', methods=['POST'])
def reload_database():
    """Recarregamento manual do banco de dados"""
    try:
        success, message = face_service.reload_database()
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 500
    except Exception as e:
        logger.error(f"Erro no recarregamento do banco: {str(e)}")
        return jsonify({
            "error": "Falha no recarregamento do banco"
        }), 500

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Informações do sistema"""
    try:
        db_status = face_service.get_detailed_database_status()
        metrics = face_service.get_performance_metrics()

        return jsonify({
            "service": "face_recognition_api",
            "version": "2.0.0",
            "status": "operational",
            "database": {
                "total_users": db_status["user_count"],
                "professores": db_status["professores_count"],
                "alunos": db_status["alunos_count"],
                "total_embeddings": db_status["total_embeddings"],
                "status": db_status["status"]
            },
            "performance": {
                "total_attempts": metrics["total_attempts"],
                "success_rate": metrics["success_rate"],
                "average_processing_time": metrics["average_processing_time"]
            },
            "compatibility": "FULL",
            "timestamp": face_service.get_current_timestamp()
        }), 200
    except Exception as e:
        logger.error(f"Erro ao obter informações do sistema: {str(e)}")
        return jsonify({
            "service": "face_recognition_api",
            "status": "degraded"
        }), 500

@app.route('/api/system/metrics', methods=['GET'])
def system_metrics():
    """Métricas detalhadas do sistema"""
    try:
        metrics = face_service.get_performance_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"Erro ao obter métricas: {str(e)}")
        return jsonify({"error": "Erro ao obter métricas do sistema"}), 500

@app.route('/api/system/detailed-metrics', methods=['GET'])
def detailed_metrics():
    """Métricas detalhadas do sistema"""
    try:
        db_status = face_service.get_detailed_database_status()
        metrics = face_service.get_performance_metrics()

        return jsonify({
            "database": db_status,
            "performance": metrics,
            "model_config": {
                "name": "VGG-Face",
                "distance_threshold": MODEL_CONFIG.DISTANCE_THRESHOLD,
                "embedding_dimension": MODEL_CONFIG.EMBEDDING_DIMENSION
            },
            "timestamp": face_service.get_current_timestamp()
        }), 200
    except Exception as e:
        logger.error(f"Erro ao obter métricas detalhadas: {str(e)}")
        return jsonify({"error": "Erro ao obter métricas"}), 500

@app.route('/api/users/list', methods=['GET'])
def list_users():
    """Lista todos os usuários cadastrados"""
    try:
        db_status = face_service.get_detailed_database_status()
        return jsonify({
            "message": "Endpoint em desenvolvimento",
            "total_users": db_status["user_count"],
            "professores": db_status["professores_count"],
            "alunos": db_status["alunos_count"]
        }), 200
    except Exception as e:
        logger.error(f"Erro ao listar usuários: {str(e)}")
        return jsonify({"error": "Erro ao listar usuários"}), 500

# Handlers de erro
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint não encontrado"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Erro interno do servidor"
    }), 500

def signal_handler(sig, frame):
    """Handler para sinais de desligamento"""
    logger.info("🛑 Recebido sinal de desligamento, limpando recursos...")
    face_service.cleanup()
    sys.exit(0)

def initialize_application():
    """Inicialização da aplicação"""
    logger.info("🚀 Inicializando API de Reconhecimento Facial...")

    try:
        if face_service.initialize():
            logger.info("✅ Serviço de reconhecimento facial inicializado com sucesso")

            # Status do banco
            db_status = face_service.get_detailed_database_status()

            logger.info("📊 Endpoints disponíveis:")
            logger.info("   POST /face-login              - Autenticação facial (legacy)")
            logger.info("   POST /api/face-login          - Autenticação facial")
            logger.info("   GET  /api/database/status     - Status do banco")
            logger.info("   POST /api/database/reload     - Recarregar banco")
            logger.info("   GET  /api/system/info         - Informações do sistema")

            logger.info("📈 ESTATÍSTICAS DO BANCO:")
            logger.info(f"   👥 Total de usuários: {db_status['user_count']}")
            logger.info(f"   👨‍🏫 Professores: {db_status['professores_count']}")
            logger.info(f"   👨‍🎓 Alunos: {db_status['alunos_count']}")
            logger.info(f"   📊 Total de embeddings: {db_status['total_embeddings']}")

            return True
        else:
            logger.error("❌ Falha na inicialização do serviço de reconhecimento facial")
            return False
    except Exception as e:
        logger.error(f"❌ Falha na inicialização da aplicação: {str(e)}")
        return False

if __name__ == "__main__":
    # Registrar handlers para sinais de desligamento
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info("🟢 INICIANDO SISTEMA DE RECONHECIMENTO FACIAL")
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
            logger.info("🛑 Servidor interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro durante execução do servidor: {str(e)}")
        finally:
            face_service.cleanup()
    else:
        logger.critical("🛑 Aplicação falhou ao iniciar - encerrando")
        exit(1)