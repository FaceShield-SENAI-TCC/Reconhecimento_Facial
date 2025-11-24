"""
Servidor Principal de Reconhecimento Facial Refatorado
Usa estrutura modular e compartilhada
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
from common.config import APP_CONFIG
from common.auth import token_required
from common.exceptions import ImageValidationError, FaceRecognitionServiceError
from face_recognition_logic import FaceRecognitionService

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('reconhecimento')

# Inicialização do serviço
face_service = FaceRecognitionService()

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check do servidor (endpoint legacy)"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/health', methods=['GET'])
def health_check_new():
    """Health check do servidor (endpoint novo)"""
    return jsonify({
        "status": "operational",
        "service": "face_recognition_api",
        "timestamp": face_service.get_current_timestamp(),
        "port": APP_CONFIG.SERVER_PORT_RECONHECIMENTO
    }), 200

@app.route('/face-login', methods=['POST'])
def face_login_legacy():
    """
    Endpoint LEGACY para autenticação facial (mantido para compatibilidade)
    """
    try:
        # Validação do payload
        if not request.is_json:
            return jsonify({"error": "Content-Type deve ser application/json"}), 400

        data = request.get_json()
        if not data or 'imagem' not in data:
            return jsonify({"error": "Campo obrigatório faltando: 'imagem'"}), 400

        # Processamento da imagem
        image_data = data['imagem']
        result = face_service.process_face_login(image_data)

        # Garantir que todos os campos estejam presentes
        if result.get('authenticated'):
            required_fields = ['id', 'username', 'tipo_usuario', 'nome', 'sobrenome', 'turma']
            for field in required_fields:
                if field not in result:
                    result[field] = None

        return jsonify(result), 200

    except ImageValidationError as e:
        logger.warning(f"VALIDACAO DE IMAGEM FALHOU: {str(e)}")
        return jsonify({
            "authenticated": False,
            "error": "Imagem inválida",
            "message": str(e)
        }), 400
    except FaceRecognitionServiceError as e:
        logger.error(f"ERRO NO SERVICO DE RECONHECIMENTO: {str(e)}")
        return jsonify({
            "error": "Erro no serviço de reconhecimento",
            "message": "Serviço temporariamente indisponível"
        }), 500
    except Exception as e:
        logger.error(f"ERRO INTERNO NO FACE LOGIN: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Erro interno do servidor",
            "message": "Serviço de autenticação temporariamente indisponível"
        }), 500

@app.route('/api/face-login', methods=['POST'])
def face_login():
    """
    Endpoint NOVO para autenticação facial com validação completa
    """
    try:
        # Validação do payload
        if not request.is_json:
            return jsonify({"error": "Content-Type deve ser application/json"}), 400

        data = request.get_json()
        if not data or 'imagem' not in data:
            return jsonify({"error": "Campo obrigatório faltando: 'imagem'"}), 400

        # Validação adicional de payload
        image_data = data['imagem']
        if not isinstance(image_data, str) or len(image_data) == 0:
            return jsonify({"error": "Dados de imagem inválidos"}), 400

        # Processar reconhecimento facial
        result = face_service.process_face_login(image_data)

        # Log detalhado do reconhecimento
        if result.get('authenticated'):
            user_id = result.get('id')
            nome = result.get('nome')
            sobrenome = result.get('sobrenome')
            username = result.get('username')
            tipo_usuario = result.get('tipo_usuario')
            turma = result.get('turma')
            confidence = result.get('confidence', 0)

            logger.info(f"RECONHECIMENTO BEM SUCEDIDO: "
                       f"ID={user_id}, "
                       f"Usuario={nome} {sobrenome}, "
                       f"Username={username}, "
                       f"Tipo={tipo_usuario}, "
                       f"Turma={turma}, "
                       f"Confianca={confidence:.2f}")

            # Garantir que todos os campos obrigatórios estejam presentes
            required_fields = ['id', 'username', 'tipo_usuario', 'nome', 'sobrenome', 'turma']
            for field in required_fields:
                if field not in result:
                    result[field] = None
                    logger.warning(f"CAMPO AUSENTE: {field} nao encontrado no resultado")

        else:
            logger.info(f"RECONHECIMENTO FALHOU: "
                       f"Razao={result.get('message', 'Nao identificado')}, "
                       f"Distancia={result.get('distance', 0):.3f}")

        return jsonify(result), 200

    except ImageValidationError as e:
        logger.warning(f"VALIDACAO DE IMAGEM FALHOU: {str(e)}")
        return jsonify({
            "authenticated": False,
            "error": "Imagem inválida",
            "message": str(e)
        }), 400
    except FaceRecognitionServiceError as e:
        logger.error(f"ERRO NO SERVICO DE RECONHECIMENTO: {str(e)}")
        return jsonify({
            "authenticated": False,
            "error": "Erro no processamento facial",
            "message": str(e)
        }), 500
    except Exception as e:
        logger.error(f"ERRO INTERNO NO FACE LOGIN: {str(e)}", exc_info=True)
        return jsonify({
            "authenticated": False,
            "error": "Erro interno do servidor",
            "message": "Falha temporária no serviço de autenticação"
        }), 500

@app.route('/database-status', methods=['GET'])
def database_status_legacy():
    """Status do banco de dados facial (endpoint legacy)"""
    try:
        status = face_service.get_database_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER STATUS DO BANCO: {str(e)}")
        return jsonify({"error": "Erro ao acessar banco de dados"}), 500

@app.route('/api/database/status', methods=['GET'])
def database_status():
    """Status do banco de dados facial (endpoint novo)"""
    try:
        status = face_service.get_database_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER STATUS DO BANCO: {str(e)}")
        return jsonify({
            "error": "Erro ao acessar banco de dados",
            "message": "Não foi possível conectar ao banco de dados"
        }), 500

@app.route('/api/database/detailed-status', methods=['GET'])
def detailed_database_status():
    """Status detalhado do banco de dados com estatísticas por tipo de usuário"""
    try:
        status = face_service.get_detailed_database_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER STATUS DETALHADO DO BANCO: {str(e)}")
        return jsonify({
            "error": "Erro ao obter status detalhado do banco",
            "message": "Não foi possível conectar ao banco de dados"
        }), 500

@app.route('/reload-database', methods=['POST'])
def reload_database_legacy():
    """Recarregamento manual do banco de dados (endpoint legacy)"""
    try:
        success, message = face_service.reload_database()
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 500
    except Exception as e:
        logger.error(f"ERRO NO RECARREGAMENTO DO BANCO: {str(e)}")
        return jsonify({"error": "Falha no recarregamento do banco"}), 500

@app.route('/api/database/reload', methods=['POST'])
def reload_database():
    """Recarregamento manual do banco de dados (endpoint novo)"""
    try:
        success, message = face_service.reload_database()
        if success:
            logger.info(f"BANCO RECARREGADO: {message}")
            return jsonify({"message": message}), 200
        else:
            logger.error(f"FALHA NO RECARREGAMENTO: {message}")
            return jsonify({"error": message}), 500
    except Exception as e:
        logger.error(f"ERRO NO RECARREGAMENTO DO BANCO: {str(e)}")
        return jsonify({
            "error": "Falha no recarregamento do banco",
            "message": "Não foi possível recarregar o banco de dados"
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
        logger.error(f"ERRO AO OBTER INFORMACOES DO SISTEMA: {str(e)}")
        return jsonify({
            "service": "face_recognition_api",
            "status": "degraded",
            "error": "Não foi possível obter informações completas do sistema"
        }), 500

@app.route('/api/system/metrics', methods=['GET'])
@token_required
def system_metrics():
    """Métricas detalhadas do sistema (requer autenticação)"""
    try:
        metrics = face_service.get_performance_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER METRICAS: {str(e)}")
        return jsonify({"error": "Erro ao obter métricas do sistema"}), 500

@app.route('/api/system/detailed-metrics', methods=['GET'])
def detailed_metrics():
    """Métricas detalhadas do sistema de reconhecimento"""
    try:
        db_status = face_service.get_detailed_database_status()
        metrics = face_service.get_performance_metrics()

        return jsonify({
            "database": db_status,
            "performance": metrics,
            "model_config": {
                "name": "VGG-Face",
                "distance_threshold": 0.60,
                "min_confidence": 0.75,
                "margin_requirement": 0.001
            },
            "timestamp": face_service.get_current_timestamp()
        }), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER METRICAS DETALHADAS: {str(e)}")
        return jsonify({"error": "Erro ao obter métricas"}), 500

@app.route('/api/users/list', methods=['GET'])
@token_required
def list_users():
    """Lista todos os usuários cadastrados (requer autenticação)"""
    try:
        db_status = face_service.get_detailed_database_status()
        return jsonify({
            "message": "Endpoint em desenvolvimento",
            "total_users": db_status["user_count"],
            "professores": db_status["professores_count"],
            "alunos": db_status["alunos_count"]
        }), 200
    except Exception as e:
        logger.error(f"ERRO AO LISTAR USUARIOS: {str(e)}")
        return jsonify({"error": "Erro ao listar usuários"}), 500

# Handlers de erro melhorados
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
    face_service.cleanup()
    sys.exit(0)

def initialize_application():
    """Inicialização da aplicação"""

    try:
        if face_service.initialize():
            # Obter status detalhado do banco
            db_status = face_service.get_detailed_database_status()

            logger.info("ESTATISTICAS DO BANCO:")
            logger.info(f"   Total de usuários: {db_status['user_count']}")
            logger.info(f"   Professores: {db_status['professores_count']}")
            logger.info(f"   Alunos: {db_status['alunos_count']}")
            logger.info(f"   Total de embeddings: {db_status['total_embeddings']}")

            logger.info("CONFIGURACAO DO MODELO:")
            logger.info(f"   Distância máxima: {0.60}")
            logger.info(f"   Confiança mínima: {0.80}")
            logger.info(f"   Margem mínima: {0.001}")

            logger.info("MONITORAMENTO: Monitoramento em tempo real do banco: ATIVO")
            return True
        else:
            logger.error("FALHA: Erro na inicializacao do servico de reconhecimento facial")
            return False
    except Exception as e:
        logger.error(f"FALHA: Erro na inicializacao da aplicacao: {str(e)}")
        return False

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
            logger.info("INTERRUPCAO: Servidor interrompido pelo usuario")
        except Exception as e:
            logger.error(f"ERRO DURANTE EXECUCAO: {str(e)}")
        finally:
            face_service.cleanup()
    else:
        logger.critical("FALHA: Aplicacao falhou ao iniciar - encerrando")
        exit(1)