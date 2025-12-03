"""
Rotas de Reconhecimento Facial - VERSÃO CORRIGIDA
"""
from flask import Blueprint, jsonify, request
import logging
from common.auth import token_required
from common.exceptions import ImageValidationError, FaceRecognitionServiceError
from common.event_loop_manager import EventLoopManager
from recognition_service import RecognitionService
from common.locker_controller import LockerController

logger = logging.getLogger(__name__)
recognition_bp = Blueprint('recognition', __name__)

# Inicialização dos serviços
# Inicialização dos serviços
recognition_service = RecognitionService()
locker_controller = None  # Não inicializa aqui, será lazy

def get_locker_controller():
    """Getter lazy para LockerController"""
    global locker_controller
    if locker_controller is None:
        from common.locker_controller import LockerController
        locker_controller = LockerController()
        logger.info("LockerController inicializado lazy")
    return locker_controller

@recognition_bp.route('/health', methods=['GET'])
def health_check():
    """Health check do servidor (endpoint legacy)"""
    return jsonify({"status": "healthy"}), 200

@recognition_bp.route('/api/health', methods=['GET'])
def health_check_new():
    """Health check do servidor (endpoint novo)"""
    from common.config import APP_CONFIG
    return jsonify({
        "status": "operational",
        "service": "face_recognition_api",
        "timestamp": recognition_service.get_current_timestamp(),
        "port": APP_CONFIG.SERVER_PORT_RECONHECIMENTO
    }), 200

@recognition_bp.route('/face-login', methods=['POST'])
def face_login_legacy():
    """Endpoint LEGACY para autenticação facial (mantido para compatibilidade)"""
    return process_face_login(request, is_legacy=True)

@recognition_bp.route('/api/face-login', methods=['POST'])
def face_login():
    """Endpoint NOVO para autenticação facial"""
    return process_face_login(request, is_legacy=False)

def process_face_login(request, is_legacy=False):
    """Processa o login facial com validação de requisição"""
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

        # Processamento da imagem via serviço
        result = recognition_service.process_face_login(image_data)

        # Log do endpoint utilizado
        endpoint_type = "LEGACY" if is_legacy else "NOVO"
        logger.info(f"=== ENDPOINT {endpoint_type} /face-login CHAMADO ===")

        # Controle da trava para alunos usando EventLoopManager
        if result.get('authenticated'):
            user_type = result.get('tipo_usuario')
            user_id = result.get('id')

            logger.info(f"=== VERIFICANDO CONDIÇÃO PARA ABRIR TRAVA ({endpoint_type}) ===")
            logger.info(f"User Type: {user_type}, User ID: {user_id}")

            if user_type and user_type.upper() == "ALUNO" and user_id:
                logger.info(f"CONDIÇÃO ATENDIDA - ALUNO DETECTADO")
                logger.info(f"INICIANDO PROCESSO DE ABERTURA DA TRAVA...")
                try:
                    # Usar EventLoopManager em vez de criar um novo loop
                    sucesso = EventLoopManager.run_async(
                        get_locker_controller().abrir_trava_aluno(user_id, user_type)
                    )

                    if sucesso:
                        logger.info(f"SUCESSO: Trava liberada via endpoint {endpoint_type} para aluno ID: {user_id}")
                    else:
                        logger.error(f"FALHA: Não foi possível liberar trava via endpoint {endpoint_type} para aluno ID: {user_id}")

                except Exception as e:
                    logger.error(f"ERRO EXCEÇÃO ao controlar trava no endpoint {endpoint_type}: {e}")
            else:
                logger.info(f"CONDIÇÃO NÃO ATENDIDA - Não é aluno ou ID inválido")

        return jsonify(result), 200

    except ImageValidationError as e:
        logger.warning(f"VALIDAÇÃO DE IMAGEM FALHOU: {str(e)}")
        return jsonify({
            "authenticated": False,
            "error": "Imagem inválida",
            "message": str(e)
        }), 400
    except FaceRecognitionServiceError as e:
        logger.error(f"ERRO NO SERVIÇO DE RECONHECIMENTO: {str(e)}")
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

@recognition_bp.route('/database-status', methods=['GET'])
def database_status_legacy():
    """Status do banco de dados facial (endpoint legacy)"""
    try:
        status = recognition_service.get_database_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER STATUS DO BANCO: {str(e)}")
        return jsonify({"error": "Erro ao acessar banco de dados"}), 500

@recognition_bp.route('/api/database/status', methods=['GET'])
def database_status():
    """Status do banco de dados facial (endpoint novo)"""
    try:
        status = recognition_service.get_database_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER STATUS DO BANCO: {str(e)}")
        return jsonify({
            "error": "Erro ao acessar banco de dados",
            "message": "Não foi possível conectar ao banco de dados"
        }), 500

@recognition_bp.route('/api/database/detailed-status', methods=['GET'])
def detailed_database_status():
    """Status detalhado do banco de dados com estatísticas por tipo de usuário"""
    try:
        status = recognition_service.get_detailed_database_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER STATUS DETALHADO DO BANCO: {str(e)}")
        return jsonify({
            "error": "Erro ao obter status detalhado do banco",
            "message": "Não foi possível conectar ao banco de dados"
        }), 500

@recognition_bp.route('/reload-database', methods=['POST'])
def reload_database_legacy():
    """Recarregamento manual do banco de dados (endpoint legacy)"""
    try:
        success, message = recognition_service.reload_database()
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 500
    except Exception as e:
        logger.error(f"ERRO NO RECARREGAMENTO DO BANCO: {str(e)}")
        return jsonify({"error": "Falha no recarregamento do banco"}), 500

@recognition_bp.route('/api/database/reload', methods=['POST'])
def reload_database():
    """Recarregamento manual do banco de dados (endpoint novo)"""
    try:
        success, message = recognition_service.reload_database()
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

@recognition_bp.route('/api/system/info', methods=['GET'])
def system_info():
    """Informações do sistema"""
    try:
        db_status = recognition_service.get_detailed_database_status()
        metrics = recognition_service.get_performance_metrics()

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
            "timestamp": recognition_service.get_current_timestamp()
        }), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER INFORMAÇÕES DO SISTEMA: {str(e)}")
        return jsonify({
            "service": "face_recognition_api",
            "status": "degraded",
            "error": "Não foi possível obter informações completas do sistema"
        }), 500

@recognition_bp.route('/api/system/metrics', methods=['GET'])
@token_required
def system_metrics():
    """Métricas detalhadas do sistema (requer autenticação)"""
    try:
        metrics = recognition_service.get_performance_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER MÉTRICAS: {str(e)}")
        return jsonify({"error": "Erro ao obter métricas do sistema"}), 500

@recognition_bp.route('/api/system/detailed-metrics', methods=['GET'])
def detailed_metrics():
    """Métricas detalhadas do sistema de reconhecimento"""
    try:
        db_status = recognition_service.get_detailed_database_status()
        metrics = recognition_service.get_performance_metrics()

        return jsonify({
            "database": db_status,
            "performance": metrics,
            "model_config": {
                "name": "VGG-Face",
                "distance_threshold": 0.60,
                "min_confidence": 0.75,
                "margin_requirement": 0.001
            },
            "timestamp": recognition_service.get_current_timestamp()
        }), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER MÉTRICAS DETALHADAS: {str(e)}")
        return jsonify({"error": "Erro ao obter métricas"}), 500

@recognition_bp.route('/api/users/list', methods=['GET'])
@token_required
def list_users():
    """Lista todos os usuários cadastrados (requer autenticação)"""
    try:
        db_status = recognition_service.get_detailed_database_status()
        return jsonify({
            "message": "Endpoint em desenvolvimento",
            "total_users": db_status["user_count"],
            "professores": db_status["professores_count"],
            "alunos": db_status["alunos_count"]
        }), 200
    except Exception as e:
        logger.error(f"ERRO AO LISTAR USUÁRIOS: {str(e)}")
        return jsonify({"error": "Erro ao listar usuários"}), 500

@recognition_bp.route('/api/locker/status', methods=['GET'])
def locker_status():
    """Retorna status atual da trava"""
    try:
        status = EventLoopManager.run_async(get_locker_controller().verificar_estado_trava())
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Erro ao verificar status da trava: {e}")
        return jsonify({"error": "Erro ao verificar status"}), 500

@recognition_bp.route('/api/debug/trava', methods=['POST'])
def debug_trava():
    """Endpoint para debug manual da trava"""
    try:
        data = request.get_json() or {}
        comando = data.get('comando', 'ABRIR_TRAVA')

        logger.info(f"DEBUG TRAVA - Comando solicitado: {comando}")

        if comando.upper() == 'ABRIR_TRAVA':
            sucesso = EventLoopManager.run_async(get_locker_controller().abrir_trava_aluno(999, "ALUNO"))
            mensagem = "Abrir trava"
        elif comando.upper() == 'FECHAR_TRAVA':
            sucesso = EventLoopManager.run_async(get_locker_controller().esp32_client.fechar_trava())
            mensagem = "Fechar trava"
        else:
            sucesso = False
            mensagem = "Comando inválido"

        return jsonify({
            "sucesso": sucesso,
            "mensagem": f"{mensagem} - {'Sucesso' if sucesso else 'Falha'}",
            "conectado": get_locker_controller().esp32_client.conectado,
            "trava_aberta": get_locker_controller().trava_aberta
        }), 200 if sucesso else 500

    except Exception as e:
        logger.error(f"ERRO no debug trava: {e}")
        return jsonify({"error": str(e)}), 500