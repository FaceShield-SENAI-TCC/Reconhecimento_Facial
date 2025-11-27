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
import asyncio
from flask import Flask, jsonify, request
from flask_cors import CORS

# Modulos compartilhados
from common.config import APP_CONFIG
from common.auth import token_required
from common.exceptions import ImageValidationError, FaceRecognitionServiceError
from face_recognition_logic import FaceRecognitionService
from common.locker_controller import LockerController

# Configuracao de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('reconhecimento')

# Inicializacao do servico
face_service = FaceRecognitionService()
locker_controller = LockerController()

# Configuracao da aplicacao Flask
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Configuracao CORS
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
    AGORA COM CONTROLE DE TRAVA
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

        # 🔥 NOVO: CONTROLE DE TRAVA NO ENDPOINT LEGACY
        logger.info("=== ENDPOINT LEGACY /face-login CHAMADO ===")

        if result.get('authenticated'):
            user_type = result.get('tipo_usuario')
            user_id = result.get('id')

            logger.info(f"=== VERIFICANDO CONDICAO PARA ABRIR TRAVA (LEGACY) ===")
            logger.info(f"User Type: {user_type}, User ID: {user_id}")

            if user_type and user_type.upper() == "ALUNO" and user_id:
                logger.info(f"CONDICAO ATENDIDA - ALUNO DETECTADO NO ENDPOINT LEGACY")
                logger.info(f"INICIANDO PROCESSO DE ABERTURA DA TRAVA...")
                try:
                    # Executa assincronamente para nao bloquear a resposta
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    sucesso = loop.run_until_complete(
                        locker_controller.abrir_trava_aluno(user_id, user_type)
                    )
                    loop.close()

                    if sucesso:
                        logger.info(f"SUCESSO: Trava liberada via endpoint LEGACY para aluno ID: {user_id}")
                    else:
                        logger.error(
                            f"FALHA: Nao foi possivel liberar trava via endpoint LEGACY para aluno ID: {user_id}")

                except Exception as e:
                    logger.error(f"ERRO EXCECAO ao controlar trava no endpoint LEGACY: {e}")
            else:
                logger.info(f"CONDICAO NAO ATENDIDA - Nao e aluno ou ID invalido (LEGACY)")

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
    Endpoint NOVO para autenticacao facial com controle de trava
    """
    try:
        # Validacao do payload (codigo existente)
        if not request.is_json:
            return jsonify({"error": "Content-Type deve ser application/json"}), 400

        data = request.get_json()
        if not data or 'imagem' not in data:
            return jsonify({"error": "Campo obrigatorio faltando: 'imagem'"}), 400

        # Validacao adicional de payload
        image_data = data['imagem']
        if not isinstance(image_data, str) or len(image_data) == 0:
            return jsonify({"error": "Dados de imagem invalidos"}), 400

        result = face_service.process_face_login(image_data)

        # NOVO: Log detalhado do resultado do reconhecimento
        logger.info(f"=== RESULTADO DO RECONHECIMENTO FACIAL ===")
        logger.info(f"Autenticado: {result.get('authenticated')}")
        logger.info(f"Tipo Usuario: {result.get('tipo_usuario')}")
        logger.info(f"User ID: {result.get('id')}")
        logger.info(f"Nome: {result.get('nome')} {result.get('sobrenome')}")

        # Log detalhado do reconhecimento
        if result.get('authenticated'):
            user_type = result.get('tipo_usuario')
            user_id = result.get('id')

            # NOVO: Controle da trava para alunos - COM MAIS LOGS
            logger.info(f"=== VERIFICANDO CONDICAO PARA ABRIR TRAVA ===")
            logger.info(f"User Type: {user_type}, User ID: {user_id}")

            if user_type and user_type.upper() == "ALUNO" and user_id:
                logger.info(f"CONDICAO ATENDIDA - ALUNO DETECTADO")
                logger.info(f"INICIANDO PROCESSO DE ABERTURA DA TRAVA...")
                try:
                    # Executa assincronamente para nao bloquear a resposta
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    sucesso = loop.run_until_complete(
                        locker_controller.abrir_trava_aluno(user_id, user_type)
                    )
                    loop.close()

                    if sucesso:
                        logger.info(f"SUCESSO: Trava liberada para aluno ID: {user_id}")
                    else:
                        logger.error(f"FALHA: Nao foi possivel liberar trava para aluno ID: {user_id}")

                except Exception as e:
                    logger.error(f"ERRO EXCECAO ao controlar trava: {e}")
            else:
                logger.info(f"CONDICAO NAO ATENDIDA - Nao e aluno ou ID invalido")

            # Log do reconhecimento (codigo existente)
            nome = result.get('nome')
            sobrenome = result.get('sobrenome')
            username = result.get('username')
            turma = result.get('turma')
            confidence = result.get('confidence', 0)

            logger.info(f"RECONHECIMENTO BEM SUCEDIDO: "
                       f"ID={user_id}, "
                       f"Usuario={nome} {sobrenome}, "
                       f"Username={username}, "
                       f"Tipo={user_type}, "
                       f"Turma={turma}, "
                       f"Confianca={confidence:.2f}")

            # Garantir que todos os campos obrigatorios estejam presentes
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
            "error": "Imagem invalida",
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
            "message": "Falha temporaria no servico de autenticacao"
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
            "message": "Nao foi possivel conectar ao banco de dados"
        }), 500

@app.route('/api/database/detailed-status', methods=['GET'])
def detailed_database_status():
    """Status detalhado do banco de dados com estatisticas por tipo de usuario"""
    try:
        status = face_service.get_detailed_database_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER STATUS DETALHADO DO BANCO: {str(e)}")
        return jsonify({
            "error": "Erro ao obter status detalhado do banco",
            "message": "Nao foi possivel conectar ao banco de dados"
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
            "message": "Nao foi possivel recarregar o banco de dados"
        }), 500

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Informacoes do sistema"""
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
            "error": "Nao foi possivel obter informacoes completas do sistema"
        }), 500

@app.route('/api/system/metrics', methods=['GET'])
@token_required
def system_metrics():
    """Metricas detalhadas do sistema (requer autenticacao)"""
    try:
        metrics = face_service.get_performance_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"ERRO AO OBTER METRICAS: {str(e)}")
        return jsonify({"error": "Erro ao obter metricas do sistema"}), 500

@app.route('/api/system/detailed-metrics', methods=['GET'])
def detailed_metrics():
    """Metricas detalhadas do sistema de reconhecimento"""
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
        return jsonify({"error": "Erro ao obter metricas"}), 500

@app.route('/api/users/list', methods=['GET'])
@token_required
def list_users():
    """Lista todos os usuarios cadastrados (requer autenticacao)"""
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
        return jsonify({"error": "Erro ao listar usuarios"}), 500

# NOVO: Endpoint para status da trava
@app.route('/api/locker/status', methods=['GET'])
def locker_status():
    """Retorna status atual da trava"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status = loop.run_until_complete(locker_controller.verificar_estado_trava())
        loop.close()

        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Erro ao verificar status da trava: {e}")
        return jsonify({"error": "Erro ao verificar status"}), 500

# NOVO: Endpoint para debug manual da trava
@app.route('/api/debug/trava', methods=['POST'])
def debug_trava():
    """Endpoint para debug manual da trava"""
    try:
        data = request.get_json() or {}
        comando = data.get('comando', 'ABRIR_TRAVA')

        logger.info(f"DEBUG TRAVA - Comando solicitado: {comando}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if comando.upper() == 'ABRIR_TRAVA':
            sucesso = loop.run_until_complete(locker_controller.abrir_trava_aluno(999, "ALUNO"))
            mensagem = "Abrir trava"
        elif comando.upper() == 'FECHAR_TRAVA':
            sucesso = loop.run_until_complete(locker_controller.esp32_client.fechar_trava())
            mensagem = "Fechar trava"
        else:
            sucesso = False
            mensagem = "Comando invalido"

        loop.close()

        return jsonify({
            "sucesso": sucesso,
            "mensagem": f"{mensagem} - {'Sucesso' if sucesso else 'Falha'}",
            "conectado": locker_controller.esp32_client.conectado,
            "trava_aberta": locker_controller.trava_aberta
        }), 200 if sucesso else 500

    except Exception as e:
        logger.error(f"ERRO no debug trava: {e}")
        return jsonify({"error": str(e)}), 500

# Handlers de erro melhorados
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint nao encontrado",
        "message": "Verifique a URL e tente novamente"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Metodo nao permitido",
        "message": "Este endpoint nao suporta o metodo HTTP utilizado"
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
    """Inicializacao da aplicacao com controle de trava"""

    try:
        if face_service.initialize():
            # Inicializar controlador da trava
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                locker_success = loop.run_until_complete(locker_controller.iniciar())
                loop.close()

                if locker_success:
                    logger.info("Controlador da trava inicializado com sucesso")
                else:
                    logger.warning("Controlador da trava nao pode se conectar a ESP32")
            except Exception as e:
                logger.warning(f"Falha na inicializacao do controlador da trava: {e}")

            # Obter status detalhado do banco (codigo existente)
            db_status = face_service.get_detailed_database_status()

            logger.info("ESTATISTICAS DO BANCO:")
            logger.info(f"   Total de usuarios: {db_status['user_count']}")
            logger.info(f"   Professores: {db_status['professores_count']}")
            logger.info(f"   Alunos: {db_status['alunos_count']}")
            logger.info(f"   Total de embeddings: {db_status['total_embeddings']}")

            logger.info("CONFIGURACAO DO MODELO:")
            logger.info(f"   Distancia maxima: {0.60}")
            logger.info(f"   Confianca minima: {0.80}")
            logger.info(f"   Margem minima: {0.001}")

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