from flask import Flask, request, jsonify
from flask_cors import CORS
import atexit
import sys
import io
import logging
import traceback

# Configurar stdout e stderr para UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Importar funções do módulo de lógica
from face_recognition_logic import (
    initialize,
    cleanup,
    process_face_login,
    get_database_status,
    reload_database,
    is_cache_valid,
    last_db_update,
    logger
)

# ====================== CONFIGURAÇÕES ======================
app = Flask(__name__)
# Configuração CORS para permitir todas as origens (apenas para desenvolvimento)
CORS(app, origins="*")  # Permite todas as origens


# ====================== ROTAS DA API ======================
def _build_cors_preflight_response():
    response = jsonify()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response


@app.route('/face-login', methods=['POST', 'OPTIONS'])
def face_login():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    logger.info("Recebendo solicitacao de login facial")

    # Obter imagem do request
    data = request.json
    if not data or 'imagem' not in data:
        logger.warning("Nenhuma imagem fornecida na solicitacao")
        return jsonify({"error": "No image provided"}), 400

    try:
        # Processar a imagem usando a lógica de reconhecimento facial
        result, status_code = process_face_login(data['imagem'])

        # Adicionar headers CORS à resposta
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, status_code

    except Exception as e:
        logger.error(f"Erro no endpoint face-login: {str(e)}")
        logger.error(traceback.format_exc())
        response = jsonify({"error": "Internal server error"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500


@app.route('/test-db', methods=['GET'])
def test_db():
    status = get_database_status()
    if not status["loaded"]:
        response = jsonify({"status": "Database not loaded"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    response = jsonify({
        "status": "Database loaded",
        "users": status["users"],
        "user_count": status["user_count"],
        "total_embeddings": status["total_embeddings"]
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/reload-db', methods=['POST'])
def reload_db():
    """Rota para recarregar manualmente o banco de dados"""
    success, message = reload_database()

    if success:
        response = jsonify({
            "success": True,
            "message": message
        })
    else:
        response = jsonify({
            "success": False,
            "error": message
        }), 500

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/health', methods=['GET'])
def health_check():
    status = get_database_status()
    response_data = {
        "status": "healthy",
        "database_loaded": status["loaded"],
        "user_count": status["user_count"],
        "cache_valid": is_cache_valid(),
        "last_update": last_db_update
    }
    response = jsonify(response_data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# ====================== INICIALIZAÇÃO E LIMPEZA ======================
@atexit.register
def cleanup_app():
    cleanup()


if __name__ == '__main__':
    try:
        # Inicializar o sistema de reconhecimento facial
        if initialize():
            logger.info("Sistema de reconhecimento facial inicializado com sucesso")
        else:
            logger.error("Falha ao inicializar o sistema de reconhecimento facial")

        # Configurar opções do servidor
        options = {
            'host': '0.0.0.0',
            'port': 5005,
            'debug': True,
            'use_reloader': False,  # Desativar reloader para evitar problemas de threading
            'threaded': True,  # Permitir múltiplas conexões
        }

        logger.info("Iniciando servidor Flask...")
        app.run(**options)

    except Exception as e:
        logger.error(f"Erro ao iniciar servidor: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Garantir que a limpeza seja feita
        cleanup()