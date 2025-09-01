import os
import threading
from flask import Flask, jsonify, request
from flask_cors import CORS
from face_recognition_logic import initialize, process_face_login, get_database_status, reload_database, cleanup
import atexit
import logging
import sys
import io
import time

# Configurar stdout e stderr para UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configurar o Flask
app = Flask(__name__)
CORS(app)  # Permite requisições de diferentes origens (CORS)


# ====================== ROTAS DA API ======================

@app.route('/health', methods=['GET'])
def health_check():
    """Verifica se o servidor está rodando."""
    return jsonify({"status": "healthy"}), 200


@app.route('/face-login', methods=['POST'])
def face_login():
    """Endpoint para login com reconhecimento facial."""
    logger.info("Recebida requisição para /face-login")
    try:
        data = request.json
        if not data or 'imagem' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data.get('imagem')

        # Correção: garantir que o prefixo seja removido de forma segura
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        result, status_code = process_face_login(image_data)

        if status_code == 200:
            logger.info(f"Login processado com sucesso. Resultado: {result['message']}")
        else:
            logger.error(
                f"Erro ao processar login. Código: {status_code}, Mensagem: {result.get('error') or result.get('message')}")

        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"Erro inesperado na rota /face-login: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route('/database-status', methods=['GET'])
def database_status():
    """Retorna o status do banco de dados facial."""
    status = get_database_status()
    return jsonify(status), 200


@app.route('/reload-database', methods=['POST'])
def reload_database_route():
    """Recarrega o banco de dados facial manualmente."""
    success, message = reload_database()
    if success:
        return jsonify({"message": message}), 200
    else:
        return jsonify({"error": message}), 500


# Nova rota para verificar o status do monitoramento
@app.route('/monitor-status', methods=['GET'])
def monitor_status():
    """Retorna o status do monitoramento do banco de dados."""
    from face_recognition_logic import db_observer, last_db_update

    status = {
        "monitor_active": db_observer is not None and db_observer.is_alive(),
        "last_update": last_db_update,
        "database_dir": "C:/Users/WBS/Desktop/Arduino/back_end/FaceId/Cadastro/Faces"
    }

    return jsonify(status), 200


# ====================== INICIALIZAÇÃO ======================

def start_app():
    """Inicializa o sistema e o servidor Flask."""
    if initialize():
        logger.info("Servidor de reconhecimento facial pronto para aceitar conexões.")
        logger.info("Monitoramento do banco de dados ativado - o sistema atualizará automaticamente.")
    else:
        logger.error("Falha na inicialização. Não foi possível iniciar o servidor.")
        sys.exit(1)


# Registrar a função de limpeza
atexit.register(cleanup)

if __name__ == "__main__":
    start_app()
    app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)