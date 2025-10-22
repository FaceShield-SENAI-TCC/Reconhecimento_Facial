"""
Servidor Principal de Reconhecimento Facial
Gerencia endpoints de autenticação facial e status do sistema
"""


import os
import logging
import signal
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
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
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


# Configuração CORS mais permissiva para desenvolvimento
CORS(app, resources={
   r"/*": {
       "origins": "*",
       "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
       "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
       "supports_credentials": True
   }
})


# Handler global para CORS preflight
@app.after_request
def after_request(response):
   response.headers.add('Access-Control-Allow-Origin', '*')
   response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
   response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
   return response


# Handler específico para OPTIONS (CORS preflight)
@app.route('/face-login', methods=['OPTIONS'])
@app.route('/api/face-login', methods=['OPTIONS'])
def options_face_login():
   return '', 200


@app.route('/health', methods=['GET'])
def health_check():
   """Health check do servidor (endpoint antigo)"""
   return jsonify({"status": "healthy"}), 200


@app.route('/api/health', methods=['GET'])
def health_check_new():
   """Health check do servidor (endpoint novo)"""
   return jsonify({
       "status": "operational",
       "service": "face_recognition_api",
       "timestamp": face_service.get_current_timestamp()
   }), 200


@app.route('/face-login', methods=['POST'])
def face_login_legacy():
   """
   Endpoint LEGACY para autenticação facial (mantido para compatibilidade)
   """
   try:
       # Validação do payload
       if not request.is_json:
           return jsonify({"error": "Content-Type must be application/json"}), 400


       data = request.get_json()
       if not data or 'imagem' not in data:
           return jsonify({"error": "Missing required field: 'imagem'"}), 400


       # Processamento da imagem
       image_data = data['imagem']
       if ',' in image_data:
           image_data = data['imagem'].split(',', 1)[1]


       # Reconhecimento facial
       result = face_service.process_face_login(image_data)
       return jsonify(result), 200


   except Exception as e:
       logger.error(f"Face login error: {str(e)}", exc_info=True)
       return jsonify({
           "error": "Internal server error",
           "message": "Authentication service temporarily unavailable"
       }), 500


@app.route('/api/face-login', methods=['POST'])
def face_login():
   """
   Endpoint NOVO para autenticação facial
   """
   try:
       # Validação do payload
       if not request.is_json:
           return jsonify({"error": "Content-Type must be application/json"}), 400


       data = request.get_json()
       if not data or 'imagem' not in data:
           return jsonify({"error": "Missing required field: 'imagem'"}), 400


       # Processamento da imagem
       image_data = data['imagem']
       if ',' in image_data:
           image_data = data['imagem'].split(',', 1)[1]


       # Reconhecimento facial
       result = face_service.process_face_login(image_data)
       return jsonify(result), 200


   except Exception as e:
       logger.error(f"Face login error: {str(e)}", exc_info=True)
       return jsonify({
           "error": "Internal server error",
           "message": "Authentication service temporarily unavailable"
       }), 500


@app.route('/database-status', methods=['GET'])
def database_status_legacy():
   """Status do banco de dados facial (endpoint antigo)"""
   status = face_service.get_database_status()
   return jsonify(status), 200


@app.route('/api/database/status', methods=['GET'])
def database_status():
   """Status do banco de dados facial (endpoint novo)"""
   status = face_service.get_database_status()
   return jsonify(status), 200


@app.route('/reload-database', methods=['POST'])
def reload_database_legacy():
   """Recarregamento manual do banco de dados (endpoint antigo)"""
   try:
       success, message = face_service.reload_database()
       if success:
           return jsonify({"message": message}), 200
       else:
           return jsonify({"error": message}), 500
   except Exception as e:
       logger.error(f"Database reload error: {str(e)}")
       return jsonify({"error": "Failed to reload database"}), 500


@app.route('/api/database/reload', methods=['POST'])
def reload_database():
   """Recarregamento manual do banco de dados (endpoint novo)"""
   try:
       success, message = face_service.reload_database()
       if success:
           return jsonify({"message": message}), 200
       else:
           return jsonify({"error": message}), 500
   except Exception as e:
       logger.error(f"Database reload error: {str(e)}")
       return jsonify({"error": "Failed to reload database"}), 500


@app.route('/api/system/info', methods=['GET'])
def system_info():
   """Informações do sistema"""
   db_status = face_service.get_database_status()
   return jsonify({
       "service": "face_recognition_api",
       "version": "1.0.0",
       "status": "operational",
       "database": db_status,
       "timestamp": face_service.get_current_timestamp()
   }), 200


# Handlers de erro melhorados
@app.errorhandler(404)
def not_found(error):
   return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
   return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
   return jsonify({"error": "Internal server error"}), 500


def signal_handler(sig, frame):
   """Handler para sinais de desligamento"""
   logger.info("🛑 Received shutdown signal, cleaning up...")
   face_service.cleanup()
   sys.exit(0)


def initialize_application():
   """Inicialização da aplicação"""
   logger.info("🚀 Initializing Face Recognition API Server...")


   try:
       if face_service.initialize():
           logger.info("✅ Face recognition service initialized successfully")
           logger.info("📊 Legacy endpoints available:")
           logger.info("   POST /face-login        - Facial authentication")
           logger.info("   GET  /database-status    - Database status")
           logger.info("   POST /reload-database    - Reload database")
           logger.info("📊 New endpoints available:")
           logger.info("   POST /api/face-login     - Facial authentication")
           logger.info("   GET  /api/database/status - Database status")
           logger.info("   POST /api/database/reload - Reload database")
           logger.info("   GET  /api/system/info    - System information")
           logger.info("🔔 Real-time database monitoring: ACTIVE")
           return True
       else:
           logger.error("❌ Failed to initialize face recognition service")
           return False
   except Exception as e:
       logger.error(f"❌ Application initialization failed: {str(e)}")
       return False


if __name__ == "__main__":
   # Registrar handlers para sinais de desligamento
   signal.signal(signal.SIGINT, signal_handler)
   signal.signal(signal.SIGTERM, signal_handler)


   if initialize_application():
       try:
           app.run(
               host='0.0.0.0',
               port=5005,
               debug=False,
               threaded=True
           )
       except KeyboardInterrupt:
           logger.info("🛑 Server stopped by user")
       finally:
           face_service.cleanup()
   else:
       logger.critical("🛑 Application failed to start - exiting")
       exit(1)




