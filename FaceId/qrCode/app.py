import asyncio
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

# Configuração do CORS para permitir seu frontend no Render e desenvolvimento local
CORS(app, origins=[
    'http://127.0.0.1:5500',
    'http://localhost:5500',
    'http://localhost:8080',
    'https://faceshield.onrender.com',  # URL frontend
    'https://*.onrender.com'  # Permite qualquer subdomínio do Render
])

# Ou para permitir TODOS os domínios (menos seguro, mas funcional):
# CORS(app)

# Importar e registrar rotas
from qr_routes import qr_bp
from iot_routes import iot_bp

app.register_blueprint(qr_bp)
app.register_blueprint(iot_bp)

@app.route('/test', methods=['GET'])
def test():
    from flask import jsonify
    return jsonify({'message': 'Backend Python funcionando!', 'status': 'ok'})

if __name__ == '__main__':
    print("INICIALIZACAO: Servidor QR Code iniciando na porta 5000...")
    print("TESTE: Acesse http://localhost:5000/test para testar")
    app.run(host='0.0.0.0', port=5000, debug=True)