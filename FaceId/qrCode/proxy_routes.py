"""
Proxy para redirecionar requisições de ferramentas para o servidor Java
"""
import requests
from flask import Blueprint, jsonify, request
import logging

logger = logging.getLogger(__name__)
proxy_bp = Blueprint('proxy_bp', __name__)

# URL base do servidor Java
JAVA_API_BASE = "http://localhost:8080"


@proxy_bp.route('/ferramentas/buscarPorQRCode/<qrcode>', methods=['GET'])
def buscar_ferramenta_por_qrcode(qrcode):
    """Proxy para buscar ferramenta por QR Code no Java"""
    try:
        logger.info(f"Proxy: Buscando ferramenta com QR Code: {qrcode}")

        # Fazer requisição para o Java
        response = requests.get(
            f"{JAVA_API_BASE}/ferramentas/buscarPorQRCode/{qrcode}",
            timeout=10
        )

        # Retornar exatamente o que o Java retornou
        return response.content, response.status_code, {'Content-Type': 'application/json'}

    except requests.exceptions.ConnectionError:
        logger.error("Proxy: Não foi possível conectar ao servidor Java")
        return jsonify({
            'error': 'Servidor Java indisponível',
            'message': 'Não foi possível conectar ao sistema de ferramentas'
        }), 503

    except Exception as e:
        logger.error(f"Proxy: Erro ao buscar ferramenta: {str(e)}")
        return jsonify({
            'error': 'Erro interno no proxy',
            'message': str(e)
        }), 500


@proxy_bp.route('/ferramentas/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_ferramentas(path):
    """Proxy genérico para todas as rotas de ferramentas"""
    try:
        # Construir URL completa
        url = f"{JAVA_API_BASE}/ferramentas/{path}"
        logger.info(f"Proxy: Redirecionando para {url}")

        # Encaminhar requisição
        method = request.method.lower()
        func = getattr(requests, method)

        response = func(
            url,
            headers={key: value for (key, value) in request.headers if key != 'Host'},
            data=request.get_data(),
            params=request.args,
            timeout=10
        )

        return response.content, response.status_code, {'Content-Type': 'application/json'}

    except Exception as e:
        logger.error(f"Proxy: Erro ao redirecionar: {str(e)}")
        return jsonify({'error': str(e)}), 500


@proxy_bp.route('/locais/<path:path>', methods=['GET'])
def proxy_locais(path):
    """Proxy para rotas de locais"""
    try:
        url = f"{JAVA_API_BASE}/locais/{path}"
        logger.info(f"Proxy: Redirecionando para {url}")

        response = requests.get(url, timeout=10)
        return response.content, response.status_code, {'Content-Type': 'application/json'}

    except Exception as e:
        logger.error(f"Proxy: Erro ao buscar locais: {str(e)}")
        return jsonify({'error': str(e)}), 500