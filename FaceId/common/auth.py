"""
Sistema de autenticação JWT unificado para ambos os serviços
"""
import jwt
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
from typing import Optional, Dict, Any

from common.config import SECURITY_CONFIG
from common.exceptions import AuthenticationError, TokenExpiredError, InvalidTokenError

logger = logging.getLogger(__name__)

class JWTAuthManager:
    """Gerenciador de autenticação JWT com métodos estáticos"""

    @staticmethod
    def generate_token(user_id: str, user_role: str = "user",
                      additional_claims: Optional[Dict] = None) -> str:
        """
        Gera token JWT para usuário autenticado

        Args:
            user_id: Identificador único do usuário
            user_role: Perfil de acesso (user/admin)
            additional_claims: Claims adicionais opcionais

        Returns:
            str: Token JWT assinado

        Raises:
            AuthenticationError: Se falhar na geração
        """
        try:
            payload = {
                'exp': datetime.utcnow() + timedelta(hours=SECURITY_CONFIG.TOKEN_EXPIRATION_HOURS),
                'iat': datetime.utcnow(),
                'sub': user_id,
                'role': user_role,
                'iss': 'face_recognition_system'
            }

            # Adicionar claims extras se fornecidas
            if additional_claims:
                payload.update(additional_claims)

            return jwt.encode(payload, SECURITY_CONFIG.SECRET_KEY, algorithm=SECURITY_CONFIG.ALGORITHM)

        except Exception as e:
            logger.error(f"Erro na geração de token: {str(e)}")
            raise AuthenticationError("Falha na geração do token de autenticação")

    @staticmethod
    def validate_token(token: str) -> Dict[str, Any]:
        """
        Valida e decodifica token JWT

        Args:
            token: Token JWT a ser validado

        Returns:
            Dict: Payload do token decodificado

        Raises:
            TokenExpiredError: Se o token expirou
            InvalidTokenError: Se o token é inválido
            AuthenticationError: Para outros erros
        """
        try:
            # Remover 'Bearer ' se presente
            if token.startswith('Bearer '):
                token = token[7:]

            payload = jwt.decode(
                token,
                SECURITY_CONFIG.SECRET_KEY,
                algorithms=[SECURITY_CONFIG.ALGORITHM]
            )
            return payload

        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token expirado")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Token inválido: {str(e)}")
        except Exception as e:
            logger.error(f"Erro na validação do token: {str(e)}")
            raise AuthenticationError("Falha na validação do token")

    @staticmethod
    def get_token_expiration() -> datetime:
        """Retorna a data de expiração padrão para tokens"""
        return datetime.utcnow() + timedelta(hours=SECURITY_CONFIG.TOKEN_EXPIRATION_HOURS)

# Decorators para endpoints protegidos
def token_required(f):
    """
    Decorator para endpoints que requerem autenticação JWT
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({
                    "error": "Autenticação necessária",
                    "message": "Header Authorization não encontrado"
                }), 401

            payload = JWTAuthManager.validate_token(auth_header)

            # Adicionar informações do usuário ao request
            request.user_id = payload['sub']
            request.user_role = payload.get('role', 'user')
            request.token_payload = payload

            return f(*args, **kwargs)

        except TokenExpiredError as e:
            return jsonify({
                "error": "Token expirado",
                "message": str(e)
            }), 401
        except InvalidTokenError as e:
            return jsonify({
                "error": "Token inválido",
                "message": str(e)
            }), 401
        except AuthenticationError as e:
            return jsonify({
                "error": "Falha de autenticação",
                "message": str(e)
            }), 401
        except Exception as e:
            logger.error(f"Erro no middleware de autenticação: {str(e)}")
            return jsonify({
                "error": "Erro de autenticação",
                "message": "Falha inesperada na autenticação"
            }), 500

    return decorated

def admin_required(f):
    """
    Decorator para endpoints que requerem privilégios de administrador
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({"error": "Autenticação necessária"}), 401

            payload = JWTAuthManager.validate_token(auth_header)

            if payload.get('role') != 'admin':
                return jsonify({
                    "error": "Permissões insuficientes",
                    "message": "Acesso de administrador necessário"
                }), 403

            request.user_id = payload['sub']
            request.user_role = payload['role']
            request.token_payload = payload

            return f(*args, **kwargs)

        except Exception as e:
            logger.error(f"Erro na verificação de admin: {str(e)}")
            return jsonify({
                "error": "Falha na verificação de permissões"
            }), 500

    return decorated