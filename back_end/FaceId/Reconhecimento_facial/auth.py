"""
Módulo de Autenticação JWT
Gerencia tokens de acesso e validação de permissões
"""

import jwt
import os
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
import logging

logger = logging.getLogger(__name__)

# Configurações de segurança
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "face-recognition-secure-key-change-in-production")
TOKEN_EXPIRATION_HOURS = int(os.getenv("TOKEN_EXPIRATION_HOURS", "24"))
ALGORITHM = "HS256"


class AuthError(Exception):
    """Exceção personalizada para erros de autenticação"""
    pass


def generate_token(user_id: str, user_role: str = "user") -> str:
    """
    Gera token JWT para usuário autenticado

    Args:
        user_id: Identificador único do usuário
        user_role: Perfil de acesso do usuário

    Returns:
        str: Token JWT assinado
    """
    try:
        payload = {
            'exp': datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS),
            'iat': datetime.utcnow(),
            'sub': user_id,
            'role': user_role,
            'iss': 'face_recognition_api'
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    except Exception as e:
        logger.error(f"Token generation error: {str(e)}")
        raise AuthError("Failed to generate authentication token")


def validate_token(token: str) -> dict:
    """
    Valida e decodifica token JWT

    Args:
        token: Token JWT a ser validado

    Returns:
        dict: Payload do token decodificado

    Raises:
        AuthError: Se o token for inválido ou expirado
    """
    try:
        if token.startswith('Bearer '):
            token = token[7:]

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {str(e)}")
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        raise AuthError("Token validation failed")


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
                    "error": "Authentication required",
                    "message": "Missing Authorization header"
                }), 401

            payload = validate_token(auth_header)
            request.user_id = payload['sub']
            request.user_role = payload.get('role', 'user')

            return f(*args, **kwargs)

        except AuthError as e:
            return jsonify({
                "error": "Authentication failed",
                "message": str(e)
            }), 401
        except Exception as e:
            logger.error(f"Authentication middleware error: {str(e)}")
            return jsonify({
                "error": "Authentication error",
                "message": "Unexpected authentication failure"
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
                return jsonify({"error": "Authentication required"}), 401

            payload = validate_token(auth_header)

            if payload.get('role') != 'admin':
                return jsonify({
                    "error": "Insufficient permissions",
                    "message": "Admin role required"
                }), 403

            request.user_id = payload['sub']
            request.user_role = payload['role']

            return f(*args, **kwargs)

        except AuthError as e:
            return jsonify({"error": str(e)}), 401
        except Exception as e:
            logger.error(f"Admin auth error: {str(e)}")
            return jsonify({"error": "Authorization check failed"}), 500

    return decorated


def get_token_expiration() -> datetime:
    """Retorna a data de expiração padrão para tokens"""
    return datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS)