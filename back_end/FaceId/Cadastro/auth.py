import sqlite3
import jwt
import os
from datetime import datetime, timedelta
from flask import request, jsonify
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import logging

logger = logging.getLogger(__name__)

# Configurações
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
TOKEN_EXPIRATION_HOURS = 24
DATABASE_PATH = "facial_auth.db"


def get_auth_connection():
    """Conexão com SQLite para autenticação"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_database():
    """Inicializa banco de autenticação"""
    try:
        conn = get_auth_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                nome TEXT NOT NULL,
                sobrenome TEXT NOT NULL,
                turma TEXT NOT NULL,
                tipo TEXT NOT NULL DEFAULT 'professor',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Criar usuário admin padrão se não existir
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        if cursor.fetchone()[0] == 0:
            password_hash = generate_password_hash("admin123")
            cursor.execute(
                "INSERT INTO users (username, password_hash, nome, sobrenome, turma, tipo) VALUES (?, ?, ?, ?, ?, ?)",
                ('admin', password_hash, 'Admin', 'Sistema', 'admin', 'admin')
            )

        conn.commit()
        conn.close()
        logger.info("✅ Banco de autenticação inicializado")
        return True

    except Exception as e:
        logger.error(f"❌ Erro no banco de autenticação: {e}")
        return False


def token_required(f):
    """Decorator para rotas protegidas"""

    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token necessário'}), 401

        try:
            if token.startswith('Bearer '):
                token = token[7:]

            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            user_id = data['sub']

            conn = get_auth_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            conn.close()

            if not user:
                return jsonify({'message': 'Usuário não encontrado'}), 401

            request.user_id = user_id
            return f(*args, **kwargs)

        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expirado'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token inválido'}), 401
        except Exception as e:
            logger.error(f"Erro na validação do token: {e}")
            return jsonify({'message': 'Erro de autenticação'}), 500

    return decorated


def generate_token(user_id):
    """Gera token JWT"""
    payload = {
        'exp': datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS),
        'iat': datetime.utcnow(),
        'sub': user_id
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')


def login():
    """Endpoint de login"""
    try:
        auth_data = request.get_json()
        if not auth_data:
            return jsonify({'message': 'Dados necessários'}), 400

        username = auth_data.get('username', '').strip()
        password = auth_data.get('password', '')

        if not username or not password:
            return jsonify({'message': 'Usuário e senha obrigatórios'}), 400

        conn = get_auth_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            token = generate_token(user['id'])
            return jsonify({
                'token': token,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'nome': user['nome'],
                    'sobrenome': user['sobrenome'],
                    'tipo': user['tipo']
                }
            })

        return jsonify({'message': 'Credenciais inválidas'}), 401

    except Exception as e:
        logger.error(f"Erro no login: {e}")
        return jsonify({'message': 'Erro interno'}), 500