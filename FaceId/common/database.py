"""
Camada de acesso a dados unificada para PostgreSQL
"""
import logging
import psycopg2
from psycopg2.extras import Json
from contextlib import contextmanager
from typing import Generator, Optional, Tuple

from common.config import DATABASE_CONFIG
from common.exceptions import DatabaseError, DatabaseConnectionError

# A linha "from common.models" foi REMOVIDA

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gerenciador centralizado de conexões com o banco de dados"""

    def __init__(self, db_config=None):
        self.db_config = db_config or DATABASE_CONFIG.to_dict()

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Context manager para conexões de banco de dados com tratamento de erro
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            yield conn
        except psycopg2.OperationalError as e:
            logger.error(f"Erro de conexão com o banco: {str(e)}")
            raise DatabaseConnectionError(f"Falha na conexão: {str(e)}")
        except Exception as e:
            logger.error(f"Erro inesperado na conexão: {str(e)}")
            raise DatabaseError(f"Erro de banco de dados: {str(e)}")
        finally:
            if conn:
                conn.close()

    def init_database(self) -> bool:
        """
        Inicializa (Cria) ou Atualiza (Altera) tabelas do banco de dados
        para garantir que todas as colunas existam.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. TENTA CRIAR A TABELA
                    # (Ela bate com a sua imagem, com 'username' e 'senha' nulos)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS usuarios(
                            id SERIAL PRIMARY KEY,
                            nome VARCHAR(100) NOT NULL,
                            sobrenome VARCHAR(100) NOT NULL,
                            tipo_usuario VARCHAR(20) NOT NULL DEFAULT 'ALUNO',
                            turma VARCHAR(50) NOT NULL,
                            username VARCHAR(100) UNIQUE,
                            senha VARCHAR(255),
                            embeddings JSONB NOT NULL,
                            foto_perfil BYTEA
                        )
                    """)

                    # 2. GARANTE QUE AS COLUNAS EXISTAM
                    cursor.execute("ALTER TABLE usuarios ADD COLUMN IF NOT EXISTS embeddings JSONB NOT NULL DEFAULT '[]'::jsonb")
                    cursor.execute("ALTER TABLE usuarios ADD COLUMN IF NOT EXISTS foto_perfil BYTEA")
                    cursor.execute("ALTER TABLE usuarios ADD COLUMN IF NOT EXISTS senha VARCHAR(255)")

                    # Garante que username é UNIQUE
                    try:
                        cursor.execute("ALTER TABLE usuarios ADD CONSTRAINT usuarios_username_key UNIQUE (username)")
                    except psycopg2.Error as e:
                        if e.pgcode == '42P07': # duplicate_table / constraint_already_exists
                            conn.rollback()
                        else:
                            raise e

                    # Cria índice para busca por nome/sobrenome/turma
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usuarios_nome_sobrenome_turma ON usuarios(nome, sobrenome, turma)")

                    conn.commit()
                    logger.info("✅ Banco de dados inicializado/atualizado com sucesso")
                    return True

        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Falha na inicialização/atualização do banco: {str(e)}")
            return False

    def check_user_exists(self, nome: str, sobrenome: str, turma: str) -> int:
        """ Verifica se usuário já existe no banco """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT COUNT(*) FROM usuarios WHERE nome = %s AND sobrenome = %s AND turma = %s",
                        (nome, sobrenome, turma)
                    )
                    count = cursor.fetchone()[0]
                    return count
        except Exception as e:
            logger.error(f"Erro ao verificar usuário: {e}")
            return 0

    def check_username_exists(self, username: str) -> int:
        """ Verifica se username já existe no banco """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT COUNT(*) FROM usuarios WHERE username = %s AND username IS NOT NULL",
                        (username,)
                    )
                    count = cursor.fetchone()[0]
                    return count
        except Exception as e:
            logger.error(f"Erro ao verificar username: {e}")
            return 0

    def count_users(self) -> int:
        """ Conta número total de usuários cadastrados """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM usuarios")
                    count = cursor.fetchone()[0]
                    return count
        except Exception as e:
            logger.error(f"Erro ao contar usuários: {e}")
            return 0

    def save_user(self, nome: str, sobrenome: str, turma: str, tipo_usuario: str,
                 embeddings: list, profile_image: bytes) -> Tuple[bool, object]:
        """
        Salva o pré-cadastro do usuário (sem username/senha)

        RETORNA:
            (True, <dicionario_com_id>) em sucesso (ex: (True, {'id': 42}))
            (False, <string_de_erro>) em falha
        """

        # --- ESTA É A CORREÇÃO MAIS IMPORTANTE ---
        # 1. username é NULL (será preenchido pelo Java)
        # 2. Usamos 'RETURNING id' para pegar o ID que o Postgres acabou de criar

        sql = """
            INSERT INTO usuarios (nome, sobrenome, turma, tipo_usuario, username, embeddings, foto_perfil)
            VALUES (%s, %s, %s, %s, NULL, %s, %s)
            ON CONFLICT (nome, sobrenome, turma) 
            DO NOTHING 
            RETURNING id 
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:

                    # Verificar se usuário já existe
                    existing_count = self.check_user_exists(nome, sobrenome, turma)
                    if existing_count > 0:
                        return False, f"Usuário {nome} {sobrenome} turma {turma} já existe."

                    # Inserir usuário
                    cursor.execute(
                        sql,
                        (nome, sobrenome, turma, tipo_usuario.upper(), Json(embeddings), profile_image)
                    )

                    # Recuperar o ID que acabou de ser inserido
                    inserted_row = cursor.fetchone()

                    if inserted_row:
                        conn.commit()
                        user_id = inserted_row[0] # Pega o ID (ex: 42)

                        # Retorna um DICIONÁRIO com o ID
                        return True, {"id": user_id}
                    else:
                        # Isso acontece se o ON CONFLICT foi ativado
                        return False, f"Usuário {nome} {sobrenome} turma {turma} já existe (ON CONFLICT)."

        except Exception as e:
            conn.rollback()
            error_msg = f"Erro ao salvar usuário: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """ Busca usuário pelo username """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, nome, sobrenome, tipo_usuario, turma, username, embeddings
                        FROM usuarios 
                        WHERE username = %s AND username IS NOT NULL
                    """, (username,))

                    result = cursor.fetchone()
                    if result:
                        return {
                            'id': result[0],
                            'nome': result[1],
                            'sobrenome': result[2],
                            'tipo_usuario': result[3],
                            'turma': result[4],
                            'username': result[5],
                            'embeddings': result[6]
                        }
                    return None
        except Exception as e:
            logger.error(f"Erro ao buscar usuário por username: {e}")
            return None

# Instância global
db_manager = DatabaseManager()