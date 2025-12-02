"""
Camada de acesso a dados unificada para PostgreSQL - VERSÃO CORRIGIDA
"""
import logging
import psycopg2
from psycopg2.extras import Json
from contextlib import contextmanager
from typing import Generator, Optional, Tuple, Union

from common.config import DATABASE_CONFIG
from common.exceptions import DatabaseError, DatabaseConnectionError

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
        """
        conn = None
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
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

                    cursor.execute("ALTER TABLE usuarios ADD COLUMN IF NOT EXISTS embeddings JSONB NOT NULL DEFAULT '[]'::jsonb")
                    cursor.execute("ALTER TABLE usuarios ADD COLUMN IF NOT EXISTS foto_perfil BYTEA")
                    cursor.execute("ALTER TABLE usuarios ADD COLUMN IF NOT EXISTS senha VARCHAR(255)")

                    try:
                        cursor.execute("ALTER TABLE usuarios ADD CONSTRAINT usuarios_username_key UNIQUE (username)")
                    except psycopg2.Error as e:
                        if e.pgcode == '42P07':
                            conn.rollback()
                        else:
                            raise e

                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usuarios_nome_sobrenome_turma ON usuarios(nome, sobrenome, turma)")

                    conn.commit()
                    logger.info(" Banco de dados inicializado/atualizado com sucesso")
                    return True

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f" Falha na inicialização/atualização do banco: {str(e)}")
            return False

    def check_user_exists(self, nome: str, sobrenome: str, turma: str) -> int:
        """Verifica se usuário já existe no banco"""
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

    def count_users(self) -> int:
        """Conta número total de usuários cadastrados"""
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
                 embeddings: list, profile_image: bytes) -> Tuple[bool, Union[int, str]]:
        """
        Salva o pré-cadastro do usuário (sem username/senha)

        RETORNO PADRONIZADO:
            - Sucesso: (True, user_id_int)
            - Falha:   (False, error_message_str)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Verificar se usuário já existe
                    cursor.execute(
                        "SELECT COUNT(*) FROM usuarios WHERE nome = %s AND sobrenome = %s AND turma = %s",
                        (nome, sobrenome, turma)
                    )
                    existing_count = cursor.fetchone()[0]

                    if existing_count > 0:
                        return False, f"Usuário {nome} {sobrenome} já está cadastrado na turma {turma}"

                    # Inserir usuário
                    cursor.execute("""
                        INSERT INTO usuarios (nome, sobrenome, turma, tipo_usuario, username, embeddings, foto_perfil)
                        VALUES (%s, %s, %s, %s, NULL, %s, %s)
                        RETURNING id
                    """, (nome, sobrenome, turma, tipo_usuario.upper(), Json(embeddings), profile_image))

                    inserted_row = cursor.fetchone()

                    if inserted_row:
                        user_id = inserted_row[0]
                        conn.commit()
                        logger.info(f"Usuário salvo no DB - ID: {user_id}")
                        return True, user_id  #  RETORNO PADRONIZADO: int ID
                    else:
                        return False, "Falha ao inserir usuário, ID não retornado."

        except Exception as e:
            error_msg = f"Erro ao salvar usuário: {str(e)}"
            logger.error(error_msg)
            return False, error_msg  #  RETORNO PADRONIZADO: string de erro

# Instância global
db_manager = DatabaseManager()