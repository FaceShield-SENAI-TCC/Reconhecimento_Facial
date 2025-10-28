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

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gerenciador centralizado de conexões com o banco de dados"""

    def __init__(self, db_config=None):
        self.db_config = db_config or DATABASE_CONFIG.to_dict()

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Context manager para conexões de banco de dados com tratamento de erro

        Yields:
            psycopg2.extensions.connection: Conexão com o banco

        Raises:
            DatabaseConnectionError: Se não conseguir conectar
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
        Inicializa tabelas do banco de dados se não existirem

        Returns:
            bool: True se bem-sucedido
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS usuarios(
                            id SERIAL PRIMARY KEY,
                            nome VARCHAR(100) NOT NULL,
                            sobrenome VARCHAR(100) NOT NULL,
                            turma VARCHAR(50) NOT NULL,
                            tipo VARCHAR(20) DEFAULT 'aluno',
                            embeddings JSONB NOT NULL,
                            foto_perfil BYTEA,
                            data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(nome, sobrenome, turma)
                        )
                    """)

                    # Criar índices para performance
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_usuarios_nome_sobrenome 
                        ON usuarios(nome, sobrenome, turma)
                    """)

                    conn.commit()
                    logger.info("✅ Banco de dados inicializado com sucesso")
                    return True

        except Exception as e:
            logger.error(f"❌ Falha na inicialização do banco: {str(e)}")
            return False

    def check_user_exists(self, nome: str, sobrenome: str, turma: str) -> int:
        """
        Verifica se usuário já existe no banco

        Args:
            nome: Nome do usuário
            sobrenome: Sobrenome do usuário
            turma: Turma do usuário

        Returns:
            int: Número de usuários encontrados
        """
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
        """
        Conta número total de usuários cadastrados

        Returns:
            int: Total de usuários
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM usuarios")
                    count = cursor.fetchone()[0]
                    return count
        except Exception as e:
            logger.error(f"Erro ao contar usuários: {e}")
            return 0

    def save_user(self, nome: str, sobrenome: str, turma: str, tipo: str,
                 embeddings: list, profile_image: bytes) -> Tuple[bool, str]:
        """
        Salva usuário no banco de dados

        Args:
            nome: Nome do usuário
            sobrenome: Sobrenome do usuário
            turma: Turma do usuário
            tipo: Tipo de usuário (aluno/professor)
            embeddings: Lista de embeddings faciais
            profile_image: Imagem de perfil em bytes

        Returns:
            Tuple[bool, str]: (sucesso, mensagem)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Inserir ou atualizar usuário
                    cursor.execute("""
                        INSERT INTO usuarios (nome, sobrenome, turma, tipo, embeddings, foto_perfil)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (nome, sobrenome, turma)
                        DO UPDATE SET embeddings = EXCLUDED.embeddings,
                                    foto_perfil = EXCLUDED.foto_perfil,
                                    tipo = EXCLUDED.tipo,
                                    data_cadastro = CURRENT_TIMESTAMP
                    """, (nome, sobrenome, turma, tipo, Json(embeddings), profile_image))

                    conn.commit()
                    return True, "Usuário salvo com sucesso"

        except Exception as e:
            error_msg = f"Erro ao salvar usuário: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

# Instância global
db_manager = DatabaseManager()