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
        Inicializa (Cria) ou Atualiza (Altera) tabelas do banco de dados
        para garantir que todas as colunas existam.

        Returns:
            bool: True se bem-sucedido
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. TENTA CRIAR A TABELA (para novas instalações)
                    # Esta é a definição "ideal" da tabela.
                    # Se a tabela já existir, este comando não faz nada.
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS usuarios(
                            id SERIAL PRIMARY KEY,
                            nome VARCHAR(100) NOT NULL,
                            sobrenome VARCHAR(100) NOT NULL,
                            tipo_usuario VARCHAR(20) NOT NULL DEFAULT 'ALUNO',
                            turma VARCHAR(50) NOT NULL,
                            username VARCHAR(100) NULL,
                            embeddings JSONB NOT NULL,
                            foto_perfil BYTEA,
                            UNIQUE(nome, sobrenome, turma)
                        )
                    """)

                    # 2. GARANTE QUE AS COLUNAS EXISTAM (para migrar tabelas antigas)
                    # ... (comandos ALTER TABLE estão corretos) ...
                    cursor.execute("""
                        ALTER TABLE usuarios
                        ADD COLUMN IF NOT EXISTS embeddings JSONB NOT NULL DEFAULT '[]'::jsonb
                    """)

                    cursor.execute("""
                        ALTER TABLE usuarios
                        ADD COLUMN IF NOT EXISTS foto_perfil BYTEA
                    """)

                    # 3. Criar índices (como no original)

                    # --- A CORREÇÃO ESTÁ AQUI ---
                    # Precisamos de um ÍNDICE ÚNICO (UNIQUE INDEX) para que
                    # o "ON CONFLICT (nome, sobrenome, turma)" funcione.
                    cursor.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_usuarios_nome_sobrenome 
                        ON usuarios(nome, sobrenome, turma)
                    """)
                    # --- FIM DA CORREÇÃO ---

                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_usuarios_username 
                        ON usuarios(username) 
                        WHERE username IS NOT NULL
                    """)

                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_usuarios_tipo 
                        ON usuarios(tipo_usuario)
                    """)

                    conn.commit()
                    logger.info("✅ Banco de dados inicializado/atualizado com sucesso")
                    return True

        except Exception as e:
            logger.error(f"❌ Falha na inicialização/atualização do banco: {str(e)}")
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

    def check_username_exists(self, username: str) -> int:
        """
        Verifica se username já existe no banco (apenas para professores)

        Args:
            username: Nome de usuário

        Returns:
            int: Número de usuários encontrados
        """
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

    def save_user(self, nome: str, sobrenome: str, turma: str, tipo_usuario: str,
                 username: Optional[str], embeddings: list, profile_image: bytes) -> Tuple[bool, str]:
        """
        Salva usuário no banco de dados

        Args:
            nome: Nome do usuário
            sobrenome: Sobrenome do usuário
            turma: Turma do usuário
            tipo_usuario: Tipo de usuário (ALUNO/PROFESSOR)
            username: Nome de usuário (apenas para professores, None para alunos)
            embeddings: Lista de embeddings faciais
            profile_image: Imagem de perfil em bytes

        Returns:
            Tuple[bool, str]: (sucesso, mensagem)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Para alunos, garantir que username seja NULL
                    if tipo_usuario.upper() == "ALUNO":
                        username = None

                    # Para professores, verificar se username foi fornecido
                    if tipo_usuario.upper() == "PROFESSOR" and not username:
                        return False, "Username é obrigatório para professores"

                    # Verificar se username já existe (apenas para professores)
                    if tipo_usuario.upper() == "PROFESSOR" and username:
                        existing_count = self.check_username_exists(username)
                        if existing_count > 0:
                            return False, f"Username '{username}' já está em uso"

                    # Inserir ou atualizar usuário
                    cursor.execute("""
                        INSERT INTO usuarios (nome, sobrenome, turma, tipo_usuario, username, embeddings, foto_perfil)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (nome, sobrenome, turma)
                        DO UPDATE SET 
                            tipo_usuario = EXCLUDED.tipo_usuario,
                            username = EXCLUDED.username,
                            embeddings = EXCLUDED.embeddings,
                            foto_perfil = EXCLUDED.foto_perfil
                    """, (nome, sobrenome, turma, tipo_usuario.upper(), username, Json(embeddings), profile_image))

                    conn.commit()

                    user_type = "aluno" if tipo_usuario.upper() == "ALUNO" else "professor"
                    return True, f"{user_type.capitalize()} {nome} {sobrenome} salvo com sucesso"

        except Exception as e:
            error_msg = f"Erro ao salvar usuário: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """
        Busca usuário pelo username (apenas professores)

        Args:
            username: Nome de usuário

        Returns:
            Optional[dict]: Dados do usuário ou None
        """
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