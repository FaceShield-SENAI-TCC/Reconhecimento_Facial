"""
Cliente WebSocket para comunicação com ESP32 - Versao Simplificada
"""
import asyncio
import websockets
import logging
from typing import Optional

from common.config import WEBSOCKET_CONFIG

logger = logging.getLogger(__name__)

class ESP32WebSocketClient:
    """Cliente WebSocket para controle da ESP32"""

    def __init__(self, ip_esp32: str = None, porta: int = None):
        # Usar configurações do config.py se não fornecidos
        self.ip_esp32 = ip_esp32 or WEBSOCKET_CONFIG.ESP32_IP
        self.porta = porta or WEBSOCKET_CONFIG.ESP32_PORT
        self.websocket = None
        self.conectado = False
        self.url = f"ws://{self.ip_esp32}:{self.porta}"

        logger.info(f"Configuracao WebSocket: {self.url}")

    async def _conectar_se_necessario(self) -> bool:
        """Conecta se não estiver conectado, reconecta se necessário"""
        if self.conectado and self.websocket and not self.websocket.closed:
            return True

        logger.info("Tentando conectar/reconectar com ESP32...")
        try:
            # Fecha conexão anterior se existir
            if self.websocket:
                await self.websocket.close()

            self.websocket = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self.conectado = True
            logger.info("Conectado a ESP32 via WebSocket")
            return True

        except Exception as e:
            logger.error(f"Falha na conexao com ESP32: {str(e)}")
            self.conectado = False
            self.websocket = None
            return False

    async def enviar_comando(self, comando: str) -> bool:
        """Envia comandos para a ESP32 com reconexão automática"""
        logger.info(f"=== TENTANDO ENVIAR COMANDO PARA ESP32 ===")
        logger.info(f"Comando: {comando}")
        logger.info(f"URL: {self.url}")

        # Tenta conectar/reconectar
        if not await self._conectar_se_necessario():
            logger.error("Nao foi possível conectar com ESP32")
            return False

        # Tenta enviar o comando
        try:
            logger.info(f"ENVIANDO: '{comando}' para ESP32")
            await self.websocket.send(comando)
            logger.info(f"SUCESSO: Comando '{comando}' enviado para ESP32")
            return True

        except Exception as e:
            logger.error(f"FALHA AO ENVIAR: Erro ao enviar comando '{comando}': {e}")
            # Marca como desconectado para tentar reconectar na próxima vez
            self.conectado = False
            self.websocket = None
            return False

    async def abrir_trava(self) -> bool:
        """Envia comando para abrir a trava"""
        logger.info("=== ENVIANDO COMANDO ABRIR_TRAVA ===")
        return await self.enviar_comando("ABRIR_TRAVA")

    async def fechar_trava(self) -> bool:
        """Envia comando para fechar a trava"""
        logger.info("=== ENVIANDO COMANDO FECHAR_TRAVA ===")
        return await self.enviar_comando("FECHAR_TRAVA")

    async def verificar_status(self) -> bool:
        """Verifica status da ESP32"""
        return await self.enviar_comando("STATUS")

    async def fechar_conexao(self):
        """Fecha a conexao WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.conectado = False
            self.websocket = None
            logger.info("Conexao WebSocket fechada")