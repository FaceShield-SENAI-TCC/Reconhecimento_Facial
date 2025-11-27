"""
Controlador da trava do armario com temporizadores
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from common.websocket_client import ESP32WebSocketClient
from common.config import WEBSOCKET_CONFIG

logger = logging.getLogger(__name__)

class LockerController:
    """Controlador da trava do armario"""

    def __init__(self):
        self.esp32_client = ESP32WebSocketClient(
            ip_esp32=WEBSOCKET_CONFIG.ESP32_IP,
            porta=WEBSOCKET_CONFIG.ESP32_PORT
        )
        self.trava_aberta = False
        self.temporizador_abertura: Optional[asyncio.Task] = None
        self.temporizador_fechamento: Optional[asyncio.Task] = None
        self.timeout_abertura = WEBSOCKET_CONFIG.TIMEOUT_ABERTURA

        logger.info(f"LockerController configurado - Timeout: {self.timeout_abertura}s")

    async def iniciar(self) -> bool:
        """Inicia a conexao com a ESP32"""
        logger.info(f"INICIANDO CONEXAO COM ESP32...")
        # Não conecta mais na inicialização - conecta na primeira tentativa de uso
        logger.info("Conexao sera estabelecida no primeiro uso")
        return True

    async def abrir_trava_aluno(self, user_id: int, user_type: str) -> bool:
        """
        Abre a trava quando aluno e reconhecido
        """
        logger.info(f"=== INICIANDO ABERTURA DE TRAVA PARA ALUNO ===")
        logger.info(f"User ID: {user_id}, Tipo: {user_type}")

        if user_type.upper() != "ALUNO":
            logger.warning(f"USUARIO NAO E ALUNO: {user_type}")
            return False

        logger.info(f"ALUNO RECONHECIDO - ID: {user_id}")
        logger.info(f"ENVIANDO COMANDO PARA ABRIR TRAVA...")

        # Envia comando para abrir a trava (com reconexão automática)
        sucesso = await self.esp32_client.abrir_trava()

        if sucesso:
            self.trava_aberta = True
            logger.info("COMANDO ABRIR_TRAVA ENVIADO COM SUCESSO")

            # Inicia temporizador de 3 minutos
            self.temporizador_abertura = asyncio.create_task(
                self._fechar_trava_apos_timeout(self.timeout_abertura)
            )
            logger.info(f"TEMPORIZADOR DE {self.timeout_abertura}s INICIADO")
            return True
        else:
            logger.error("FALHA: COMANDO ABRIR_TRAVA NAO FOI ENVIADO")
            return False

    async def abrir_trava_qrcode(self) -> bool:
        """
        Abre a trava quando QR Code e lido
        """
        logger.info("=== INICIANDO ABERTURA DE TRAVA VIA QR CODE ===")

        # Envia comando para abrir a trava
        sucesso = await self.esp32_client.abrir_trava()
        if sucesso:
            self.trava_aberta = True

            # Cancela temporizador de abertura anterior se existir
            if self.temporizador_abertura:
                self.temporizador_abertura.cancel()
                self.temporizador_abertura = None

            logger.info("TRAVA ABERTA VIA QR CODE - AGUARDANDO FECHAMENTO DO ARMARIO")
            return True
        else:
            logger.error("FALHA AO ABRIR TRAVA VIA QR CODE")
            return False

    async def _fechar_trava_apos_timeout(self, segundos: int):
        """Fecha a trava apos o tempo especificado"""
        await asyncio.sleep(segundos)
        if self.trava_aberta:
            logger.info("TEMPO DE ABERTURA EXPIRADO - FECHANDO TRAVA")
            await self.fechar_trava()

    async def fechar_trava(self) -> bool:
        """Fecha a trava do armario"""
        sucesso = await self.esp32_client.fechar_trava()
        if sucesso:
            self.trava_aberta = False
            logger.info("TRAVA FECHADA COM SUCESSO")

            # Cancela temporizadores ativos
            if self.temporizador_abertura:
                self.temporizador_abertura.cancel()
                self.temporizador_abertura = None
            if self.temporizador_fechamento:
                self.temporizador_fechamento.cancel()
                self.temporizador_fechamento = None

            return True
        else:
            logger.error("FALHA AO FECHAR TRAVA")
            return False

    async def verificar_estado_trava(self) -> dict:
        """Retorna estado atual da trava"""
        return {
            "trava_aberta": self.trava_aberta,
            "conectado": self.esp32_client.conectado,
            "url": self.esp32_client.url
        }