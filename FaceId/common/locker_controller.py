"""
Controlador da trava do armário - VERSÃO SIMPLIFICADA
"""
import asyncio
import logging
import threading
from typing import Optional
from common.websocket_client import ESP32WebSocketClient
from common.config import WEBSOCKET_CONFIG

logger = logging.getLogger(__name__)

class LockerController:
    """Controlador da trava do armário"""

    def __init__(self):
        self.esp32_client = ESP32WebSocketClient(
            ip_esp32=WEBSOCKET_CONFIG.ESP32_IP,
            porta=WEBSOCKET_CONFIG.ESP32_PORT
        )
        self.trava_aberta = False
        self._lock = threading.RLock()

        logger.info(f"LockerController inicializado para {WEBSOCKET_CONFIG.ESP32_IP}:{WEBSOCKET_CONFIG.ESP32_PORT}")

    async def abrir_trava_aluno(self, user_id: int, user_type: str) -> bool:
        """Abre trava para aluno reconhecido (MODO FACIAL - 10s)"""
        logger.info(f"Abrindo trava modo FACIAL para usuário {user_id}")
        try:
            sucesso = await self.esp32_client.abrir_trava()
            if sucesso:
                self.trava_aberta = True
                logger.info("✅ Trava aberta no modo FACIAL")
            else:
                logger.error("❌ Falha ao abrir trava modo FACIAL")
            return sucesso
        except Exception as e:
            logger.error(f"Erro ao abrir trava facial: {str(e)}")
            return False

    async def abrir_trava_qrcode(self) -> bool:
        """Abre trava via QR Code (MODO IOT - fecha com microswitch)"""
        logger.info("Abrindo trava modo IOT (QR Code)")
        try:
            sucesso, resposta = await self.esp32_client.abrir_trava_iot()
            if sucesso:
                self.trava_aberta = True
                logger.info(f"✅ Trava aberta no modo IOT - {resposta}")
            else:
                logger.error(f"❌ Falha ao abrir trava modo IOT - {resposta}")
            return sucesso
        except Exception as e:
            logger.error(f"Erro ao abrir trava IOT: {str(e)}")
            return False

    async def fechar_trava(self) -> bool:
        """Fecha a trava do armário"""
        logger.info("Fechando trava")
        try:
            sucesso = await self.esp32_client.fechar_trava()
            if sucesso:
                self.trava_aberta = False
                logger.info("✅ Trava fechada com sucesso")
            else:
                logger.error("❌ Falha ao fechar trava")
            return sucesso
        except Exception as e:
            logger.error(f"Erro ao fechar trava: {str(e)}")
            return False

    async def verificar_estado_trava(self) -> dict:
        """Retorna estado atual da trava"""
        try:
            sucesso, resposta = await self.esp32_client.verificar_status()

            if sucesso:
                # Atualiza estado baseado na resposta
                if "ABERTA" in resposta.upper():
                    self.trava_aberta = True
                elif "FECHADA" in resposta.upper():
                    self.trava_aberta = False

                return {
                    "trava_aberta": self.trava_aberta,
                    "conectado": True,
                    "url": self.esp32_client.url,
                    "resposta_esp32": resposta,
                    "status": "online"
                }
            else:
                return {
                    "trava_aberta": self.trava_aberta,
                    "conectado": False,
                    "url": self.esp32_client.url,
                    "resposta_esp32": resposta,
                    "status": "offline"
                }
        except Exception as e:
            logger.error(f"Erro ao verificar estado: {str(e)}")
            return {
                "trava_aberta": self.trava_aberta,
                "conectado": False,
                "url": self.esp32_client.url,
                "resposta_esp32": f"Erro: {str(e)}",
                "status": "error"
            }

    async def testar_conexao(self) -> dict:
        """Testa a conexão com a ESP32"""
        return await self.verificar_estado_trava()