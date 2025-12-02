"""
Cliente WebSocket para comunicação com ESP32 - VERSÃO COM RECONEXÃO
"""
import asyncio
import websockets
import logging
import time
from typing import Optional, Tuple

from common.config import WEBSOCKET_CONFIG

logger = logging.getLogger(__name__)

class ESP32WebSocketClient:
    """Cliente WebSocket para controle da ESP32 com reconexão automática"""

    def __init__(self, ip_esp32: str = None, porta: int = None):
        self.ip_esp32 = ip_esp32 or WEBSOCKET_CONFIG.ESP32_IP
        self.porta = porta or WEBSOCKET_CONFIG.ESP32_PORT
        self.websocket = None
        self.conectado = False
        self.url = f"ws://{self.ip_esp32}:{self.porta}"
        self.last_connection_attempt = 0
        self.connection_lock = asyncio.Lock()

        logger.info(f"Configuração WebSocket: {self.url}")

    async def _conectar_se_necessario(self) -> bool:
        """Conecta se não estiver conectado, reconecta se necessário"""
        async with self.connection_lock:
            current_time = time.time()

            # Se já está conectado e o websocket está aberto, retorna True
            if (self.conectado and self.websocket and
                not self.websocket.closed and
                self.websocket.open):
                return True

            # Evita tentativas muito frequentes (mínimo 5 segundos entre tentativas)
            if current_time - self.last_connection_attempt < 5:
                logger.debug("Aguardando para reconexão...")
                return False

            logger.info("Tentando conectar com ESP32...")
            self.last_connection_attempt = current_time

            try:
                # Fecha conexão anterior se existir
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except:
                        pass
                    self.websocket = None

                # Tenta conectar com timeout
                logger.info(f"Conectando a {self.url}...")
                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.url,
                        ping_interval=30,
                        ping_timeout=10,
                        close_timeout=5
                    ),
                    timeout=5.0
                )

                self.conectado = True
                logger.info("✅ Conectado a ESP32 via WebSocket")
                return True

            except asyncio.TimeoutError:
                logger.error(f"Timeout ao conectar com ESP32 em {self.url}")
                self.conectado = False
                return False
            except ConnectionRefusedError:
                logger.error(f"Conexão recusada pela ESP32 em {self.url}")
                self.conectado = False
                return False
            except OSError as e:
                logger.error(f"Erro de rede ao conectar com ESP32: {str(e)}")
                self.conectado = False
                return False
            except Exception as e:
                logger.error(f"Erro inesperado ao conectar: {str(e)}")
                self.conectado = False
                return False

    async def enviar_comando(self, comando: str) -> Tuple[bool, str]:
        """
        Envia comandos para a ESP32 com reconexão automática
        """
        logger.info(f"Enviando comando: {comando}")

        # Tenta conectar
        if not await self._conectar_se_necessario():
            return False, "Não foi possível conectar com ESP32"

        try:
            # Envia comando
            logger.info(f"📤 ENVIANDO: '{comando}'")
            await self.websocket.send(comando)

            # Tenta receber resposta com timeout maior
            try:
                resposta = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=10.0  # Aumentado para 10 segundos
                )
                logger.info(f"📥 RESPOSTA: '{resposta}'")
                return True, resposta.strip()
            except asyncio.TimeoutError:
                logger.warning(f"Timeout ao aguardar resposta para '{comando}'")
                # Mesmo sem resposta, considera sucesso se o comando foi enviado
                return True, "Sem resposta (comando enviado)"

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Conexão fechada durante envio, tentando reconectar...")
            self.conectado = False
            return False, "Conexão fechada"
        except Exception as e:
            logger.error(f"Erro ao enviar comando: {str(e)}")
            self.conectado = False
            return False, str(e)

    async def abrir_trava(self) -> bool:
        """Envia comando para abrir a trava no modo FACIAL"""
        logger.info("Abrindo trava (modo FACIAL)")
        sucesso, resposta = await self.enviar_comando("ABRIR_TRAVA")
        return sucesso and "TRAVA_ABERTA_FACIAL" in resposta

    async def abrir_trava_iot(self) -> Tuple[bool, str]:
        """Abre trava no modo IOT - só fecha quando armário fechar"""
        logger.info("Abrindo trava (modo IOT)")
        sucesso, resposta = await self.enviar_comando("ABRIR_TRAVA_IOT")
        return sucesso, resposta

    async def fechar_trava(self) -> bool:
        """Envia comando para fechar a trava"""
        logger.info("Fechando trava")
        sucesso, resposta = await self.enviar_comando("FECHAR_TRAVA")
        return sucesso

    async def verificar_status(self) -> Tuple[bool, str]:
        """Verifica status da ESP32"""
        return await self.enviar_comando("STATUS")

    async def fechar_conexao(self):
        """Fecha a conexão WebSocket"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.conectado = False
            self.websocket = None
            logger.info("Conexão WebSocket fechada")