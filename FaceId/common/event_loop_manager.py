"""
Gerenciador de loop de eventos para Flask - VERSÃO SIMPLIFICADA
"""
import asyncio
import threading
import logging
from typing import Any

logger = logging.getLogger(__name__)

class EventLoopManager:
    """Gerencia loop de eventos assíncrono para Flask"""

    _loop = None
    _thread = None

    @classmethod
    def get_event_loop(cls):
        """Obtém ou cria um loop de eventos"""
        if cls._loop is None or cls._loop.is_closed():
            cls._loop = asyncio.new_event_loop()
            logger.info("Novo loop de eventos criado")
        return cls._loop

    @classmethod
    def run_async(cls, coroutine):
        """
        Executa uma corotina de forma síncrona
        Útil para chamadas assíncronas em código síncrono do Flask
        """
        try:
            # Tenta usar loop existente
            loop = cls.get_event_loop()

            # Se não estiver rodando, cria uma thread para rodá-lo
            if not loop.is_running():
                if cls._thread is None or not cls._thread.is_alive():
                    cls._thread = threading.Thread(
                        target=loop.run_forever,
                        daemon=True
                    )
                    cls._thread.start()
                    logger.info("Thread do loop de eventos iniciada")

            # Executa a corotina no loop
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            return future.result(timeout=10)  # Timeout de 10 segundos

        except asyncio.TimeoutError:
            logger.error("Timeout na execução assíncrona")
            return False
        except Exception as e:
            logger.error(f"Erro ao executar assíncrono: {str(e)}")
            return False