"""
Gerenciador de loop de eventos para Flask - VERSÃO ROBUSTA
"""
import asyncio
import threading
import logging
import time
import sys
from typing import Any

logger = logging.getLogger(__name__)

class EventLoopManager:
    """Gerencia loop de eventos assíncrono para Flask"""

    _loop = None
    _thread = None
    _running = False
    _lock = threading.RLock()
    _initialized = False

    @classmethod
    def initialize(cls):
        """Inicializa o gerenciador de loop (chamar antes de usar)"""
        with cls._lock:
            if not cls._initialized:
                # Configura política de event loop para Windows se necessário
                if sys.platform == 'win32':
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

                cls._initialized = True
                logger.info("EventLoopManager inicializado")

    @classmethod
    def get_event_loop(cls):
        """Obtém ou cria um loop de eventos"""
        cls.initialize()  # Garante inicialização

        with cls._lock:
            if cls._loop is None or cls._loop.is_closed():
                try:
                    # Tenta obter o loop existente
                    cls._loop = asyncio.get_event_loop()
                except RuntimeError:
                    # Cria novo loop se não existir
                    cls._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(cls._loop)
                    logger.info("Novo loop de eventos criado")
                except Exception as e:
                    logger.error(f"Erro ao obter loop: {str(e)}")
                    cls._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(cls._loop)

            # Inicia a thread do loop se não estiver rodando
            if not cls._loop.is_running() and not cls._running:
                cls._start_loop_thread()

            return cls._loop

    @classmethod
    def _start_loop_thread(cls):
        """Inicia o loop em uma thread separada"""
        with cls._lock:
            if cls._thread is None or not cls._thread.is_alive():
                cls._thread = threading.Thread(
                    target=cls._run_loop,
                    daemon=True,
                    name="EventLoopThread"
                )
                cls._running = True
                cls._thread.start()

                # Aguarda um pouco para garantir que a thread iniciou
                time.sleep(0.5)
                logger.info("Thread do loop de eventos iniciada")

    @classmethod
    def _run_loop(cls):
        """Executa o loop de eventos"""
        loop = cls._loop
        try:
            loop.run_forever()
        except Exception as e:
            logger.error(f"Erro no loop de eventos: {str(e)}")
        finally:
            cls._running = False
            logger.info("Loop de eventos finalizado")

    @classmethod
    def run_async(cls, coroutine):
        """
        Executa uma corotina de forma síncrona
        Útil para chamadas assíncronas em código síncrono do Flask
        """
        cls.initialize()  # Garante inicialização

        try:
            loop = cls.get_event_loop()

            # Se não estiver rodando, tenta iniciar
            if not loop.is_running():
                logger.warning("Loop não está rodando, tentando iniciar thread...")
                cls._start_loop_thread()
                time.sleep(1)  # Aguarda thread iniciar

            # Verifica novamente
            if not loop.is_running():
                logger.error("Não foi possível iniciar o loop de eventos")
                return False

            # Executa a corotina no loop
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            return future.result(timeout=15)  # Timeout de 15 segundos

        except asyncio.TimeoutError:
            logger.error("Timeout na execução assíncrona")
            return False
        except Exception as e:
            logger.error(f"Erro ao executar assíncrono: {str(e)}", exc_info=True)
            return False

    @classmethod
    def shutdown(cls):
        """Desliga o gerenciador de loop"""
        with cls._lock:
            if cls._loop and cls._loop.is_running():
                cls._loop.call_soon_threadsafe(cls._loop.stop)
                cls._running = False
                logger.info("Loop de eventos parado")