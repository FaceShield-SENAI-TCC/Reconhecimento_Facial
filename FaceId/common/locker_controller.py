import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable
from common.websocket_client import ESP32WebSocketClient


class LockerController:
    def __init__(self):
        self.esp32_client = ESP32WebSocketClient()
        self.trava_aberta = False
        self.temporizador_abertura = None
        self.temporizador_fechamento = None

    async def abrir_trava_aluno(self, user_id: int, user_type: str) -> bool:
        """Abre trava quando aluno é reconhecido - 3min timeout"""
        if user_type.upper() == "ALUNO":
            # Implementar lógica de 3 minutos
            pass

    async def abrir_trava_qrcode(self) -> bool:
        """Abre trava quando QR Code é lido"""
        pass

    async def verificar_estado_trava(self) -> dict:
        """Retorna estado atual da trava"""
        pass