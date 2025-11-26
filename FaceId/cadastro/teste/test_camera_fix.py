import asyncio
import websockets
import time

class ESP32WebSocketClient:
    def __init__(self, ip_esp32="10.110.22.9", porta=8765):
        self.ip_esp32 = ip_esp32
        self.porta = porta
        self.websocket = None
        self.conectado = False
        self.estado_led = False
        self.url = f"ws://{ip_esp32}:{porta}"

    async def conectar(self):
        """Estabelece conexão com o ESP32"""
        try:
            print(f"Conectando a {self.url}...")
            self.websocket = await websockets.connect(self.url)
            self.conectado = True
            print("Conectado ao ESP32 com sucesso!")

            # Inicia tarefa para ouvir mensagens
            asyncio.create_task(self.ouvir_mensagens())

            # Solicita estado atual do LED
            await self.enviar_mensagem("STATUS")
            return True

        except Exception as e:
            print(f"Erro na conexão: {e}")
            self.conectado = False
            return False

    async def enviar_mensagem(self, mensagem):
        """Envia mensagem para o ESP32"""
        if self.conectado and self.websocket:
            try:
                await self.websocket.send(mensagem)
                print(f"Mensagem enviada: {mensagem}")
            except Exception as e:
                print(f"Erro ao enviar mensagem: {e}")
                self.conectado = False

    async def ouvir_mensagens(self):
        """Ouve mensagens do ESP32"""
        try:
            async for mensagem in self.websocket:
                await self.processar_mensagem(mensagem)
        except Exception as e:
            print(f"Erro ao ouvir mensagens: {e}")
            self.conectado = False

    async def processar_mensagem(self, mensagem):
        """Processa mensagens recebidas do ESP32"""
        mensagem = mensagem.strip()
        print(f"Recebido do ESP32: {mensagem}")

        if "LED ligado" in mensagem:
            self.estado_led = True
        elif "LED desligado" in mensagem:
            self.estado_led = False

        # Exibe mensagens de erro em laranja (ANSI code)
        if "Erro:" in mensagem:
            print(f"\033[33m{mensagem}\033[0m")
        else:
            print(f"\033[32m{mensagem}\033[0m")

    async def controlar_led(self, estado):
        """
        Controla o LED baseado em um valor booleano
        True = Liga LED, False = Desliga LED
        """
        if not self.conectado:
            print("Erro: Não conectado ao ESP32")
            return False

        comando = "ON" if estado else "OFF"
        await self.enviar_mensagem(comando)
        return True

    async def fechar_conexao(self):
        """Fecha a conexão com o ESP32"""
        if self.websocket:
            await self.websocket.close()
            self.conectado = False
            print("Conexão fechada")

# Exemplo de uso
async def main():
    # Cria instância do cliente
    esp32 = ESP32WebSocketClient()

    # Conecta ao ESP32
    if await esp32.conectar():
        # Aguarda um pouco para estabilização
        await asyncio.sleep(2)

        # Exemplo de controle por método booleano
        estados_led = [True, False, True, False]

        for estado in estados_led:
            print(f"\n--- Alterando LED para: {estado} ---")
            await esp32.controlar_led(estado)

            # Aguarda 3 segundos entre cada comando
            await asyncio.sleep(3)

        # Fecha conexão
        await esp32.fechar_conexao()

if __name__ == "__main__":
    # Executa o exemplo
    asyncio.run(main())