"""
Sistema Principal de Gerenciamento de Serviços
Controla todos os serviços do sistema de reconhecimento facial
"""
import os
import sys
import signal
import subprocess
import threading
import time
from datetime import datetime

# Adicionar diretório raiz ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from common.config import APP_CONFIG
except ImportError as e:
    print(f"ERRO: Falha na importacao de configuracoes: {e}")
    sys.exit(1)

class ServiceManager:
    """Gerenciador de serviços do sistema"""

    def __init__(self):
        self.processes = {}
        self.is_running = False

        # Configuração dos serviços
        self.service_config = {
            'cadastro': {
                'name': 'Servidor de Cadastro Facial',
                'port': APP_CONFIG.SERVER_PORT_CADASTRO,
                'script': 'cadastro/app.py',
                'path': current_dir
            },
            'reconhecimento': {
                'name': 'Servidor de Reconhecimento Facial',
                'port': APP_CONFIG.SERVER_PORT_RECONHECIMENTO,
                'script': 'reconhecimento/app.py',
                'path': current_dir
            },
            'qrcode': {
                'name': 'Servidor de QR Code',
                'port': 5000,
                'script': 'qrCode/app.py',
                'path': current_dir
            }
        }

    def start_service(self, service_name):
        """Inicia um serviço específico"""
        if service_name not in self.service_config:
            print(f"ERRO: Servico desconhecido: {service_name}")
            return False

        config = self.service_config[service_name]

        # Verificar se já está rodando
        if service_name in self.processes and self.processes[service_name].poll() is None:
            print(f"INFO: {config['name']} ja esta em execucao")
            return True

        try:
            print(f"INICIANDO: {config['name']}")

            # Verificar se arquivo existe
            script_path = os.path.join(config['path'], config['script'])
            if not os.path.exists(script_path):
                print(f"ERRO: Arquivo nao encontrado: {script_path}")
                return False

            # Configurar encoding para UTF-8
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            # Executar processo
            process = subprocess.Popen(
                [sys.executable, config['script']],
                cwd=config['path'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
                env=env,
                encoding='utf-8'
            )

            self.processes[service_name] = process

            # Thread para monitorar stdout - sem filtro de emojis
            def log_output(service_name, process):
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line.strip():
                            # Remove apenas quebras de linha extras, mantém o conteúdo original
                            clean_line = line.strip()
                            print(f"[{service_name}] {clean_line}")
                except Exception as e:
                    print(f"ERRO: Falha ao ler stdout de {service_name}: {e}")

            stdout_thread = threading.Thread(
                target=log_output,
                args=(service_name, process),
                daemon=True
            )
            stdout_thread.start()

            # Thread para monitorar stderr - sem filtro de emojis
            def log_errors(service_name, process):
                try:
                    for line in iter(process.stderr.readline, ''):
                        if line.strip():
                            clean_line = line.strip()
                            print(f"[{service_name}] {clean_line}")
                except Exception as e:
                    print(f"ERRO: Falha ao ler stderr de {service_name}: {e}")

            stderr_thread = threading.Thread(
                target=log_errors,
                args=(service_name, process),
                daemon=True
            )
            stderr_thread.start()

            # Aguardar para verificar se iniciou corretamente
            time.sleep(3)
            if process.poll() is not None:
                print(f"ERRO: Falha ao iniciar {config['name']} - processo terminou")
                return False

            print(f"SUCESSO: {config['name']} iniciado na porta {config['port']}")
            return True

        except Exception as e:
            print(f"ERRO: Falha ao iniciar {config['name']}: {str(e)}")
            return False

    def stop_service(self, service_name):
        """Para um serviço específico"""
        if service_name in self.processes:
            config = self.service_config[service_name]
            print(f"PARANDO: {config['name']}")

            process = self.processes[service_name]
            process.terminate()

            try:
                process.wait(timeout=10)
                print(f"SUCESSO: {config['name']} parado")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"AVISO: {config['name']} forcado a parar")

            del self.processes[service_name]
            return True

        print(f"AVISO: Servico {service_name} nao estava em execucao")
        return False

    def start_all_services(self):
        """Inicia todos os serviços"""
        print("INICIANDO TODOS OS SERVICOS")

        success_count = 0
        services_order = ['cadastro', 'reconhecimento', 'qrcode']

        for service_name in services_order:
            if self.start_service(service_name):
                success_count += 1
            time.sleep(2)

        print(f"RESUMO: Servicos iniciados {success_count}/{len(services_order)}")
        return success_count == len(services_order)

    def stop_all_services(self):
        """Para todos os serviços"""
        print("PARANDO TODOS OS SERVICOS")

        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)

        print("SUCESSO: Todos os servicos parados")

    def get_service_status(self):
        """Obtém status de todos os serviços"""
        status = {}

        for service_name, config in self.service_config.items():
            is_running = (
                service_name in self.processes and
                self.processes[service_name].poll() is None
            )

            status[service_name] = {
                'name': config['name'],
                'port': config['port'],
                'running': is_running
            }

        return status

    def monitor_services(self):
        """Monitora continuamente os serviços"""
        check_count = 0
        while self.is_running:
            try:
                time.sleep(30)
                check_count += 1

                if check_count % 10 == 0:
                    print(f"MONITOR: Verificacao {check_count} - Sistemas estaveis")

                for service_name in list(self.processes.keys()):
                    if (service_name in self.processes and
                        self.processes[service_name].poll() is not None):
                        print(f"RECUPERACAO: {service_name} parou - Reiniciando...")
                        self.start_service(service_name)

            except Exception as e:
                print(f"ERRO MONITOR: {str(e)}")

    def print_status(self):
        """Exibe status formatado dos serviços"""
        status = self.get_service_status()

        print("\n" + "=" * 70)
        print("STATUS DOS SERVICOS - SISTEMA DE RECONHECIMENTO FACIAL")
        print("=" * 70)

        for service_name, info in status.items():
            status_text = "ATIVO" if info['running'] else "INATIVO"
            status_indicator = "[X]" if info['running'] else "[ ]"

            print(f"{status_indicator} {info['name']}")
            print(f"    Porta: {info['port']}")
            print(f"    Status: {status_text}")
            print()

        print("=" * 70)

def signal_handler(sig, frame):
    """Manipula sinais de desligamento gracioso"""
    print("\nSINAL: Recebido sinal de desligamento...")
    service_manager.is_running = False
    service_manager.stop_all_services()
    print("SISTEMA: Finalizado com sucesso")
    sys.exit(0)

# Instância global do gerenciador
service_manager = ServiceManager()

def main():
    """Função principal"""
    print("=" * 70)
    print("SISTEMA PRINCIPAL DE RECONHECIMENTO FACIAL")
    print("=" * 70)

    # Registrar handlers de sinal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        service_manager.start_all_services()
        service_manager.print_status()

        service_manager.is_running = True
        monitor_thread = threading.Thread(
            target=service_manager.monitor_services,
            daemon=True
        )
        monitor_thread.start()
        print("MONITOR: Monitoramento automatico ativado")

        print("CONTROLE: Pressione Ctrl+C para encerrar todos os servicos")
        print("SISTEMA: Executando...")

        while service_manager.is_running:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nCONTROLE: Interrompido pelo usuario")
    except Exception as e:
        print(f"ERRO CRITICO: {str(e)}")
    finally:
        service_manager.is_running = False
        service_manager.stop_all_services()
        print("SISTEMA: Finalizado")

if __name__ == "__main__":
    main()