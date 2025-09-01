import http.server
import socketserver
import os
import socket
import webbrowser
import time
import threading
import random

# Configurações
PORT = 3000
ALTERNATIVE_PORTS = [3001, 3002, 3003, 8080, 8000, 5000, 5001]
BACKEND_URL = "http://localhost:5005"  # URL do backend

# Usar caminho absoluto para o diretório front-end correto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../../front_end/Login")

# Verificar se o diretório existe
if not os.path.exists(FRONTEND_DIR):
    print(f"Erro: Diretório não encontrado: {FRONTEND_DIR}")
    print("Verificando estrutura de pastas...")
    # Tentar encontrar o caminho correto
    possible_paths = [
        os.path.join(BASE_DIR, "front_end"),
        os.path.join(BASE_DIR, "../front_end"),
        os.path.join(BASE_DIR, "../../front_end"),
        os.path.join(BASE_DIR, "../../../front_end"),
        os.path.join(BASE_DIR, "../../../front_end/Login"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            FRONTEND_DIR = path
            print(f"Usando caminho alternativo: {FRONTEND_DIR}")
            break
    else:
        print("Nenhum diretório front-end encontrado. Criando diretório padrão...")
        os.makedirs(FRONTEND_DIR, exist_ok=True)

print(f"Servindo arquivos de: {FRONTEND_DIR}")
print(f"Caminho absoluto: {os.path.abspath(FRONTEND_DIR)}")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def end_headers(self):
        # Adicionar headers CORS para permitir requisições do front-end
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Personalizar logs para melhor debug
        print(f"{self.log_date_time_string()} - {self.address_string()} - {format % args}")

    def do_GET(self):
        # Servir arquivos estáticos
        if self.path == '/':
            self.path = '/CameraLogin.html'

        # Verificar se o arquivo existe
        file_path = os.path.join(FRONTEND_DIR, self.path[1:])
        if not os.path.exists(file_path) and self.path.endswith('.html'):
            # Se não encontrar o arquivo, servir a página principal
            self.path = '/CameraLogin.html'

        return super().do_GET()

def is_port_available(port):
    """Verifica se a porta está disponível"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except socket.error:
            return False

def check_backend_connection():
    """Verifica se o backend está respondendo"""
    import requests
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ Backend conectado com sucesso!")
            return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Não foi possível conectar ao backend: {e}")
        print("Certifique-se de que o backend está rodando na porta 5005")
        return False

def find_available_port():
    """Encontra uma porta disponível de forma mais confiável"""
    ports_to_try = [PORT] + ALTERNATIVE_PORTS

    for port in ports_to_try:
        if is_port_available(port):
            return port

    # Se nenhuma porta padrão estiver disponível, tentar uma aleatória
    for _ in range(10):
        random_port = random.randint(10000, 65535)
        if is_port_available(random_port):
            return random_port

    return None

def start_frontend_server():
    """Inicia o servidor frontend de forma mais robusta"""
    # Encontrar uma porta disponível
    available_port = find_available_port()

    if available_port is None:
        print("❌ Nenhuma porta disponível encontrada.")
        return False

    try:
        # Configurar o servidor com allow_reuse_address
        socketserver.TCPServer.allow_reuse_address = True

        with socketserver.TCPServer(("", available_port), Handler) as httpd:
            url = f"http://localhost:{available_port}"
            print(f"🌐 Servidor frontend iniciado em: {url}")
            print("📁 Servindo arquivos de:", FRONTEND_DIR)
            print("🎯 Página principal:", f"{url}/CameraLogin.html")
            print("🛑 Pressione Ctrl+C para parar o servidor")

            # Verificar conexão com o backend
            if not check_backend_connection():
                print("\n⚠️  Aviso: Frontend funcionará, mas não se conectará ao backend")
                print("   Execute o backend com: python appFaceId.py")

            # Abrir navegador após um breve delay
            threading.Timer(1.0, lambda: webbrowser.open(f"{url}/CameraLogin.html")).start()

            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n👋 Servidor parado pelo usuário")
            except Exception as e:
                print(f"❌ Erro no servidor: {e}")
            finally:
                # Fechar o servidor corretamente
                httpd.shutdown()
                httpd.server_close()

        return True

    except OSError as e:
        print(f"❌ Erro ao iniciar servidor: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Iniciando servidor frontend...")

    # Verificar se há arquivos frontend
    main_page = os.path.join(FRONTEND_DIR, "CameraLogin.html")
    if not os.path.exists(main_page):
        print(f"⚠️  Aviso: Arquivo principal não encontrado: {main_page}")
        print("   O servidor iniciará, mas pode não funcionar corretamente")

    # Iniciar servidor
    success = start_frontend_server()

    if not success:
        print("❌ Falha ao iniciar o servidor frontend")
        print("💡 Dicas:")
        print("   - Verifique se outra aplicação não está usando as portas")
        print("   - Tente fechar outros servidores locais")
        print("   - Execute com privilégios de administrador se necessário")