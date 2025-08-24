import http.server
import socketserver
import os
import socket

# Tente usar a porta 5005, se não estiver disponível, use uma alternativa
PORT = 5005
ALTERNATIVE_PORTS = [5006, 5007, 5008, 8000, 8080]

# Usar caminho absoluto para o diretório front-end correto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../../../front_end/Login")

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
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:5005')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def log_message(self, format, *args):
        # Personalizar logs para melhor debug
        print(f"{self.log_date_time_string()} - {self.address_string()} - {format % args}")


def is_port_available(port):
    """Verifica se a porta está disponível"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except socket.error:
            return False


# Encontrar uma porta disponível
available_port = PORT
if not is_port_available(PORT):
    print(f"Porta {PORT} já está em uso. Procurando porta alternativa...")
    for port in ALTERNATIVE_PORTS:
        if is_port_available(port):
            available_port = port
            print(f"Usando porta alternativa: {available_port}")
            break
    else:
        print("Nenhuma porta alternativa disponível. Encerrando.")
        exit(1)

try:
    with socketserver.TCPServer(("", available_port), Handler) as httpd:
        print(f"Servindo em http://localhost:{available_port}")
        print(f"Acesse http://localhost:{available_port}/Login/CameraLogin.html")
        print("Pressione Ctrl+C para parar o servidor")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor parado")
except OSError as e:
    print(f"Erro ao iniciar servidor: {e}")
    print("Tentando usar outra porta...")

    # Tentar uma última vez com uma porta aleatória
    import random

    random_port = random.randint(8000, 9000)
    print(f"Tentando porta: {random_port}")

    try:
        with socketserver.TCPServer(("", random_port), Handler) as httpd:
            print(f"Servindo em http://localhost:{random_port}")
            print(f"Acesse http://localhost:{random_port}/Login/CameraLogin.html")
            print("Pressione Ctrl+C para parar o servidor")
            httpd.serve_forever()
    except Exception as e:
        print(f"Erro crítico: {e}")
        print("Não foi possível iniciar o servidor. Verifique as portas em uso.")