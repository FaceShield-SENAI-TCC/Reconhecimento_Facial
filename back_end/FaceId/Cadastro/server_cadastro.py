import http.server
import socketserver
import os

# Altere este caminho para o diretório do seu frontend
FRONTEND_DIR = "../../../front_end/Cadastro"  # Exemplo: "../frontend-cadastro"

PORT = 8001
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:7001')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Credentials', 'true')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Servindo frontend de cadastro em http://localhost:{PORT}")
    print(f"Diretório: {FRONTEND_DIR}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor parado")