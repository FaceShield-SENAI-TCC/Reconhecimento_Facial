import http.server
import socketserver
import os

PORT = 8000
DIRECTORY = "C:/Users/WBS/Desktop/Arduino/front_end"  # Serve from current directory


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Add CORS headers to allow requests from your frontend
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:5000')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Servindo em http://localhost:{PORT}")
    print("Acesse http://localhost:8000/Login/CameraLogin.html")
    print("Pressione Ctrl+C para parar o servidor")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor parado")