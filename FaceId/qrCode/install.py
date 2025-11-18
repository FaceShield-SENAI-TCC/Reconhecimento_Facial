# install.py
import sys
import subprocess
import importlib.util
from pathlib import Path


def check_python_version():
    version = sys.version_info
    if version.major == 3 and version.minor == 9:
        print("✓ Python 3.9 detectado")
        return True
    else:
        print(f"⚠ Versão do Python: {version.major}.{version.minor}.{version.micro}")
        print("Recomendado: Python 3.9")
        return True


def check_and_install(package, pip_name=None):
    if pip_name is None:
        pip_name = package

    try:
        spec = importlib.util.find_spec(package)
        if spec is None:
            print(f"Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✓ {package} instalado com sucesso")
        else:
            print(f"✓ {package} já instalado")
        return True
    except Exception as e:
        print(f"✗ Erro ao instalar {package}: {e}")
        return False


def create_requirements_file():
    requirements_content = """Flask==2.3.3
flask-cors==4.0.0
opencv-python==4.8.1.78
numpy==1.24.4
"""
    try:
        with open('requirements.txt', 'w') as f:
            f.write(requirements_content)
        print("✓ Arquivo requirements.txt criado")
        return True
    except Exception as e:
        print(f"✗ Erro ao criar requirements.txt: {e}")
        return False


def main():
    print("=== Instalação do Leitor Automático de QR Code ===\n")

    check_python_version()

    print("\n1. Criando arquivo de dependências...")
    create_requirements_file()

    print("\n2. Verificando dependências...")

    packages = [
        ("flask", "Flask"),
        ("flask_cors", "flask-cors"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
    ]

    success = True
    for import_name, pip_name in packages:
        if not check_and_install(import_name, pip_name):
            success = False

    if success:
        print("\n🎉 Todas as dependências foram instaladas com sucesso!")
        print("\n🚀 Para executar:")
        print("   1. python app.py")
        print("   2. Use o frontend com leitura automática")
        print("\n📡 Endpoints disponíveis:")
        print("   - POST http://localhost:5000/decode_qr")
        print("   - GET  http://localhost:5000/qrcodes/latest")
        print("\n⚠ Dica: Certifique-se de que a câmera está apontada para um QR Code bem iluminado e nítido")
    else:
        print("\n⚠ Algumas dependências não puderam ser instaladas automaticamente.")
        print("Tente instalar manualmente: pip install -r requirements.txt")


if __name__ == "__main__":
    main()