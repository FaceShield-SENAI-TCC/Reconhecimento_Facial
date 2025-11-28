from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import asyncio
import sys
import os

# Adiciona o diretório raiz ao path para importar módulos comuns
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.locker_controller import LockerController

app = Flask(__name__)

# Configuração do CORS para permitir seu frontend
CORS(app, origins=['http://127.0.0.1:5500', 'http://localhost:5500', 'http://localhost:8080'])

# Instância do controlador da trava
locker_controller = LockerController()


class QRCodeDecoder:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def decode_qr_code(self, image: np.ndarray) -> str:
        """Decodifica QR Code com tratamento de erro"""
        try:
            data, points, _ = self.detector.detectAndDecode(image)
            return data.strip() if data else ""
        except Exception as e:
            print(f"Erro na decodificação QR Code: {e}")
            return ""


# Instância global do decoder
qr_decoder = QRCodeDecoder()


@app.route('/read-qrcode', methods=['POST', 'OPTIONS'])
def read_qrcode():
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response

    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Nenhuma imagem enviada.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nenhum arquivo selecionado.'}), 400

    try:
        print("PROCESSAMENTO: Processando imagem recebida...")

        # Converte a imagem para um array numpy
        image_bytes = file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'error': 'Não foi possível decodificar a imagem.'}), 400

        print(f"DECODIFICACAO: Imagem decodificada - Dimensoes: {image.shape}")

        # Decodifica QR Code
        data = qr_decoder.decode_qr_code(image)

        print(f"DETECCAO QR CODE: Conteudo detectado: '{data}'")

        if data and data.strip():
            print(f"SUCESSO: QR Code detectado - Conteudo: {data.strip()}")
            return jsonify({
                'success': True,
                'qrCode': data.strip(),
                'message': 'QR Code detectado com sucesso!'
            })
        else:
            print("AVISO: Nenhum QR Code detectado na imagem")
            return jsonify({
                'success': False,
                'error': 'Nenhum QR Code detectado. Aponte para um QR Code válido.'
            }), 200

    except Exception as e:
        print(f"ERRO NO PROCESSAMENTO: {str(e)}")
        return jsonify({'success': False, 'error': f'Erro no servidor: {str(e)}'}), 500


# 🔥 NOVO ENDPOINT ADICIONADO AQUI
@app.route('/iot/control', methods=['POST', 'OPTIONS'])
def iot_control():
    """Endpoint para controle IOT da trava - só fecha quando armário fechar"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Dados JSON necessários'}), 400

        command = data.get('command', '').upper()
        print(f"COMANDO IOT RECEBIDO: {command}")

        if command == "ABRIR_TRAVA_IOT":
            print("INICIANDO ABERTURA DE TRAVA NO MODO IOT...")

            # Executa assincronamente
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            sucesso = loop.run_until_complete(locker_controller.abrir_trava_iot())
            loop.close()

            if sucesso:
                print("TRAVA ABERTA NO MODO IOT - AGUARDANDO FECHAMENTO DO ARMARIO")
                return jsonify({
                    'success': True,
                    'message': 'Trava liberada no modo IOT'
                })
            else:
                print("FALHA AO ABRIR TRAVA NO MODO IOT")
                return jsonify({
                    'success': False,
                    'error': 'Falha ao comunicar com a ESP32'
                }), 500

        else:
            return jsonify({
                'success': False,
                'error': f'Comando não reconhecido: {command}'
            }), 400

    except Exception as e:
        print(f"ERRO NO CONTROLE IOT: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Erro interno: {str(e)}'
        }), 500


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Backend Python funcionando!', 'status': 'ok'})


if __name__ == '__main__':
    print("INICIALIZACAO: Servidor QR Code iniciando na porta 5000...")
    print("TESTE: Acesse http://localhost:5000/test para testar")
    app.run(host='0.0.0.0', port=5000, debug=True)