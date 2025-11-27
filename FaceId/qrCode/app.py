from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import asyncio
from common.locker_controller import LockerController

app = Flask(__name__)

# Configuracao do CORS para permitir seu frontend
CORS(app, origins=['http://127.0.0.1:5500', 'http://localhost:5500', 'http://localhost:8080'])


class QRCodeDecoder:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Aplica multiplas tecnicas de pre-processamento para melhorar a deteccao"""
        # Converte para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Filtro para reduzir ruido
        denoised = cv2.medianBlur(gray, 3)

        # 2. Equalizacao de histograma para melhorar contraste
        equalized = cv2.equalizeHist(denoised)

        # 3. Filtro bilateral (preserva bordas)
        bilateral = cv2.bilateralFilter(equalized, 9, 75, 75)

        return bilateral

    def resize_image(self, image: np.ndarray, target_size: int = 800) -> np.ndarray:
        """Redimensiona imagem mantendo aspect ratio"""
        height, width = image.shape[:2]
        max_dim = max(height, width)

        if max_dim > target_size:
            scale = target_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        elif max_dim < 300:
            scale = 300 / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return image

    def apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Aplica filtro de sharpening para realcar bordas"""
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def try_multiple_decodings(self, image: np.ndarray) -> str:
        """Tenta multiplas estrategias de decodificacao"""
        strategies = [
            ("Original", image),
            ("Pre-processada", self.preprocess_image(image)),
            ("Redimensionada", self.resize_image(image)),
            ("Sharpened", self.apply_sharpening(self.preprocess_image(image))),
            ("Combinada", self.resize_image(self.preprocess_image(image)))
        ]

        for strategy_name, processed_img in strategies:
            try:
                # Tenta detectar com a imagem processada
                data, points, _ = self.detector.detectAndDecode(processed_img)

                if data and data.strip():
                    print(f"QR Code detectado com estrategia: {strategy_name}")
                    return data.strip()

                # Se nao detectou, tenta com diferentes niveis de threshold
                if len(processed_img.shape) == 2:  # Se e escala de cinza
                    # Tenta diferentes metodos de threshold
                    _, thresh1 = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY)
                    _, thresh2 = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    thresh3 = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)

                    for i, thresh_img in enumerate([thresh1, thresh2, thresh3]):
                        data, points, _ = self.detector.detectAndDecode(thresh_img)
                        if data and data.strip():
                            print(f"QR Code detectado com threshold {i + 1} na estrategia: {strategy_name}")
                            return data.strip()

            except Exception as e:
                print(f"Erro na estrategia {strategy_name}: {e}")
                continue

        return ""


# Instancia global do decoder e controlador da trava
qr_decoder = QRCodeDecoder()
locker_controller = LockerController()


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
            return jsonify({'success': False, 'error': 'Nao foi possivel decodificar a imagem.'}), 400

        print(f"DECODIFICACAO: Imagem decodificada - Dimensoes: {image.shape}")

        # Usa o decoder melhorado
        data = qr_decoder.try_multiple_decodings(image)

        print(f"DETECCAO QR CODE: Conteudo detectado: '{data}'")

        if data and data.strip():
            print(f"SUCESSO: QR Code detectado - Conteudo: {data.strip()}")

            # Abrir trava quando QR Code for lido
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(locker_controller.abrir_trava_qrcode())
                loop.close()
                print("Trava liberada via QR Code")
            except Exception as e:
                print(f"Erro ao controlar trava: {e}")

            return jsonify({
                'success': True,
                'qrCode': data.strip(),
                'message': 'QR Code detectado com sucesso! Trava liberada.'
            })
        else:
            print("AVISO: Nenhum QR Code detectado na imagem apos multiplas tentativas")
            return jsonify({
                'success': False,
                'error': 'Nenhum QR Code detectado. Tente: 1) Melhorar a iluminacao 2) Centralizar o QR Code 3) Aproximar a camera'
            }), 200

    except Exception as e:
        print(f"ERRO NO PROCESSAMENTO: {str(e)}")
        return jsonify({'success': False, 'error': f'Erro no servidor: {str(e)}'}), 500


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Backend Python funcionando!', 'status': 'ok'})


if __name__ == '__main__':
    print("INICIALIZACAO: Servidor QR Code iniciando na porta 5000...")
    print("TESTE: Acesse http://localhost:5000/test para testar")
    app.run(host='0.0.0.0', port=5000, debug=True)