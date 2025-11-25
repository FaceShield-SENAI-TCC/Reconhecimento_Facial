from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)

# Configuração do CORS para permitir seu frontend
CORS(app, origins=['http://127.0.0.1:5500', 'http://localhost:5500', 'http://localhost:8080'])


class QRCodeDecoder:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Aplica múltiplas técnicas de pré-processamento para melhorar a detecção"""
        # Converte para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Filtro para reduzir ruído
        denoised = cv2.medianBlur(gray, 3)

        # 2. Equalização de histograma para melhorar contraste
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
        """Aplica filtro de sharpening para realçar bordas"""
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def try_multiple_decodings(self, image: np.ndarray) -> str:
        """Tenta múltiplas estratégias de decodificação"""
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
                    print(f"✓ QR Code detectado com estratégia: {strategy_name}")
                    return data.strip()

                # Se não detectou, tenta com diferentes níveis de threshold
                if len(processed_img.shape) == 2:  # Se é escala de cinza
                    # Tenta diferentes métodos de threshold
                    _, thresh1 = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY)
                    _, thresh2 = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    thresh3 = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)

                    for i, thresh_img in enumerate([thresh1, thresh2, thresh3]):
                        data, points, _ = self.detector.detectAndDecode(thresh_img)
                        if data and data.strip():
                            print(f"✓ QR Code detectado com threshold {i + 1} na estratégia: {strategy_name}")
                            return data.strip()

            except Exception as e:
                print(f"✗ Erro na estratégia {strategy_name}: {e}")
                continue

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

        # Usa o decoder melhorado
        data = qr_decoder.try_multiple_decodings(image)

        print(f"DETECCAO QR CODE: Conteudo detectado: '{data}'")

        if data and data.strip():
            print(f"SUCESSO: QR Code detectado - Conteudo: {data.strip()}")
            return jsonify({
                'success': True,
                'qrCode': data.strip(),
                'message': 'QR Code detectado com sucesso!'
            })
        else:
            print("AVISO: Nenhum QR Code detectado na imagem após múltiplas tentativas")
            return jsonify({
                'success': False,
                'error': 'Nenhum QR Code detectado. Tente: 1) Melhorar a iluminação 2) Centralizar o QR Code 3) Aproximar a câmera'
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