from flask import Blueprint, request, jsonify
import cv2
import numpy as np

qr_bp = Blueprint('qr_bp', __name__)

@qr_bp.route('/read-qrcode', methods=['POST', 'OPTIONS'])
def read_qrcode():
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
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

        # Cria o detector de QR Code
        detector = cv2.QRCodeDetector()

        # Método compatível com todas as versões do OpenCV
        result = detector.detectAndDecode(image)

        # OpenCV 4.x+ retorna uma tupla, OpenCV 3.x pode retornar direto
        if isinstance(result, tuple):
            data = result[0]
        else:
            data = result

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