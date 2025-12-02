"""
Rotas para leitura de QR Code - VERSÃO CORRIGIDA
"""
from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)
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

    logger.info("=== INICIANDO PROCESSAMENTO DE QR CODE ===")

    # Verificar se há arquivo de imagem
    if 'image' not in request.files:
        logger.error("Nenhuma imagem enviada")
        return jsonify({'success': False, 'error': 'Nenhuma imagem enviada.'}), 400

    file = request.files['image']

    if file.filename == '':
        logger.error("Nome de arquivo vazio")
        return jsonify({'success': False, 'error': 'Nenhum arquivo selecionado.'}), 400

    try:
        logger.info(f"Processando arquivo: {file.filename}")

        # Ler imagem
        image_bytes = file.read()
        logger.info(f"Tamanho da imagem: {len(image_bytes)} bytes")

        # Converter para numpy array
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Falha ao decodificar imagem")
            return jsonify({'success': False, 'error': 'Não foi possível decodificar a imagem.'}), 400

        logger.info(f"Imagem decodificada - Dimensões: {image.shape}")

        # Criar detector de QR Code
        detector = cv2.QRCodeDetector()

        # Método compatível
        result = detector.detectAndDecode(image)

        # OpenCV 4.x+ retorna uma tupla, OpenCV 3.x pode retornar direto
        if isinstance(result, tuple):
            data = result[0]
            logger.debug(f"Resultado tuple: {result}")
        else:
            data = result
            logger.debug(f"Resultado direto: {result}")

        logger.info(f"QR Code detectado: '{data}'")

        if data and data.strip():
            qr_data = data.strip()
            logger.info(f"✅ QR Code válido detectado: {qr_data}")
            return jsonify({
                'success': True,
                'qrCode': qr_data,
                'message': 'QR Code detectado com sucesso!'
            })
        else:
            logger.warning("Nenhum QR Code detectado na imagem")
            return jsonify({
                'success': False,
                'error': 'Nenhum QR Code detectado. Aponte para um QR Code válido.'
            }), 200

    except Exception as e:
        logger.error(f"❌ ERRO NO PROCESSAMENTO: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Erro no servidor: {str(e)}'}), 500