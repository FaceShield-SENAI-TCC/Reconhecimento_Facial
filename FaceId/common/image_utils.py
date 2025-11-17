"""
Utilitários para processamento de imagens - VERSÃO LEVE
"""
from typing import Tuple, Optional

import cv2
import numpy as np
import base64
import re
import logging

logger = logging.getLogger(__name__)

class ImageValidator:
    """Validador de imagens base64"""

    @staticmethod
    def validate_base64_image(image_data: str) -> Tuple[bool, str]:
        """Valida dados de imagem base64"""
        try:
            if not image_data or not isinstance(image_data, str):
                return False, "Dados de imagem vazios"

            if len(image_data) > 7 * 1024 * 1024:
                return False, "Imagem muito grande"

            if 'data:image' in image_data:
                image_data = image_data.split(',', 1)[1]

            base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
            if not base64_pattern.match(image_data):
                return False, "Formato base64 inválido"

            return True, "Imagem válida"

        except Exception as e:
            return False, f"Erro na validação: {str(e)}"

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calcula a nitidez da imagem"""
        if image is None or image.size == 0:
            return 0.0

        try:
            small_img = cv2.resize(image, (100, 100))
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 0.0

    @staticmethod
    def decode_base64_image(image_data: str) -> Optional[np.ndarray]:
        """Decodifica imagem base64 para array numpy"""
        try:
            is_valid, message = ImageValidator.validate_base64_image(image_data)
            if not is_valid:
                return None

            if 'data:image' in image_data:
                image_data = image_data.split(',', 1)[1]

            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            return image if image is not None and image.size > 0 else None

        except Exception as e:
            logger.error(f"Erro na decodificação: {str(e)}")
            return None

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """Pré-processamento básico da imagem"""
        try:
            # Redimensionar se necessário
            if image.shape[0] > 800 or image.shape[1] > 800:
                scale = 800 / max(image.shape[0], image.shape[1])
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height))

            return image
        except:
            return image

class FaceQualityValidator:
    """Validador de qualidade facial - VERSÃO LEVE"""

    def __init__(self, min_face_size: Tuple[int, int] = (100, 100), min_sharpness: float = 100.0):
        self.min_face_size = min_face_size
        self.min_sharpness = min_sharpness

    def validate_face_image(self, face_img: np.ndarray) -> Tuple[bool, str]:
        """Valida qualidade da imagem facial"""
        if face_img is None or face_img.size == 0:
            return False, "Imagem vazia"

        height, width = face_img.shape[:2]
        if height < self.min_face_size[0] or width < self.min_face_size[1]:
            return False, f"Rosto muito pequeno: {width}x{height}"

        sharpness = ImageValidator.calculate_sharpness(face_img)
        if sharpness < self.min_sharpness:
            return False, f"Imagem muito borrada: {sharpness:.1f}"

        return True, f"Qualidade aceitável: {sharpness:.1f}"