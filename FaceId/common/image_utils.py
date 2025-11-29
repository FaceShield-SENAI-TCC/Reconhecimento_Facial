"""
Utilitários para processamento e validação de imagens
"""
import cv2
import numpy as np
import base64
import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ImageValidator:
    """Validador de imagens base64 com verificações de segurança"""

    @staticmethod
    def validate_base64_image(image_data: str) -> Tuple[bool, str]:
        """
        Valida dados de imagem base64 com verificações completas

        Args:
            image_data: String base64 da imagem

        Returns:
            Tuple[bool, str]: (é_válido, mensagem_erro)
        """
        try:
            if not image_data or not isinstance(image_data, str):
                return False, "Dados de imagem vazios ou inválidos"

            if len(image_data) > 7 * 1024 * 1024:  # 7MB
                return False, "Imagem muito grande (máximo 5MB)"

            # Extrair dados base64 se vier com header
            if 'data:image' in image_data:
                image_data = image_data.split(',', 1)[1]

            # Validar formato base64
            base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
            if not base64_pattern.match(image_data):
                return False, "Formato base64 inválido"

            # Testar decodificação
            try:
                decoded = base64.b64decode(image_data)
                if len(decoded) == 0:
                    return False, "Dados base64 vazios após decodificação"
            except Exception as e:
                return False, f"Falha na decodificação base64: {str(e)}"

            return True, "Imagem válida"

        except Exception as e:
            logger.error(f"Erro na validação de imagem: {str(e)}")
            return False, f"Erro na validação: {str(e)}"

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """
        Calcula a nitidez da imagem usando o operador Laplacian

        Args:
            image: Array numpy da imagem

        Returns:
            float: Score de nitidez (quanto maior, mais nítida)
        """
        if image is None or image.size == 0:
            return 0.0

        try:
            # Reduzir tamanho para cálculo eficiente
            small_img = cv2.resize(image, (100, 100))
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception as e:
            logger.warning(f"Erro ao calcular nitidez: {e}")
            return 0.0

    @staticmethod
    def decode_base64_image(image_data: str) -> Optional[np.ndarray]:
        """
        Decodifica imagem base64 para array numpy com tratamento de erro

        Args:
            image_data: String base64 da imagem

        Returns:
            Optional[np.ndarray]: Array da imagem ou None se falhar
        """
        try:
            # Validar antes de decodificar
            is_valid, message = ImageValidator.validate_base64_image(image_data)
            if not is_valid:
                logger.error(f"Imagem inválida: {message}")
                return None

            # Extrair dados base64 se vier com header
            if 'data:image' in image_data:
                image_data = image_data.split(',', 1)[1]

            # Decodificar
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None or image.size == 0:
                logger.error("Falha ao decodificar imagem - dados corrompidos")
                return None

            return image

        except Exception as e:
            logger.error(f"Erro na decodificação de imagem: {str(e)}")
            return None

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Aplica pré-processamento para melhorar qualidade da imagem

        Args:
            image: Imagem original

        Returns:
            np.ndarray: Imagem processada
        """
        try:
            # Redimensionar se necessário
            if image.shape[0] > 800 or image.shape[1] > 800:
                scale = 800 / max(image.shape[0], image.shape[1])
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height))

            # Melhorar contraste com CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            return image

        except Exception as e:
            logger.warning(f"Pré-processamento de imagem falhou: {str(e)}")
            return image


class FaceQualityValidator:
    """Validador de qualidade facial com critérios padronizados"""

    def __init__(self, min_face_size: Tuple[int, int] = (100, 100), min_sharpness: float = 120.0):
        self.min_face_size = min_face_size
        self.min_sharpness = min_sharpness

    def validate_face_image(self, face_img: np.ndarray) -> Tuple[bool, str]:
        """
        Valida qualidade da imagem facial usando critérios padronizados

        Args:
            face_img: ROI do rosto

        Returns:
            Tuple[bool, str]: (é_válido, mensagem)
        """
        if face_img is None or face_img.size == 0:
            return False, "Imagem vazia"

        # Verificar tamanho mínimo
        height, width = face_img.shape[:2]
        if height < self.min_face_size[0] or width < self.min_face_size[1]:
            return False, f"Rosto muito pequeno: {width}x{height}"

        # Verificar nitidez
        sharpness = ImageValidator.calculate_sharpness(face_img)
        if sharpness < self.min_sharpness:
            return False, f"Imagem muito borrada: {sharpness:.1f}"

        return True, f"Qualidade aceitável: {sharpness:.1f}"


class AntiSpoofingValidator:
    """Validador anti-spoofing para impedir reconhecimento por fotos"""

    def __init__(self):
        self.min_face_ratio = 0.15  # Rosto deve ocupar pelo menos 15% da imagem
        self.max_face_ratio = 0.85  # Rosto não pode ocupar mais que 85%
        self.min_sharpness_live = 120.0  # Nitidez mínima para câmera ao vivo
        self.max_aspect_ratio = 1.8  # Proporção máxima largura/altura

    def is_live_camera_face(self, image: np.ndarray, face_roi: np.ndarray) -> Tuple[bool, str]:
        """
        Valida se o rosto é de uma câmera ao vivo e não de uma foto
        """
        try:
            img_height, img_width = image.shape[:2]
            face_height, face_width = face_roi.shape[:2]

            # 1. Verificar proporção do rosto na imagem
            face_ratio = (face_width * face_height) / (img_width * img_height)

            if face_ratio < self.min_face_ratio:
                return False, f"Rosto muito pequeno na imagem ({face_ratio:.1%})"

            if face_ratio > self.max_face_ratio:
                return False, f"Rosto muito grande na imagem ({face_ratio:.1%}) - possível foto"

            # 2. Verificar nitidez (fotos tendem a ser mais borradas)
            sharpness = ImageValidator.calculate_sharpness(face_roi)
            if sharpness < self.min_sharpness_live:
                return False, f"Nitidez insuficiente ({sharpness:.1f}) - possível foto"

            # 3. Verificar proporção aspecto (fotos têm proporções diferentes)
            aspect_ratio = face_width / face_height
            if aspect_ratio > self.max_aspect_ratio or aspect_ratio < (1/self.max_aspect_ratio):
                return False, f"Proporção do rosto inválida ({aspect_ratio:.2f})"

            # 4. Verificar iluminação (fotos têm iluminação mais uniforme)
            brightness_std = np.std(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
            if brightness_std < 25:  # Muito uniforme = possível foto
                return False, "Iluminação muito uniforme - possível foto"

            return True, "Rosto validado como câmera ao vivo"

        except Exception as e:
            return False, f"Erro na validação anti-spoofing: {str(e)}"