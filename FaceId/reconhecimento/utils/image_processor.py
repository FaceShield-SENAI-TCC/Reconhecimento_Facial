"""
Utilitários para Processamento de Imagem
"""
import base64
import numpy as np
import cv2
from typing import Optional, Tuple

def base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Converte uma string base64 em uma imagem OpenCV

    Args:
        base64_string: String base64 contendo a imagem

    Returns:
        np.ndarray: Imagem OpenCV ou None se falhar
    """
    try:
        # Remover prefixo se existir
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decodificar base64
        img_data = base64.b64decode(base64_string)

        # Converter para array numpy
        np_arr = np.frombuffer(img_data, np.uint8)

        # Decodificar para imagem OpenCV
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Falha ao decodificar imagem")

        return img

    except Exception as e:
        print(f"Erro ao converter base64 para imagem: {e}")
        return None

def preprocess_image(image: np.ndarray, target_size: tuple = (640, 480)) -> np.ndarray:
    """
    Pré-processa a imagem para reconhecimento facial

    Args:
        image: Imagem OpenCV
        target_size: Tamanho alvo para redimensionamento

    Returns:
        np.ndarray: Imagem pré-processada
    """
    # Redimensionar se necessário
    if image.shape[0] > target_size[0] or image.shape[1] > target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Converter para RGB se estiver em BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def validate_image_size(image: np.ndarray, min_size: tuple = (100, 100)) -> bool:
    """
    Valida o tamanho mínimo da imagem

    Args:
        image: Imagem OpenCV
        min_size: Tamanho mínimo permitido (largura, altura)

    Returns:
        bool: True se a imagem atender ao tamanho mínimo
    """
    if image is None:
        return False

    height, width = image.shape[:2]
    return width >= min_size[0] and height >= min_size[1]

def crop_face_region(image: np.ndarray, face_rect: tuple) -> np.ndarray:
    """
    Recorta a região do rosto da imagem

    Args:
        image: Imagem completa
        face_rect: Retângulo do rosto (x, y, width, height)

    Returns:
        np.ndarray: ROI do rosto
    """
    x, y, w, h = face_rect

    # Garantir que as coordenadas estão dentro dos limites da imagem
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    return image[y:y+h, x:x+w]

def calculate_image_sharpness(image: np.ndarray) -> float:
    """
    Calcula a nitidez da imagem usando o operador Laplaciano

    Args:
        image: Imagem OpenCV

    Returns:
        float: Valor da nitidez
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Aplicar o operador Laplaciano
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calcular a variância
    return float(laplacian.var())

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normaliza os valores dos pixels para o intervalo [0, 1]

    Args:
        image: Imagem OpenCV

    Returns:
        np.ndarray: Imagem normalizada
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    return image

def detect_face_boundaries(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detecta os limites do rosto na imagem usando Haar Cascade

    Args:
        image: Imagem OpenCV

    Returns:
        Optional[Tuple]: (x, y, width, height) ou None
    """
    # Carregar classificador Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detectar rostos
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        # Retornar o primeiro rosto detectado
        x, y, w, h = faces[0]
        return x, y, w, h

    return None