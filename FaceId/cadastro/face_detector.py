"""
Módulo de Detecção Facial Otimizado
"""
import time
import cv2
from deepface import DeepFace
from common.image_utils import FaceQualityValidator

import logging
logger = logging.getLogger(__name__)

class FaceDetector:
    """Detector facial otimizado com cache"""

    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 0.2  # 5 FPS para detecção
        self.cached_faces = []

    def detect_faces(self, frame):
        """Detecta rostos no frame com otimização"""
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.cached_faces

        try:
            small_frame = cv2.resize(frame, (320, 240))
            detected_faces = DeepFace.extract_faces(
                img_path=small_frame,
                detector_backend="opencv",
                enforce_detection=False,
                align=False
            )

            faces = []
            quality_validator = FaceQualityValidator()

            for face in detected_faces:
                if 'facial_area' in face:
                    x = int(face['facial_area']['x'] * frame.shape[1] / 320)
                    y = int(face['facial_area']['y'] * frame.shape[0] / 240)
                    w = int(face['facial_area']['w'] * frame.shape[1] / 320)
                    h = int(face['facial_area']['h'] * frame.shape[0] / 240)

                    # Validar qualidade do rosto
                    face_roi = frame[y:y + h, x:x + w]
                    is_valid, validation_msg = quality_validator.validate_face_image(face_roi)

                    if is_valid:
                        faces.append((x, y, w, h))

            self.cached_faces = faces
            self.last_detection_time = current_time
            return faces

        except Exception as e:
            logger.warning(f"Erro na detecção: {e}")
            return []