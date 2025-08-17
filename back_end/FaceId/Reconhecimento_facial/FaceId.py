import os
import numpy as np
import time
import cv2
from deepface import DeepFace

# ====================== CONFIGURAÇÕES ======================
DATABASE_DIR = "../Cadastro/facial_database"
MODEL_NAME = "VGG-Face"
DISTANCE_THRESHOLD = 0.40
MIN_FACE_SIZE = (100, 100)
RECOGNITION_COOLDOWN = 1.5


# ====================== CARREGAR BANCO DE DADOS FACIAL ======================
def load_facial_database():
    print("Carregando banco de dados facial...")
    database = {}

    for user_folder in os.listdir(DATABASE_DIR):
        user_path = os.path.join(DATABASE_DIR, user_folder)

        if os.path.isdir(user_path):
            user_name = " ".join([part.capitalize() for part in user_folder.split('_')])
            embeddings = []

            for face_file in os.listdir(user_path):
                if face_file.endswith((".jpg", ".png", ".jpeg")):
                    face_path = os.path.join(user_path, face_file)

                    try:
                        embedding_obj = DeepFace.represent(
                            img_path=face_path,
                            model_name=MODEL_NAME,
                            detector_backend="opencv",
                            enforce_detection=False
                        )

                        # Verificação robusta da estrutura de retorno
                        if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                            embedding_data = embedding_obj[0]
                            if "embedding" in embedding_data:
                                embedding = np.array(embedding_data["embedding"])
                                normalized_embedding = embedding / np.linalg.norm(embedding)
                                embeddings.append(normalized_embedding)
                            else:
                                print(f"⚠️ Estrutura de embedding inválida em: {face_path}")
                        else:
                            print(f"⚠️ Nenhum rosto detectado em: {face_path}")

                    except Exception as e:
                        print(f"Erro ao processar {face_path}: {str(e)}")
                        continue

            if embeddings:
                database[user_name] = embeddings
                print(f"» {user_name}: {len(embeddings)} amostras carregadas")

    if not database:
        print("❌ Banco de dados vazio ou nenhum rosto válido encontrado!")
        return None

    return database


# ====================== FUNÇÃO DE RECONHECIMENTO ======================
def recognize_face(face_img, database):
    if database is None:
        return None, None

    try:
        embedding_obj = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=False
        )

        if not isinstance(embedding_obj, list) or len(embedding_obj) == 0:
            return None, None

        embedding_data = embedding_obj[0]
        if "embedding" not in embedding_data:
            return None, None

        captured_embedding = np.array(embedding_data["embedding"])
        captured_embedding_normalized = captured_embedding / np.linalg.norm(captured_embedding)

        best_match = None
        min_distance = float('inf')

        for user_name, embeddings in database.items():
            for db_embedding in embeddings:
                cosine_similarity = np.dot(captured_embedding_normalized, db_embedding)
                cosine_distance = 1 - cosine_similarity

                if cosine_distance < min_distance and cosine_distance < DISTANCE_THRESHOLD:
                    min_distance = cosine_distance
                    best_match = user_name

        return best_match, min_distance

    except Exception as e:
        print(f"🚨 Erro no reconhecimento: {str(e)}")
        return None, None


# ====================== SISTEMA PRINCIPAL ======================
def facial_recognition_system():
    facial_db = load_facial_database()
    if facial_db is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Erro ao abrir câmera!")
        return

    print("\n✅ Sistema ativado. Pressione ESC para sair.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '../Recursos/haarcascade_frontalface_default.xml')

    last_recognition = 0
    current_user = None
    confidence = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE
        )

        recognition_status = "Procurando..."
        color = (0, 165, 255)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            current_time = time.time()
            if current_time - last_recognition > RECOGNITION_COOLDOWN:
                face_img = frame[y:y + h, x:x + w]
                current_user, distance = recognize_face(face_img, facial_db)
                last_recognition = current_time

                if current_user:
                    recognition_status = f"Bem-vindo(a), {current_user}!"
                    color = (0, 255, 0)
                    confidence = 1 - distance
                else:
                    recognition_status = "Desconhecido"
                    color = (0, 0, 255)
                    confidence = None
            elif current_user:
                recognition_status = f"Bem-vindo(a), {current_user}!"
                color = (0, 255, 0)

        cv2.putText(frame, recognition_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if confidence is not None:
            cv2.putText(frame, f"Confiança: {confidence:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, "Pressione ESC para sair", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Sistema de Segurança Facial', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n🛑 Sistema encerrado.")


if __name__ == "__main__":
    facial_recognition_system()