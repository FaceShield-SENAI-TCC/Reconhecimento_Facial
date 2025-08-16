import cv2


def test_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERRO: Não foi possível abrir a câmera")
        return

    print("Câmera aberta com sucesso!")
    print("Pressione 'q' para sair...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro na captura do frame")
            break

        cv2.imshow('Teste de Câmera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()