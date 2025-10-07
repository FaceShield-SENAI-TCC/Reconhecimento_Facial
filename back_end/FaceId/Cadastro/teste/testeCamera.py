import cv2
import os

print("Iniciando o teste de câmera...")

# Tenta capturar o vídeo da câmera (o '0' geralmente se refere à webcam padrão)
cap = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta com sucesso
if not cap.isOpened():
    print("ERRO: Não foi possível abrir a câmera.")
    print("Verifique se a câmera está conectada, se os drivers estão instalados e se não está sendo usada por outro programa.")
    exit()

print("Câmera aberta com sucesso. Pressione 'q' na janela para sair.")

while True:
    # Lê um frame da câmera
    ret, frame = cap.read()

    # Se não for possível ler o frame, encerra o loop
    if not ret:
        print("ERRO: Não foi possível ler o frame da câmera. Encerrando.")
        break

    # Exibe o frame em uma janela chamada 'Teste de Câmera'
    cv2.imshow('Teste de Camera', frame)

    # Espera 1ms por uma tecla. Se a tecla for 'q', o loop é encerrado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o objeto de captura e fecha todas as janelas do OpenCV
cap.release()
cv2.destroyAllWindows()

print("Teste de câmera finalizado.")