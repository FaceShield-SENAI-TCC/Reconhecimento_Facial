"""
Teste específico para diagnosticar problemas da câmera no cadastro
"""
import cv2
import base64
import requests
import json


def test_camera_direct():
    """Testa acesso direto à câmera"""
    print("🔍 TESTE DIRETO DA CÂMERA")
    print("=" * 40)

    for i in range(3):
        print(f"\n📷 Testando câmera índice {i}...")
        cap = cv2.VideoCapture(i)

        if cap.isOpened():
            print(f"✅ Câmera {i} ABERTA")

            # Testar leitura de frames
            frames_ok = 0
            for j in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames_ok += 1
                    print(f"   ✅ Frame {j + 1}: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    print(f"   ❌ Frame {j + 1}: FALHOU")

            print(f"📊 Resultado: {frames_ok}/5 frames OK")

            if frames_ok > 0:
                # Testar codificação base64
                small_frame = cv2.resize(frame, (320, 240))
                _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])

                if buffer is not None:
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    data_url = f"data:image/jpeg;base64,{jpg_as_text}"
                    print(f"✅ Base64 OK: {len(jpg_as_text)} bytes")
                    print(f"✅ Data URL: {data_url[:50]}...")
                else:
                    print("❌ Falha na codificação JPEG")

            cap.release()
        else:
            print(f"❌ Câmera {i} NÃO ABRE")


def test_websocket_connection():
    """Testa se o WebSocket está respondendo"""
    print("\n🔗 TESTE DE CONEXÃO WEBSOCKET")
    print("=" * 40)

    try:
        # Testar health endpoint
        response = requests.get('http://localhost:7001/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint: OK")
            data = response.json()
            print(f"   📊 Clientes ativos: {data.get('active_clients', 0)}")
            print(f"   📊 Capturas ativas: {data.get('active_captures', 0)}")
        else:
            print(f"❌ Health endpoint: ERRO {response.status_code}")
    except Exception as e:
        print(f"❌ Não conseguiu conectar ao servidor: {e}")


def test_frame_generation():
    """Testa geração de frames como será feito no WebSocket"""
    print("\n🎯 TESTE DE GERAÇÃO DE FRAMES")
    print("=" * 40)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Câmera não disponível para teste")
        return

    ret, frame = cap.read()
    if ret:
        # Processamento igual ao do servidor
        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (320, 240))
        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])

        if buffer is not None:
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{jpg_as_text}"

            print(f"✅ Frame processado: {frame.shape[1]}x{frame.shape[0]} -> 320x240")
            print(f"✅ Tamanho base64: {len(jpg_as_text)} bytes")
            print(f"✅ Data URL inicia com: {data_url[:50]}...")
            print(f"✅ Data URL termina com: ...{data_url[-50:]}")

            # Verificar formato
            if data_url.startswith('data:image/jpeg;base64,'):
                print("✅ Formato Data URL: CORRETO")
            else:
                print("❌ Formato Data URL: INCORRETO")
        else:
            print("❌ Falha na codificação do frame")
    else:
        print("❌ Não foi possível ler frame da câmera")

    cap.release()


if __name__ == "__main__":
    print("🎯 DIAGNÓSTICO COMPLETO - CÂMERA CADASTRO")
    print("=" * 50)

    test_camera_direct()
    test_websocket_connection()
    test_frame_generation()

    print("\n💡 CONCLUSÃO:")
    print("✅ Se todos os testes passaram, o problema está no front-end")
    print("❌ Se algum teste falhou, o problema está no back-end")