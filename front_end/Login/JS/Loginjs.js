const video = document.getElementById("video");
const mensagem = document.getElementById("mensagem");
const scanAnimation = document.getElementById("scanAnimation");
const faceScanWidget = document.querySelector('.face-scan-widget');
let stream = null;
let recognitionInterval = null;

async function iniciarCamera() {
    try {
        // Limpar qualquer intervalo anterior
        if (recognitionInterval) {
            clearInterval(recognitionInterval);
        }
        
        // Parar qualquer stream anterior
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        
        // Solicitar acesso à câmera
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            }
        });
        
        video.srcObject = stream;
        mensagem.textContent = "Posicione seu rosto na câmera";
        mensagem.style.color = "white";
        
        // Iniciar reconhecimento facial após 1 segundo
        setTimeout(iniciarReconhecimentoFacial, 1000);
        
    } catch (error) {
        console.error("Erro ao acessar câmera:", error);
        mensagem.textContent = "Erro ao acessar a câmera. Verifique as permissões.";
        mensagem.style.color = "#ff4d7d";
    }
}

function iniciarReconhecimentoFacial() {
    // Verificar se a câmera está funcionando
    if (!video.srcObject) {
        mensagem.textContent = "Câmera não disponível";
        return;
    }
    
    // Configurar intervalo para reconhecimento (a cada 2 segundos)
    recognitionInterval = setInterval(async () => {
        try {
            // Adicionar efeito visual de escaneamento
            faceScanWidget.style.boxShadow = "0 0 20px #00e0ff";
            scanAnimation.style.display = "block";
            
            // Capturar frame
            const imageData = await capturarFrame();
            
            // Enviar para o backend
            const resultado = await enviarParaReconhecimento(imageData);
            
            // Processar resultado
            if (resultado && resultado.authenticated) {
                clearInterval(recognitionInterval);
                mensagem.textContent = `Bem-vindo, ${resultado.user}!`;
                mensagem.style.color = "#00e0ff";
                
                // Efeito visual de sucesso
                faceScanWidget.style.boxShadow = "0 0 30px #00ff7f";
                
                // Redirecionar após 2 segundos
                setTimeout(() => {
                    window.location.href = "../MenuAluno/Menu.html";
                }, 2000);
            } else if (resultado && !resultado.authenticated) {
                mensagem.textContent = resultado.message || "Usuário não reconhecido";
                mensagem.style.color = "#ff4d7d";
                
                // Efeito visual de falha
                faceScanWidget.style.boxShadow = "0 0 20px #ff4d7d";
                setTimeout(() => {
                    faceScanWidget.style.boxShadow = "0 0 20px #00e0ff";
                    mensagem.textContent = "Posicione seu rosto na câmera";
                    mensagem.style.color = "white";
                }, 2000);
            }
        } catch (error) {
            console.error("Erro no reconhecimento:", error);
            mensagem.textContent = "Erro no reconhecimento. Tente novamente.";
            mensagem.style.color = "#ff4d7d";
        } finally {
            // Remover efeito visual após um tempo
            setTimeout(() => {
                scanAnimation.style.display = "none";
            }, 500);
        }
    }, 2000); // Verificar a cada 2 segundos
}

async function capturarFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    
    // Desenhar a imagem espelhada (a câmera frontal geralmente espelha a imagem)
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Converter para base64 (sem o prefixo data:image/jpeg;base64,)
    return canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
}

async function enviarParaReconhecimento(imageData) {
    try {
        const response = await fetch('http://localhost:5005/face-login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ imagem: imageData })
        });

        if (!response.ok) {
            throw new Error(`Erro HTTP: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error("Erro na requisição:", error);
        
        // Verificar se é erro de CORS
        if (error.message.includes("Failed to fetch") || error.message.includes("CORS")) {
            mensagem.textContent = "Erro de conexão com o servidor. Verifique se o servidor está rodando.";
            mensagem.style.color = "#ff4d7d";
        }
        
        return null;
    }
}

function pararCamera() {
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// Iniciar quando a página carregar
document.addEventListener('DOMContentLoaded', iniciarCamera);

// Parar a câmera quando a página for fechada
window.addEventListener('beforeunload', pararCamera);
window.addEventListener('pagehide', pararCamera);

// Recarregar a câmera se a página for reativada
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        // Pequeno atraso para garantir que a página esteja completamente visível
        setTimeout(iniciarCamera, 300);
    } else {
        pararCamera();
    }
});