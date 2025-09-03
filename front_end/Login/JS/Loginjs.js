const video = document.getElementById("video");
const mensagem = document.getElementById("mensagem");
const scanAnimation = document.getElementById("scanAnimation");
const faceScanWidget = document.querySelector('.face-scan-widget');
const botaoTentar = document.getElementById("botaoTentar");
const buttonTentar = document.getElementById("buttonTentar");
const statusConexao = document.getElementById("statusConexao");
let stream = null;
let recognitionInterval = null;
let isBackendOnline = false;

// Configurações
const BACKEND_URL = 'http://localhost:5005';
const CHECK_INTERVAL = 1000; // Reduzido para 1 segundo para melhor detecção
const REQUIRED_CONSECUTIVE_TIME = 5000; // 5 segundos de detecção contínua
const MAX_RETRIES = 3;
let retryCount = 0;

// Variáveis para controle de detecção contínua
let lastRecognizedUser = null;
let recognitionStartTime = null;
let recognitionTimer = null;

// Verificar status do backend
async function verificarBackend() {
    try {
        const response = await fetch(`${BACKEND_URL}/health`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (response.ok) {
            const data = await response.json();
            isBackendOnline = data.status === 'healthy';
            atualizarStatusConexao(isBackendOnline);
            return isBackendOnline;
        }
    } catch (error) {
        console.error("Erro ao verificar backend:", error);
        isBackendOnline = false;
        atualizarStatusConexao(false);
        return false;
    }
    return false;
}

// Atualizar indicador de status
function atualizarStatusConexao(online) {
    if (online) {
        statusConexao.textContent = 'Backend Online';
        statusConexao.className = 'status-conexao status-online';
    } else {
        statusConexao.textContent = 'Backend Offline';
        statusConexao.className = 'status-conexao status-offline';
    }
}

// Mostrar botão de tentar novamente
function mostrarBotaoTentar() {
    botaoTentar.style.display = 'block';
}

// Esconder botão de tentar novamente
function esconderBotaoTentar() {
    botaoTentar.style.display = 'none';
}

// Configurar botão de tentar novamente
buttonTentar.addEventListener('click', function() {
    esconderBotaoTentar();
    resetarReconhecimento();
    iniciarSistema();
});

// Resetar o estado de reconhecimento
function resetarReconhecimento() {
    lastRecognizedUser = null;
    recognitionStartTime = null;
    if (recognitionTimer) {
        clearTimeout(recognitionTimer);
        recognitionTimer = null;
    }
    mensagem.textContent = "Posicione seu rosto na câmera";
    mensagem.style.color = "white";
    faceScanWidget.style.boxShadow = "0 0 20px rgba(0, 224, 255, 0.5)";
}

async function iniciarSistema() {
    mensagem.textContent = "Verificando conexão...";
    mensagem.style.color = "white";
    
    // Verificar se o backend está online
    const backendDisponivel = await verificarBackend();
    
    if (!backendDisponivel) {
        mensagem.textContent = "Servidor de reconhecimento não disponível. Verifique se o backend está rodando.";
        mensagem.style.color = "#ff4d7d";
        mostrarBotaoTentar();
        return;
    }
    
    // Se o backend está online, iniciar a câmera
    iniciarCamera();
}

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
        resetarReconhecimento();
       
        // Iniciar reconhecimento facial após 1 segundo
        setTimeout(iniciarReconhecimentoFacial, 1000);
       
    } catch (error) {
        console.error("Erro ao acessar câmera:", error);
        mensagem.textContent = "Erro ao acessar a câmera. Verifique as permissões.";
        mensagem.style.color = "#ff4d7d";
        mostrarBotaoTentar();
    }
}

function iniciarReconhecimentoFacial() {
    // Verificar se a câmera está funcionando
    if (!video.srcObject) {
        mensagem.textContent = "Câmera não disponível";
        return;
    }
   
    // Configurar intervalo para reconhecimento (a cada 1 segundo)
    recognitionInterval = setInterval(async () => {
        try {
            // Verificar se o backend ainda está online
            if (!await verificarBackend()) {
                mensagem.textContent = "Conexão com o servidor perdida";
                mensagem.style.color = "#ff4d7d";
                clearInterval(recognitionInterval);
                mostrarBotaoTentar();
                return;
            }
            
            // Adicionar efeito visual de escaneamento
            faceScanWidget.style.boxShadow = "0 0 20px #00e0ff";
            scanAnimation.style.display = "block";
           
            // Capturar frame
            const imageData = await capturarFrame();
           
            // Enviar para o backend
            const resultado = await enviarParaReconhecimento(imageData);
           
            // Processar resultado
            console.log("Resultado do reconhecimento:", resultado);
            
            if (resultado && resultado.authenticated) {
                const currentUser = resultado.user;
                const currentTime = new Date().getTime();
                
                // Se é um usuário diferente do anterior, reiniciar o contador
                if (currentUser !== lastRecognizedUser) {
                    lastRecognizedUser = currentUser;
                    recognitionStartTime = currentTime;
                    mensagem.textContent = `Reconhecido: ${currentUser}. Mantenha-se na câmera...`;
                    mensagem.style.color = "#00e0ff";
                }
                
                // Calcular quanto tempo o mesmo usuário foi detectado
                const detectionTime = currentTime - recognitionStartTime;
                
                // Atualizar mensagem com contagem regressiva
                const timeLeft = Math.max(0, REQUIRED_CONSECUTIVE_TIME - detectionTime);
                mensagem.textContent = `Reconhecido: ${currentUser}. Redirecionando em ${(timeLeft/1000).toFixed(1)}s...`;
                
                // Se o mesmo usuário foi detectado por 3 segundos, redirecionar
                if (detectionTime >= REQUIRED_CONSECUTIVE_TIME) {
                    clearInterval(recognitionInterval);
                    mensagem.textContent = `Bem-vindo, ${currentUser}!`;
                    mensagem.style.color = "#00e0ff";
                   
                    // Efeito visual de sucesso
                    faceScanWidget.style.boxShadow = "0 0 30px #00ff7f";
                   
                    // Redirecionar após 1 segundo
                    setTimeout(() => {
                        window.location.href = "../MenuProf/Menu.html";
                    }, 1000);
                }
            } else if (resultado && !resultado.authenticated) {
                // Resetar reconhecimento se não autenticado
                resetarReconhecimento();
                mensagem.textContent = resultado.message || "Usuário não reconhecido";
                mensagem.style.color = "#ff4d7d";
               
                // Efeito visual de falha
                faceScanWidget.style.boxShadow = "0 0 20px #ff4d7d";
                setTimeout(() => {
                    faceScanWidget.style.boxShadow = "0 0 20px #00e0ff";
                    mensagem.textContent = "Posicione seu rosto na câmera";
                    mensagem.style.color = "white";
                }, 2000);
            } else {
                // Caso resultado seja null (erro na requisição)
                resetarReconhecimento();
                mensagem.textContent = "Erro no reconhecimento. Tente novamente.";
                mensagem.style.color = "#ff4d7d";
            }
        } catch (error) {
            console.error("Erro no reconhecimento:", error);
            resetarReconhecimento();
            mensagem.textContent = "Erro no reconhecimento. Tente novamente.";
            mensagem.style.color = "#ff4d7d";
        } finally {
            // Remover efeito visual após um tempo
            setTimeout(() => {
                scanAnimation.style.display = "none";
            }, 500);
        }
    }, CHECK_INTERVAL); // Verificar a cada 1 segundo
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

        // Verificar se a resposta é válida
        if (!response.ok) {
            const errorText = await response.text();
            console.error("Erro na resposta do servidor:", response.status, errorText);
            
            // Tentar parsear como JSON se possível
            try {
                const errorData = JSON.parse(errorText);
                throw new Error(errorData.error || `Erro HTTP: ${response.status}`);
            } catch (e) {
                throw new Error(`Erro HTTP: ${response.status} - ${errorText}`);
            }
        }

        // Parsear la respuesta JSON
        const result = await response.json();
        return result;
        
    } catch (error) {
        console.error("Erro na requisição:", error);
       
        // Verificar se é erro de CORS
        if (error.message.includes("Failed to fetch") || error.message.includes("CORS") || error.message.includes("NetworkError")) {
            mensagem.textContent = "Erro de conexão com o servidor. Verifique se o servidor está rodando.";
            mensagem.style.color = "#ff4d7d";
            isBackendOnline = false;
            atualizarStatusConexao(false);
        } else if (error.message.includes("Unexpected token") || error.message.includes("JSON")) {
            mensagem.textContent = "Erro no formato da resposta do servidor.";
            mensagem.style.color = "#ff4d7d";
        } else {
            mensagem.textContent = `Erro: ${error.message}`;
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
   
    if (recognitionTimer) {
        clearTimeout(recognitionTimer);
        recognitionTimer = null;
    }
   
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    resetarReconhecimento();
}

// Iniciar quando a página carregar
document.addEventListener('DOMContentLoaded', iniciarSistema);

// Parar a câmera quando a página for fechada
window.addEventListener('beforeunload', pararCamera);
window.addEventListener('pagehide', pararCamera);

// Recarregar a câmera se a página for reativada
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        // Pequeno atraso para garantir que a página esteja completamente visível
        setTimeout(iniciarSistema, 300);
    } else {
        pararCamera();
    }
});

// Adicionar timeout para fetch
(function() {
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        const [url, options] = args;
        const timeout = options?.timeout || 5000;
        
        const controller = new AbortController();
        const signal = controller.signal;
        
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        const fetchOptions = {
            ...options,
            signal
        };
        
        return originalFetch(url, fetchOptions)
            .then(response => {
                clearTimeout(timeoutId);
                return response;
            })
            .catch(error => {
                clearTimeout(timeoutId);
                throw error;
            });
    };
})();