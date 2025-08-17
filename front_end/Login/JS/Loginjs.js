const video = document.getElementById("video");
const mensagem = document.getElementById("mensagem");
const successOverlay = document.getElementById("success-overlay");

// Estados do sistema
let stream = null;
let recognitionInterval = null;
let currentUser = null;
let recognitionStartTime = 0;
let recognitionProgress = 0;
const REQUIRED_RECOGNITION_TIME = 5000; // 5 segundos em milissegundos

// Adicionar barra de progresso
const progressContainer = document.createElement("div");
progressContainer.className = "progress-container";
const progressBar = document.createElement("div");
progressBar.className = "progress-bar";
progressContainer.appendChild(progressBar);
mensagem.parentNode.insertBefore(progressContainer, mensagem.nextSibling);

async function iniciarCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 }
      } 
    });
    video.srcObject = stream;
    mensagem.textContent = "Câmera ativada. Posicione seu rosto.";
    
    // Iniciar o reconhecimento contínuo
    startContinuousRecognition();
  } catch (error) {
    mensagem.textContent = "Não foi possível ativar a câmera.";
    console.error("Erro ao acessar câmera:", error);
  }
}

function startContinuousRecognition() {
  // Limpar intervalo anterior se existir
  if (recognitionInterval) clearInterval(recognitionInterval);
  
  // Verificar a cada 500ms
  recognitionInterval = setInterval(async () => {
    try {
      // Capturar frame
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      
      // Espelhar a imagem para corresponder ao preview
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Converter para base64 (70% de qualidade)
      const imagemBase64 = canvas.toDataURL('image/jpeg', 0.7);
      
      // Enviar para o backend
      const response = await fetch('http://localhost:5000/face-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imagem: imagemBase64 })
      });
      
      if (!response.ok) {
        throw new Error(`Erro no servidor: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.authenticated) {
        const user = data.user;
        const confidencePercent = Math.round(data.confidence * 100);
        
        // Se for o mesmo usuário
        if (currentUser === user) {
          const elapsedTime = Date.now() - recognitionStartTime;
          recognitionProgress = Math.min(100, (elapsedTime / REQUIRED_RECOGNITION_TIME) * 100);
          
          mensagem.textContent = `Reconhecido: ${user} (${confidencePercent}%) - ${Math.round(recognitionProgress)}%`;
          progressBar.style.width = `${recognitionProgress}%`;
          
          // Se passou 5 segundos
          if (elapsedTime >= REQUIRED_RECOGNITION_TIME) {
            clearInterval(recognitionInterval);
            showSuccessAnimation();
            // Redirecionar após 2 segundos
            setTimeout(() => {
              window.location.href = '../MenuProf/Menu.html';
            }, 2000);
          }
        } else {
          // Novo usuário detectado
          currentUser = user;
          recognitionStartTime = Date.now();
          recognitionProgress = 0;
          progressBar.style.width = "0%";
          mensagem.textContent = `Reconhecido: ${user} (${confidencePercent}%)`;
        }
      } else {
        // Resetar se não reconheceu
        currentUser = null;
        recognitionStartTime = 0;
        recognitionProgress = 0;
        progressBar.style.width = "0%";
        mensagem.textContent = data.message || "Posicione seu rosto na câmera.";
      }
    } catch (error) {
      console.error('Erro:', error);
      mensagem.textContent = "Erro temporário. Continuando...";
    }
  }, 500); // Verificar a cada 500ms
}

function showSuccessAnimation() {
  // Mostrar overlay de sucesso
  successOverlay.classList.add('active');
  mensagem.textContent = "Autenticação bem-sucedida! Redirecionando...";
  progressBar.style.width = "100%";
  
  // Adicionar animação ao checkmark
  setTimeout(() => {
    successOverlay.querySelector('.checkmark__circle').style.animation = 'stroke 0.6s cubic-bezier(0.65, 0, 0.45, 1) forwards';
    successOverlay.querySelector('.checkmark__check').style.animation = 'stroke 0.3s cubic-bezier(0.65, 0, 0.45, 1) 0.8s forwards';
  }, 100);
}

// Iniciar câmera quando a página carregar
window.addEventListener('DOMContentLoaded', iniciarCamera);

// Parar a câmera ao sair da página
window.addEventListener('beforeunload', () => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
  if (recognitionInterval) clearInterval(recognitionInterval);
});