document.addEventListener("DOMContentLoaded", () => {
  // Elementos do formulário
  const form = document.getElementById("cadastroForm");
  if (!form) {
    console.error("Elemento #cadastroForm não encontrado!");
    return;
  }

  const feedback = document.getElementById("feedback");
  const tipoUsuarioSelect = document.getElementById("tipoUsuario");
  const usernameGroup = document.getElementById("username-group");
  const usernameInput = document.getElementById("username");
  const scanWidget = document.querySelector(".face-scan-widget");
  const scanInstruction = scanWidget.querySelector(".upload-instruction");
  const senhaGroup = document.getElementById("senha-group");
  const senhaInput = document.getElementById("senha");
  const turmaInput = document.getElementById("turma");
  
  // Estados da captura
  let isScanning = false;
  let captureSessionId = null;
  let faceCaptureComplete = false;
  let faceCaptureSuccess = false;

  // Elementos para visualização da câmera
  const cameraFeed = document.createElement('img');
  cameraFeed.id = 'camera-feed';
  cameraFeed.style.display = 'none';
  cameraFeed.style.width = '100%';
  cameraFeed.style.height = '100%';
  cameraFeed.style.objectFit = 'cover';
  cameraFeed.style.borderRadius = '8px';
  scanWidget.appendChild(cameraFeed);

  // Elemento de checkmark
  const checkmark = document.createElement('div');
  checkmark.innerHTML = '&#10004;';
  checkmark.style.display = 'none';
  checkmark.style.position = 'absolute';
  checkmark.style.top = '50%';
  checkmark.style.left = '50%';
  checkmark.style.transform = 'translate(-50%, -50%)';
  checkmark.style.fontSize = '5rem';
  checkmark.style.color = '#00ffaa';
  checkmark.style.textShadow = '0 0 20px rgba(0, 255, 170, 0.8)';
  checkmark.style.zIndex = '10';
  scanWidget.appendChild(checkmark);

  const API_URL = "http://localhost:8080/usuarios/novoUsuario";

  // Conectar ao WebSocket do servidor de captura
  const socket = io("http://localhost:5000", {
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 3000
  });

  // ====================== EVENTOS DO WEBSOCKET ======================
  socket.on("connect", () => {
    console.log("Conectado ao servidor de captura facial");
  });

  socket.on("capture_progress", (data) => {
    if (data.session_id === captureSessionId) {
      const progress = data.progress;
      scanInstruction.textContent = `Capturando... ${progress}%`;
      
      scanWidget.style.background = `linear-gradient(
        to right, 
        rgba(0, 224, 255, 0.5) ${progress}%, 
        rgba(30, 41, 59, 0.3) ${progress}%
      )`;
    }
  });

  socket.on('capture_frame', (data) => {
    if (data.session_id === captureSessionId) {
      cameraFeed.src = `data:image/jpeg;base64,${data.frame}`;
      cameraFeed.style.display = 'block';
      scanInstruction.style.display = 'none';
    }
  });

  socket.on("capture_complete", (data) => {
    if (data.session_id === captureSessionId) {
      isScanning = false;
      faceCaptureComplete = true;
      
      if (data.success) {
        faceCaptureSuccess = true;
        cameraFeed.style.display = 'none';
        checkmark.style.display = 'block';
        scanInstruction.style.display = 'none';
        
        // Animação de confirmação
        setTimeout(() => {
          checkmark.style.transition = 'all 1s ease';
          checkmark.style.transform = 'translate(-50%, -50%) scale(1.5)';
          checkmark.style.opacity = '0';
        }, 2000);
        
        setTimeout(() => {
          checkmark.style.display = 'none';
          scanInstruction.style.display = 'block';
          scanInstruction.textContent = "Biometria capturada!";
          scanWidget.style.background = "rgba(0, 255, 170, 0.1)";
          scanWidget.classList.add('capture-success');
        }, 3000);
      } else {
        faceCaptureSuccess = false;
        cameraFeed.style.display = 'none';
        scanInstruction.style.display = 'block';
        scanInstruction.textContent = "Falha na captura. Clique para tentar novamente";
        scanWidget.style.background = "rgba(255, 77, 125, 0.2)";
        showFeedback("error", `Erro na captura biométrica: ${data.message}`);
      }
    }
  });

  // ====================== EVENTO DE CLIQUE NO WIDGET ======================
  scanWidget.addEventListener("click", async () => {
    if (isScanning) return;
    
    const nome = document.getElementById("nome").value.trim();
    const sobrenome = document.getElementById("sobrenome").value.trim();
    
    if (!nome || !sobrenome) {
      showFeedback("error", "Preencha nome e sobrenome antes da captura biométrica");
      return;
    }

    isScanning = true;
    faceCaptureComplete = false;
    faceCaptureSuccess = false;
    captureSessionId = Date.now().toString();
    
    // Resetar elementos visuais
    cameraFeed.style.display = 'none';
    checkmark.style.display = 'none';
    scanInstruction.style.display = 'block';
    scanInstruction.textContent = "Preparando câmera...";
    scanWidget.style.background = "rgba(30, 41, 59, 0.5)";
    scanWidget.classList.remove('capture-success');

    try {
      // Conectar ao WebSocket com o session_id
      socket.emit("join", { session_id: captureSessionId });

      // Iniciar o processo de captura no back-end
      const response = await fetch("http://localhost:5000/start_capture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `${nome}_${sobrenome}`,
          session_id: captureSessionId
        })
      });

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || "Erro ao iniciar captura");
      }
      
      scanInstruction.textContent = "Siga as instruções...";
    } catch (error) {
      isScanning = false;
      scanInstruction.textContent = "Erro na conexão. Clique para tentar";
      console.error("Erro na captura biométrica:", error);
      showFeedback("error", "Falha ao iniciar captura biométrica");
    }
  });

  // ====================== FUNÇÕES AUXILIARES ======================
  function toggleUsernameField() {
    if (!tipoUsuarioSelect) return;

    const isProfessor = tipoUsuarioSelect.value === "2";

    setTimeout(() => {
      if (usernameGroup)
        usernameGroup.style.display = isProfessor ? "block" : "none";
      if (senhaGroup) senhaGroup.style.display = isProfessor ? "block" : "none";

      if (senhaInput) senhaInput.required = isProfessor;
      if (usernameInput) usernameInput.required = isProfessor;

      if (!isProfessor) {
        if (usernameInput) usernameInput.value = "";
        if (senhaInput) senhaInput.value = "";
      }
    }, 300);
  }

  function validarCampo(input) {
    const parent = input.parentElement;
    if (!parent) return;

    if (input.checkValidity()) {
      parent.classList.add("valid");
      parent.classList.remove("invalid");
    } else {
      parent.classList.add("invalid");
      parent.classList.remove("valid");
    }
  }

  function showFeedback(tipo, mensagem) {
    if (!feedback) return;

    feedback.textContent = mensagem;
    feedback.className = "feedback " + tipo;
    feedback.style.display = "block";

    setTimeout(() => {
      feedback.style.display = "none";
    }, 5000);
  }

  // ====================== CONFIGURAÇÃO INICIAL ======================
  if (tipoUsuarioSelect) {
    tipoUsuarioSelect.addEventListener("change", toggleUsernameField);
    toggleUsernameField();
  }

  document.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", () => validarCampo(input));
  });

  // ====================== SUBMIT DO FORMULÁRIO ======================
  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    
    // Validar se a captura biométrica foi realizada com sucesso
    if (!faceCaptureComplete) {
      showFeedback("error", "Realize a captura biométrica antes de cadastrar");
      return;
    }
    
    if (!faceCaptureSuccess) {
      showFeedback("error", "Captura biométrica não concluída com sucesso");
      return;
    }

    // Validação dos campos
    const tipoUsuario = tipoUsuarioSelect ? tipoUsuarioSelect.value : "";
    const campos = ["nome", "sobrenome", "turma"];

    if (tipoUsuario === "2") {
      campos.push("username", "senha");
    }

    const invalidos = [];
    let isValid = true;

    campos.forEach((id) => {
      const campo = document.getElementById(id);
      if (!campo) return;

      const erroMensagem = campo.parentElement.querySelector(".error-message");

      if (!campo.value.trim()) {
        invalidos.push(id);
        campo.classList.add("input-error");
        isValid = false;

        if (!erroMensagem) {
          const errorElement = document.createElement("div");
          errorElement.classList.add("error-message");
          errorElement.textContent = "Este campo é obrigatório.";
          campo.parentElement.appendChild(errorElement);
        }
      } else {
        campo.classList.remove("input-error");
        if (erroMensagem) {
          erroMensagem.remove();
        }
      }
    });

    if (!tipoUsuario) {
      showFeedback("error", "Selecione o tipo de usuário");
      return;
    }

    if (!isValid) {
      const firstInvalid = document.getElementById(invalidos[0]);
      if (firstInvalid) firstInvalid.focus();
      showFeedback(
        "error",
        "Por favor, preencha todos os campos obrigatórios."
      );
      return;
    }

    const formData = {
      nome: document.getElementById("nome").value.trim(),
      sobrenome: document.getElementById("sobrenome").value.trim(),
      turma: document.getElementById("turma").value.trim(),
      tipoUsuario: tipoUsuario === "1" ? "ALUNO" : "PROFESSOR",
      username:
        tipoUsuario === "2"
          ? document.getElementById("username").value.trim()
          : null,
      senha:
        tipoUsuario === "2" ? document.getElementById("senha").value : null,
    };

    try {
      console.log("Enviando dados para o backend:", formData);

      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        let errorMessage = `Erro HTTP! Status: ${response.status}`;
        try {
          const errorData = await response.json();
          if (errorData.message) {
            errorMessage += ` - ${errorData.message}`;
          }
        } catch (e) {}
        throw new Error(errorMessage);
      }

      const responseData = await response.json();
      console.log("Resposta completa:", {
        status: response.status,
        data: responseData,
      });

      if (response.status === 201 && responseData.id) {
        showFeedback("success", "Cadastro realizado com sucesso!");
        form.reset();
        toggleUsernameField();

        // Resetar estado da captura biométrica
        faceCaptureComplete = false;
        faceCaptureSuccess = false;
        scanInstruction.textContent = "Clique para captura biométrica";
        scanWidget.style.background = "";
        scanWidget.classList.remove('capture-success');

        setTimeout(() => {
          const firstInput = form.querySelector("input");
          if (firstInput) firstInput.focus();
        }, 100);
      } else {
        const errorMsg =
          responseData.message ||
          "Cadastro aparentemente realizado, mas sem confirmação do servidor";
        showFeedback("warning", errorMsg);
      }
    } catch (error) {
      console.error("Erro completo na requisição:", error);

      let errorMessage;
      if (
        error.message.includes("Failed to fetch") ||
        error.message.includes("ERR_CONNECTION_REFUSED")
      ) {
        errorMessage =
          "Servidor offline! Verifique se o backend está rodando na porta 8080.";
      } else if (error.message.includes("PropertyValueException")) {
        errorMessage =
          "Erro de validação: Campo obrigatório não preenchido no servidor";
      } else if (error.message.includes("Erro HTTP")) {
        errorMessage = `Erro no servidor: ${error.message}`;
      } else {
        errorMessage = error.message || "Erro ao processar o cadastro";
      }

      showFeedback("error", errorMessage);

      console.error("Detalhes do erro:", {
        name: error.name,
        message: error.message,
        stack: error.stack,
      });
    }
  });
});