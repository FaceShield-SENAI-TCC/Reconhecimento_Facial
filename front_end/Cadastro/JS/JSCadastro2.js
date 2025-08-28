document.addEventListener("DOMContentLoaded", () => {
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
  const nomeInput = document.getElementById("nome");
  const sobrenomeInput = document.getElementById("sobrenome");
  const turmaInput = document.getElementById("turma");

  const API_URL = "http://localhost:8080/usuarios/novoUsuario";
  const CAPTURE_API_URL = "http://localhost:7001";

  let isScanning = false;
  let isViewingCamera = false; // Novo estado para controle da visualização da câmera
  let captureSessionId = null;
  let faceCaptureComplete = false;
  let faceCaptureSuccess = false;

  // Adicionar botão para visualizar câmera
  const viewCameraBtn = document.createElement('button');
  viewCameraBtn.textContent = 'Visualizar Câmera';
  viewCameraBtn.style.marginTop = '10px';
  viewCameraBtn.style.padding = '8px 16px';
  viewCameraBtn.style.background = 'rgba(0, 224, 255, 0.3)';
  viewCameraBtn.style.color = 'white';
  viewCameraBtn.style.border = '1px solid rgba(0, 224, 255, 0.5)';
  viewCameraBtn.style.borderRadius = '4px';
  viewCameraBtn.style.cursor = 'pointer';
  scanWidget.parentNode.insertBefore(viewCameraBtn, scanWidget.nextSibling);

  const cameraFeed = document.createElement('img');
  cameraFeed.id = 'camera-feed';
  cameraFeed.style.display = 'none';
  cameraFeed.style.width = '100%';
  cameraFeed.style.height = '100%';
  cameraFeed.style.objectFit = 'cover';
  cameraFeed.style.borderRadius = '8px';
  scanWidget.appendChild(cameraFeed);

  const checkmark = document.createElement('div');
  checkmark.innerHTML = '&#10004;';
  checkmark.style.position = 'absolute';
  checkmark.style.top = '50%';
  checkmark.style.left = '50%';
  checkmark.style.transform = 'translate(-50%, -50%) scale(0)';
  checkmark.style.fontSize = '0rem';
  checkmark.style.color = '#00ffaa';
  checkmark.style.textShadow = '0 0 20px rgba(0, 255, 170, 0.8)';
  checkmark.style.zIndex = '10';
  checkmark.style.opacity = '0';
  checkmark.style.transition = 'all 0.8s ease-out';
  scanWidget.appendChild(checkmark);

  const socket = io(CAPTURE_API_URL, {
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 3000,
    autoConnect: false
  });

  socket.on("connect", () => {
    console.log("Conectado ao servidor de captura facial.");
    
    if (isViewingCamera) {
      // Modo apenas visualização - não inicia captura
      scanInstruction.textContent = "Visualizando câmera...";
      return;
    }
    
    // Modo captura biométrica
    if (!nomeInput.value.trim() || !sobrenomeInput.value.trim() || !turmaInput.value.trim()) {
      showFeedback("error", "Preencha nome, sobrenome e turma antes da captura biométrica");
      isScanning = false;
      socket.disconnect();
      return;
    }

    socket.emit('start_camera', {
        nome: nomeInput.value.trim(),
        sobrenome: sobrenomeInput.value.trim(),
        turma: turmaInput.value.trim(),
        session_id: captureSessionId
    });

    scanInstruction.textContent = "Siga as instruções...";
  });

  socket.on("connect_error", (err) => {
    console.error(`Erro de conexão com Socket.IO: ${err.message}`);
    showFeedback("error", "Não foi possível conectar ao servidor de captura. Verifique se o back-end está rodando na porta 7001.");
    isScanning = false;
    isViewingCamera = false;
    scanInstruction.textContent = "Falha na conexão. Tente novamente.";
    scanWidget.style.background = "rgba(255, 77, 125, 0.2)";
  });

  socket.on("capture_progress", (data) => {
    // Só mostra progresso se estiver no modo de captura, não apenas visualização
    if (!isViewingCamera) {
      const progress = Math.min(100, Math.round((data.captured / data.total) * 100));
      scanInstruction.textContent = `Capturando... ${progress}%`;
     
      scanWidget.style.background = `linear-gradient(
          to right,
          rgba(0, 224, 255, 0.5) ${progress}%,
          rgba(30, 41, 59, 0.3) ${progress}%
      )`;
    }
  });

  socket.on('capture_frame', (data) => {
    cameraFeed.src = `data:image/jpeg;base64,${data.frame}`;
    cameraFeed.style.display = 'block';
    scanInstruction.style.display = 'none';
    checkmark.style.display = 'none';
  });

  socket.on("capture_complete", (data) => {
    isScanning = false;
    faceCaptureComplete = true;

    if (data.success) {
      faceCaptureSuccess = true;
      cameraFeed.style.display = 'none';
      scanInstruction.style.display = 'none';
     
      checkmark.style.display = 'block';
      checkmark.style.opacity = '1';
      checkmark.style.fontSize = '5rem';
      checkmark.style.transform = 'translate(-50%, -50%) scale(1)';

      setTimeout(() => {
        checkmark.style.opacity = '0';
        checkmark.style.transform = 'translate(-50%, -50%) scale(0.5)';
        scanInstruction.style.display = 'block';
        scanInstruction.textContent = "Biometria capturada!";
        scanWidget.style.background = "rgba(0, 255, 170, 0.1)";
        scanWidget.classList.add('capture-success');
      }, 2500);

      setTimeout(() => {
        checkmark.style.display = 'none';
      }, 3300);
     
    } else {
      faceCaptureSuccess = false;
      cameraFeed.style.display = 'none';
      checkmark.style.display = 'none';
      scanInstruction.style.display = 'block';
      scanInstruction.textContent = "Falha na captura. Clique para tentar novamente";
      scanWidget.style.background = "rgba(255, 77, 125, 0.2)";
      showFeedback("error", `Erro na captura biométrica: ${data.message}`);
    }
    console.log("Captura completa. Resultado: ", data.success ? "Sucesso" : "Falha", "Mensagem:", data.message);
    socket.disconnect();
  });

  // Função para visualizar a câmera sem iniciar captura
  viewCameraBtn.addEventListener("click", () => {
    if (isViewingCamera) {
      // Se já está visualizando, para a visualização
      socket.disconnect();
      isViewingCamera = false;
      viewCameraBtn.textContent = 'Visualizar Câmera';
      cameraFeed.style.display = 'none';
      scanInstruction.style.display = 'block';
      scanInstruction.textContent = "Clique para captura biométrica";
      return;
    }

    isViewingCamera = true;
    captureSessionId = Date.now().toString() + "_view";

    cameraFeed.src = '';
    cameraFeed.style.display = 'none';
    checkmark.style.display = 'none';
    checkmark.style.opacity = '0';
    checkmark.style.fontSize = '0rem';
    checkmark.style.transform = 'translate(-50%, -50%) scale(0)';
   
    scanInstruction.style.display = 'block';
    scanInstruction.textContent = "Preparando câmera...";
    scanWidget.style.background = "rgba(30, 41, 59, 0.5)";
    scanWidget.classList.remove('capture-success');

    socket.io.opts.query = { session_id: captureSessionId };
    socket.connect();
    viewCameraBtn.textContent = 'Parar Visualização';
  });

  scanWidget.addEventListener("click", () => {
    if (isScanning || isViewingCamera) return;

    if (!nomeInput.value.trim() || !sobrenomeInput.value.trim() || !turmaInput.value.trim()) {
      showFeedback("error", "Preencha nome, sobrenome e turma antes da captura biométrica");
      return;
    }

    isScanning = true;
    isViewingCamera = false; // Garante que não está no modo visualização
    faceCaptureComplete = false;
    faceCaptureSuccess = false;
    captureSessionId = Date.now().toString();

    cameraFeed.src = '';
    cameraFeed.style.display = 'none';
    checkmark.style.display = 'none';
    checkmark.style.opacity = '0';
    checkmark.style.fontSize = '0rem';
    checkmark.style.transform = 'translate(-50%, -50%) scale(0)';
   
    scanInstruction.style.display = 'block';
    scanInstruction.textContent = "Preparando câmera...";
    scanWidget.style.background = "rgba(30, 41, 59, 0.5)";
    scanWidget.classList.remove('capture-success');

    socket.io.opts.query = { session_id: captureSessionId };
    socket.connect();
    console.log("Iniciando conexão com Socket.IO...");
  });

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

  if (tipoUsuarioSelect) {
    tipoUsuarioSelect.addEventListener("change", toggleUsernameField);
    toggleUsernameField();
  }

  document.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", () => validarCampo(input));
  });

  form.addEventListener("submit", async function (event) {
    event.preventDefault();

    if (!faceCaptureComplete) {
      showFeedback("error", "Realize a captura biométrica antes de cadastrar");
      return;
    }

    if (!faceCaptureSuccess) {
      showFeedback("error", "Captura biométrica não concluída com sucesso");
      return;
    }

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
      if (!campo.value.trim()) {
        invalidos.push(id);
        isValid = false;
      }
    });

    if (!tipoUsuario) {
      showFeedback("error", "Selecione o tipo de usuário");
      return;
    }

    if (!isValid) {
      const firstInvalid = document.getElementById(invalidos[0]);
      if (firstInvalid) firstInvalid.focus();
      showFeedback("error", "Por favor, preencha todos os campos obrigatórios.");
      return;
    }

    const formData = {
      nome: nomeInput.value.trim(),
      sobrenome: sobrenomeInput.value.trim(),
      turma: turmaInput.value.trim(),
      tipoUsuario: tipoUsuario === "1" ? "ALUNO" : "PROFESSOR",
      username: tipoUsuario === "2" ? usernameInput.value.trim() : null,
      senha: tipoUsuario === "2" ? senhaInput.value : null,
    };

    try {
      showFeedback("info", "Enviando dados de cadastro...");
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        let errorMessage = `Erro HTTP! Status: ${response.status}`;
        try {
          const errorData = await response.json();
          if (errorData.message) { errorMessage += ` - ${errorData.message}`; }
        } catch (e) {}
        throw new Error(errorMessage);
      }

      const responseData = await response.json();

      if (response.status === 201 && responseData.id) {
        showFeedback("success", "Cadastro realizado com sucesso!");
        form.reset();
        toggleUsernameField();
        
        faceCaptureComplete = false;
        faceCaptureSuccess = false;
        scanInstruction.textContent = "Clique para captura biométrica";
        scanWidget.style.background = "";
        scanWidget.classList.remove('capture-success');

        cameraFeed.style.display = 'none';
        checkmark.style.display = 'none';
        checkmark.style.opacity = '0';
        checkmark.style.fontSize = '0rem';
        checkmark.style.transform = 'translate(-50%, -50%) scale(0)';

        setTimeout(() => {
          const firstInput = form.querySelector("input");
          if (firstInput) firstInput.focus();
        }, 100);
      } else {
        const errorMsg = responseData.message || "Cadastro aparentemente realizado, mas sem confirmação do servidor";
        showFeedback("warning", errorMsg);
      }
    } catch (error) {
      console.error("Erro completo na requisição:", error);
      let errorMessage;
      if (error.message.includes("Failed to fetch") || error.message.includes("ERR_CONNECTION_REFUSED")) {
        errorMessage = "Servidor de cadastro offline! Verifique se o backend está rodando na porta 8080.";
      } else if (error.message.includes("PropertyValueException")) {
        errorMessage = "Erro de validação: Campo obrigatório não preenchido no servidor";
      } else if (error.message.includes("Erro HTTP") && !error.message.includes("400")) {
        errorMessage = `Erro no servidor: ${error.message}`;
      } else {
        errorMessage = error.message || "Erro ao processar o cadastro";
      }
      showFeedback("error", errorMessage);
    }
  });
});