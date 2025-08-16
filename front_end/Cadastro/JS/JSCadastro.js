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
  const senhaGroup = document.getElementById("senha-group");
  const senhaInput = document.getElementById("senha");
  const turmaInput = document.getElementById("turma");
  let isScanning = false;

  const API_URL = "http://localhost:8080/usuarios/novoUsuario";

  // Função para mostrar/ocultar campos de professor
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

  // Event listeners
  if (tipoUsuarioSelect) {
    tipoUsuarioSelect.addEventListener("change", toggleUsernameField);
    toggleUsernameField(); // Chamada inicial para configurar campos
  }

  // Validação em tempo real
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

  document.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", () => validarCampo(input));
  });

  // Exibir mensagens de feedback
  function showFeedback(tipo, mensagem) {
    if (!feedback) return;

    feedback.textContent = mensagem;
    feedback.className = "feedback " + tipo;
    feedback.style.display = "block";

    setTimeout(() => {
      feedback.style.display = "none";
    }, 5000);
  }

  // Submit do formulário
  form.addEventListener("submit", async function (event) {
    event.preventDefault();

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

      // Verificação inicial da resposta
      if (!response.ok) {
        // Tentar obter a mensagem de erro do servidor
        let errorMessage = `Erro HTTP! Status: ${response.status}`;
        try {
          const errorData = await response.json();
          if (errorData.message) {
            errorMessage += ` - ${errorData.message}`;
          }
        } catch (e) {}
        throw new Error(errorMessage);
      }

      // Tentar parsear a resposta como JSON
      const responseData = await response.json();
      console.log("Resposta completa:", {
        status: response.status,
        data: responseData,
      });

      // Verificação robusta de sucesso
      if (response.status === 201 && responseData.id) {
        showFeedback("success", "Cadastro realizado com sucesso!");
        form.reset();
        toggleUsernameField();

        // Foco no primeiro campo após sucesso
        setTimeout(() => {
          const firstInput = form.querySelector("input");
          if (firstInput) firstInput.focus();
        }, 100);
      } else {
        // Tratar casos onde o status é 201 mas sem confirmação real
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

      // Log detalhado para debug
      console.error("Detalhes do erro:", {
        name: error.name,
        message: error.message,
        stack: error.stack,
      });
    }
  });
});
