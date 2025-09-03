async function initAuth(type) {
  const loading = document.querySelector(".loading-overlay");
  loading.style.display = "grid";

  try {
    await new Promise((resolve) => setTimeout(resolve, 2000));

    if (type === "login") {
      await handleLogin();
    } else {
      await handleRegistration();
    }
  } catch (error) {
    showError(error.message);
  } finally {
    loading.style.display = "none";
  }
}

async function handleLogin() {
  await new Promise((resolve) => setTimeout(resolve, 1500));
  showFeedback("success", "Login realizado com sucesso!");
}

async function handleRegistration() {
  await new Promise((resolve) => setTimeout(resolve, 2500));
  showFeedback("success", "Perfil cadastrado com sucesso!");
}

function showFeedback(type, message) {
  const feedback = document.createElement("div");
  feedback.className = `feedback ${type}`;
  feedback.textContent = message;
  document.body.appendChild(feedback);
  setTimeout(() => feedback.remove(), 3000);
}

function showError(message) {
  showFeedback("error", `Erro: ${message}`);
}
