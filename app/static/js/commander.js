const el = id => document.getElementById(id);
let lastSignature = "";
let oneShotRecognition = null;
let continuousRecognition = null;
let continuousEnabled = false;
let restartTimer = null;

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = String(value);
  return node.innerHTML;
}

function renderConversation(items) {
  const signature = JSON.stringify(items);
  if (signature === lastSignature) return;
  lastSignature = signature;

  const box = el("conversation");
  box.innerHTML = items.map(message => `
    <article class="message ${message.role}">
      <div>${message.role === "assistant" ? "C" : "You"}</div>
      <section>
        <small>${message.role === "assistant" ? "COMMANDER" : "DRIVER"} · ${message.time}</small>
        <p>${escapeHtml(message.content)}</p>
      </section>
    </article>
  `).join("");

  box.scrollTop = box.scrollHeight;
}

function setListening(active, title, detail = "") {
  el("listening-strip").classList.toggle("active", active);
  el("listening-title").textContent = title;
  el("heard-text").textContent = detail || "Say “Commander” followed by your command";
}

async function send(message) {
  const clean = message.trim();
  if (!clean) return;

  el("commander-input").value = "";
  setListening(true, "Commander is processing", clean);

  try {
    const result = await DG.post("/api/commander/message", {message: clean});
    renderConversation(result.conversation);

    const voice = DG.state?.settings?.voice_output;
    if (voice && "speechSynthesis" in window) {
      speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(result.response);
      utterance.rate = 1.02;
      speechSynthesis.speak(utterance);
    }
  } catch (error) {
    DG.toast("Commander could not process the request", "error");
    console.error(error);
  } finally {
    setListening(continuousEnabled, continuousEnabled ? "Listening for “Commander”" : "Commander is standing by");
  }
}

function extractWakeCommand(transcript) {
  const normal = transcript.trim();
  const match = normal.match(/\bcommander\b[\s,:-]*(.*)$/i);
  return match ? match[1].trim() : null;
}

function createRecognition({continuous = false} = {}) {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return null;

  const recognition = new Recognition();
  recognition.lang = "en-GB";
  recognition.continuous = continuous;
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  return recognition;
}

function configureOneShot() {
  oneShotRecognition = createRecognition();
  if (!oneShotRecognition) {
    el("browser-mic").disabled = true;
    return;
  }

  oneShotRecognition.onstart = () => {
    el("browser-mic").classList.add("active");
    setListening(true, "Listening for one command");
  };

  oneShotRecognition.onresult = event => {
    const transcript = event.results[event.results.length - 1][0].transcript;
    el("heard-text").textContent = transcript;
    send(transcript);
  };

  oneShotRecognition.onerror = event => {
    DG.toast(`Microphone: ${event.error}`, "error");
  };

  oneShotRecognition.onend = () => {
    el("browser-mic").classList.remove("active");
    if (!continuousEnabled) setListening(false, "Commander is standing by");
  };

  el("browser-mic").addEventListener("click", () => {
    try { oneShotRecognition.start(); } catch (_) {}
  });
}

function configureContinuous() {
  continuousRecognition = createRecognition({continuous: true});

  if (!continuousRecognition) {
    el("handsfree-start").disabled = true;
    el("speech-support-note").textContent =
      "Continuous browser recognition is unavailable. Use Chrome or Edge, or the backend listener.";
    return;
  }

  continuousRecognition.onstart = () => {
    el("handsfree-badge").textContent = "LISTENING";
    el("handsfree-badge").classList.add("success");
    setListening(true, "Listening for “Commander”");
  };

  continuousRecognition.onresult = event => {
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      if (!event.results[index].isFinal) continue;
      const transcript = event.results[index][0].transcript;
      el("heard-text").textContent = transcript;
      const command = extractWakeCommand(transcript);
      if (command !== null) {
        if (command) send(command);
        else DG.toast("Commander wake word detected");
      }
    }
  };

  continuousRecognition.onerror = event => {
    if (event.error !== "no-speech" && event.error !== "aborted") {
      DG.toast(`Hands-free microphone: ${event.error}`, "error");
    }
  };

  continuousRecognition.onend = () => {
    if (continuousEnabled) {
      window.clearTimeout(restartTimer);
      restartTimer = window.setTimeout(() => {
        try { continuousRecognition.start(); } catch (_) {}
      }, 350);
    } else {
      el("handsfree-badge").textContent = "OFF";
      el("handsfree-badge").classList.remove("success");
      setListening(false, "Commander is standing by");
    }
  };

  el("handsfree-start").addEventListener("click", () => {
    continuousEnabled = true;
    el("handsfree-start").disabled = true;
    el("handsfree-stop").disabled = false;
    try { continuousRecognition.start(); } catch (_) {}
  });

  el("handsfree-stop").addEventListener("click", () => {
    continuousEnabled = false;
    el("handsfree-start").disabled = false;
    el("handsfree-stop").disabled = true;
    window.clearTimeout(restartTimer);
    try { continuousRecognition.stop(); } catch (_) {}
  });
}

DG.subscribe(({conversation, voice}) => {
  renderConversation(conversation);
  el("wake-status").textContent = voice.enabled
    ? "Backend wake word listening"
    : voice.available
      ? "Backend wake word ready"
      : "Browser voice ready";

  el("wake-detail").textContent = voice.available
    ? (voice.detail || voice.status)
    : "Use hands-free browser mode or typed commands";
});

el("commander-form").addEventListener("submit", event => {
  event.preventDefault();
  send(el("commander-input").value);
});

document.querySelectorAll("[data-command]").forEach(button =>
  button.addEventListener("click", () => send(button.dataset.command))
);

el("clear-chat").addEventListener("click", async () => {
  const result = await DG.post("/api/commander/clear");
  lastSignature = "";
  renderConversation(result.conversation);
});

el("wake-start").addEventListener("click", async () => {
  const result = await DG.post("/api/commander/voice/start");
  DG.toast(result.enabled ? "Backend wake word enabled" : result.detail, result.enabled ? "success" : "error");
});

el("wake-stop").addEventListener("click", async () => {
  await DG.post("/api/commander/voice/stop");
  DG.toast("Backend wake word disabled");
});

configureOneShot();
configureContinuous();
