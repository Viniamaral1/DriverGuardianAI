const el = id => document.getElementById(id);

let lastSignature = "";
let oneShotRecognition = null;
let continuousRecognition = null;
let continuousEnabled = false;
let restartTimer = null;
let silenceTimer = null;
let pendingWakeCommand = "";
let processingVoice = false;
let selectedVoice = null;

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

function setVoiceState(state, title, detail = "") {
  el("listening-strip").dataset.state = state;
  el("listening-strip").classList.toggle("active", state === "listening");
  el("listening-strip").classList.toggle("processing", state === "processing");
  el("listening-strip").classList.toggle("speaking", state === "speaking");
  el("listening-title").textContent = title;
  el("heard-text").textContent = detail || "Say “Commander” followed by your command";
}

function loadVoices() {
  if (!("speechSynthesis" in window)) return;
  const voices = speechSynthesis.getVoices();
  selectedVoice =
    voices.find(voice => /en-GB/i.test(voice.lang) && /male|daniel|george/i.test(voice.name)) ||
    voices.find(voice => /en-GB/i.test(voice.lang)) ||
    voices.find(voice => /^en/i.test(voice.lang)) ||
    voices[0] ||
    null;
}

if ("speechSynthesis" in window) {
  loadVoices();
  speechSynthesis.onvoiceschanged = loadVoices;
}

function speakResponse(text) {
  return new Promise(resolve => {
    const enabled = DG.state?.settings?.voice_output !== false;
    if (!enabled || !("speechSynthesis" in window)) {
      resolve();
      return;
    }

    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 0.95;
    utterance.volume = 1.0;
    if (selectedVoice) utterance.voice = selectedVoice;

    utterance.onstart = () => setVoiceState("speaking", "Commander is speaking", text);
    utterance.onend = () => resolve();
    utterance.onerror = () => resolve();

    // A tiny delay improves reliability in Chromium after recognition stops.
    window.setTimeout(() => speechSynthesis.speak(utterance), 120);
  });
}

async function send(message, {fromVoice = false} = {}) {
  const clean = message.trim();
  if (!clean || processingVoice) return;

  processingVoice = fromVoice;
  el("commander-input").value = "";
  setVoiceState("processing", "Commander is processing", clean);

  try {
    const result = await DG.post("/api/commander/message", {message: clean});
    renderConversation(result.conversation);
    await speakResponse(result.response);
  } catch (error) {
    DG.toast("Commander could not process the request", "error");
    console.error(error);
  } finally {
    processingVoice = false;
    if (continuousEnabled) {
      window.setTimeout(restartContinuousRecognition, 650);
    } else {
      setVoiceState("idle", "Commander is standing by");
    }
  }
}

function extractWakeCommand(transcript) {
  const match = transcript.trim().match(/\bcommander\b[\s,:-]*(.*)$/i);
  return match ? match[1].trim() : null;
}

function createRecognition({continuous = false, interim = false} = {}) {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return null;

  const recognition = new Recognition();
  recognition.lang = "en-GB";
  recognition.continuous = continuous;
  recognition.interimResults = interim;
  recognition.maxAlternatives = 1;
  return recognition;
}

function stopContinuousForProcessing() {
  window.clearTimeout(silenceTimer);
  try {
    continuousRecognition?.stop();
  } catch (_) {}
}

function scheduleSilenceProcessing() {
  window.clearTimeout(silenceTimer);
  silenceTimer = window.setTimeout(() => {
    const command = pendingWakeCommand.trim();
    pendingWakeCommand = "";
    stopContinuousForProcessing();

    if (command) {
      send(command, {fromVoice: true});
    } else {
      setVoiceState("idle", "No command received", "Listening paused after 5 seconds of silence");
      if (continuousEnabled) window.setTimeout(restartContinuousRecognition, 900);
    }
  }, 5000);
}

function restartContinuousRecognition() {
  if (!continuousEnabled || processingVoice) return;
  pendingWakeCommand = "";
  try {
    continuousRecognition.start();
  } catch (_) {
    window.setTimeout(restartContinuousRecognition, 500);
  }
}

function configureOneShot() {
  oneShotRecognition = createRecognition();
  if (!oneShotRecognition) {
    el("browser-mic").disabled = true;
    return;
  }

  oneShotRecognition.onstart = () => {
    el("browser-mic").classList.add("active");
    setVoiceState("listening", "Listening for one command");
  };

  oneShotRecognition.onresult = event => {
    const transcript = event.results[event.results.length - 1][0].transcript;
    el("heard-text").textContent = transcript;
    send(transcript, {fromVoice: true});
  };

  oneShotRecognition.onerror = event => {
    if (event.error !== "aborted") DG.toast(`Microphone: ${event.error}`, "error");
  };

  oneShotRecognition.onend = () => {
    el("browser-mic").classList.remove("active");
    if (!processingVoice && !continuousEnabled) setVoiceState("idle", "Commander is standing by");
  };

  el("browser-mic").addEventListener("click", () => {
    try { oneShotRecognition.start(); } catch (_) {}
  });
}

function configureContinuous() {
  continuousRecognition = createRecognition({continuous: true, interim: true});

  if (!continuousRecognition) {
    el("handsfree-start").disabled = true;
    el("speech-support-note").textContent =
      "Continuous browser recognition is unavailable. Use Chrome or Edge, or the backend listener.";
    return;
  }

  continuousRecognition.onstart = () => {
    if (processingVoice) return;
    el("handsfree-badge").textContent = "LISTENING";
    el("handsfree-badge").classList.add("success");
    setVoiceState("listening", "Listening for “Commander”");
  };

  continuousRecognition.onresult = event => {
    let latestTranscript = "";

    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      latestTranscript = event.results[index][0].transcript;
      const command = extractWakeCommand(latestTranscript);

      if (command !== null) {
        pendingWakeCommand = command;
        el("heard-text").textContent = latestTranscript;

        if (command) {
          setVoiceState("listening", "Wake word detected", command);
          scheduleSilenceProcessing();
        } else {
          setVoiceState("listening", "Yes?", "Waiting up to 5 seconds for your command");
          scheduleSilenceProcessing();
        }

        if (event.results[index].isFinal && command) {
          window.clearTimeout(silenceTimer);
          stopContinuousForProcessing();
          const finalCommand = pendingWakeCommand;
          pendingWakeCommand = "";
          window.setTimeout(() => send(finalCommand, {fromVoice: true}), 120);
        }
      }
    }
  };

  continuousRecognition.onerror = event => {
    if (!["no-speech", "aborted"].includes(event.error)) {
      DG.toast(`Hands-free microphone: ${event.error}`, "error");
    }
  };

  continuousRecognition.onend = () => {
    if (!continuousEnabled || processingVoice) return;
    window.setTimeout(restartContinuousRecognition, 450);
  };

  el("handsfree-start").addEventListener("click", () => {
    continuousEnabled = true;
    el("handsfree-start").disabled = true;
    el("handsfree-stop").disabled = false;
    restartContinuousRecognition();
  });

  el("handsfree-stop").addEventListener("click", () => {
    continuousEnabled = false;
    pendingWakeCommand = "";
    window.clearTimeout(silenceTimer);
    window.clearTimeout(restartTimer);
    el("handsfree-start").disabled = false;
    el("handsfree-stop").disabled = true;
    el("handsfree-badge").textContent = "OFF";
    el("handsfree-badge").classList.remove("success");
    try { continuousRecognition.stop(); } catch (_) {}
    setVoiceState("idle", "Commander is standing by");
  });
}

DG.subscribe(({conversation, voice}) => {
  renderConversation(conversation);

  const backendStart = el("wake-start");
  backendStart.disabled = !voice.available;
  backendStart.title = voice.available ? "Enable Python host microphone" : (voice.detail || "Backend microphone unavailable");

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
  try {
    const result = await DG.post("/api/commander/voice/start");
    DG.toast(
      result.enabled ? "Backend wake word enabled" : result.detail,
      result.enabled ? "success" : "error"
    );
  } catch (error) {
    DG.toast("Backend listener is unavailable", "error");
    console.error(error);
  }
});

el("wake-stop").addEventListener("click", async () => {
  try {
    await DG.post("/api/commander/voice/stop");
    DG.toast("Backend wake word disabled");
  } catch (error) {
    DG.toast("Backend listener could not be stopped", "error");
    console.error(error);
  }
});

configureOneShot();
configureContinuous();

el("stop-speaking")?.addEventListener("click", () => {
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  setVoiceState("idle", "Commander stopped speaking");
  DG.toast("Commander speech stopped");
});
