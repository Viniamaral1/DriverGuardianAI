const byId = id => document.getElementById(id);

let cameraStreamActive = false;
let startInProgress = false;
let commanderRecognition = null;
let commanderEnabled = false;
let commanderProcessing = false;
let waitingForCommand = false;
let commandSilenceTimer = null;
let recognitionRestartTimer = null;

function activeLifecycle(metrics) {
  return Boolean(metrics.monitoring || metrics.starting || metrics.stopping);
}

function setCameraStream(active) {
  const image = byId("live-camera-feed");
  const placeholder = document.querySelector(".camera-placeholder");

  if (active && !cameraStreamActive) {
    cameraStreamActive = true;
    image.src = `/api/camera/stream?ts=${Date.now()}`;
  }

  if (!active && cameraStreamActive) {
    cameraStreamActive = false;
    image.removeAttribute("src");
  }

  image.classList.toggle("visible", active);
  placeholder?.classList.toggle("hidden-camera-placeholder", active);
}

function renderCalibration(metrics) {
  const overlay = byId("calibration-overlay");
  const calibrating =
    Boolean(metrics.monitoring) && !Boolean(metrics.calibration_complete);

  overlay.classList.toggle("visible", calibrating);
  if (!calibrating) return;

  const remaining = Number(metrics.calibration_remaining || 0);
  const progress = Math.max(0, Math.min(1, (10 - remaining) / 10));

  byId("calibration-seconds").textContent = remaining.toFixed(1);
  byId("calibration-ring").style.setProperty(
    "--calibration",
    `${progress * 360}deg`
  );
  byId("calibration-message").textContent = metrics.face_detected
    ? "Hold a natural expression while Guardian OS learns your baseline."
    : "Move into view so calibration can continue.";
}

function renderState(metrics) {
  const lifecycle = activeLifecycle(metrics);
  const risk = Number(metrics.fatigue_probability || 0);

  byId("camera-status").textContent = metrics.camera_status || "STANDBY";
  byId("driver-state").textContent = metrics.state || "READY";
  byId("camera-message").textContent =
    metrics.error ||
    (metrics.starting
      ? "Connecting to the camera…"
      : metrics.stopping
        ? "Releasing the camera…"
        : metrics.monitoring
          ? metrics.calibration_complete
            ? "Live V3 pipeline active"
            : `Personal calibration: ${Number(
                metrics.calibration_remaining || 0
              ).toFixed(1)}s remaining`
          : "Camera standing by");

  byId("camera-fps").textContent = Number(metrics.fps || 0).toFixed(1);
  byId("face-status").textContent =
    metrics.face_detected ? "FACE LOCK" : "NO FACE";

  byId("camera-stage").classList.toggle("active", Boolean(metrics.monitoring));
  setCameraStream(
    Boolean(metrics.monitoring) &&
    metrics.camera_status === "CONNECTED"
  );
  renderCalibration(metrics);

  byId("risk-value").textContent = `${Math.round(risk * 100)}%`;
  byId("risk-gauge").style.setProperty("--risk", `${risk * 360}deg`);
  byId("state-value").textContent = metrics.state || "READY";
  byId("state-banner").className =
    `state-banner ${String(metrics.state || "ready")
      .toLowerCase()
      .replaceAll(" ", "-")}`;

  byId("ear-value").textContent = Number(metrics.ear || 0).toFixed(3);
  byId("blink-value").textContent =
    Number(metrics.blink_rate || 0).toFixed(1);
  byId("yawn-value").textContent =
    Number(metrics.yawn_score || 0).toFixed(3);
  byId("tilt-value").textContent =
    `${Number(metrics.head_tilt || 0).toFixed(1)}°`;

  byId("start-monitoring").disabled = lifecycle;
  byId("stop-monitoring").disabled =
    !lifecycle || Boolean(metrics.stopping);
}

async function startCamera() {
  if (startInProgress) return;

  startInProgress = true;
  byId("start-monitoring").disabled = true;
  byId("stop-monitoring").disabled = false;
  byId("camera-message").textContent = "Connecting to the camera…";
  DG.toast("Connecting to the real V3 camera pipeline…");

  try {
    const state = await DG.post("/api/monitoring/start");
    DG.emit(state);
    DG.toast(state.message || "Live monitoring started", "success");
  } catch (error) {
    let message = "Monitoring could not start";

    try {
      const payload = JSON.parse(error.message);
      message = payload.detail || payload.message || message;
    } catch (_) {
      if (error.message) message = error.message;
    }

    byId("camera-message").textContent = message;
    byId("start-monitoring").disabled = false;
    byId("stop-monitoring").disabled = true;
    DG.toast(message, "error");
    console.error("Camera start failed:", error);
  } finally {
    startInProgress = false;
  }
}

async function stopCamera() {
  byId("stop-monitoring").disabled = true;
  DG.stopSpeech();

  try {
    const state = await DG.post("/api/monitoring/stop");
    setCameraStream(false);
    DG.emit(state);
    DG.toast(state.message || "Monitoring stopped", "success");
  } catch (error) {
    byId("stop-monitoring").disabled = false;
    DG.toast("Monitoring could not stop", "error");
    console.error("Camera stop failed:", error);
  }
}


DG.subscribe(({metrics}) => {
  renderState(metrics);
});

byId("start-monitoring").addEventListener("click", startCamera);

byId("stop-monitoring").addEventListener("click", stopCamera);

function setCommanderState(state, title, detail) {
  const panel = byId("commander-listening-state");
  panel.dataset.state = state;
  byId("commander-state-title").textContent = title;
  byId("commander-state-detail").textContent = detail;
}

function createCommanderRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return null;

  const recognition = new Recognition();
  recognition.lang = "en-GB";
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  return recognition;
}

function stopRecognition() {
  window.clearTimeout(commandSilenceTimer);
  window.clearTimeout(recognitionRestartTimer);
  try { commanderRecognition?.stop(); } catch (_) {}
}

function restartRecognition(delay = 350) {
  if (!commanderEnabled || commanderProcessing) return;
  window.clearTimeout(recognitionRestartTimer);
  recognitionRestartTimer = window.setTimeout(() => {
    if (!commanderEnabled || commanderProcessing) return;
    try { commanderRecognition.start(); } catch (_) {
      restartRecognition(500);
    }
  }, delay);
}

function extractWakeCommand(transcript) {
  const match = String(transcript).trim().match(/\bcommander\b[\s,:-]*(.*)$/i);
  return match ? match[1].trim() : null;
}

async function commanderSpeak(text, options = {}) {
  stopRecognition();
  setCommanderState("speaking", "Commander speaking", text);
  await DG.speak(text, options);
}

async function processCommanderCommand(command) {
  const clean = String(command || "").trim();
  if (!clean || commanderProcessing) return;

  commanderProcessing = true;
  waitingForCommand = false;
  stopRecognition();
  setCommanderState("processing", "Processing command", clean);

  try {
    const result = await DG.post("/api/commander/message", {message: clean});
    await commanderSpeak(result.response);
  } catch (error) {
    DG.toast("Commander could not process the command", "error");
    await commanderSpeak("I could not process that command.");
    console.error(error);
  } finally {
    commanderProcessing = false;
    if (commanderEnabled) {
      setCommanderState(
        "listening",
        "Listening for ‘Commander’",
        "Say Commander followed by a command."
      );
      restartRecognition(500);
    }
  }
}

function startCommandSilenceTimer() {
  window.clearTimeout(commandSilenceTimer);
  commandSilenceTimer = window.setTimeout(() => {
    waitingForCommand = false;
    setCommanderState(
      "listening",
      "Listening for ‘Commander’",
      "No command heard. Say Commander to try again."
    );
  }, 5000);
}

function configureCommanderRecognition() {
  commanderRecognition = createCommanderRecognition();

  if (!commanderRecognition) {
    byId("commander-voice-toggle").disabled = true;
    setCommanderState(
      "unsupported",
      "Voice recognition unavailable",
      "Use Chrome or Edge for browser voice commands."
    );
    return;
  }

  commanderRecognition.onstart = () => {
    if (!commanderEnabled || commanderProcessing) return;
    setCommanderState(
      "listening",
      waitingForCommand ? "Listening for your command" : "Listening for ‘Commander’",
      waitingForCommand ? "Speak your request now." : "Hands-free voice command is active."
    );
  };

  commanderRecognition.onresult = event => {
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const transcript = event.results[index][0].transcript.trim();
      const final = event.results[index].isFinal;

      if (waitingForCommand) {
        setCommanderState("listening", "Command heard", transcript);
        if (final && transcript) {
          window.clearTimeout(commandSilenceTimer);
          processCommanderCommand(transcript);
        }
        continue;
      }

      const command = extractWakeCommand(transcript);
      if (command === null) continue;

      if (command) {
        setCommanderState("listening", "Wake word detected", command);
        if (final) processCommanderCommand(command);
        continue;
      }

      if (final) {
        waitingForCommand = true;
        stopRecognition();
        setCommanderState("speaking", "Commander heard you", "Yes?");
        DG.speak("Yes?", {rate: 1.05}).then(() => {
          if (!commanderEnabled) return;
          setCommanderState("listening", "Listening for your command", "Speak within five seconds.");
          restartRecognition(250);
          startCommandSilenceTimer();
        });
      }
    }
  };

  commanderRecognition.onerror = event => {
    if (!["no-speech", "aborted"].includes(event.error)) {
      DG.toast(`Commander microphone: ${event.error}`, "error");
    }
  };

  commanderRecognition.onend = () => {
    if (commanderEnabled && !commanderProcessing) restartRecognition(400);
  };
}

async function enableCommander() {
  if (!commanderRecognition) return;
  commanderEnabled = true;
  waitingForCommand = false;
  byId("commander-voice-toggle").textContent = "Turn Commander off";
  byId("commander-voice-toggle").classList.remove("primary");
  byId("commander-voice-toggle").classList.add("secondary");

  setCommanderState("speaking", "Voice commands on", "Commander is preparing to listen.");
  await DG.speak("Voice commands on.");

  if (commanderEnabled) {
    setCommanderState("listening", "Listening for ‘Commander’", "Hands-free voice command is active.");
    restartRecognition(250);
  }
}

async function disableCommander() {
  commanderEnabled = false;
  waitingForCommand = false;
  commanderProcessing = false;
  stopRecognition();
  byId("commander-voice-toggle").textContent = "Turn Commander on";
  byId("commander-voice-toggle").classList.remove("secondary");
  byId("commander-voice-toggle").classList.add("primary");
  setCommanderState("off", "Voice commands off", "Commander is not listening.");
  await DG.speak("Voice commands off.");
}

byId("commander-voice-toggle").addEventListener("click", async () => {
  if (commanderEnabled) await disableCommander();
  else await enableCommander();
});

byId("stop-speaking").addEventListener("click", () => {
  DG.stopSpeech();
  commanderProcessing = false;
  if (commanderEnabled) {
    setCommanderState("listening", "Listening for ‘Commander’", "Commander speech stopped.");
    restartRecognition(250);
  } else {
    setCommanderState("off", "Voice commands off", "Commander speech stopped.");
  }
  DG.toast("Commander speech stopped");
});

configureCommanderRecognition();
