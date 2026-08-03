const el = id => document.getElementById(id);
const pct = value => `${Math.round(Number(value || 0) * 100)}%`;

let streamActive = false;
let riskHistory = [];
let previousState = "READY";
let lastCriticalAnnouncement = 0;
let recognition = null;
let privacyApproved = sessionStorage.getItem("dg-monitoring-privacy") === "true";

function setStream(active) {
  const image = el("monitor-camera-feed");
  if (active && !streamActive) {
    streamActive = true;
    image.src = `/api/camera/stream?ts=${Date.now()}`;
  }
  if (!active && streamActive) {
    streamActive = false;
    image.removeAttribute("src");
  }
  image.classList.toggle("visible", active);
  el("monitor-placeholder").classList.toggle("hidden-camera-placeholder", active);
}

function drawTrend() {
  const canvas = el("monitor-trend");
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || 900;
  const height = canvas.clientHeight || 190;
  const dpr = window.devicePixelRatio || 1;

  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const styles = getComputedStyle(document.documentElement);
  const line = styles.getPropertyValue("--line").trim();
  const accent = styles.getPropertyValue("--accent").trim();
  const warning = styles.getPropertyValue("--warning").trim();
  const danger = styles.getPropertyValue("--danger").trim();
  const muted = styles.getPropertyValue("--muted").trim();

  ctx.strokeStyle = line;
  [0.25, .45, .65, .82].forEach(value => {
    const y = height - value * height;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  });

  ctx.font = "10px system-ui";
  ctx.fillStyle = muted;
  ctx.fillText("82% critical", 8, height - .82 * height - 5);
  ctx.fillText("65% warning", 8, height - .65 * height - 5);

  if (riskHistory.length < 2) return;
  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, accent);
  gradient.addColorStop(.65, warning);
  gradient.addColorStop(1, danger);
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 2.8;
  ctx.beginPath();
  riskHistory.forEach((value, index) => {
    const x = index * width / Math.max(1, riskHistory.length - 1);
    const y = height - Math.min(1, Math.max(0, value)) * height;
    index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

function showSafetyAlert(level, title, message) {
  el("safety-alert-title").textContent = title;
  el("safety-alert-message").textContent = message;
  el("safety-alert-overlay").dataset.level = level;
  el("safety-alert-overlay").classList.add("visible");
}

function handleStateAlert(metrics) {
  const state = String(metrics.state || "READY");
  if (!metrics.calibration_complete || state === previousState) {
    previousState = state;
    return;
  }

  const name = String(DG.state?.settings?.driver_name || "").trim();
  const greeting = name ? `${name}, ` : "";
  const volume = DG.state?.settings?.alert_volume ?? 80;

  if (state === "WARNING") {
    const message = `${greeting}fatigue indicators are increasing. Stay attentive and consider a safe break.`;
    DG.playAlertTone("warning", volume);
    DG.speak(message);
    showSafetyAlert("warning", "Fatigue indicators increasing", "Stay attentive and consider taking a safe break.");
  }

  if (state === "CRITICAL") {
    const now = Date.now();
    if (now - lastCriticalAnnouncement > 8000) {
      lastCriticalAnnouncement = now;
      const message = `${greeting}heavy fatigue detected. Please stop somewhere safe and take a break.`;
      DG.playAlertTone("critical", volume);
      DG.speak(message, {rate: .96});
      showSafetyAlert("critical", "Heavy fatigue detected", "Please stop somewhere safe and take a break.");
    }
  }

  previousState = state;
}

function renderCalibration(metrics) {
  const calibrating = metrics.monitoring && !metrics.calibration_complete;
  el("monitor-calibration").classList.toggle("visible", calibrating);
  if (!calibrating) return;

  const remaining = Number(metrics.calibration_remaining || 0);
  const progress = Math.max(0, Math.min(1, (10 - remaining) / 10));
  el("monitor-calibration-seconds").textContent = remaining.toFixed(1);
  el("monitor-calibration-ring").style.setProperty("--calibration", `${progress * 360}deg`);
  el("monitor-calibration-message").textContent = metrics.face_detected
    ? "Keep a natural expression while Guardian OS learns your personal baseline."
    : "Move into view so calibration can continue.";
}

async function startMonitoring() {
  DG.unlockAudio();
  el("monitor-start").disabled = true;
  el("monitor-stop").disabled = false;
  DG.toast("Connecting to the local V3 pipeline…");
  try {
    const result = await DG.post("/api/monitoring/start");
    DG.emit(result);
    DG.toast(result.message || "Monitoring started", "success");
  } catch (error) {
    el("monitor-start").disabled = false;
    el("monitor-stop").disabled = true;
    DG.toast("Monitoring could not start", "error");
    console.error(error);
  }
}

async function stopMonitoring() {
  DG.stopSpeech();
  el("monitor-stop").disabled = true;
  try {
    const result = await DG.post("/api/monitoring/stop");
    setStream(false);
    DG.emit(result);
    DG.toast(result.message || "Monitoring stopped", "success");
  } catch (error) {
    el("monitor-stop").disabled = false;
    DG.toast("Monitoring could not stop", "error");
    console.error(error);
  }
}

async function askCommander(message) {
  const clean = String(message || "").trim();
  if (!clean) return;
  el("monitor-commander-input").value = "";
  el("monitor-commander-reply").textContent = "Commander is thinking…";
  try {
    const result = await DG.post("/api/commander/message", {message: clean});
    el("monitor-commander-reply").textContent = result.response;
    await DG.speak(result.response);
  } catch (error) {
    el("monitor-commander-reply").textContent = "Commander could not process the request.";
    console.error(error);
  }
}

DG.subscribe(({metrics}) => {
  const lifecycle = metrics.monitoring || metrics.starting || metrics.stopping;
  el("monitor-camera-status").textContent = metrics.camera_status;
  el("monitor-state").textContent = metrics.state;
  el("monitor-camera-message").textContent =
    metrics.error ||
    (metrics.starting ? "Connecting camera…" :
     metrics.stopping ? "Releasing camera…" :
     metrics.monitoring ? "Live V3 monitoring active" : "Camera standing by");

  el("monitor-fps").textContent = Number(metrics.fps || 0).toFixed(1);
  el("monitor-backend").textContent = metrics.camera_backend || "—";
  el("monitor-face-status").textContent = metrics.face_detected ? "FACE LOCK" : "NO FACE";
  setStream(metrics.monitoring && metrics.camera_status === "CONNECTED");
  renderCalibration(metrics);

  const risk = Number(metrics.fatigue_probability || 0);
  el("monitor-risk").textContent = pct(risk);
  el("monitor-risk-gauge").style.setProperty("--risk", `${risk * 360}deg`);
  el("monitor-state-value").textContent = metrics.state;
  el("monitor-state-banner").className =
    `state-banner ${String(metrics.state).toLowerCase().replaceAll(" ", "-")}`;
  el("monitor-ear").textContent = Number(metrics.ear || 0).toFixed(3);
  el("monitor-blinks").textContent = Number(metrics.blink_rate || 0).toFixed(1);
  el("monitor-yawn").textContent = Number(metrics.yawn_score || 0).toFixed(3);
  el("monitor-tilt").textContent = `${Number(metrics.head_tilt || 0).toFixed(1)}°`;
  el("monitor-alert-count").textContent = `${metrics.alert_count || 0} alerts`;

  el("monitor-start").disabled = lifecycle;
  el("monitor-stop").disabled = !lifecycle || metrics.stopping;

  if (metrics.monitoring && metrics.calibration_complete) {
    riskHistory.push(risk);
    if (riskHistory.length > 120) riskHistory.shift();
  } else if (!metrics.monitoring) {
    riskHistory = [];
  }
  drawTrend();
  handleStateAlert(metrics);
});

el("privacy-understood").addEventListener("change", event => {
  el("privacy-continue").disabled = !event.target.checked;
});
el("privacy-continue").addEventListener("click", () => {
  privacyApproved = true;
  sessionStorage.setItem("dg-monitoring-privacy", "true");
  el("privacy-modal").classList.add("hidden");
  startMonitoring();
});
if (privacyApproved) el("privacy-modal").classList.add("hidden");

el("monitor-start").addEventListener("click", () => {
  if (!privacyApproved) {
    el("privacy-modal").classList.remove("hidden");
    return;
  }
  startMonitoring();
});
el("monitor-stop").addEventListener("click", stopMonitoring);
el("monitor-stop-speaking").addEventListener("click", () => {
  DG.stopSpeech();
  DG.toast("Commander speech stopped");
});
el("monitor-mute-alerts").addEventListener("click", event => {
  DG.setAlertsMuted(!DG.alertsMuted);
  event.currentTarget.textContent = DG.alertsMuted ? "Unmute alerts" : "Mute alerts";
  DG.toast(DG.alertsMuted ? "Alert tones muted" : "Alert tones enabled");
});
el("monitor-mute-alerts").textContent = DG.alertsMuted ? "Unmute alerts" : "Mute alerts";
el("dismiss-safety-alert").addEventListener("click", () => {
  el("safety-alert-overlay").classList.remove("visible");
});
el("monitor-commander-form").addEventListener("submit", event => {
  event.preventDefault();
  askCommander(el("monitor-commander-input").value);
});

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (Recognition) {
  recognition = new Recognition();
  recognition.lang = "en-GB";
  recognition.onstart = () => el("monitor-mic").classList.add("active");
  recognition.onend = () => el("monitor-mic").classList.remove("active");
  recognition.onresult = event => askCommander(event.results[event.results.length - 1][0].transcript);
  el("monitor-mic").addEventListener("click", () => {
    try { recognition.start(); } catch (_) {}
  });
} else {
  el("monitor-mic").disabled = true;
}

window.addEventListener("resize", drawTrend);
