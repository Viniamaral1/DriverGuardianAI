const el = id => document.getElementById(id);
const pct = value => `${Math.round(value * 100)}%`;
const timeText = seconds =>
  `${String(Math.floor(seconds / 60)).padStart(2,"0")}:${String(seconds % 60).padStart(2,"0")}`;

let cameraStreamActive = false;
let riskHistory = [];
let lastCalibrationComplete = false;
let dashboardRecognition = null;

function setCameraStream(active) {
  const image = el("live-camera-feed");
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
  if (placeholder) {
    placeholder.classList.toggle("hidden-camera-placeholder", active);
  }
}

function isActiveLifecycle(metrics) {
  return Boolean(metrics.monitoring || metrics.starting || metrics.stopping);
}

function drawTrend() {
  const canvas = el("risk-trend-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 720;
  const height = canvas.clientHeight || 150;

  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const styles = getComputedStyle(document.documentElement);
  const line = styles.getPropertyValue("--line").trim() || "rgba(180,220,240,.12)";
  const accent = styles.getPropertyValue("--accent").trim() || "#38e8ed";
  const warning = styles.getPropertyValue("--warning").trim() || "#ffba49";
  const danger = styles.getPropertyValue("--danger").trim() || "#ff5f6d";
  const muted = styles.getPropertyValue("--muted").trim() || "#8494a4";

  ctx.strokeStyle = line;
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.65, 0.82].forEach(value => {
    const y = height - value * height;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
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
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  riskHistory.forEach((value, index) => {
    const x = index * width / Math.max(1, riskHistory.length - 1);
    const y = height - Math.min(1, Math.max(0, value)) * height;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderCalibration(metrics) {
  const overlay = el("calibration-overlay");
  const calibrating = metrics.monitoring && !metrics.calibration_complete;
  overlay.classList.toggle("visible", calibrating);

  if (!calibrating) return;

  const remaining = Number(metrics.calibration_remaining || 0);
  const progress = Math.max(0, Math.min(1, (10 - remaining) / 10));
  el("calibration-seconds").textContent = remaining.toFixed(1);
  el("calibration-ring").style.setProperty("--calibration", `${progress * 360}deg`);
  el("calibration-message").textContent = metrics.face_detected
    ? "Hold a natural expression while Guardian OS learns your personal baseline."
    : "Move into view so calibration can continue.";

  if (lastCalibrationComplete !== metrics.calibration_complete) {
    lastCalibrationComplete = metrics.calibration_complete;
  }
}

function renderReport(report) {
  const state = report?.state || "IDLE";
  el("report-state").textContent = `REPORT ${state}`;
  el("report-state").className =
    `badge ${state === "COMPLETE" ? "success" : state === "ERROR" ? "danger" : ""}`;
}

function renderGreeting(settings) {
  const name = String(settings?.driver_name || "").trim();
  el("dashboard-greeting").textContent = name ? `Good to see you, ${name}` : "Assistant online";
  if (!name && sessionStorage.getItem("dg-profile-dismissed") !== "true") {
    el("profile-modal").classList.remove("hidden");
  }
}

async function askCommander(message) {
  const clean = String(message || "").trim();
  if (!clean) return;

  const reply = el("dashboard-commander-reply");
  reply.textContent = "Commander is thinking…";
  el("dashboard-commander-input").value = "";

  try {
    const result = await DG.post("/api/commander/message", {message: clean});
    reply.textContent = result.response;

    await DG.speak(result.response);
  } catch (error) {
    reply.textContent = "Commander could not process the request.";
    DG.toast("Commander request failed", "error");
    console.error(error);
  }
}

DG.subscribe(({metrics, events, voice, report, settings}) => {
  const activeLifecycle = isActiveLifecycle(metrics);

  el("camera-status").textContent = metrics.camera_status;
  el("camera-message").textContent =
    metrics.error ||
    (metrics.starting
      ? "Connecting to the camera…"
      : metrics.stopping
        ? "Releasing the camera…"
        : metrics.monitoring
          ? metrics.calibration_complete
            ? "Live V3 pipeline active"
            : `Personal calibration: ${Number(metrics.calibration_remaining || 0).toFixed(1)}s remaining`
          : "Camera standing by");

  el("face-status").textContent = metrics.face_detected ? "FACE LOCK" : "NO FACE";
  el("camera-fps").textContent = Number(metrics.fps || 0).toFixed(1);
  el("camera-stage").classList.toggle("active", metrics.monitoring);
  setCameraStream(metrics.monitoring && metrics.camera_status === "CONNECTED");
  renderCalibration(metrics);

  const risk = Number(metrics.fatigue_probability || 0);
  el("risk-value").textContent = pct(risk);
  el("risk-gauge").style.setProperty("--risk", `${risk * 360}deg`);
  el("driver-state").textContent = metrics.state;
  el("state-banner").className =
    `state-banner ${String(metrics.state).toLowerCase().replaceAll(" ", "-")}`;

  el("session-time").textContent = timeText(Number(metrics.session_seconds || 0));
  el("alert-count").textContent = metrics.alert_count;
  el("model-status").textContent = metrics.model_status;

  el("ear-value").textContent = Number(metrics.ear || 0).toFixed(3);
  el("blink-value").textContent = Number(metrics.blink_rate || 0).toFixed(1);
  el("yawn-value").textContent = Number(metrics.yawn_score || 0).toFixed(3);
  el("tilt-value").textContent = Number(metrics.head_tilt || 0).toFixed(1);

  el("ear-meter").style.width =
    `${Math.min(100, Number(metrics.ear || 0) / .36 * 100)}%`;
  el("blink-meter").style.width =
    `${Math.min(100, Number(metrics.blink_rate || 0) / 35 * 100)}%`;
  el("yawn-meter").style.width =
    `${Math.min(100, Number(metrics.yawn_score || 0) * 100)}%`;
  el("tilt-meter").style.width =
    `${Math.min(100, Number(metrics.head_tilt || 0) / 25 * 100)}%`;

  if (el("start-monitoring")) el("start-monitoring").disabled = activeLifecycle;
  el("stop-monitoring").disabled = !activeLifecycle || metrics.stopping;

  if (metrics.monitoring && metrics.calibration_complete) {
    riskHistory.push(risk);
    if (riskHistory.length > 60) riskHistory.shift();
  } else if (!metrics.monitoring && !metrics.stopping) {
    riskHistory = [];
  }
  drawTrend();

  el("event-count").textContent = `${events.length} events`;
  el("event-list").innerHTML = events.map(event => `
    <article class="${event.level}">
      <time>${event.time}</time>
      <div><b>${event.source}</b><span>${event.message}</span></div>
    </article>
  `).join("") || `<div class="empty-state">No events yet.</div>`;

  el("dashboard-commander-detail").textContent =
    voice.enabled ? "Listening for the Commander wake word." : "Ask without leaving the live session.";

  renderReport(report);
  renderGreeting(settings);
});

const dashboardStopButton = el("stop-monitoring");

dashboardStopButton?.addEventListener("click", async () => {
  el("stop-monitoring").disabled = true;
  DG.toast("Stopping monitoring and preparing the session report…");

  try {
    const state = await DG.post("/api/monitoring/stop");
    setCameraStream(false);
    DG.emit(state);
    DG.toast(
      state.message || "Monitoring stopped",
      state.success ? "success" : "normal"
    );
  } catch (error) {
    DG.toast("Monitoring could not stop", "error");
    if (el("stop-monitoring")) el("stop-monitoring").disabled = false;
    console.error(error);
  }
});

el("dashboard-commander-form").addEventListener("submit", event => {
  event.preventDefault();
  askCommander(el("dashboard-commander-input").value);
});

document.querySelectorAll("[data-dashboard-command]").forEach(button => {
  button.addEventListener("click", () => askCommander(button.dataset.dashboardCommand));
});

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (Recognition) {
  dashboardRecognition = new Recognition();
  dashboardRecognition.lang = "en-GB";
  dashboardRecognition.interimResults = false;
  dashboardRecognition.onstart = () => el("dashboard-mic").classList.add("active");
  dashboardRecognition.onend = () => el("dashboard-mic").classList.remove("active");
  dashboardRecognition.onresult = event => {
    askCommander(event.results[event.results.length - 1][0].transcript);
  };
  el("dashboard-mic").addEventListener("click", () => {
    try { dashboardRecognition.start(); } catch (_) {}
  });
} else {
  el("dashboard-mic").disabled = true;
}

el("profile-skip").addEventListener("click", () => {
  sessionStorage.setItem("dg-profile-dismissed", "true");
  el("profile-modal").classList.add("hidden");
});

el("profile-form").addEventListener("submit", async event => {
  event.preventDefault();
  const name = el("profile-name").value.trim();
  if (!name) return;

  try {
    const response = await fetch("/api/settings", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({driver_name: name}),
    });
    if (!response.ok) throw new Error(await response.text());
    el("profile-modal").classList.add("hidden");
    DG.toast(`Welcome, ${name}`, "success");
  } catch (error) {
    DG.toast("The greeting could not be saved", "error");
    console.error(error);
  }
});

el("dashboard-stop-speaking")?.addEventListener("click", () => {
  DG.stopSpeech();
  DG.toast("Commander speech stopped");
});

window.addEventListener("resize", drawTrend);
