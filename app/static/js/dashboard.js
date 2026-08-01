const el = id => document.getElementById(id);
const pct = value => `${Math.round(value * 100)}%`;
const timeText = seconds =>
  `${String(Math.floor(seconds / 60)).padStart(2,"0")}:${String(seconds % 60).padStart(2,"0")}`;

let cameraStreamActive = false;

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

DG.subscribe(({metrics, events, voice}) => {
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

  el("risk-value").textContent = pct(Number(metrics.fatigue_probability || 0));
  el("risk-gauge").style.setProperty(
    "--risk",
    `${Number(metrics.fatigue_probability || 0) * 360}deg`
  );
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

  el("start-monitoring").disabled = activeLifecycle;
  el("stop-monitoring").disabled = !activeLifecycle || metrics.stopping;

  el("event-count").textContent = `${events.length} events`;
  el("event-list").innerHTML = events.map(event => `
    <article class="${event.level}">
      <time>${event.time}</time>
      <div><b>${event.source}</b><span>${event.message}</span></div>
    </article>
  `).join("") || `<div class="empty-state">No events yet.</div>`;

  el("commander-status-title").textContent =
    voice.enabled ? "Listening for Commander" : "Assistant online";
  el("commander-status-detail").textContent = voice.detail;
});

el("start-monitoring").addEventListener("click", async () => {
  el("start-monitoring").disabled = true;
  el("stop-monitoring").disabled = false;
  DG.toast("Connecting to the real V3 camera pipeline…");

  try {
    const state = await DG.post("/api/monitoring/start");
    DG.emit(state);
    DG.toast(state.message || "Live monitoring started", "success");
  } catch (error) {
    let message = "Monitoring could not start";
    try {
      const parsed = JSON.parse(error.message);
      message = parsed.detail || message;
    } catch (_) {}
    DG.toast(message, "error");
    el("start-monitoring").disabled = false;
    el("stop-monitoring").disabled = true;
    console.error(error);
  }
});

el("stop-monitoring").addEventListener("click", async () => {
  el("stop-monitoring").disabled = true;
  DG.toast("Stopping monitoring and releasing the camera…");

  try {
    const state = await DG.post("/api/monitoring/stop");
    setCameraStream(false);
    DG.emit(state);
    DG.toast(state.message || "Monitoring stopped", state.success ? "success" : "normal");
  } catch (error) {
    DG.toast("Monitoring could not stop", "error");
    el("stop-monitoring").disabled = false;
    console.error(error);
  }
});
