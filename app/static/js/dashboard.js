const el = id => document.getElementById(id);
const pct = value => `${Math.round(value * 100)}%`;
const timeText = seconds => `${String(Math.floor(seconds / 60)).padStart(2,"0")}:${String(seconds % 60).padStart(2,"0")}`;

DG.subscribe(({metrics, events, voice}) => {
  el("camera-status").textContent = metrics.camera_status;
  el("camera-message").textContent = metrics.monitoring ? "Live pipeline active" : "Camera standing by";
  el("face-status").textContent = metrics.face_detected ? "FACE LOCK" : "NO FACE";
  el("camera-stage").classList.toggle("active", metrics.monitoring);

  el("risk-value").textContent = pct(metrics.fatigue_probability);
  el("risk-gauge").style.setProperty("--risk", `${metrics.fatigue_probability * 360}deg`);
  el("driver-state").textContent = metrics.state;
  el("state-banner").className = `state-banner ${metrics.state.toLowerCase()}`;

  el("session-time").textContent = timeText(metrics.session_seconds);
  el("alert-count").textContent = metrics.alert_count;
  el("model-status").textContent = metrics.model_status;

  el("ear-value").textContent = metrics.ear.toFixed(3);
  el("blink-value").textContent = metrics.blink_rate.toFixed(1);
  el("yawn-value").textContent = metrics.yawn_score.toFixed(3);
  el("tilt-value").textContent = metrics.head_tilt.toFixed(1);

  el("ear-meter").style.width = `${Math.min(100, metrics.ear / .36 * 100)}%`;
  el("blink-meter").style.width = `${Math.min(100, metrics.blink_rate / 35 * 100)}%`;
  el("yawn-meter").style.width = `${metrics.yawn_score * 100}%`;
  el("tilt-meter").style.width = `${Math.min(100, metrics.head_tilt / 25 * 100)}%`;

  el("start-monitoring").disabled = metrics.monitoring;
  el("stop-monitoring").disabled = !metrics.monitoring;

  el("event-count").textContent = `${events.length} events`;
  el("event-list").innerHTML = events.map(event => `
    <article class="${event.level}">
      <time>${event.time}</time>
      <div><b>${event.source}</b><span>${event.message}</span></div>
    </article>
  `).join("") || `<div class="empty-state">No events yet.</div>`;

  el("commander-status-title").textContent = voice.enabled ? "Listening for Commander" : "Assistant online";
  el("commander-status-detail").textContent = voice.detail;
});

el("start-monitoring").addEventListener("click", async () => {
  try {
    const state = await DG.post("/api/monitoring/start");
    DG.emit(state);
    DG.toast("Monitoring started", "success");
  } catch (error) {
    DG.toast("Monitoring could not start", "error");
    console.error(error);
  }
});

el("stop-monitoring").addEventListener("click", async () => {
  try {
    const state = await DG.post("/api/monitoring/stop");
    DG.emit(state);
    DG.toast("Monitoring stopped");
  } catch (error) {
    DG.toast("Monitoring could not stop", "error");
    console.error(error);
  }
});
