const d = id => document.getElementById(id);

function row(label, value, good = null) {
  const status = good === true ? "ok" : good === false ? "bad" : "";
  return `<div class="diagnostic-row ${status}"><span>${label}</span><strong>${value ?? "—"}</strong></div>`;
}

function badge(id, text, good) {
  const node = d(id);
  node.textContent = text;
  node.className = `badge ${good ? "success" : good === false ? "danger" : ""}`;
}

async function refreshDiagnostics() {
  try {
    const response = await fetch("/api/diagnostics");
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    const m = data.monitoring || {};
    const voice = data.voice || {};
    const report = data.report || {};

    const cameraGood = !m.error && ["CONNECTED","STANDBY"].includes(m.camera_status);
    badge("diag-camera-badge", m.camera_status || "UNKNOWN", cameraGood);
    d("diag-camera").innerHTML = [
      row("Camera status", m.camera_status, cameraGood),
      row("Backend", m.camera_backend || "Not selected"),
      row("Thread alive", m.thread_alive ? "Yes" : "No", !m.monitoring || m.thread_alive),
      row("Frame available", m.frame_available ? "Yes" : "No", !m.monitoring || m.frame_available),
      row("Face detected", m.face_detected ? "Yes" : "No"),
      row("FPS", Number(m.fps || 0).toFixed(1)),
      row("Error", m.error || "None", !m.error),
    ].join("");

    const modelGood = !m.model_warning && m.model_status !== "ERROR";
    badge("diag-model-badge", m.model_status || "READY", modelGood);
    d("diag-model").innerHTML = [
      row("Model status", m.model_status || "READY", modelGood),
      row("State", m.state || "READY"),
      row("Calibration", m.calibration_complete ? "Complete" : "Not active"),
      row("Baseline EAR", Number(m.baseline_ear || 0).toFixed(3)),
      row("Model warning", m.model_warning || "None", !m.model_warning),
    ].join("");

    badge("diag-voice-badge", voice.enabled ? "LISTENING" : voice.available ? "READY" : "BROWSER", true);
    d("diag-voice").innerHTML = [
      row("Backend available", voice.available ? "Yes" : "No"),
      row("Backend enabled", voice.enabled ? "Yes" : "No"),
      row("Status", voice.status || "OFFLINE"),
      row("Detail", voice.detail || "Browser voice mode available"),
    ].join("");

    const reportGood = report.state !== "ERROR";
    badge("diag-report-badge", report.state || "IDLE", reportGood);
    d("diag-report").innerHTML = [
      row("Automatic reports", data.settings?.automatic_reports ? "Enabled" : "Disabled"),
      row("Pipeline state", report.state || "IDLE", reportGood),
      row("Message", report.message || "No report activity"),
      row("Latest HTML", report.html_report ? "Ready" : "None"),
      row("Error", report.error || "None", !report.error),
    ].join("");

    const packages = data.packages || {};
    d("diag-environment").innerHTML = [
      row("Python executable", data.system?.python),
      row("Python version", data.system?.python_version),
      row("Platform", data.system?.platform),
      row("OpenCV", packages.cv2),
      row("MediaPipe", packages.mediapipe),
      row("scikit-learn", packages.sklearn),
      row("FastAPI", packages.fastapi),
      row("Camera index", data.settings?.camera_index),
    ].join("");
  } catch (error) {
    DG.toast("Diagnostics could not be loaded", "error");
    console.error(error);
  }
}

d("refresh-diagnostics").addEventListener("click", refreshDiagnostics);
refreshDiagnostics();
window.setInterval(refreshDiagnostics, 5000);
