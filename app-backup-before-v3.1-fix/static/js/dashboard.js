const el = id => document.getElementById(id);
let dashboardRecognition = null;

function renderGreeting(settings) {
  const name = String(settings?.driver_name || "").trim();
  el("dashboard-greeting").textContent = name
    ? `Welcome back, ${name}`
    : "Welcome to Guardian OS";

  if (!name && sessionStorage.getItem("dg-profile-dismissed") !== "true") {
    el("profile-modal").classList.remove("hidden");
  }
}

function renderHomeState({metrics, events, voice, report, settings}) {
  const active = Boolean(metrics.monitoring || metrics.starting || metrics.stopping);

  el("home-monitoring-state").textContent =
    metrics.starting ? "Starting" :
    metrics.stopping ? "Stopping" :
    metrics.monitoring ? metrics.state : "Ready";

  el("home-monitoring-detail").textContent =
    metrics.monitoring
      ? `${Number(metrics.fatigue_probability || 0) * 100 < 1
          ? "Live session active"
          : `${Math.round(Number(metrics.fatigue_probability || 0) * 100)}% current risk`}`
      : "Camera is not active";

  el("home-commander-state").textContent = voice?.enabled ? "Listening" : "Online";
  el("home-commander-detail").textContent =
    voice?.detail || "Local assistant ready";

  const reportState = report?.state || "IDLE";
  el("home-report-state").textContent =
    reportState === "COMPLETE" ? "Report ready" :
    reportState === "GENERATING" ? "Generating" :
    reportState === "ERROR" ? "Report error" :
    "No new report";

  el("home-report-detail").textContent =
    report?.detail ||
    (reportState === "COMPLETE"
      ? "Open Reports or Metrics to review it"
      : "Start a session to create one");

  el("home-health-state").textContent =
    metrics.error ? "Attention required" : "Ready";
  el("home-health-detail").textContent =
    metrics.error || `${metrics.model_status || "READY"} model · ${metrics.camera_status || "STANDBY"} camera`;

  el("event-count").textContent = `${events.length} events`;
  el("event-list").innerHTML = events.slice(0, 8).map(event => `
    <article class="${event.level}">
      <time>${event.time}</time>
      <div><b>${event.source}</b><span>${event.message}</span></div>
    </article>
  `).join("") || `<div class="empty-state">No recent activity.</div>`;

  renderGreeting(settings);
}

async function askCommander(message) {
  const clean = String(message || "").trim();
  if (!clean) return;

  el("dashboard-commander-input").value = "";
  el("dashboard-commander-reply").textContent = "Commander is thinking…";

  try {
    const result = await DG.post("/api/commander/message", {message: clean});
    el("dashboard-commander-reply").textContent = result.response;
    await DG.speak(result.response);
  } catch (error) {
    el("dashboard-commander-reply").textContent =
      "Commander could not process the request.";
    DG.toast("Commander request failed", "error");
    console.error(error);
  }
}

DG.subscribe(renderHomeState);

el("dashboard-commander-form")?.addEventListener("submit", event => {
  event.preventDefault();
  askCommander(el("dashboard-commander-input").value);
});

document.querySelectorAll("[data-dashboard-command]").forEach(button => {
  button.addEventListener("click", () => {
    askCommander(button.dataset.dashboardCommand);
  });
});

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (Recognition) {
  dashboardRecognition = new Recognition();
  dashboardRecognition.lang = "en-GB";
  dashboardRecognition.interimResults = false;
  dashboardRecognition.onstart = () => el("dashboard-mic")?.classList.add("active");
  dashboardRecognition.onend = () => el("dashboard-mic")?.classList.remove("active");
  dashboardRecognition.onresult = event => {
    askCommander(event.results[event.results.length - 1][0].transcript);
  };
  el("dashboard-mic")?.addEventListener("click", () => {
    try { dashboardRecognition.start(); } catch (_) {}
  });
} else if (el("dashboard-mic")) {
  el("dashboard-mic").disabled = true;
}

el("dashboard-stop-speaking")?.addEventListener("click", () => {
  DG.stopSpeech();
  DG.toast("Commander speech stopped");
});

el("profile-skip")?.addEventListener("click", () => {
  sessionStorage.setItem("dg-profile-dismissed", "true");
  el("profile-modal").classList.add("hidden");
});

el("profile-form")?.addEventListener("submit", async event => {
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
