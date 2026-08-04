const edgeEl = id => document.getElementById(id);
let edgeSnapshot = null;

const edgePercent = value => `${(Number(value || 0) * 100).toFixed(1)}%`;

function edgeDuration(seconds) {
  const total = Math.round(Number(seconds || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function edgeDate(value) {
  if (!value) return "Unknown time";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function setContextForm(context) {
  const form = edgeEl("edge-context-form");
  Object.entries(context || {}).forEach(([key, value]) => {
    if (!form.elements[key]) return;
    if (form.elements[key].type === "checkbox") form.elements[key].checked = Boolean(value);
    else form.elements[key].value = value ?? "";
  });
  document.querySelectorAll(".manual-context-field").forEach(field => {
    field.classList.toggle("context-disabled", !form.elements.manual_override.checked);
  });
}

function contextAge(seconds) {
  if (seconds === null || seconds === undefined) return "No current reading";
  const minutes = Math.max(0, Math.round(Number(seconds) / 60));
  return minutes < 1 ? "Updated less than a minute ago" : `Updated ${minutes}m ago`;
}

function renderAutomaticContext(context) {
  const automatic = context.automatic_weather || {};
  edgeEl("auto-weather-value").textContent = String((context.weather || {}).value || "unknown").replaceAll("_", " ");
  edgeEl("auto-weather-source").textContent = `${(context.weather || {}).source || "Unknown"}${automatic.location ? ` · ${automatic.location}` : ""}`;
  edgeEl("auto-road-value").textContent = String((context.road_condition || {}).value || "unknown").replaceAll("_", " ");
  edgeEl("auto-road-source").textContent = (context.road_condition || {}).source || "Unknown";
  edgeEl("auto-temperature").textContent = automatic.temperature_c === null || automatic.temperature_c === undefined ? "—" : `${Number(automatic.temperature_c).toFixed(1)}°C`;
  edgeEl("auto-weather-age").textContent = automatic.error || contextAge(automatic.age_seconds);
  edgeEl("automatic-context-status").dataset.state = automatic.available ? (automatic.fresh ? "fresh" : "stale") : "offline";
}

function renderEdge(snapshot) {
  edgeSnapshot = snapshot;
  const insights = snapshot.insights || {};
  const sessions = snapshot.recent_sessions || [];

  edgeEl("edge-updated").textContent =
    `Updated ${edgeDate(snapshot.updated_at)}`;
  edgeEl("edge-session-count").textContent = insights.session_count || 0;
  edgeEl("edge-average-risk").textContent = edgePercent(insights.average_risk);
  edgeEl("edge-highest-risk").textContent = edgePercent(insights.highest_risk);
  edgeEl("edge-pending-sync").textContent = snapshot.pending_sync || 0;
  edgeEl("edge-summary-message").textContent = insights.summary || "No insight available.";
  edgeEl("edge-total-time").textContent = edgeDuration(insights.total_duration_seconds);
  edgeEl("edge-average-duration").textContent = edgeDuration(insights.average_duration_seconds);
  edgeEl("edge-total-alerts").textContent = insights.total_alerts || 0;
  edgeEl("edge-alert-rate").textContent = edgePercent(insights.alert_session_rate);
  edgeEl("edge-baseline-ear").textContent =
    Number(insights.average_baseline_ear || 0).toFixed(3);
  edgeEl("edge-risk-period").textContent =
    String(insights.highest_risk_period || "not enough data")
      .replaceAll("_", " ");

  const latest = insights.latest_session;
  edgeEl("edge-latest-session").innerHTML = latest ? `
    <p class="eyebrow">LATEST SESSION</p>
    <div class="edge-latest-grid">
      <div><span>Completed</span><b>${edgeDate(latest.generated_at)}</b></div>
      <div><span>Duration</span><b>${edgeDuration(latest.duration_seconds)}</b></div>
      <div><span>Maximum risk</span><b>${edgePercent(latest.maximum_risk)}</b></div>
      <div><span>Alerts</span><b>${latest.alert_count || 0}</b></div>
      <div><span>Dominant signal</span><b>${latest.dominant_signal || "unknown"}</b></div>
    </div>
  ` : `
    <p class="eyebrow">LATEST SESSION</p>
    <p class="muted">No completed session is available.</p>
  `;

  edgeEl("edge-recent-count").textContent = `${sessions.length} records`;
  edgeEl("edge-session-list").innerHTML = sessions.length
    ? sessions.map(session => `
      <article class="edge-session-item">
        <div class="edge-session-risk">
          <strong>${edgePercent(session.maximum_risk)}</strong>
          <span>peak</span>
        </div>
        <div>
          <b>${edgeDate(session.generated_at)}</b>
          <span>${edgeDuration(session.duration_seconds)} · ${session.alert_count || 0} alerts · ${session.dominant_signal || "unknown"}</span>
        </div>
        <a href="/reports">View report →</a>
      </article>
    `).join("")
    : `<div class="empty-state">No completed sessions have been imported yet.</div>`;

  const pending = Number(snapshot.pending_sync || 0);
  edgeEl("edge-queue-label").textContent = `${pending} pending`;
  edgeEl("edge-sync-status").textContent = pending ? "PENDING LOCAL" : "QUEUE CLEAR";
  edgeEl("edge-mark-synced").disabled = pending === 0;

  setContextForm(snapshot.manual_context || {});
  renderAutomaticContext(snapshot.context || {});
}

async function loadEdge() {
  try {
    const response = await fetch("/api/edge", {cache: "no-store"});
    if (!response.ok) throw new Error(await response.text());
    renderEdge(await response.json());
  } catch (error) {
    DG.toast("Edge Memory could not be loaded", "error");
    console.error(error);
  }
}

edgeEl("edge-refresh").addEventListener("click", async () => {
  edgeEl("edge-refresh").disabled = true;
  try {
    const result = await DG.post("/api/edge/refresh");
    renderEdge(result.snapshot);
    DG.toast(`${result.session_count} local sessions indexed`, "success");
  } catch (error) {
    DG.toast("Local memory refresh failed", "error");
    console.error(error);
  } finally {
    edgeEl("edge-refresh").disabled = false;
  }
});

edgeEl("edge-context-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = Object.fromEntries(new FormData(form).entries());
  payload.automatic_enabled = form.elements.automatic_enabled.checked;
  payload.manual_override = form.elements.manual_override.checked;

  try {
    const response = await fetch("/api/edge/context", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    renderEdge(result.snapshot);
    DG.toast("Journey context saved locally", "success");
  } catch (error) {
    DG.toast("Journey context could not be saved", "error");
    console.error(error);
  }
});

edgeEl("edge-refresh-weather").addEventListener("click", async event => {
  event.currentTarget.disabled = true;
  try {
    const result = await DG.post("/api/edge/context/refresh-weather");
    renderEdge(result.snapshot);
    DG.toast(result.context.automatic_weather?.available ? "Current weather refreshed" : "Weather unavailable; context set to Unknown", result.context.automatic_weather?.available ? "success" : "warning");
  } catch (error) {
    DG.toast("Automatic weather refresh failed", "error");
    console.error(error);
  } finally {
    event.currentTarget.disabled = false;
  }
});

edgeEl("edge-context-form").elements.manual_override.addEventListener("change", event => {
  document.querySelectorAll(".manual-context-field").forEach(field => field.classList.toggle("context-disabled", !event.currentTarget.checked));
});

edgeEl("edge-mark-synced").addEventListener("click", async () => {
  try {
    const response = await fetch("/api/edge/sync/mark-complete", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_ids: null}),
    });
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    renderEdge(result.snapshot);
    DG.toast(`${result.updated} records marked as synced`, "success");
  } catch (error) {
    DG.toast("Sync queue could not be updated", "error");
    console.error(error);
  }
});

loadEdge();
