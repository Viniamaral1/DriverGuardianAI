const intelligenceEl = id => document.getElementById(id);
const intelligencePercent = value =>
  `${(Number(value || 0) * 100).toFixed(1)}%`;

function intelligenceLabel(value) {
  return String(value ?? "unknown").replaceAll("_", " ");
}

function renderContributions(items) {
  const container = intelligenceEl("intelligence-contributions");
  if (!items?.length) {
    container.innerHTML =
      `<div class="empty-state">No calibrated contribution is active.</div>`;
    return;
  }

  container.innerHTML = items.map(item => `
    <article class="intelligence-contribution">
      <div>
        <span>${item.label}</span>
        <b>${intelligencePercent(item.share)}</b>
      </div>
      <div class="meter">
        <i style="width:${Math.min(100, Number(item.share || 0) * 100)}%"></i>
      </div>
      <small>Evidence value ${Number(item.value || 0).toFixed(3)}</small>
    </article>
  `).join("");
}

function renderIntelligence(snapshot) {
  const live = snapshot.live || {};
  const quality = snapshot.signal_quality || {};
  const outlook = snapshot.outlook || {};
  const baseline = snapshot.baseline_comparison || {};
  const context = snapshot.context || {};
  const caution = snapshot.journey_caution || {};
  const history = snapshot.history || {};
  const explanation = snapshot.explanation || {};
  const environment = snapshot.environment || {};

  intelligenceEl("intelligence-generated").textContent =
    `Updated ${new Date(snapshot.generated_at).toLocaleTimeString("en-GB")}`;
  intelligenceEl("intelligence-live-state").textContent =
    live.monitoring ? live.state : "READY";
  intelligenceEl("intelligence-orb").dataset.state =
    String(live.state || "ready").toLowerCase();

  intelligenceEl("intelligence-risk").textContent =
    intelligencePercent(live.risk);
  intelligenceEl("intelligence-risk-state").textContent =
    live.monitoring ? `${live.state} · ${live.alert_count || 0} alerts` :
      "Monitoring standby";

  intelligenceEl("intelligence-quality").textContent =
    intelligenceLabel(quality.level);
  intelligenceEl("intelligence-quality-score").textContent =
    quality.score ? `${intelligencePercent(quality.score)} · ${quality.summary}` :
      quality.summary || "No live signal";

  intelligenceEl("intelligence-outlook").textContent =
    intelligenceLabel(outlook.level);
  intelligenceEl("intelligence-outlook-score").textContent =
    outlook.horizon || "No forecast";

  intelligenceEl("intelligence-session-count").textContent =
    `${history.session_count || 0} sessions`;
  intelligenceEl("intelligence-history-risk").textContent =
    `Average ${intelligencePercent(history.average_risk)}`;

  intelligenceEl("intelligence-explanation-summary").textContent =
    explanation.summary || "No explanation is available.";
  renderContributions(explanation.contributions || []);

  const outlookScore = Number(outlook.score || 0);
  intelligenceEl("outlook-score-ring").style.setProperty(
    "--outlook-score",
    `${outlookScore * 360}deg`
  );
  intelligenceEl("outlook-score-ring").dataset.level = outlook.level || "standby";
  intelligenceEl("outlook-score-value").textContent =
    intelligencePercent(outlookScore);
  intelligenceEl("outlook-horizon").textContent =
    outlook.horizon || "No live forecast";
  intelligenceEl("outlook-summary").textContent =
    outlook.summary || "No outlook is available.";
  intelligenceEl("outlook-method").textContent =
    outlook.method || "Transparent rule-based outlook.";
  intelligenceEl("intelligence-outlook-level").textContent =
    String(outlook.level || "standby").toUpperCase();

  intelligenceEl("baseline-current").textContent =
    baseline.current ? Number(baseline.current).toFixed(3) : "—";
  intelligenceEl("baseline-historical").textContent =
    baseline.historical ? Number(baseline.historical).toFixed(3) : "—";
  intelligenceEl("baseline-difference").textContent =
    baseline.available
      ? `${Number(baseline.difference_percent || 0).toFixed(1)}%`
      : "—";
  intelligenceEl("baseline-summary").textContent =
    baseline.summary || "No baseline comparison is available.";

  intelligenceEl("context-period").textContent =
    intelligenceLabel(context.automatic_time_period);
  intelligenceEl("context-auto-light").textContent =
    intelligenceLabel(context.automatic_external_light);
  intelligenceEl("context-weather").textContent =
    intelligenceLabel(context.weather);
  intelligenceEl("context-weather-source").textContent =
    context.sources?.weather?.source || "Unknown source";
  intelligenceEl("context-road").textContent =
    intelligenceLabel(context.road_condition);
  intelligenceEl("context-road-source").textContent =
    context.sources?.road_condition?.source || "Unknown source";
  intelligenceEl("context-light-source").textContent =
    context.sources?.external_light?.source || "Local clock estimate";
  intelligenceEl("context-cabin").textContent =
    intelligenceLabel(context.cabin_light);
  intelligenceEl("context-occlusion").textContent =
    intelligenceLabel(context.occlusion);
  intelligenceEl("context-summary").textContent =
    caution.summary || "No additional journey caution is recorded.";
  intelligenceEl("context-caution-badge").textContent =
    caution.elevated ? "ELEVATED CAUTION" : "CONTEXT NORMAL";
  intelligenceEl("context-caution-badge").className =
    `badge ${caution.elevated ? "warning" : "success"}`;

  intelligenceEl("environment-quality-badge").textContent =
    String(environment.quality || "standby").toUpperCase();
  intelligenceEl("environment-quality-badge").className =
    `badge ${
      environment.quality === "good"
        ? "success"
        : environment.quality === "moderate"
          ? "warning"
          : environment.quality === "limited"
            ? "danger"
            : ""
    }`;
  intelligenceEl("environment-light").textContent =
    intelligenceLabel(environment.cabin_light);
  intelligenceEl("environment-sharpness").textContent =
    intelligenceLabel(environment.sharpness);
  intelligenceEl("environment-brightness").textContent =
    Number(environment.brightness || 0).toFixed(0);
  intelligenceEl("environment-contrast").textContent =
    Number(environment.contrast || 0).toFixed(1);
  intelligenceEl("environment-underexposed").textContent =
    intelligencePercent(environment.underexposed_ratio);
  intelligenceEl("environment-overexposed").textContent =
    intelligencePercent(environment.overexposed_ratio);
  intelligenceEl("environment-glare").textContent =
    intelligencePercent(environment.glare_ratio);
  intelligenceEl("environment-score").textContent =
    intelligencePercent(environment.quality_score);
  intelligenceEl("environment-summary").textContent =
    environment.summary || "No image-quality assessment is available.";

  intelligenceEl("history-average-risk").textContent =
    intelligencePercent(history.average_risk);
  intelligenceEl("history-highest-risk").textContent =
    intelligencePercent(history.highest_risk);
  intelligenceEl("history-risk-period").textContent =
    intelligenceLabel(history.highest_risk_period);
  intelligenceEl("history-summary").textContent =
    history.summary || "No historical insight is available.";

  intelligenceEl("intelligence-safety-note").textContent =
    snapshot.safety_note || "Live monitoring remains the primary safety input.";
}

async function loadIntelligence() {
  try {
    const response = await fetch("/api/intelligence", {cache: "no-store"});
    if (!response.ok) throw new Error(await response.text());
    renderIntelligence(await response.json());
  } catch (error) {
    DG.toast("Guardian Intelligence could not be loaded", "error");
    console.error(error);
  }
}

intelligenceEl("intelligence-refresh")?.addEventListener("click", async event => {
  event.currentTarget.disabled = true;
  await loadIntelligence();
  event.currentTarget.disabled = false;
  DG.toast("Guardian Intelligence refreshed", "success");
});

DG.subscribe(() => {
  if (document.visibilityState === "visible") loadIntelligence();
});

loadIntelligence();
window.setInterval(loadIntelligence, 4000);
