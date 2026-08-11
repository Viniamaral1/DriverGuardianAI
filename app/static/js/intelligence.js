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

  const autoOcclusion = context.automatic_occlusion || {};
  const resolvedSource =
    context.occlusion_source ||
    context.sources?.occlusion?.source ||
    "unknown";
  const detail = intelligenceEl("context-occlusion-detail");
  if (detail) {
    detail.textContent =
      `Auto: ${intelligenceLabel(autoOcclusion.value || "unknown")} · ` +
      `${intelligencePercent(autoOcclusion.confidence || 0)} · ` +
      `Source: ${resolvedSource}`;
  }
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

  renderV8DecisionEngine(snapshot.decision_engine || {});
  v81AddLivePoint(snapshot);
  v81RenderMemoryStatus(snapshot.decision_memory || {});
}

function renderV8DecisionEngine(engine) {
  engine = engine || {};
  const confidence = engine.confidence || {};
  const caution = engine.context_caution || {};
  const evidence = engine.evidence || [];
  const timeline = engine.timeline || [];
  const contract = engine.safety_contract || {};

  intelligenceEl("v8-risk-score").textContent =
    intelligencePercent(engine.risk_score || 0);
  intelligenceEl("v8-risk-band").textContent =
    intelligenceLabel(engine.risk_band || "standby");

  intelligenceEl("v8-confidence-score").textContent =
    intelligencePercent(confidence.score || 0);
  intelligenceEl("v8-confidence-level").textContent =
    intelligenceLabel(confidence.level || "standby");
  intelligenceEl("v8-confidence-summary").textContent =
    confidence.summary || "No live confidence estimate.";

  const badge = intelligenceEl("v8-engine-confidence-badge");
  badge.textContent = intelligenceLabel(confidence.level || "standby");
  badge.className = `badge ${
    confidence.level === "high" ? "success" :
    confidence.level === "limited" ? "warning" : ""
  }`;

  const reasons = caution.reasons || [];
  intelligenceEl("v8-context-level").textContent =
    intelligenceLabel(caution.level || "normal");
  intelligenceEl("v8-context-count").textContent =
    `${reasons.length} factor${reasons.length === 1 ? "" : "s"}`;
  intelligenceEl("v8-context-summary").textContent =
    caution.summary || "No additional context caution.";

  intelligenceEl("v8-evidence-ledger").innerHTML = evidence.length
    ? evidence.map(item => `
      <div class="v8-evidence-row">
        <div>
          <span>${item.label}</span>
          <b>${intelligencePercent(item.value || 0)}</b>
          <small>${intelligenceLabel(item.role || "supporting")}</small>
        </div>
        <div class="v8-evidence-meter">
          <i style="width:${Math.max(0, Math.min(100, Number(item.contribution || 0) * 100))}%"></i>
        </div>
        <p>${item.explanation || ""}</p>
      </div>
    `).join("")
    : `<div class="empty-state">No live evidence is currently active.</div>`;

  intelligenceEl("v8-explanation").textContent =
    engine.explanation || "The advisory engine is waiting for live signals.";
  intelligenceEl("v8-action").textContent =
    engine.action || "Start Monitoring when ready.";
  intelligenceEl("v8-safety-contract").textContent =
    contract.description || "V8 remains advisory.";

  intelligenceEl("v8-decision-timeline").innerHTML = timeline.length
    ? timeline.slice(0, 8).map(item => `
      <div class="v8-timeline-item">
        <time>${item.time || "—"}</time>
        <b>${intelligenceLabel(item.risk_band || "unknown")}</b>
        <span>${intelligencePercent(item.score || 0)}</span>
        <small>${item.reason || ""} · ${intelligenceLabel(item.confidence || "unknown")} confidence</small>
      </div>
    `).join("")
    : `<div class="empty-state">Risk-band changes will appear here.</div>`;
}


const v81LiveSeries = [];
let v81MemorySessions = [];
let v81SelectedMemory = null;

function v81Clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, Number(value || 0)));
}

function v81Path(points, xFor, yFor) {
  if (!points.length) return "";
  return points.map((point, index) =>
    `${index ? "L" : "M"} ${xFor(point, index).toFixed(1)} ${yFor(point, index).toFixed(1)}`
  ).join(" ");
}

function v81RenderLineChart(svg, series, options = {}) {
  if (!svg) return;
  const width = options.width || 640;
  const height = options.height || 190;
  const left = 38, right = 12, top = 12, bottom = 25;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const rows = options.rows || [];
  const yMin = Number(options.yMin ?? 0);
  const yMax = Number(options.yMax ?? 1);
  const count = Math.max(2, rows.length);

  const xFor = (_, index) => left + (index / (count - 1)) * plotWidth;
  const yForValue = value => {
    const ratio = (Number(value || 0) - yMin) / Math.max(0.000001, yMax - yMin);
    return top + (1 - v81Clamp(ratio)) * plotHeight;
  };

  let markup = "";
  for (let i = 0; i <= 4; i++) {
    const y = top + (i / 4) * plotHeight;
    const value = yMax - ((yMax - yMin) * i / 4);
    markup += `
      <line x1="${left}" y1="${y}" x2="${width-right}" y2="${y}"
        class="v81-grid-line"></line>
      <text x="3" y="${y + 4}" class="v81-axis-label">
        ${options.formatY ? options.formatY(value) : value.toFixed(1)}
      </text>`;
  }

  series.forEach((item, seriesIndex) => {
    const valid = rows.map((row, index) => ({
      value: Number(item.value(row) || 0),
      index,
    }));
    const path = v81Path(
      valid,
      point => xFor(point, point.index),
      point => yForValue(point.value)
    );
    markup += `<path d="${path}" class="v81-series v81-series-${seriesIndex + 1}"></path>`;
  });

  if (options.markerIndex !== undefined && rows.length) {
    const index = Math.max(0, Math.min(rows.length - 1, options.markerIndex));
    const x = xFor(rows[index], index);
    markup += `<line x1="${x}" y1="${top}" x2="${x}" y2="${height-bottom}"
      class="v81-marker"></line>`;
  }

  svg.innerHTML = markup;
}

function v81AddLivePoint(snapshot) {
  const engine = snapshot.decision_engine || {};
  const evidence = Object.fromEntries(
    (engine.evidence || []).map(item => [item.key, item.value])
  );
  const live = snapshot.live || {};
  const environment = snapshot.environment || {};
  const quality = snapshot.signal_quality || {};
  const baseline = engine.personal_baseline || {};

  v81LiveSeries.push({
    time: Date.now(),
    risk: Number(engine.risk_score || 0),
    confidence: Number(engine.confidence?.score || 0),
    ear: Number(baseline.current_ear || 0),
    baseline: Number(baseline.baseline_ear || 0),
    yawn: Number(evidence.yawn || 0),
    tilt: Number(evidence.head_pose || 0),
    signalQuality: Number(quality.score || 0),
    imageQuality: Number(environment.quality_score || 0),
  });

  while (v81LiveSeries.length > 30) v81LiveSeries.shift();
  v81RenderLiveCharts();
}

function v81RenderLiveCharts() {
  if (!v81LiveSeries.length) return;
  const last = v81LiveSeries[v81LiveSeries.length - 1];

  intelligenceEl("v81-risk-now").textContent =
    intelligencePercent(last.risk);
  intelligenceEl("v81-ear-now").textContent =
    last.ear ? last.ear.toFixed(3) : "—";
  intelligenceEl("v81-behaviour-now").textContent =
    `Yawn ${intelligencePercent(last.yawn)}`;
  intelligenceEl("v81-quality-now").textContent =
    intelligencePercent(last.signalQuality);

  v81RenderLineChart(
    intelligenceEl("v81-risk-chart"),
    [
      {value: row => row.risk},
      {value: row => row.confidence},
    ],
    {
      rows: v81LiveSeries,
      yMin: 0,
      yMax: 1,
      formatY: value => `${Math.round(value * 100)}%`,
    }
  );

  const earValues = v81LiveSeries
    .flatMap(row => [row.ear, row.baseline])
    .filter(value => value > 0);
  const earMin = earValues.length ? Math.max(0, Math.min(...earValues) - 0.04) : 0;
  const earMax = earValues.length ? Math.max(...earValues) + 0.04 : 0.4;

  v81RenderLineChart(
    intelligenceEl("v81-ear-chart"),
    [
      {value: row => row.ear},
      {value: row => row.baseline},
    ],
    {
      rows: v81LiveSeries,
      yMin: earMin,
      yMax: earMax,
      formatY: value => value.toFixed(2),
    }
  );

  v81RenderLineChart(
    intelligenceEl("v81-behaviour-chart"),
    [
      {value: row => row.yawn},
      {value: row => row.tilt},
    ],
    {
      rows: v81LiveSeries,
      yMin: 0,
      yMax: 1,
      formatY: value => `${Math.round(value * 100)}%`,
    }
  );

  v81RenderLineChart(
    intelligenceEl("v81-quality-chart"),
    [
      {value: row => row.signalQuality},
      {value: row => row.imageQuality},
    ],
    {
      rows: v81LiveSeries,
      yMin: 0,
      yMax: 1,
      formatY: value => `${Math.round(value * 100)}%`,
    }
  );
}

function v81RenderMemoryStatus(memory) {
  const badge = intelligenceEl("v81-memory-status");
  if (!badge) return;
  if (memory?.recording) {
    badge.textContent = `RECORDING · ${memory.sample_count || 0}`;
    badge.className = "badge success";
  } else {
    badge.textContent = "MEMORY STANDBY";
    badge.className = "badge";
  }
}

async function v81LoadMemoryList() {
  try {
    const response = await fetch("/api/intelligence/memory", {cache: "no-store"});
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    v81MemorySessions = payload.sessions || [];
    v81RenderMemoryList();
    v81RenderCompareOptions();
  } catch (error) {
    console.error(error);
  }
}

function v81RenderMemoryList() {
  const container = intelligenceEl("v81-memory-list");
  if (!container) return;

  container.innerHTML = v81MemorySessions.length
    ? v81MemorySessions.slice(0, 20).map(session => `
      <button type="button" class="v81-memory-item"
        data-memory-id="${session.id}">
        <div>
          <b>${new Date(session.started_at).toLocaleString("en-GB")}</b>
          <span>${session.driver_profile || "Guest"}</span>
        </div>
        <div>
          <strong>${intelligencePercent(session.maximum_advisory_risk)}</strong>
          <small>${session.sample_count} samples · ${session.active ? "LIVE" : "saved"}</small>
        </div>
      </button>
    `).join("")
    : `<div class="empty-state">No Decision Memory sessions yet.</div>`;

  container.querySelectorAll("[data-memory-id]").forEach(button => {
    button.addEventListener("click", () => v81LoadMemorySession(button.dataset.memoryId));
  });
}

async function v81LoadMemorySession(sessionId) {
  try {
    const response = await fetch(
      `/api/intelligence/memory/${encodeURIComponent(sessionId)}`,
      {cache: "no-store"}
    );
    if (!response.ok) throw new Error(await response.text());
    v81SelectedMemory = await response.json();

    intelligenceEl("v81-selected-session").textContent =
      new Date(v81SelectedMemory.started_at).toLocaleString("en-GB");

    const slider = intelligenceEl("v81-replay-slider");
    const samples = v81SelectedMemory.samples || [];
    slider.max = Math.max(0, samples.length - 1);
    slider.value = Math.max(0, samples.length - 1);

    const jsonLink = intelligenceEl("v81-export-json");
    const csvLink = intelligenceEl("v81-export-csv");
    jsonLink.href = `/api/intelligence/memory/${encodeURIComponent(sessionId)}/json`;
    csvLink.href = `/api/intelligence/memory/${encodeURIComponent(sessionId)}/csv`;
    jsonLink.classList.remove("disabled");
    csvLink.classList.remove("disabled");

    v81RenderReplay(Number(slider.value));
  } catch (error) {
    DG.toast("Decision Memory session could not be loaded", "error");
    console.error(error);
  }
}

function v81RenderReplay(index) {
  const samples = v81SelectedMemory?.samples || [];
  if (!samples.length) return;
  index = Math.max(0, Math.min(samples.length - 1, Number(index || 0)));
  const row = samples[index];

  intelligenceEl("v81-replay-time").textContent =
    new Date(row.timestamp).toLocaleTimeString("en-GB");
  intelligenceEl("v81-replay-risk").textContent =
    intelligencePercent(row.advisory_risk);
  intelligenceEl("v81-replay-confidence").textContent =
    intelligencePercent(row.decision_confidence);
  intelligenceEl("v81-replay-ear").textContent =
    Number(row.ear || 0).toFixed(3);
  intelligenceEl("v81-replay-evidence").textContent =
    row.dominant_evidence || "—";
  intelligenceEl("v81-replay-context").textContent =
    [row.weather, row.road_condition, row.external_light]
      .filter(Boolean).join(" · ");
  intelligenceEl("v81-replay-action").textContent =
    row.recommended_action || "No recommendation stored.";

  v81RenderLineChart(
    intelligenceEl("v81-replay-chart"),
    [
      {value: item => item.advisory_risk},
      {value: item => item.decision_confidence},
      {value: item => item.raw_model_probability},
    ],
    {
      rows: samples,
      yMin: 0,
      yMax: 1,
      width: 900,
      height: 220,
      markerIndex: index,
      formatY: value => `${Math.round(value * 100)}%`,
    }
  );
}

function v81RenderCompareOptions() {
  const options = v81MemorySessions.map(session =>
    `<option value="${session.id}">
      ${new Date(session.started_at).toLocaleString("en-GB")} · ${session.driver_profile}
    </option>`
  ).join("");

  const a = intelligenceEl("v81-compare-a");
  const b = intelligenceEl("v81-compare-b");
  if (!a || !b) return;
  a.innerHTML = options;
  b.innerHTML = options;

  if (v81MemorySessions.length > 1) {
    a.selectedIndex = 1;
    b.selectedIndex = 0;
  }
}

async function v81CompareSessions() {
  const a = intelligenceEl("v81-compare-a")?.value;
  const b = intelligenceEl("v81-compare-b")?.value;
  if (!a || !b || a === b) {
    DG.toast("Choose two different Decision Memory sessions", "warning");
    return;
  }

  try {
    const response = await fetch(
      `/api/intelligence/memory/compare/${encodeURIComponent(a)}/${encodeURIComponent(b)}`,
      {cache: "no-store"}
    );
    if (!response.ok) throw new Error(await response.text());
    const comparison = await response.json();
    const first = comparison.first || {};
    const second = comparison.second || {};
    const delta = comparison.delta_second_minus_first || {};

    const signedPct = value => {
      const number = Number(value || 0) * 100;
      return `${number >= 0 ? "+" : ""}${number.toFixed(1)}%`;
    };

    intelligenceEl("v81-comparison-result").innerHTML = `
      <div class="v81-compare-card">
        <span>Average risk</span>
        <b>${intelligencePercent(first.summary?.average_advisory_risk)}</b>
        <strong>→ ${intelligencePercent(second.summary?.average_advisory_risk)}</strong>
        <small>${signedPct(delta.average_advisory_risk)}</small>
      </div>
      <div class="v81-compare-card">
        <span>Peak risk</span>
        <b>${intelligencePercent(first.summary?.maximum_advisory_risk)}</b>
        <strong>→ ${intelligencePercent(second.summary?.maximum_advisory_risk)}</strong>
        <small>${signedPct(delta.maximum_advisory_risk)}</small>
      </div>
      <div class="v81-compare-card">
        <span>Confidence</span>
        <b>${intelligencePercent(first.summary?.average_confidence)}</b>
        <strong>→ ${intelligencePercent(second.summary?.average_confidence)}</strong>
        <small>${signedPct(delta.average_confidence)}</small>
      </div>
      <div class="v81-compare-card">
        <span>Average EAR</span>
        <b>${Number(first.summary?.average_ear || 0).toFixed(3)}</b>
        <strong>→ ${Number(second.summary?.average_ear || 0).toFixed(3)}</strong>
        <small>${Number(delta.average_ear || 0) >= 0 ? "+" : ""}${Number(delta.average_ear || 0).toFixed(3)}</small>
      </div>`;
  } catch (error) {
    DG.toast("Decision Memory comparison failed", "error");
    console.error(error);
  }
}

intelligenceEl("v81-replay-slider")?.addEventListener("input", event => {
  v81RenderReplay(Number(event.currentTarget.value));
});

intelligenceEl("v81-memory-refresh")?.addEventListener("click", async () => {
  await v81LoadMemoryList();
  DG.toast("Decision Memory refreshed", "success");
});

intelligenceEl("v81-compare-button")?.addEventListener("click", v81CompareSessions);

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

// Core WebSocket already maintains the global READY/MONITORING state.
// V8.2 intentionally avoids re-fetching the full Intelligence payload on every
// WebSocket frame; the 4-second timer below is enough for advisory charts and
// materially reduces CPU, JSON serialization and Decision Memory disk traffic.

loadIntelligence();
v81LoadMemoryList();
window.setInterval(loadIntelligence, 4000);
window.setInterval(v81LoadMemoryList, 20000);
