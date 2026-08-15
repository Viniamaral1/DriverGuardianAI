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
  const perception = snapshot.perception_confidence || {};
  const predictive = snapshot.predictive_guardian || {};
  const passportValidation = snapshot.passport_validation || {};

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
      `Stable: ${intelligenceLabel(autoOcclusion.value || "unknown")} · ` +
      `${intelligencePercent(autoOcclusion.confidence || 0)} · ` +
      `Raw: ${intelligenceLabel(autoOcclusion.raw_value || "unknown")} · ` +
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

  const perceptionState = String(perception.state || "standby").toLowerCase();
  const perceptionBadge = intelligenceEl("v86-perception-badge");
  perceptionBadge.textContent = perceptionState.toUpperCase();
  perceptionBadge.className = `badge ${
    perceptionState === "trusted" ? "success" :
    perceptionState === "degraded" ? "warning" :
    perceptionState === "insufficient" ? "danger" : ""
  }`;
  intelligenceEl("v86-perception-score").textContent =
    intelligencePercent(perception.score || 0);
  intelligenceEl("v86-perception-state").textContent =
    intelligenceLabel(perception.state || "standby");
  intelligenceEl("v86-perception-policy").textContent =
    intelligenceLabel(perception.observation_mode || "standby");
  intelligenceEl("v86-perception-absence").textContent =
    perception.absence_interpretation || "No live visual observation.";
  intelligenceEl("v86-perception-eyes").textContent =
    intelligencePercent(perception.components?.eye_visibility || 0);
  intelligenceEl("v86-perception-eye-note").textContent =
    perception.can_trust_absence
      ? "Eye-region non-detection can be interpreted normally."
      : "Missing eye cues may reflect limited visibility.";
  intelligenceEl("v86-perception-frame").textContent =
    intelligencePercent(perception.components?.image_quality || 0);
  intelligenceEl("v86-perception-frame-note").textContent =
    environment.summary || "No image-quality assessment is available.";
  intelligenceEl("v86-perception-summary").textContent =
    perception.summary || "No perception-confidence assessment is available.";
  intelligenceEl("v86-perception-boundary").textContent =
    perception.safety_boundary ||
    "Perception Confidence does not alter fatigue probability or alerts.";
  const perceptionReasons = perception.reason_codes || [];
  intelligenceEl("v86-perception-reasons").innerHTML = perceptionReasons.length
    ? perceptionReasons.map(item => `
      <span class="v86-reason ${item.severity || "info"}">
        <b>${intelligenceLabel(item.code || "reason")}</b>
        <small>${item.message || ""}</small>
      </span>`).join("")
    : `<span class="v86-reason success"><b>Observation clear</b><small>No current perception limitation was identified.</small></span>`;

  renderV89PredictiveGuardian(predictive, passportValidation);

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

function renderV89ForecastChart(projection) {
  const svg = intelligenceEl("v89-forecast-chart");
  if (!svg) return;
  const points = Array.isArray(projection) ? projection : [];
  if (!points.length) {
    svg.innerHTML = `<text x="20" y="90" class="v81-axis-label">No forecast projection available.</text>`;
    return;
  }

  const width = 620, height = 180;
  const left = 42, right = 18, top = 18, bottom = 30;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const maxMinute = Math.max(15, ...points.map(p => Number(p.minute || 0)));
  const x = minute => left + (Number(minute || 0) / maxMinute) * plotW;
  const y = risk => top + (1 - Math.max(0, Math.min(1, Number(risk || 0)))) * plotH;

  let markup = "";
  [0, .25, .5, .75, 1].forEach(value => {
    const yy = y(value);
    markup += `<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" class="v81-grid-line"></line>`;
    markup += `<text x="3" y="${yy+4}" class="v81-axis-label">${Math.round(value*100)}%</text>`;
  });
  const thresholdY = y(.65);
  markup += `<line x1="${left}" y1="${thresholdY}" x2="${width-right}" y2="${thresholdY}" class="v89-threshold"></line>`;
  markup += `<text x="${width-right-88}" y="${thresholdY-5}" class="v89-threshold-label">elevated 65%</text>`;

  const path = points.map((p, i) =>
    `${i ? "L" : "M"} ${x(p.minute).toFixed(1)} ${y(p.risk).toFixed(1)}`
  ).join(" ");
  markup += `<path d="${path}" class="v89-forecast-line"></path>`;
  points.forEach(p => {
    markup += `<circle cx="${x(p.minute)}" cy="${y(p.risk)}" r="5" class="v89-forecast-point"></circle>`;
    markup += `<text x="${x(p.minute)-8}" y="${height-8}" class="v81-axis-label">${p.minute}m</text>`;
  });
  svg.innerHTML = markup;
}

function renderV89PredictiveGuardian(predictive, passportValidation) {
  const status = String(predictive.status || "standby").toLowerCase();
  const direction = String(
    predictive.direction || predictive.forecast_state || "uncertain"
  ).toLowerCase();
  const badge = intelligenceEl("v89-predictive-status");
  badge.textContent = status === "withheld"
    ? "WITHHELD"
    : direction.toUpperCase();
  badge.className = `badge ${
    status === "withheld" ? "danger" :
    direction === "rising" ? "warning" :
    direction === "falling" ? "success" :
    direction === "stable" ? "success" : ""
  }`;

  intelligenceEl("v89-direction").textContent =
    intelligenceLabel(direction);
  intelligenceEl("v89-horizon").textContent =
    predictive.horizon || "No forecast";
  intelligenceEl("v89-confidence").textContent =
    intelligencePercent(predictive.confidence || 0);
  intelligenceEl("v89-forecast-risk").textContent =
    intelligencePercent(predictive.forecast_risk || 0);

  const timeTo = predictive.time_to_elevated_minutes;
  intelligenceEl("v89-time-to-risk").textContent =
    timeTo !== null && timeTo !== undefined
      ? Number(timeTo) < 1
        ? `Estimated elevated-risk window ~${Math.max(1, Math.round(Number(timeTo) * 60))} sec`
        : `Estimated elevated-risk window ~${Number(timeTo).toFixed(1)} min`
      : "No justified escalation-time estimate";

  const historical = predictive.historical_pattern || {};
  intelligenceEl("v89-history-count").textContent =
    `${historical.session_count || 0} sessions`;
  const medianElevatedSeconds = historical.median_first_elevated_seconds;
  const medianElevatedMinutes = historical.median_first_elevated_minutes;
  intelligenceEl("v89-history-window").textContent =
    medianElevatedSeconds !== null && medianElevatedSeconds !== undefined
      ? Number(medianElevatedSeconds) < 60
        ? `Median elevated-risk timing ${Math.round(Number(medianElevatedSeconds))} sec`
        : `Median elevated-risk timing ${Number(medianElevatedMinutes).toFixed(1)} min`
      : "No historical escalation window yet";

  intelligenceEl("v89-passport-trust").textContent =
    passportValidation?.label
      ? `Passport: ${passportValidation.label}`
      : "Passport trust unavailable";
  intelligenceEl("v89-summary").textContent =
    predictive.summary || "Predictive Guardian is waiting for evidence.";
  intelligenceEl("v89-action").textContent =
    predictive.recommended_action
      || "Live Guardian monitoring remains authoritative.";
  intelligenceEl("v89-boundary").textContent =
    predictive.safety_boundary
      || "Predictive Guardian does not replace the trained model or existing alerts.";

  const withheld = intelligenceEl("v89-withheld");
  const reasons = predictive.withheld_reasons || [];
  if (reasons.length) {
    withheld.hidden = false;
    withheld.innerHTML = `<b>Forecast withheld</b>${reasons.map(reason =>
      `<span>${reason}</span>`
    ).join("")}`;
  } else {
    withheld.hidden = true;
    withheld.innerHTML = "";
  }

  const factors = predictive.factors || [];
  intelligenceEl("v89-factors").innerHTML = factors.length
    ? factors.map(item => `
      <div class="v89-factor">
        <div><span>${item.label}</span><b>${intelligencePercent(item.value || 0)}</b></div>
        <small>${item.detail || ""}</small>
      </div>
    `).join("")
    : `<div class="empty-state">Waiting for predictive evidence.</div>`;

  renderV89ForecastChart(predictive.projection || []);
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
let v841EvidenceGroup = null;
let v841EvidenceFrameIndex = 0;
let v841EvidencePlaybackTimer = null;

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

  const eventMarkers = Array.isArray(options.eventMarkers)
    ? options.eventMarkers
    : [];
  eventMarkers.forEach(event => {
    const index = Math.max(
      0,
      Math.min(rows.length - 1, Number(event.index || 0))
    );
    if (!rows.length) return;
    const x = xFor(rows[index], index);
    const type = String(event.type || "event")
      .replace(/[^a-z0-9_-]/gi, "")
      .toLowerCase();
    markup += `
      <g class="v841-chart-event v841-chart-event-${type}"
        data-replay-index="${index}" tabindex="0"
        role="button" aria-label="Jump to ${type} event">
        <circle cx="${x}" cy="${top + 8}" r="6"></circle>
        <line x1="${x}" y1="${top + 15}" x2="${x}" y2="${height-bottom}"
          class="v841-chart-event-line"></line>
      </g>`;
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
          <small>${session.sample_count} samples · ${Number(session.near_miss_count || 0)} near-miss · ${session.active ? "LIVE" : "saved"}</small>
        </div>
      </button>
    `).join("")
    : `<div class="empty-state">No Decision Memory sessions yet.</div>`;

  container.querySelectorAll("[data-memory-id]").forEach(button => {
    button.addEventListener("click", () => v81LoadMemorySession(button.dataset.memoryId));
  });
}

function v85NearMissLabel(type) {
  const labels = {
    near_alert: "Near-alert",
    recovery: "Recovery",
    escalation: "Escalation",
    weak_signal_accumulation: "Weak-signal accumulation",
    repeated_uncertainty: "Repeated uncertainty",
    baseline_drift: "Baseline drift",
  };
  return labels[type] || intelligenceLabel(type || "Near miss");
}

function v85RenderNearMissMemory(activeIndex = null) {
  const memory = v81SelectedMemory?.near_miss_memory || {};
  const episodes = memory.episodes || [];
  const count = intelligenceEl("v85-near-miss-count");
  const summary = intelligenceEl("v85-near-miss-summary");
  const list = intelligenceEl("v85-near-miss-list");
  if (!count || !summary || !list) return;

  count.textContent =
    `${Number(memory.episode_count || 0)} episode${Number(memory.episode_count || 0) === 1 ? "" : "s"}`;
  summary.textContent =
    memory.summary || "No Near-Miss analysis is available for this session.";

  if (!episodes.length) {
    list.innerHTML = `
      <div class="empty-state">
        No Near-Miss pattern met the current retrospective criteria.
      </div>`;
    return;
  }

  list.innerHTML = episodes.map((episode, order) => {
    const peakIndex = Number(episode.peak_index || 0);
    const active = activeIndex !== null
      && Math.abs(Number(activeIndex) - peakIndex) <= 1;
    const contributors = (episode.contributing_evidence || [])
      .slice(0, 3)
      .map(item => `${item.label} ${intelligencePercent(item.peak)}`)
      .join(" · ");
    const peakTime = episode.peak_timestamp
      ? new Date(episode.peak_timestamp).toLocaleTimeString("en-GB")
      : `Sample ${peakIndex + 1}`;
    const evidence = episode.visual_evidence_available
      ? " · Visual evidence nearby"
      : "";
    const repeat = episode.repeated_in_session
      ? ` · repeated ${episode.same_type_episode_count}×`
      : "";

    return `
      <button type="button"
        class="v85-near-miss-card${active ? " is-active" : ""}"
        data-near-miss-index="${peakIndex}"
        data-near-miss-order="${order}">
        <div class="v85-near-miss-card-head">
          <span>${v85NearMissLabel(episode.type)}</span>
          <b>${intelligencePercent(episode.peak_risk)}</b>
        </div>
        <strong>${episode.title || "Near-Miss episode"}</strong>
        <small>${peakTime} · ${Number(episode.duration_seconds || 0)}s · confidence ${intelligencePercent(episode.confidence)}${repeat}${evidence}</small>
        <p>${episode.explanation || ""}</p>
        ${contributors ? `<em>${contributors}</em>` : ""}
      </button>`;
  }).join("");
}

function v85NearMissMarkers() {
  return (v81SelectedMemory?.near_miss_memory?.episodes || []).map(
    episode => ({
      index: Number(episode.peak_index || 0),
      type: "near_miss",
      title: episode.title || "Near-Miss episode",
      detail: episode.explanation || "",
      level: episode.type || "near_miss",
    })
  );
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

    v841StopEvidencePlayback();
    v841EvidenceGroup = null;
    v841EvidenceFrameIndex = 0;
    v85RenderNearMissMemory(Number(slider.value));
    v81RenderReplay(Number(slider.value));
  } catch (error) {
    DG.toast("Decision Memory session could not be loaded", "error");
    console.error(error);
  }
}

function v841StopEvidencePlayback() {
  if (v841EvidencePlaybackTimer) {
    window.clearInterval(v841EvidencePlaybackTimer);
    v841EvidencePlaybackTimer = null;
  }
}

function v841JumpToReplayIndex(index) {
  const samples = v81SelectedMemory?.samples || [];
  if (!samples.length) return;
  const safeIndex = Math.max(
    0,
    Math.min(samples.length - 1, Number(index || 0))
  );
  const slider = intelligenceEl("v81-replay-slider");
  if (slider) slider.value = safeIndex;
  v81RenderReplay(safeIndex);
}

function v842ReplayEventAtIndex(index) {
  return (v81SelectedMemory?.events || []).filter(
    event => Number(event.index || 0) === Number(index)
  );
}

function v842ReplayIndexFromPointer(event) {
  const svg = intelligenceEl("v81-replay-chart");
  const samples = v81SelectedMemory?.samples || [];
  if (!svg || !samples.length) return null;

  const rect = svg.getBoundingClientRect();
  const viewWidth = svg.viewBox?.baseVal?.width || 900;
  const pointerX =
    ((event.clientX - rect.left) / Math.max(1, rect.width)) * viewWidth;
  const plotLeft = 38;
  const plotRight = 12;
  const plotWidth = viewWidth - plotLeft - plotRight;
  const ratio = Math.max(
    0,
    Math.min(1, (pointerX - plotLeft) / Math.max(1, plotWidth))
  );
  return Math.round(ratio * Math.max(0, samples.length - 1));
}

function v842RenderReplayTooltip(event) {
  const tooltip = intelligenceEl("v842-replay-tooltip");
  const wrap = tooltip?.parentElement;
  const samples = v81SelectedMemory?.samples || [];
  if (!tooltip || !wrap || !samples.length) return;

  const index = v842ReplayIndexFromPointer(event);
  if (index === null) return;
  const row = samples[index];
  const events = [
    ...v842ReplayEventAtIndex(index),
    ...v85NearMissMarkers().filter(
      item => Number(item.index || 0) === Number(index)
    ),
  ];
  const eventMarkup = events.length
    ? `<div class="v842-tooltip-events">${events.map(item =>
        `<b>${String(item.title || item.type || "Event")}</b>`
      ).join("")}</div>`
    : "";

  tooltip.innerHTML = `
    <strong>${new Date(row.timestamp).toLocaleTimeString("en-GB")}</strong>
    <div><span>Risk</span><b>${intelligencePercent(row.advisory_risk)}</b></div>
    <div><span>Confidence</span><b>${intelligencePercent(row.decision_confidence)}</b></div>
    ${row.perception_state ? `<div><span>Perception</span><b>${intelligenceLabel(row.perception_state)} · ${intelligencePercent(row.perception_confidence)}</b></div>` : ""}
    <div><span>EAR</span><b>${Number(row.ear || 0).toFixed(3)}</b></div>
    <div><span>Baseline</span><b>${Number(row.baseline_ear || 0).toFixed(3)}</b></div>
    ${eventMarkup}
  `;

  const wrapRect = wrap.getBoundingClientRect();
  const x = event.clientX - wrapRect.left;
  const y = event.clientY - wrapRect.top;
  const maxLeft = Math.max(8, wrapRect.width - 230);
  tooltip.style.left = `${Math.max(8, Math.min(maxLeft, x + 12))}px`;
  tooltip.style.top = `${Math.max(8, y - 18)}px`;
  tooltip.hidden = false;
}

function v842HideReplayTooltip() {
  const tooltip = intelligenceEl("v842-replay-tooltip");
  if (tooltip) tooltip.hidden = true;
}

function v842PreloadEvidence(files) {
  (files || []).forEach(filename => {
    const image = new Image();
    image.decoding = "async";
    image.src = v841EvidenceUrl(filename);
  });
}

function v841EvidenceUrl(filename) {
  if (!v81SelectedMemory?.id || !filename) return "";
  return `/api/intelligence/memory/${encodeURIComponent(v81SelectedMemory.id)}/evidence/${encodeURIComponent(filename)}`;
}

function v841RenderEvidencePlayer() {
  const player = intelligenceEl("v841-evidence-player");
  const image = intelligenceEl("v841-player-image");
  const empty = intelligenceEl("v841-player-empty");
  const position = intelligenceEl("v841-player-position");
  const prev = intelligenceEl("v841-player-prev");
  const play = intelligenceEl("v841-player-play");
  const pause = intelligenceEl("v841-player-pause");
  const next = intelligenceEl("v841-player-next");
  if (!player || !image || !empty || !position) return;

  const files = v841EvidenceGroup?.files || [];
  if (!files.length) {
    v841StopEvidencePlayback();
    v841EvidenceGroup = null;
    v841EvidenceFrameIndex = 0;
    player.classList.add("is-empty");
    image.removeAttribute("src");
    image.hidden = true;
    empty.hidden = false;
    position.textContent = "0 / 0";
    [prev, play, pause, next].forEach(button => {
      if (button) button.disabled = true;
    });
    return;
  }

  v841EvidenceFrameIndex = Math.max(
    0,
    Math.min(files.length - 1, v841EvidenceFrameIndex)
  );
  player.classList.remove("is-empty");
  image.hidden = false;
  empty.hidden = true;
  image.src = v841EvidenceUrl(files[v841EvidenceFrameIndex]);
  position.textContent =
    `${v841EvidenceFrameIndex + 1} / ${files.length}`;

  if (prev) prev.disabled = files.length < 2;
  if (play) play.disabled = files.length < 2;
  if (pause) pause.disabled = files.length < 2;
  if (next) next.disabled = files.length < 2;

  document.querySelectorAll(".v84-evidence-frame").forEach((frame, index) => {
    frame.classList.toggle("is-current", index === v841EvidenceFrameIndex);
  });
}

function v841SetEvidenceFrame(index) {
  const files = v841EvidenceGroup?.files || [];
  if (!files.length) return;
  v841EvidenceFrameIndex = (
    (Number(index) % files.length) + files.length
  ) % files.length;
  v841RenderEvidencePlayer();
}

function v841PlayEvidence() {
  const files = v841EvidenceGroup?.files || [];
  if (files.length < 2) return;
  v841StopEvidencePlayback();
  v841EvidencePlaybackTimer = window.setInterval(() => {
    if (v841EvidenceFrameIndex >= files.length - 1) {
      v841StopEvidencePlayback();
      return;
    }
    v841SetEvidenceFrame(v841EvidenceFrameIndex + 1);
  }, 600);
}

function v84EvidenceForReplayIndex(index) {
  const groups = v81SelectedMemory?.visual_evidence?.events || [];
  let best = null;
  let bestDistance = Infinity;
  for (const group of groups) {
    const distance = Math.abs(Number(group.index || 0) - Number(index || 0));
    if (distance < bestDistance) {
      best = group;
      bestDistance = distance;
    }
  }
  return bestDistance <= 2 ? best : null;
}

function v84RenderVisualEvidence(index) {
  const strip = intelligenceEl("v84-evidence-strip");
  const status = intelligenceEl("v84-evidence-status");
  const deleteButton = intelligenceEl("v84-delete-evidence");
  if (!strip || !status) return;

  const visual = v81SelectedMemory?.visual_evidence || {};
  if (deleteButton) deleteButton.disabled = !visual.available;

  const group = v84EvidenceForReplayIndex(index);
  if (!group || !(group.files || []).length) {
    status.textContent = visual.available
      ? "No captured evidence near this replay point"
      : "No visual evidence captured for this session";
    strip.innerHTML = `<div class="empty-state">${
      visual.available
        ? "Move the replay slider near a captured event."
        : "Enable Visual Evidence in Settings before Monitoring."
    }</div>`;
    v841EvidenceGroup = null;
    v841RenderEvidencePlayer();
    return;
  }

  const sameGroup =
    v841EvidenceGroup
    && Number(v841EvidenceGroup.index) === Number(group.index)
    && String(v841EvidenceGroup.type) === String(group.type);

  if (!sameGroup) {
    v841StopEvidencePlayback();
    v841EvidenceFrameIndex = 0;
  }
  v841EvidenceGroup = group;
  if (!sameGroup) {
    v842PreloadEvidence(group.files || []);
  }

  status.textContent =
    `${group.title || group.type || "Event"} · ${group.files.length} image${group.files.length === 1 ? "" : "s"}`;

  const sessionId = encodeURIComponent(v81SelectedMemory.id);
  strip.innerHTML = group.files.map((filename, order) => `
    <figure class="v84-evidence-frame${order === v841EvidenceFrameIndex ? " is-current" : ""}"
      data-evidence-frame="${order}" tabindex="0" role="button"
      aria-label="Show evidence frame ${order + 1}">
      <img src="/api/intelligence/memory/${sessionId}/evidence/${encodeURIComponent(filename)}"
        alt="Guardian event evidence frame ${order + 1}" loading="lazy">
      <figcaption>${order === group.files.length - 1 ? "Event frame" : "Pre-event"}</figcaption>
    </figure>
  `).join("");

  v841RenderEvidencePlayer();
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
  intelligenceEl("v86-replay-perception").textContent = row.perception_state
    ? `${intelligenceLabel(row.perception_state)} · ${intelligencePercent(row.perception_confidence)}`
    : "Legacy session";
  intelligenceEl("v81-replay-ear").textContent =
    Number(row.ear || 0).toFixed(3);
  intelligenceEl("v81-replay-evidence").textContent =
    row.dominant_evidence || "—";
  intelligenceEl("v81-replay-context").textContent =
    [row.weather, row.road_condition, row.external_light]
      .filter(Boolean).join(" · ");
  intelligenceEl("v81-replay-action").textContent =
    row.recommended_action || "No recommendation stored.";
  v84RenderVisualEvidence(index);
  v85RenderNearMissMemory(index);

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
      eventMarkers: [
        ...(v81SelectedMemory?.events || []),
        ...v85NearMissMarkers(),
      ],
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

intelligenceEl("v85-near-miss-list")?.addEventListener("click", event => {
  const card = event.target.closest("[data-near-miss-index]");
  if (!card) return;
  v841JumpToReplayIndex(Number(card.dataset.nearMissIndex));
});

intelligenceEl("v81-replay-slider")?.addEventListener("input", event => {
  v841JumpToReplayIndex(Number(event.currentTarget.value));
});


intelligenceEl("v81-replay-chart")?.addEventListener("click", event => {
  const marker = event.target.closest("[data-replay-index]");
  if (marker) {
    v841JumpToReplayIndex(Number(marker.dataset.replayIndex));
    return;
  }
  const index = v842ReplayIndexFromPointer(event);
  if (index !== null) v841JumpToReplayIndex(index);
});

intelligenceEl("v81-replay-chart")?.addEventListener("pointermove", v842RenderReplayTooltip);
intelligenceEl("v81-replay-chart")?.addEventListener("pointerleave", v842HideReplayTooltip);

intelligenceEl("v81-replay-chart")?.addEventListener("keydown", event => {
  const marker = event.target.closest("[data-replay-index]");
  if (!marker || !["Enter", " "].includes(event.key)) return;
  event.preventDefault();
  v841JumpToReplayIndex(Number(marker.dataset.replayIndex));
});

intelligenceEl("v84-evidence-strip")?.addEventListener("click", event => {
  const frame = event.target.closest("[data-evidence-frame]");
  if (!frame) return;
  v841StopEvidencePlayback();
  v841SetEvidenceFrame(Number(frame.dataset.evidenceFrame));
});

intelligenceEl("v84-evidence-strip")?.addEventListener("keydown", event => {
  const frame = event.target.closest("[data-evidence-frame]");
  if (!frame || !["Enter", " "].includes(event.key)) return;
  event.preventDefault();
  v841StopEvidencePlayback();
  v841SetEvidenceFrame(Number(frame.dataset.evidenceFrame));
});

intelligenceEl("v841-player-prev")?.addEventListener("click", () => {
  v841StopEvidencePlayback();
  v841SetEvidenceFrame(v841EvidenceFrameIndex - 1);
});
intelligenceEl("v841-player-play")?.addEventListener("click", v841PlayEvidence);
intelligenceEl("v841-player-pause")?.addEventListener("click", v841StopEvidencePlayback);
intelligenceEl("v841-player-next")?.addEventListener("click", () => {
  v841StopEvidencePlayback();
  v841SetEvidenceFrame(v841EvidenceFrameIndex + 1);
});

intelligenceEl("v84-delete-evidence")?.addEventListener("click", async () => {
  if (!v81SelectedMemory?.id) return;
  const confirmed = window.confirm(
    "Delete all local visual evidence for this session? Decision Memory metrics will be preserved."
  );
  if (!confirmed) return;
  try {
    const response = await fetch(
      `/api/intelligence/memory/${encodeURIComponent(v81SelectedMemory.id)}/evidence`,
      {method: "DELETE"}
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await v81LoadMemorySession(v81SelectedMemory.id);
  } catch (error) {
    console.error("Could not delete visual evidence", error);
  }
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
