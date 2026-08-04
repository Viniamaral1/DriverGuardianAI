const list = document.getElementById("metrics-report-list");
const content = document.getElementById("metrics-content");
const title = document.getElementById("metrics-title");
const jsonLink = document.getElementById("metrics-json");
const htmlLink = document.getElementById("metrics-html");
const pdfLink = document.getElementById("metrics-pdf");

const percent = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
const number = (value, digits = 3) => Number(value || 0).toFixed(digits);

function duration(seconds) {
  const total = Math.round(Number(seconds || 0));
  return `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, "0")}s`;
}

async function selectReport(report, button) {
  document.querySelectorAll(".metrics-report-item").forEach(item => item.classList.remove("active"));
  button.classList.add("active");
  title.textContent = report.name;
  content.className = "metrics-content";
  content.innerHTML = `<div class="empty-state">Loading metrics…</div>`;

  jsonLink.href = `/api/reports/${report.id}/data`;
  jsonLink.download = `${report.id}.json`;
  jsonLink.classList.remove("hidden");

  if (report.html_file) {
    htmlLink.href = `/api/reports/${report.id}/view`;
    htmlLink.classList.remove("hidden");
  } else {
    htmlLink.classList.add("hidden");
  }

  pdfLink.href = `/api/reports/${report.id}/pdf`;
  pdfLink.classList.remove("hidden");

  try {
    const response = await fetch(`/api/reports/${report.id}/data`);
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    renderMetrics(data);
  } catch (error) {
    content.innerHTML = `<div class="empty-state">This report could not be loaded.</div>`;
    console.error(error);
  }
}

function renderMetrics(data) {
  const states = data.state_percentages || {};
  const stateBars = Object.entries(states).map(([state, value]) => `
    <article class="state-metric">
      <div><span>${state.replaceAll("_", " ")}</span><b>${Number(value).toFixed(1)}%</b></div>
      <div class="meter"><i style="width:${Math.min(100, Number(value))}%"></i></div>
    </article>
  `).join("");

  content.innerHTML = `
    <div class="metrics-summary-grid">
      <article><span>Duration</span><strong>${duration(data.duration_seconds)}</strong></article>
      <article><span>Maximum risk</span><strong>${percent(data.maximum_smoothed_probability)}</strong></article>
      <article><span>Alerts</span><strong>${data.alert_count || 0}</strong></article>
      <article><span>Dominant signal</span><strong>${data.dominant_risk_signal || "unknown"}</strong></article>
    </div>

    <div class="metrics-section">
      <div class="panel-head"><div><p class="eyebrow">RISK</p><h2>Probability metrics</h2></div></div>
      <div class="metric-table">
        <div><span>Average smoothed risk</span><b>${percent(data.average_smoothed_probability)}</b></div>
        <div><span>Maximum smoothed risk</span><b>${percent(data.maximum_smoothed_probability)}</b></div>
        <div><span>Average model probability</span><b>${percent(data.average_raw_model_probability)}</b></div>
        <div><span>Maximum decision probability</span><b>${percent(data.maximum_decision_probability)}</b></div>
        <div><span>Warning episodes</span><b>${data.warning_episodes || 0}</b></div>
        <div><span>Critical episodes</span><b>${data.critical_episodes || 0}</b></div>
      </div>
    </div>

    <div class="metrics-section">
      <div class="panel-head"><div><p class="eyebrow">SIGNALS</p><h2>Behavioural measurements</h2></div></div>
      <div class="metric-table">
        <div><span>Baseline EAR</span><b>${number(data.baseline_ear)}</b></div>
        <div><span>Average EAR</span><b>${number(data.average_ear)}</b></div>
        <div><span>Minimum EAR</span><b>${number(data.minimum_ear)}</b></div>
        <div><span>Maximum yawn score</span><b>${number(data.maximum_yawn_score)}</b></div>
        <div><span>Baseline head tilt</span><b>${number(data.baseline_tilt, 2)}°</b></div>
        <div><span>Maximum head tilt</span><b>${number(data.maximum_head_tilt, 2)}°</b></div>
      </div>
    </div>

    <div class="metrics-section">
      <div class="panel-head"><div><p class="eyebrow">TIMELINE</p><h2>Time by state</h2></div></div>
      <div class="state-metrics">${stateBars || "<p class='muted'>No state timeline was recorded.</p>"}</div>
    </div>

    <div class="metrics-privacy-note">
      <b>Export note</b>
      <span>The PDF contains metrics, alerts and summaries only. Raw camera frames are not included.</span>
    </div>
  `;
}

fetch("/api/reports")
  .then(response => response.json())
  .then(({reports}) => {
    document.getElementById("metrics-report-count").textContent = reports.length;
    document.getElementById("overview-session-count").textContent = reports.length;
    document.getElementById("overview-alert-count").textContent =
      reports.reduce((sum, report) => sum + Number(report.alert_count || 0), 0);
    document.getElementById("overview-max-risk").textContent =
      `${(Math.max(0, ...reports.map(report => Number(report.maximum_risk || 0))) * 100).toFixed(0)}%`;
    if (!reports.length) {
      list.innerHTML = `<div class="empty-state">No generated sessions are available yet.</div>`;
      return;
    }

    list.innerHTML = "";
    reports.forEach((report, index) => {
      const button = document.createElement("button");
      button.className = "metrics-report-item";
      button.innerHTML = `
        <span class="report-item-icon">⌁</span>
        <div><b>${report.name}</b><span>${duration(report.duration_seconds)} · ${report.alert_count} alerts</span></div>
        <strong>${percent(report.maximum_risk)}</strong>
      `;
      button.addEventListener("click", () => selectReport(report, button));
      list.appendChild(button);
      if (index === 0) selectReport(report, button);
    });
  })
  .catch(error => {
    list.innerHTML = `<div class="empty-state">Metrics could not be loaded.</div>`;
    console.error(error);
  });
