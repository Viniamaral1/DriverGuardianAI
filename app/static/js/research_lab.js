const researchEl = id => document.getElementById(id);
const fmt = value => Number(value || 0).toLocaleString("en-GB");
const pct = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
const label = value => String(value ?? "unknown").replaceAll("_", " ");

function bars(items, valueKey = "share") {
  if (!items?.length) return `<div class="empty-state">No values available.</div>`;
  const max = Math.max(...items.map(item => Number(item[valueKey] || 0)), 0.000001);
  return items.map(item => `<div class="research-bar"><div><span>${item.label ?? item.feature}</span><b>${valueKey === "share" ? fmt(item.count) : Number(item[valueKey] || 0).toFixed(4)}</b></div><div class="meter"><i style="width:${Math.min(100, Number(item[valueKey] || 0) / max * 100)}%"></i></div>${valueKey === "share" ? `<small>${pct(item.share)}</small>` : ""}</div>`).join("");
}

function renderLifecycle(status) {
  const lifecycle = status.lifecycle || {};
  const model = lifecycle.model_card || {};
  const protocol = lifecycle.training_protocol || {};
  const decision = lifecycle.live_decision_layer || {};
  const evidence = status.deployment_evidence || {};
  const deployment = status.deployment || {};

  researchEl("research-mode").textContent = String(deployment.mode || "research").toUpperCase();
  researchEl("research-mode-status").textContent = deployment.research_enabled ? "Research tools enabled locally" : "Research tools disabled";
  researchEl("research-deployment-boundary").textContent = deployment.public_boundary || "";
  researchEl("research-audit-form").classList.toggle("research-disabled", !deployment.research_enabled);
  researchEl("research-export").classList.toggle("disabled", !deployment.research_enabled);

  researchEl("model-name").textContent = model.name || "—";
  researchEl("model-family").textContent = model.family || "—";
  researchEl("model-task").textContent = model.task || "—";
  researchEl("model-features").textContent = (model.features || []).join(", ") || "—";
  researchEl("model-packaging-note").textContent = model.packaging_note || "";
  researchEl("model-artifact-badge").textContent = model.artifact_available ? "ARTIFACT AVAILABLE" : "NOT IN SOURCE ZIP";
  researchEl("model-artifact-badge").className = `badge ${model.artifact_available ? "success" : "warning"}`;

  const protocolLabels = {
    participant_and_session_aware: "Participant and session groups separated",
    train_only_preprocessing: "Preprocessing fitted on training data only",
    threshold_selected_on_calibration_split: "Threshold selected on calibration split",
    untouched_test_split: "Untouched test split retained",
    deployment_transfer_evaluation: "Live transfer evaluated separately",
  };
  researchEl("training-protocol").innerHTML = Object.entries(protocolLabels).map(([key,text]) => `<div><span>${text}</span><b>${protocol[key] ? "✓" : "—"}</b></div>`).join("");

  const stages = [
    ["Trained model", true],
    ["Personal baseline calibration", decision.personal_calibration],
    ["Temporal smoothing", decision.temporal_smoothing],
    ["Controlled alert state", decision.controlled_alert_state_machine],
  ];
  researchEl("decision-layer").innerHTML = stages.map(([name,active], index) => `${index ? "<i>→</i>" : ""}<div class="decision-node" data-active="${Boolean(active)}">${name}</div>`).join("");
  researchEl("decision-explanation").textContent = decision.explanation || "";

  researchEl("deployment-sessions").textContent = fmt(evidence.session_count);
  researchEl("deployment-alerts").textContent = fmt(evidence.total_alerts);
  researchEl("deployment-duration").textContent = `${(Number(evidence.total_duration_seconds || 0) / 60).toFixed(1)} min`;
  researchEl("deployment-highest-risk").textContent = pct(evidence.highest_smoothed_risk);
  researchEl("deployment-baseline").textContent = evidence.average_baseline_ear ? Number(evidence.average_baseline_ear).toFixed(3) : "—";
  researchEl("deployment-summary").textContent = evidence.summary || "";
}

function renderAudit(audit) {
  const dataset = audit.dataset || {};
  const dist = audit.distributions || {};
  researchEl("research-rows").textContent = fmt(dataset.rows);
  researchEl("research-file").textContent = dataset.name || "Dataset";
  researchEl("research-participants").textContent = fmt(dataset.participants);
  researchEl("research-sessions").textContent = fmt(dataset.sessions);
  researchEl("life-data").textContent = `${fmt(dataset.rows)} observations`;
  researchEl("research-distributions").innerHTML = `<section><h3>State labels</h3>${bars(dist.states)}</section><section><h3>Conditions</h3>${bars(dist.conditions)}</section><section><h3>Fatigue levels</h3>${bars(dist.fatigue_levels)}</section><section><h3>Participants</h3>${bars(dist.participants)}</section>`;
  researchEl("research-association").innerHTML = bars(audit.feature_association?.state || [], "score");
  const rows = audit.condition_profile || [];
  researchEl("research-condition-table").innerHTML = rows.length ? rows.map(row => `<tr><td><b>${label(row.condition)}</b></td><td>${fmt(row.rows)}</td><td>${fmt(row.participants)}</td><td>${fmt(row.sessions)}</td><td>${Number(row.mean_ear || 0).toFixed(3)}</td><td>${Number(row.mean_yawn_score || 0).toFixed(3)}</td><td>${Number(row.mean_head_tilt || 0).toFixed(2)}°</td><td>${pct(row.drowsy_share)}</td></tr>`).join("") : `<tr><td colspan="8">No condition profile available.</td></tr>`;
  researchEl("research-readiness").innerHTML = Object.entries(audit.readiness || {}).map(([key,value]) => `<div><span>${label(key)}</span><b class="research-status" data-state="${value}">${label(value)}</b></div>`).join("");
  const privacy = [
    ...(audit.boundaries || []),
    "Public deployments should expose aggregate model-card facts only.",
    "Participant IDs, session IDs, dataset paths and raw rows remain research-only.",
  ];
  researchEl("research-boundaries").innerHTML = privacy.map(item => `<li>${item}</li>`).join("");
}

async function loadResearchStatus() {
  const response = await fetch("/api/research-lab", {cache:"no-store"});
  if (!response.ok) throw new Error(await response.text());
  const status = await response.json();
  renderLifecycle(status);
  const reference = status.reference || {};
  if (reference.dataset?.rows) renderAudit(reference);
  if (status.last_audit) renderAudit(status.last_audit);
}

researchEl("research-audit-form")?.addEventListener("submit", async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  try {
    const response = await fetch("/api/research-lab/audit", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({dataset_path:researchEl("research-dataset-path").value.trim()})});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Audit failed.");
    renderAudit(payload);
    DG.toast("AI lifecycle evidence updated", "success");
  } catch (error) {
    DG.toast(error.message || "AI research audit failed", "error");
    console.error(error);
  } finally { button.disabled = false; }
});

loadResearchStatus().catch(error => console.error(error));
async function evaluationResponsePayload(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch (_) {
    return {
      detail: text.startsWith("Internal Server Error")
        ? "Guardian returned an invalid evaluation response. Install the V7.2.1 compatibility fix and run again."
        : text,
    };
  }
}

function metricValue(value) {
  return value === null || value === undefined ? "—" : pct(value);
}

function renderConfusion(matrix) {
  const values = [
    ["True normal", matrix?.true_negative],
    ["False alert", matrix?.false_positive],
    ["Missed fatigue", matrix?.false_negative],
    ["True fatigue", matrix?.true_positive],
  ];
  researchEl("evaluation-confusion").innerHTML = values
    .map(([name, value]) => `<div><small>${name}</small><b>${fmt(value)}</b></div>`)
    .join("");
}

function renderEvaluation(result) {
  if (!result?.available) return;

  const metrics = result.metrics || {};
  const model = result.model || {};
  const split = result.test_split || {};

  researchEl("evaluation-status-badge").textContent = "EVALUATED";
  researchEl("evaluation-status-badge").className = "badge success";
  researchEl("evaluation-export").classList.remove("disabled");

  researchEl("eval-balanced-accuracy").textContent =
    metricValue(metrics.balanced_accuracy);
  researchEl("eval-accuracy").textContent = metricValue(metrics.accuracy);
  researchEl("eval-precision").textContent = metricValue(metrics.precision);
  researchEl("eval-recall").textContent = metricValue(metrics.recall);
  researchEl("eval-f1").textContent = metricValue(metrics.f1);
  researchEl("eval-roc-auc").textContent = metricValue(metrics.roc_auc);

  renderConfusion(metrics.confusion_matrix || {});

  researchEl("evaluation-contract").innerHTML = `
    <div><span>Model</span><b>${model.name || "—"}</b></div>
    <div><span>Features</span><b>${(model.features || []).join(", ") || "—"}</b></div>
    <div><span>Threshold</span><b>${Number(model.saved_threshold ?? 0.5).toFixed(3)}</b></div>
    <div><span>Test rows</span><b>${fmt(split.rows)}</b></div>
  `;

  const conditions = result.condition_performance || [];
  researchEl("evaluation-condition-table").innerHTML = conditions.length
    ? conditions.map(row => `
      <tr>
        <td><b>${label(row.label)}</b></td>
        <td>${fmt(row.rows)}</td>
        <td>${metricValue(row.balanced_accuracy)}</td>
        <td>${metricValue(row.precision)}</td>
        <td>${metricValue(row.recall)}</td>
        <td>${metricValue(row.f1)}</td>
        <td>${fmt(row.confusion_matrix?.false_positive)}</td>
        <td>${fmt(row.confusion_matrix?.false_negative)}</td>
      </tr>
    `).join("")
    : `<tr><td colspan="8">The test split does not include a condition column.</td></tr>`;

  const participants = (result.participant_performance || [])
    .sort((a, b) => Number(b.rows || 0) - Number(a.rows || 0));
  researchEl("evaluation-participant-bars").innerHTML = participants.length
    ? participants.map(row => `
      <div class="research-bar">
        <div><span>${row.label}</span><b>${metricValue(row.balanced_accuracy)}</b></div>
        <div class="meter"><i style="width:${Math.min(100, Number(row.balanced_accuracy || 0) * 100)}%"></i></div>
        <small>${fmt(row.rows)} rows · F1 ${metricValue(row.f1)}</small>
      </div>
    `).join("")
    : `<div class="empty-state">No participant column was found in the test split.</div>`;

  const calibration = result.probability_calibration || [];
  researchEl("evaluation-calibration-bars").innerHTML = calibration.length
    ? calibration.map(row => `
      <div class="calibration-row">
        <span>${Number(row.lower).toFixed(1)}–${Number(row.upper).toFixed(1)}</span>
        <div>
          <i style="width:${Math.min(100, Number(row.mean_probability || 0) * 100)}%"></i>
          <em style="left:${Math.min(100, Number(row.observed_positive_rate || 0) * 100)}%"></em>
        </div>
        <small>predicted ${pct(row.mean_probability)} · observed ${pct(row.observed_positive_rate)} · ${fmt(row.rows)} rows</small>
      </div>
    `).join("")
    : `<div class="empty-state">No calibration bins are available.</div>`;
}

async function loadEvaluationStatus() {
  try {
    const response = await fetch("/api/research-lab/evaluation", {cache:"no-store"});
    if (!response.ok) return;
    const status = await response.json();
    const candidates = status.candidate_paths || {};

    const setCandidate = (id, values) => {
      const input = researchEl(id);
      if (!input || !values?.length) return;
      const existing = input.value.trim();
      if (!existing || existing.startsWith("data\\") || existing.startsWith("models\\")) {
        input.value = values.find(value => value) || existing;
      }
    };

    setCandidate("evaluation-model-path", candidates.model);
    setCandidate("evaluation-test-path", candidates.test);
    setCandidate("evaluation-calibration-path", candidates.calibration);

    if (status.export_available) {
      researchEl("evaluation-export").classList.remove("disabled");
    }
    if (status.last_evaluation) renderEvaluation(status.last_evaluation);
  } catch (error) {
    console.error(error);
  }
}

researchEl("evaluation-form")?.addEventListener("submit", async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  researchEl("evaluation-status-badge").textContent = "RUNNING";
  researchEl("evaluation-status-badge").className = "badge warning";

  try {
    const response = await fetch("/api/research-lab/evaluation", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        model_path: researchEl("evaluation-model-path").value.trim(),
        test_path: researchEl("evaluation-test-path").value.trim(),
        calibration_path:
          researchEl("evaluation-calibration-path").value.trim() || null,
      }),
    });
    const payload = await evaluationResponsePayload(response);
    if (!response.ok) {
      throw new Error(payload.detail || "Model evaluation failed.");
    }
    renderEvaluation(payload);
    DG.toast("Held-out model evaluation completed", "success");
  } catch (error) {
    researchEl("evaluation-status-badge").textContent = "FAILED";
    researchEl("evaluation-status-badge").className = "badge danger";
    DG.toast(error.message || "Model evaluation failed", "error");
    console.error(error);
  } finally {
    button.disabled = false;
  }
});

loadEvaluationStatus();
