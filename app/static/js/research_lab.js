const researchEl = id => document.getElementById(id);
const fmt = value => Number(value || 0).toLocaleString("en-GB");
const pct = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
const label = value => String(value ?? "unknown").replaceAll("_", " ");

function bars(items, valueKey = "share") {
  if (!items?.length) return `<div class="empty-state">No values available.</div>`;
  const max = Math.max(...items.map(item => Number(item[valueKey] || 0)), 0.000001);
  return items.map(item => `
    <div class="research-bar">
      <div><span>${item.label ?? item.feature}</span><b>${
        valueKey === "share" ? fmt(item.count) : Number(item[valueKey] || 0).toFixed(4)
      }</b></div>
      <div class="meter"><i style="width:${Math.min(100, Number(item[valueKey] || 0) / max * 100)}%"></i></div>
      ${valueKey === "share" ? `<small>${pct(item.share)}</small>` : ""}
    </div>
  `).join("");
}

function renderAudit(audit) {
  const dataset = audit.dataset || {};
  const dist = audit.distributions || {};

  researchEl("research-rows").textContent = fmt(dataset.rows);
  researchEl("research-file").textContent = dataset.name || "Dataset";
  researchEl("research-participants").textContent = fmt(dataset.participants);
  researchEl("research-sessions").textContent = fmt(dataset.sessions);
  researchEl("research-conditions").textContent = fmt(dist.conditions?.length);
  researchEl("research-missing").textContent = fmt(dataset.missing_cells);

  researchEl("research-distributions").innerHTML = `
    <section><h3>State labels</h3>${bars(dist.states)}</section>
    <section><h3>Conditions</h3>${bars(dist.conditions)}</section>
    <section><h3>Fatigue levels</h3>${bars(dist.fatigue_levels)}</section>
    <section><h3>Participants</h3>${bars(dist.participants)}</section>
  `;

  researchEl("research-association").innerHTML =
    bars(audit.feature_association?.state || [], "score");

  const rows = audit.condition_profile || [];
  researchEl("research-condition-table").innerHTML = rows.length ? rows.map(row => `
    <tr>
      <td><b>${label(row.condition)}</b></td>
      <td>${fmt(row.rows)}</td>
      <td>${fmt(row.participants)}</td>
      <td>${fmt(row.sessions)}</td>
      <td>${Number(row.mean_ear || 0).toFixed(3)}</td>
      <td>${Number(row.mean_yawn_score || 0).toFixed(3)}</td>
      <td>${Number(row.mean_head_tilt || 0).toFixed(2)}°</td>
      <td>${pct(row.drowsy_share)}</td>
    </tr>
  `).join("") : `<tr><td colspan="8">No condition profile available.</td></tr>`;

  researchEl("research-readiness").innerHTML =
    Object.entries(audit.readiness || {}).map(([key, value]) => `
      <div><span>${label(key)}</span><b class="research-status" data-state="${value}">${label(value)}</b></div>
    `).join("");

  researchEl("research-boundaries").innerHTML =
    (audit.boundaries || []).map(item => `<li>${item}</li>`).join("");
}

async function loadResearchStatus() {
  const response = await fetch("/api/research-lab", {cache: "no-store"});
  if (!response.ok) throw new Error(await response.text());
  const status = await response.json();
  const reference = status.reference || {};
  if (reference.dataset?.rows) {
    researchEl("research-reference-rows").textContent = fmt(reference.dataset.rows);
  }
  if (status.last_audit) renderAudit(status.last_audit);
}

researchEl("research-audit-form")?.addEventListener("submit", async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  try {
    const response = await fetch("/api/research-lab/audit", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        dataset_path: researchEl("research-dataset-path").value.trim()
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Audit failed.");
    renderAudit(payload);
    DG.toast("AI research audit completed", "success");
  } catch (error) {
    DG.toast(error.message || "AI research audit failed", "error");
    console.error(error);
  } finally {
    button.disabled = false;
  }
});

loadResearchStatus().catch(error => console.error(error));
