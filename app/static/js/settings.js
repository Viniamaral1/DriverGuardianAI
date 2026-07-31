const form = document.getElementById("settings-form");
const sensitivity = form.elements.sensitivity;
const volume = form.elements.alert_volume;
const updateOutputs = () => {
  document.getElementById("sensitivity-output").textContent = `${sensitivity.value}%`;
  document.getElementById("volume-output").textContent = `${volume.value}%`;
};
[sensitivity, volume].forEach(x => x.addEventListener("input", updateOutputs));

fetch("/api/settings").then(r => r.json()).then(s => {
  Object.entries(s).forEach(([key, value]) => {
    const input = form.elements[key];
    if (!input) return;
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = value;
  });
  updateOutputs();
});
form.addEventListener("submit", async e => {
  e.preventDefault();
  const body = {
    theme: form.elements.theme.value,
    accent: form.elements.accent.value,
    voice_output: form.elements.voice_output.checked,
    camera_index: Number(form.elements.camera_index.value),
    sensitivity: Number(form.elements.sensitivity.value),
    alert_volume: Number(form.elements.alert_volume.value),
  };
  const response = await fetch("/api/settings", {method:"PUT", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  if (!response.ok) throw new Error(await response.text());
  DG.applyTheme(body.theme, body.accent);
  DG.toast("Settings saved");
});
