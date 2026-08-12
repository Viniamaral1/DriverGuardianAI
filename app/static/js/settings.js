const form = document.getElementById("settings-form");
const sensitivity = form.elements.sensitivity;
const volume = form.elements.alert_volume;
const theme = form.elements.theme;
const accent = form.elements.accent;
const message = document.getElementById("settings-message");

function updateOutputs() {
  document.getElementById("sensitivity-output").textContent = `${sensitivity.value}%`;
  document.getElementById("volume-output").textContent = `${volume.value}%`;
}

function previewTheme() {
  DG.applyTheme(theme.value, accent.value);
  message.textContent = "Preview active. Save to keep these preferences.";
}

[sensitivity, volume].forEach(input => input.addEventListener("input", updateOutputs));
theme.addEventListener("change", previewTheme);
accent.addEventListener("change", previewTheme);

fetch("/api/settings")
  .then(response => response.json())
  .then(settings => {
    Object.entries(settings).forEach(([key, value]) => {
      const input = form.elements[key];
      if (!input || key === "theme" || key === "accent") return;
      if (input.type === "checkbox") input.checked = Boolean(value);
      else input.value = value ?? "";
    });

    theme.value = document.documentElement.dataset.theme || "dark";
    accent.value = document.documentElement.dataset.accent || "cyan";
    updateOutputs();
  })
  .catch(error => {
    theme.value = document.documentElement.dataset.theme || "dark";
    accent.value = document.documentElement.dataset.accent || "cyan";
    updateOutputs();
    message.textContent = "Some saved settings could not be loaded.";
    console.error(error);
  });

form.addEventListener("submit", async event => {
  event.preventDefault();

  const body = {
    theme: theme.value,
    accent: accent.value,
    voice_output: form.elements.voice_output.checked,
    camera_index: Number(form.elements.camera_index.value),
    sensitivity: Number(sensitivity.value),
    alert_volume: Number(volume.value),
    driver_name: form.elements.driver_name.value.trim(),
    automatic_reports: form.elements.automatic_reports.checked,
    persistent_calibration_enabled: form.elements.persistent_calibration_enabled.checked,
    visual_evidence_enabled: form.elements.visual_evidence_enabled.checked,
  };

  try {
    const response = await fetch("/api/settings", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)
    });

    if (!response.ok) throw new Error(await response.text());

    DG.applyTheme(body.theme, body.accent);
    message.textContent = "Preferences saved successfully.";
    DG.toast("Settings saved", "success");
  } catch (error) {
    message.textContent = "Settings could not be saved.";
    DG.toast("Settings could not be saved", "error");
    console.error(error);
  }
});
