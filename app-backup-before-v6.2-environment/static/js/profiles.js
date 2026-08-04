const profileEl = id => document.getElementById(id);
let profileSnapshot = null;

async function profileError(response, fallback) {
  try {
    const payload = await response.json();
    return payload.detail || fallback;
  } catch (_) {
    return fallback;
  }
}

function profileDate(value) {
  if (!value) return "Never";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleString("en-GB");
}

function renderProfiles(snapshot) {
  profileSnapshot = snapshot;
  const active = snapshot.active_profile;
  const profiles = snapshot.profiles || [];

  profileEl("profiles-active-name").textContent =
    active?.name || "Guest";
  profileEl("profiles-active-status").textContent =
    active?.calibration ? "QUICK VERIFICATION READY" : "FULL CALIBRATION";

  profileEl("profiles-active-summary").textContent = active
    ? active.calibration
      ? `A saved baseline is available. The next session begins with a three-second verification. Last updated ${profileDate(active.calibration.updated_at)}.`
      : "This profile has no saved baseline yet. The next session will use the original full calibration."
    : "Guest mode always uses the existing full calibration and does not save or reuse a personal baseline.";

  const calibration = active?.calibration || {};
  profileEl("profile-baseline-ear").textContent =
    calibration.baseline_ear ? Number(calibration.baseline_ear).toFixed(3) : "—";
  profileEl("profile-baseline-yawn").textContent =
    calibration.baseline_yawn !== undefined
      ? Number(calibration.baseline_yawn).toFixed(3)
      : "—";
  profileEl("profile-baseline-tilt").textContent =
    calibration.baseline_tilt !== undefined
      ? `${Number(calibration.baseline_tilt).toFixed(1)}°`
      : "—";
  profileEl("profile-observations").textContent =
    calibration.observation_count || 0;
  profileEl("profile-verifications").textContent =
    active?.verification_count || 0;
  profileEl("profile-full-count").textContent =
    active?.full_calibration_count || 0;
  profileEl("profiles-reset-active").disabled =
    !active?.calibration;

  profileEl("profiles-count").textContent =
    `${profiles.length} profile${profiles.length === 1 ? "" : "s"}`;

  profileEl("profiles-list").innerHTML = profiles.length
    ? profiles.map(profile => `
      <article class="profile-list-item ${profile.active ? "active" : ""}">
        <div class="profile-avatar">${profile.name.slice(0, 2).toUpperCase()}</div>
        <div class="profile-list-copy">
          <b>${profile.name}</b>
          <span>
            ${profile.has_calibration ? "Saved baseline ready" : "Full calibration required"}
            · ${profile.verification_count || 0} quick verifications
          </span>
        </div>
        <div class="profile-list-actions">
          <button
            class="button ${profile.active ? "secondary" : "primary"}"
            type="button"
            data-profile-select="${profile.id}"
            ${profile.active ? "disabled" : ""}
          >${profile.active ? "Active" : "Select"}</button>
          <button
            class="icon-button profile-delete"
            type="button"
            title="Delete profile"
            data-profile-delete="${profile.id}"
          >×</button>
        </div>
      </article>
    `).join("")
    : `<div class="empty-state">No local driver profile exists yet.</div>`;
}

async function loadProfiles() {
  const response = await fetch("/api/profiles", {cache: "no-store"});
  if (!response.ok) {
    throw new Error(
      await profileError(response, "Active profile could not be changed.")
    );
  }
  renderProfiles(await response.json());
}

async function setActiveProfile(profileId) {
  const response = await fetch("/api/profiles/active", {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({profile_id: profileId}),
  });
  if (!response.ok) throw new Error(await response.text());
  renderProfiles(await response.json());
}

profileEl("profile-create-form")?.addEventListener("submit", async event => {
  event.preventDefault();

  const form = event.currentTarget;
  const submitButton = form.querySelector('button[type="submit"]');
  const name = new FormData(form).get("name");

  submitButton.disabled = true;

  try {
    const response = await fetch("/api/profiles", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name}),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Driver profile could not be created.");
    }

    renderProfiles(payload);
    form.reset();
    DG.toast("Driver profile created and selected", "success");
  } catch (error) {
    DG.toast(error.message || "Driver profile could not be created", "error");
    console.error(error);
  } finally {
    submitButton.disabled = false;
  }
});

profileEl("profiles-list")?.addEventListener("click", async event => {
  const select = event.target.closest("[data-profile-select]");
  const remove = event.target.closest("[data-profile-delete]");

  try {
    if (select) {
      await setActiveProfile(select.dataset.profileSelect);
      DG.toast("Active driver profile changed", "success");
    }

    if (remove) {
      const profile = profileSnapshot?.profiles?.find(
        item => item.id === remove.dataset.profileDelete
      );
      if (!window.confirm(`Delete the local profile “${profile?.name || "driver"}”?`)) {
        return;
      }
      const response = await fetch(
        `/api/profiles/${encodeURIComponent(remove.dataset.profileDelete)}`,
        {method: "DELETE"}
      );
      if (!response.ok) throw new Error(await response.text());
      renderProfiles(await response.json());
      DG.toast("Driver profile deleted", "success");
    }
  } catch (error) {
    DG.toast("Profile change failed. Stop Monitoring first.", "error");
    console.error(error);
  }
});

profileEl("profiles-use-guest")?.addEventListener("click", async () => {
  try {
    await setActiveProfile(null);
    DG.toast("Guest mode selected", "success");
  } catch (error) {
    DG.toast("Stop Monitoring before changing profile", "error");
    console.error(error);
  }
});

profileEl("profiles-reset-active")?.addEventListener("click", async () => {
  const profileId = profileSnapshot?.active_profile_id;
  if (!profileId) return;
  if (!window.confirm("Reset this saved calibration? The next session will use the full calibration.")) {
    return;
  }

  try {
    const response = await fetch(
      `/api/profiles/${encodeURIComponent(profileId)}/reset-calibration`,
      {method: "POST"}
    );
    if (!response.ok) throw new Error(await response.text());
    renderProfiles(await response.json());
    DG.toast("Saved calibration reset", "success");
  } catch (error) {
    DG.toast("Calibration could not be reset", "error");
    console.error(error);
  }
});

profileEl("profiles-refresh")?.addEventListener("click", async () => {
  try {
    await loadProfiles();
    DG.toast("Driver profiles refreshed", "success");
  } catch (error) {
    DG.toast("Profiles could not be loaded", "error");
    console.error(error);
  }
});

loadProfiles().catch(error => {
  DG.toast("Profiles could not be loaded", "error");
  console.error(error);
});
