const profileEl = id => document.getElementById(id);
let profileSnapshot = null;
let passportSnapshot = null;

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

function passportPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function passportTags(items, emptyText, keyName) {
  if (!items?.length) return `<em>${emptyText}</em>`;
  return items.map(item => {
    const label = item[keyName] || item.code || "Unknown";
    return `<span>${label}${item.count ? ` · ${item.count}` : ""}</span>`;
  }).join("");
}

function renderPassport(passport) {
  passportSnapshot = passport;
  const active = profileSnapshot?.active_profile;
  const available = Boolean(active && passport);

  profileEl("passport-status").textContent = available
    ? String(passport.origin || "local").toUpperCase()
    : "NO ACTIVE PROFILE";

  profileEl("passport-summary").textContent = available
    ? `Versioned local passport for ${active.name}. Updated ${profileDate(passport.updated_at)}.`
    : "Select a local driver profile to build a versioned Calibration Passport.";

  profileEl("passport-id").textContent =
    passport?.passport_id || "—";
  profileEl("passport-schema").textContent =
    passport?.schema || "—";

  const baseline = passport?.personal_visual_baseline || {};
  profileEl("passport-ear").textContent =
    baseline.baseline_ear
      ? Number(baseline.baseline_ear).toFixed(3)
      : "—";
  profileEl("passport-observations").textContent =
    baseline.observation_count || 0;

  const perception = passport?.perception_reliability_profile || {};
  profileEl("passport-perception").textContent =
    perception.average_confidence
      ? passportPercent(perception.average_confidence)
      : perception.included_in_export === false
        ? "Excluded"
        : "—";
  profileEl("passport-trusted").textContent =
    perception.trusted_rate !== undefined
      ? passportPercent(perception.trusted_rate)
      : "—";
  profileEl("passport-eyes").textContent =
    perception.average_eye_visibility
      ? passportPercent(perception.average_eye_visibility)
      : "—";

  const history = passport?.calibration_history || {};
  profileEl("passport-history").textContent =
    `${history.decision_memory_sessions || 0} sessions`;

  const visibility = passport?.visibility_profile || {};
  profileEl("passport-conditions").innerHTML = passportTags(
    visibility.known_conditions,
    visibility.sunglasses_seen ? "Sunglasses observed" : "None recorded yet",
    "condition",
  );
  profileEl("passport-limitations").innerHTML = passportTags(
    perception.common_limitations,
    "None recorded yet",
    "code",
  );

  const privacy = passport?.privacy_and_retention || {};
  profileEl("passport-allow-export").checked =
    Boolean(privacy.allow_export ?? true);
  profileEl("passport-include-perception").checked =
    Boolean(privacy.include_perception_history ?? true);

  profileEl("passport-save-privacy").disabled = !available;
  profileEl("passport-export").disabled =
    !available || !Boolean(privacy.allow_export ?? true);
  profileEl("passport-reset").disabled = !available;
  profileEl("passport-import-file").disabled = !active;
}

async function loadPassport() {
  const profileId = profileSnapshot?.active_profile_id;
  if (!profileId) {
    renderPassport(null);
    return;
  }
  const response = await fetch(
    `/api/profiles/${encodeURIComponent(profileId)}/passport`,
    {cache: "no-store"},
  );
  if (!response.ok) {
    throw new Error(
      await profileError(response, "Calibration Passport could not be loaded.")
    );
  }
  renderPassport(await response.json());
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
  await loadPassport();
}

async function setActiveProfile(profileId) {
  const response = await fetch("/api/profiles/active", {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({profile_id: profileId}),
  });
  if (!response.ok) throw new Error(await response.text());
  renderProfiles(await response.json());
  await loadPassport();
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
    await loadPassport();
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
      await loadPassport();
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
    await loadPassport();
    DG.toast("Saved calibration reset", "success");
  } catch (error) {
    DG.toast("Calibration could not be reset", "error");
    console.error(error);
  }
});

profileEl("passport-save-privacy")?.addEventListener("click", async () => {
  const profileId = profileSnapshot?.active_profile_id;
  if (!profileId) return;
  try {
    const response = await fetch(
      `/api/profiles/${encodeURIComponent(profileId)}/passport/privacy`,
      {
        method: "PUT",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          allow_export: profileEl("passport-allow-export").checked,
          include_perception_history:
            profileEl("passport-include-perception").checked,
        }),
      },
    );
    if (!response.ok) {
      throw new Error(await profileError(response, "Privacy choices could not be saved."));
    }
    renderPassport(await response.json());
    DG.toast("Passport privacy choices saved", "success");
  } catch (error) {
    DG.toast(error.message || "Passport privacy update failed", "error");
    console.error(error);
  }
});

profileEl("passport-export")?.addEventListener("click", async () => {
  const profileId = profileSnapshot?.active_profile_id;
  if (!profileId) return;
  try {
    const response = await fetch(
      `/api/profiles/${encodeURIComponent(profileId)}/passport/export`,
      {cache: "no-store"},
    );
    if (!response.ok) {
      throw new Error(await profileError(response, "Passport export failed."));
    }
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/i);
    const filename = match?.[1] || "Guardian-Calibration-Passport.json";
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    DG.toast("Calibration Passport exported", "success");
  } catch (error) {
    DG.toast(error.message || "Passport export failed", "error");
    console.error(error);
  }
});

profileEl("passport-import-file")?.addEventListener("change", async event => {
  const file = event.currentTarget.files?.[0];
  const profileId = profileSnapshot?.active_profile_id;
  if (!file || !profileId) return;

  const activeName = profileSnapshot?.active_profile?.name || "selected driver";
  const confirmed = window.confirm(
    `Import this Calibration Passport into ${activeName}? This replaces the saved behavioural baseline for that profile.`
  );
  if (!confirmed) {
    event.currentTarget.value = "";
    return;
  }

  try {
    const passport = JSON.parse(await file.text());
    const response = await fetch(
      `/api/profiles/${encodeURIComponent(profileId)}/passport/import`,
      {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({passport}),
      },
    );
    if (!response.ok) {
      throw new Error(await profileError(response, "Passport import failed."));
    }
    renderPassport(await response.json());
    await loadProfiles();
    DG.toast("Calibration Passport imported", "success");
  } catch (error) {
    DG.toast(error.message || "Passport import failed", "error");
    console.error(error);
  } finally {
    event.currentTarget.value = "";
  }
});

profileEl("passport-reset")?.addEventListener("click", async () => {
  const profileId = profileSnapshot?.active_profile_id;
  if (!profileId) return;
  if (!window.confirm(
    "Reset Passport metadata and privacy choices? The local driver calibration itself will be preserved."
  )) {
    return;
  }
  try {
    const response = await fetch(
      `/api/profiles/${encodeURIComponent(profileId)}/passport/reset`,
      {method: "POST"},
    );
    if (!response.ok) {
      throw new Error(await profileError(response, "Passport reset failed."));
    }
    renderPassport(await response.json());
    DG.toast("Passport metadata reset", "success");
  } catch (error) {
    DG.toast(error.message || "Passport reset failed", "error");
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
