# Guardian OS V8.4.3 — UX & Privacy Patch

This patch closes two usability/privacy issues before V8.5 Near-Miss Memory.

## Sidebar navigation
- The desktop sidebar remains fixed.
- The navigation list now scrolls independently inside the viewport.
- Settings/About remain reachable on shorter screens.
- The status card remains pinned below the scrollable navigation.
- Existing collapsed-sidebar and mobile navigation behaviour is preserved.

## Visual Evidence consent
Visual Evidence remains OFF by default.

The first time a user attempts to enable it, Guardian displays a consent dialog
explaining:
- event-linked camera images may be captured;
- evidence is stored locally;
- Guardian is not recording continuous full-session video;
- evidence can be deleted independently from Decision Memory metrics;
- cancelling leaves Visual Evidence disabled.

The consent acknowledgement is stored in browser localStorage so the warning
does not repeatedly interrupt the same browser after it has been accepted.

Turning Visual Evidence off stops future capture after settings are saved, but
does not silently delete existing evidence.

## Protected / unchanged
- fatigue model
- PersonalCalibration
- TemporalStateEngine
- AlertManager
- camera ownership
- Visual Evidence capture backend
- event thresholds
- replay functionality
- Decision Memory sampling
