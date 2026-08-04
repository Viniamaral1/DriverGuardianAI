# Guardian OS V6.3 — Automatic Journey Context

Built from the uploaded V6.2 stable source.

## Added

- keyless online current-weather integration using Open-Meteo;
- configurable city or locality on Edge Intelligence;
- current weather, temperature, precipitation, cloud, wind and visibility metadata;
- conservative road-condition estimate from precipitation and temperature;
- source, confidence, location and freshness metadata;
- explicit offline fallback to `Unknown`;
- manual values only when **Use manual override** is enabled;
- a manual **Refresh weather now** action;
- source labels on Guardian Intelligence.

## Stale-data safety

Guardian never keeps an old `Rain` value indefinitely. Online context refreshes
approximately every 15 minutes. A cached reading is usable only for a short
30-minute window and is visibly labelled stale if refresh fails. After that,
weather and road condition become `Unknown`.

## Privacy and architecture

The configured city/locality and coordinates resolved by the weather provider
are used only for current conditions. No device location permission, GPS or
browser tracking is added. The fatigue model, camera, calibration, profiles,
alerts and reports are unchanged.

## Configure

1. Open Edge Intelligence.
2. Enter a city or locality, for example `London, UK`.
3. Keep **Automatic online weather** enabled.
4. Keep **Use manual override** disabled for genuinely automatic values.
5. Save context settings and click **Refresh weather now**.

Internet access is required for weather. When offline, Guardian clearly shows
`Unknown` rather than presenting an old condition as current.
