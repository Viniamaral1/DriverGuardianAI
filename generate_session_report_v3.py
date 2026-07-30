from __future__ import annotations

import argparse
import base64
import json
import math
import webbrowser
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs" / "v3"
ALERT_DIR = LOG_DIR / "alerts"
REPORT_DIR = ROOT / "reports" / "v3"


def latest(directory: Path, pattern: str) -> Path | None:
    files = [p for p in directory.glob(pattern) if p.is_file() and p.stat().st_size > 0]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def load_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def row_durations(df: pd.DataFrame) -> np.ndarray:
    t = df["elapsed_seconds"].to_numpy(float)
    if len(t) < 2:
        return np.zeros(len(t))
    diffs = np.diff(t)
    valid = diffs[np.isfinite(diffs) & (diffs > 0)]
    fallback = float(np.median(valid)) if len(valid) else 0.0
    durations = np.append(diffs, fallback)
    durations = np.where(np.isfinite(durations) & (durations >= 0), durations, fallback)
    if fallback > 0:
        durations = np.minimum(durations, fallback * 5)
    return durations


def episodes(states: pd.Series, target: str) -> int:
    mask = states.astype(str).eq(target)
    return int((mask & ~mask.shift(fill_value=False)).sum())


def safe(series: pd.Series, fn: str, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return default
    return float(getattr(values, fn)())


def img_uri(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def risk_chart(session: pd.DataFrame, alerts: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(13.5, 4.5))
    x = session["elapsed_seconds"]
    ax.plot(x, session["raw_model_probability"], label="Raw model", alpha=0.45, linewidth=1)
    ax.plot(x, session["decision_probability"], label="Calibrated risk", linewidth=1.5)
    ax.plot(x, session["smoothed_probability"], label="Smoothed risk", linewidth=2)
    ax.axhline(0.65, linestyle="--", linewidth=1, label="Warning threshold")
    ax.axhline(0.82, linestyle="--", linewidth=1, label="Critical threshold")

    if not alerts.empty and "elapsed_seconds" in alerts:
        for i, value in enumerate(alerts["elapsed_seconds"].dropna()):
            ax.axvline(
                float(value),
                linestyle=":",
                linewidth=1.5,
                label="Triggered alert" if i == 0 else None,
            )

    ax.set(
        title="Risk Timeline",
        xlabel="Elapsed time (seconds)",
        ylabel="Probability / risk",
        ylim=(-0.02, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return img_uri(fig)


def state_chart(state_seconds: dict[str, float]) -> str:
    labels = ["Monitoring", "Warning", "Critical", "Calibrating", "No face"]
    keys = ["MONITORING", "WARNING", "CRITICAL", "CALIBRATING", "NO FACE"]
    values = [state_seconds.get(k, 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    bars = ax.bar(labels, values)
    ax.set(title="Time Spent in Each State", ylabel="Seconds")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return img_uri(fig)


def signal_chart(session: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(13.5, 4.5))
    x = session["elapsed_seconds"]
    ax.plot(x, session["eye_risk"], label="Eye risk", linewidth=1.8)
    ax.plot(x, session["yawn_risk"], label="Yawn risk", linewidth=1.5)
    ax.plot(x, session["tilt_risk"], label="Head-tilt risk", linewidth=1.5)
    ax.set(
        title="Behavioural Risk Contributions",
        xlabel="Elapsed time (seconds)",
        ylabel="Risk contribution",
        ylim=(-0.02, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return img_uri(fig)


def ear_chart(session: pd.DataFrame, baseline: float) -> str:
    fig, ax = plt.subplots(figsize=(13.5, 4.5))
    x = session["elapsed_seconds"]
    ax.plot(x, session["ear"], label="Live EAR", linewidth=1.7)
    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1.5,
        label=f"Personal baseline ({baseline:.3f})",
    )
    ax.set(
        title="Eye Aspect Ratio Relative to Personal Baseline",
        xlabel="Elapsed time (seconds)",
        ylabel="EAR",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return img_uri(fig)


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "—"

    total = max(0, int(round(value)))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def alert_table(alerts: pd.DataFrame) -> str:
    if alerts.empty:
        return '<div class="empty">No controlled alerts were triggered.</div>'

    rows = []

    for i, row in alerts.iterrows():
        rows.append(
            "<tr>"
            f"<td>#{int(row.get('event_id', i + 1))}</td>"
            f"<td>{float(row.get('elapsed_seconds', math.nan)):.2f}s</td>"
            f"<td>{float(row.get('critical_duration_seconds', math.nan)):.2f}s</td>"
            f"<td>{float(row.get('smoothed_probability', math.nan)):.3f}</td>"
            f"<td>{float(row.get('ear', math.nan)):.3f}</td>"
            f"<td>{float(row.get('eye_risk', math.nan)):.3f}</td>"
            f"<td>{float(row.get('yawn_risk', math.nan)):.3f}</td>"
            f"<td>{float(row.get('tilt_risk', math.nan)):.3f}</td>"
            f"<td>{'Yes' if bool(row.get('sound_played', False)) else 'No'}</td>"
            "</tr>"
        )

    return (
        '<div class="table-wrap"><table><thead><tr>'
        '<th>Event</th><th>Time</th><th>Critical duration</th><th>Smoothed risk</th>'
        '<th>EAR</th><th>Eye risk</th><th>Yawn risk</th><th>Tilt risk</th><th>Sound</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a polished DriverGuardianAI V3 session report."
    )
    parser.add_argument("--session", type=Path)
    parser.add_argument("--alerts", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    session_path = (
        args.session.resolve()
        if args.session
        else latest(LOG_DIR, "driver_guardian_v3_*.csv")
    )

    if session_path is None or not session_path.exists():
        raise FileNotFoundError(f"No session log found in {LOG_DIR}")

    alert_path = (
        args.alerts.resolve()
        if args.alerts
        else latest(ALERT_DIR, "alert_events_*.csv")
    )

    session = load_csv(session_path)
    alerts = load_csv(alert_path)

    required = {
        "timestamp",
        "elapsed_seconds",
        "ear",
        "yawn_score",
        "head_tilt",
        "baseline_ear",
        "baseline_yawn",
        "baseline_tilt",
        "raw_model_probability",
        "decision_probability",
        "smoothed_probability",
        "eye_risk",
        "yawn_risk",
        "tilt_risk",
        "temporal_state",
    }

    missing = sorted(required - set(session.columns))

    if missing:
        raise ValueError(f"Missing session columns: {missing}")

    session["timestamp"] = pd.to_datetime(session["timestamp"], errors="coerce")

    numeric(
        session,
        [c for c in required if c not in {"timestamp", "temporal_state"}]
        + ["alert_count", "critical_duration_seconds", "cooldown_remaining_seconds"],
    )

    numeric(
        alerts,
        [
            "event_id",
            "elapsed_seconds",
            "critical_duration_seconds",
            "smoothed_probability",
            "raw_model_probability",
            "decision_probability",
            "ear",
            "baseline_ear",
            "eye_risk",
            "yawn_risk",
            "tilt_risk",
            "sound_played",
        ],
    )

    session = session.sort_values("elapsed_seconds").reset_index(drop=True)
    session["row_duration_seconds"] = row_durations(session)

    duration = float(
        session["elapsed_seconds"].iloc[-1]
        - session["elapsed_seconds"].iloc[0]
    )

    if duration <= 0:
        duration = float(session["row_duration_seconds"].sum())

    grouped = (
        session.groupby(session["temporal_state"].astype(str))[
            "row_duration_seconds"
        ]
        .sum()
        .to_dict()
    )

    state_seconds = {
        state: float(grouped.get(state, 0.0))
        for state in ["MONITORING", "WARNING", "CRITICAL", "CALIBRATING", "NO FACE"]
    }

    baseline_ear = safe(session["baseline_ear"], "median")
    baseline_yawn = safe(session["baseline_yawn"], "median")
    baseline_tilt = safe(session["baseline_tilt"], "median")

    alert_count = len(alerts)

    if alert_count == 0 and "alert_count" in session:
        alert_count = int(safe(session["alert_count"], "max"))

    risk_means = {
        "Eye closure": safe(session["eye_risk"], "mean"),
        "Yawning": safe(session["yawn_risk"], "mean"),
        "Head tilt": safe(session["tilt_risk"], "mean"),
    }

    dominant_signal = max(risk_means, key=risk_means.get)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "session_file": str(session_path),
        "alert_file": str(alert_path) if alert_path else None,
        "duration_seconds": duration,
        "logged_frames": int(len(session)),
        "estimated_fps": float(len(session) / duration) if duration > 0 else 0.0,
        "baseline_ear": baseline_ear,
        "baseline_yawn": baseline_yawn,
        "baseline_tilt": baseline_tilt,
        "average_ear": safe(session["ear"], "mean"),
        "minimum_ear": safe(session["ear"], "min"),
        "maximum_yawn_score": safe(session["yawn_score"], "max"),
        "maximum_head_tilt": safe(session["head_tilt"], "max"),
        "average_raw_model_probability": safe(
            session["raw_model_probability"], "mean"
        ),
        "maximum_raw_model_probability": safe(
            session["raw_model_probability"], "max"
        ),
        "average_decision_probability": safe(
            session["decision_probability"], "mean"
        ),
        "maximum_decision_probability": safe(
            session["decision_probability"], "max"
        ),
        "average_smoothed_probability": safe(
            session["smoothed_probability"], "mean"
        ),
        "maximum_smoothed_probability": safe(
            session["smoothed_probability"], "max"
        ),
        "state_seconds": state_seconds,
        "state_percentages": {
            state: (100 * seconds / duration if duration > 0 else 0.0)
            for state, seconds in state_seconds.items()
        },
        "warning_episodes": episodes(session["temporal_state"], "WARNING"),
        "critical_episodes": episodes(session["temporal_state"], "CRITICAL"),
        "alert_count": int(alert_count),
        "dominant_risk_signal": dominant_signal,
        "dominant_risk_mean": risk_means[dominant_signal],
    }

    risk_img = risk_chart(session, alerts)
    state_img = state_chart(state_seconds)
    signal_img = signal_chart(session)
    ear_img = ear_chart(session, baseline_ear)
    table_html = alert_table(alerts)

    monitoring_pct = summary["state_percentages"]["MONITORING"]
    warning_pct = summary["state_percentages"]["WARNING"]
    critical_pct = summary["state_percentages"]["CRITICAL"]

    css = """
    :root{--bg:#0b1020;--surface:#121a2e;--surface2:#19233b;--border:#2a3958;
    --text:#eef4ff;--muted:#9cabc4;--accent:#43b9e8;--good:#51d58a;
    --warn:#f5bd4f;--danger:#ef5f6c;--shadow:0 18px 45px rgba(0,0,0,.24)}
    *{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Segoe UI,sans-serif;
    background:radial-gradient(circle at top right,rgba(67,185,232,.13),transparent 30%),
    radial-gradient(circle at top left,rgba(81,213,138,.08),transparent 28%),var(--bg);
    color:var(--text);line-height:1.55}.page{width:min(1440px,calc(100% - 32px));margin:auto;
    padding:32px 0 56px}.hero{display:grid;grid-template-columns:1.6fr 1fr;gap:24px;margin-bottom:24px}
    .card,.panel,.metric{background:linear-gradient(145deg,rgba(25,35,59,.98),rgba(18,26,46,.98));
    border:1px solid var(--border);box-shadow:var(--shadow)}.card{padding:32px;border-radius:24px}
    .eyebrow{color:var(--accent);font-weight:700;letter-spacing:.14em;text-transform:uppercase;
    font-size:.78rem;margin-bottom:10px}h1{margin:0;font-size:clamp(2.2rem,5vw,4.2rem);line-height:1;
    letter-spacing:-.04em}h2{margin:0 0 18px;font-size:1.35rem}.muted{color:var(--muted)}
    .badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}.badge{padding:8px 12px;border:1px solid var(--border);
    border-radius:999px;color:var(--muted);font-size:.84rem}.big{font-size:3.1rem;font-weight:800}
    .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
    .metric{padding:22px;border-radius:18px}.label{color:var(--muted);font-size:.78rem;text-transform:uppercase;
    letter-spacing:.09em}.value{font-size:1.65rem;font-weight:800;margin-top:6px}.detail{color:var(--muted);
    font-size:.88rem;margin-top:4px}.panel{padding:24px;border-radius:22px;margin-bottom:24px}.chart{width:100%;
    display:block;border-radius:14px;background:#fff}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px}
    .states{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.state{padding:16px;border:1px solid var(--border);
    border-radius:14px;background:rgba(255,255,255,.025)}.table-wrap{overflow-x:auto;border:1px solid var(--border);
    border-radius:14px}table{width:100%;border-collapse:collapse;min-width:900px}th,td{padding:13px 14px;
    border-bottom:1px solid var(--border);text-align:left;font-size:.88rem}th{color:var(--muted);
    background:rgba(255,255,255,.03)}.empty{padding:26px;border:1px dashed var(--border);
    border-radius:14px;text-align:center;color:var(--muted)}.footer{text-align:center;color:var(--muted);
    font-size:.84rem}.good{color:var(--good)}.warn{color:var(--warn)}.danger{color:var(--danger)}
    @media(max-width:1000px){.hero,.grid2{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}
    .states{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.metrics,.states{grid-template-columns:1fr}}
    """

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriverGuardianAI Session Report</title>
<style>{css}</style>
</head>
<body>
<div class="page">

<section class="hero">
  <div class="card">
    <div class="eyebrow">DriverGuardianAI V3</div>
    <h1>Session Analytics</h1>
    <p class="muted">
      Calibrated behavioural monitoring, temporal fatigue reasoning,
      controlled alert management, and explainable risk contributions.
    </p>
    <div class="badges">
      <div class="badge">Frames: {len(session):,}</div>
      <div class="badge">Estimated FPS: {summary['estimated_fps']:.1f}</div>
      <div class="badge">Dominant signal: {dominant_signal}</div>
    </div>
  </div>

  <div class="card">
    <div class="eyebrow">Session duration</div>
    <div class="big">{fmt_seconds(duration)}</div>
    <p class="muted">Generated {summary['generated_at']}</p>
  </div>
</section>

<section class="metrics">
  <div class="metric">
    <div class="label">Maximum risk</div>
    <div class="value">{100 * summary['maximum_smoothed_probability']:.1f}%</div>
    <div class="detail">Peak temporally smoothed risk</div>
  </div>

  <div class="metric">
    <div class="label">Critical time</div>
    <div class="value warn">{fmt_seconds(state_seconds['CRITICAL'])}</div>
    <div class="detail">{critical_pct:.1f}% of session</div>
  </div>

  <div class="metric">
    <div class="label">Controlled alerts</div>
    <div class="value {'danger' if alert_count else 'good'}">{alert_count}</div>
    <div class="detail">Triggered after sustained evidence</div>
  </div>

  <div class="metric">
    <div class="label">Minimum EAR</div>
    <div class="value">{summary['minimum_ear']:.3f}</div>
    <div class="detail">Baseline {baseline_ear:.3f}</div>
  </div>
</section>

<section class="panel">
  <h2>Risk Timeline</h2>
  <img class="chart" src="{risk_img}">
</section>

<section class="grid2">
  <div class="panel">
    <h2>State Distribution</h2>
    <img class="chart" src="{state_img}">
  </div>

  <div class="panel">
    <h2>Session States</h2>
    <div class="states">
      <div class="state">
        <div class="label">Monitoring</div>
        <div class="value">{fmt_seconds(state_seconds['MONITORING'])}</div>
        <div class="detail">{monitoring_pct:.1f}%</div>
      </div>

      <div class="state">
        <div class="label">Warning</div>
        <div class="value">{fmt_seconds(state_seconds['WARNING'])}</div>
        <div class="detail">{warning_pct:.1f}%</div>
      </div>

      <div class="state">
        <div class="label">Critical</div>
        <div class="value">{fmt_seconds(state_seconds['CRITICAL'])}</div>
        <div class="detail">{critical_pct:.1f}%</div>
      </div>

      <div class="state">
        <div class="label">Calibrating</div>
        <div class="value">{fmt_seconds(state_seconds['CALIBRATING'])}</div>
      </div>

      <div class="state">
        <div class="label">No face</div>
        <div class="value">{fmt_seconds(state_seconds['NO FACE'])}</div>
      </div>
    </div>
  </div>
</section>

<section class="panel">
  <h2>Behavioural Risk Contributions</h2>
  <img class="chart" src="{signal_img}">
</section>

<section class="panel">
  <h2>Eye Behaviour</h2>
  <img class="chart" src="{ear_img}">
</section>

<section class="panel">
  <h2>Alert Events</h2>
  {table_html}
</section>

<section class="panel">
  <h2>Technical Summary</h2>
  <div class="states">
    <div class="state">
      <div class="label">Raw model avg</div>
      <div class="value">{100 * summary['average_raw_model_probability']:.1f}%</div>
    </div>

    <div class="state">
      <div class="label">Decision avg</div>
      <div class="value">{100 * summary['average_decision_probability']:.1f}%</div>
    </div>

    <div class="state">
      <div class="label">Smoothed avg</div>
      <div class="value">{100 * summary['average_smoothed_probability']:.1f}%</div>
    </div>

    <div class="state">
      <div class="label">Warning episodes</div>
      <div class="value">{summary['warning_episodes']}</div>
    </div>

    <div class="state">
      <div class="label">Critical episodes</div>
      <div class="value">{summary['critical_episodes']}</div>
    </div>
  </div>
</section>

<div class="footer">
  Portfolio demonstration only — not validated as a production automotive safety device.
</div>

</div>
</body>
</html>
"""

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"session_report_{session_path.stem}_{stamp}"

    html_path = args.output_dir / f"{base}.html"
    json_path = args.output_dir / f"{base}.json"

    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("DriverGuardianAI V3 Session Report")
    print("=" * 72)
    print(f"Session:        {session_path}")
    print(f"Alerts:         {alert_path or 'None'}")
    print(f"Duration:       {fmt_seconds(duration)}")
    print(f"Frames:         {len(session):,}")
    print(f"Alerts:         {alert_count}")
    print(f"Maximum risk:   {100 * summary['maximum_smoothed_probability']:.1f}%")
    print(f"HTML report:    {html_path}")
    print(f"JSON summary:   {json_path}")
    print("=" * 72)

    if args.open:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()