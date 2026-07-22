"""
Export DriverGuardianAI feature-extraction source files.

Creates one report containing:
1. The complete current src/vision_agent.py.
2. Its checkpoint version when available.
3. High-scoring candidate collection-code snippets.

Output:
results/v2/feature_code_search/feature_source_export.txt
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "feature_code_search"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "feature_source_export.txt"
)

FULL_SOURCE_CANDIDATES = [
    PROJECT_ROOT / "src" / "vision_agent.py",
    (
        PROJECT_ROOT
        / "src"
        / ".ipynb_checkpoints"
        / "vision_agent-checkpoint.py"
    ),
    PROJECT_ROOT / "realtime_driver_guardian.py",
    (
        PROJECT_ROOT
        / ".ipynb_checkpoints"
        / "realtime_driver_guardian-checkpoint.py"
    ),
]

SUPPORTED_SUFFIXES = {
    ".py",
    ".ipynb",
}

EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "env",
    "venv",
    ".venv",
    "site-packages",
    "node_modules",
    "results",
    "logs",
    "models",
    "data",
}

FEATURE_TERMS = [
    "eye_aspect_ratio",
    "ear_left",
    "ear_right",
    "yawn_score",
    "interocular",
    "head_tilt",
    "left_mid_y",
    "right_mid_y",
    "hands_detected",
    "hands_on_wheel",
    "wrist.y",
    "mean_light",
    "low_light",
    "face_confidence",
    "blink_count",
    "closed_frames",
]

COLLECTION_HINTS = [
    "csv.writer",
    "to_csv",
    "dictwriter",
    "fatigue_level",
    "fatigue_score",
    "recording",
    "collection",
    "capture",
]


def read_python_file(filepath):
    return filepath.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()


def read_notebook_file(filepath):
    with filepath.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        notebook = json.load(file)

    lines = []

    for cell_index, cell in enumerate(
        notebook.get("cells", [])
    ):
        if cell.get("cell_type") != "code":
            continue

        lines.append(
            f"# ===== Notebook code cell {cell_index} ====="
        )

        source = cell.get("source", [])

        if isinstance(source, str):
            lines.extend(source.splitlines())
        else:
            for source_line in source:
                lines.extend(
                    str(source_line).splitlines()
                )

    return lines


def read_source(filepath):
    if filepath.suffix.lower() == ".py":
        return read_python_file(filepath)

    if filepath.suffix.lower() == ".ipynb":
        return read_notebook_file(filepath)

    return []


def should_skip(filepath):
    relative_parts = filepath.relative_to(
        PROJECT_ROOT
    ).parts[:-1]

    return any(
        part.lower() in EXCLUDED_DIRECTORIES
        for part in relative_parts
    )


def discover_source_files():
    files = []

    for filepath in PROJECT_ROOT.rglob("*"):
        if not filepath.is_file():
            continue

        if filepath.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if should_skip(filepath):
            continue

        files.append(filepath)

    return sorted(files)


def score_source(lines):
    text = "\n".join(lines).lower()

    matched_features = [
        term
        for term in FEATURE_TERMS
        if term.lower() in text
    ]

    matched_hints = [
        term
        for term in COLLECTION_HINTS
        if term.lower() in text
    ]

    score = (
        3 * len(matched_features)
        + 2 * len(matched_hints)
    )

    return score, matched_features, matched_hints


def matching_line_numbers(lines):
    numbers = []

    terms = [
        *FEATURE_TERMS,
        *COLLECTION_HINTS,
    ]

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        lower_line = line.lower()

        if any(
            term.lower() in lower_line
            for term in terms
        ):
            numbers.append(line_number)

    return numbers


def merge_context_ranges(
    line_numbers,
    line_count,
    context=18,
):
    ranges = []

    for line_number in sorted(set(line_numbers)):
        start = max(1, line_number - context)
        end = min(line_count, line_number + context)

        if (
            ranges
            and start <= ranges[-1][1] + 3
        ):
            ranges[-1] = (
                ranges[-1][0],
                max(ranges[-1][1], end),
            )
        else:
            ranges.append((start, end))

    return ranges


def format_numbered_lines(
    lines,
    start=1,
    end=None,
):
    if end is None:
        end = len(lines)

    return "\n".join(
        (
            f"{line_number:5d}: "
            f"{lines[line_number - 1]}"
        )
        for line_number in range(
            start,
            end + 1,
        )
    )


def main():
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Feature Source Export")
    print("=" * 72)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = [
        "DriverGuardianAI V2",
        "Feature Source Export",
        "=" * 72,
        "",
    ]

    exported_full_files = []

    for filepath in FULL_SOURCE_CANDIDATES:
        if not filepath.exists():
            continue

        lines = read_source(filepath)
        relative_path = filepath.relative_to(
            PROJECT_ROOT
        )

        exported_full_files.append(
            str(relative_path)
        )

        report.extend(
            [
                "",
                "#" * 88,
                f"FULL FILE: {relative_path}",
                "#" * 88,
                "",
                format_numbered_lines(lines),
            ]
        )

    full_candidate_set = {
        filepath.resolve()
        for filepath in FULL_SOURCE_CANDIDATES
        if filepath.exists()
    }

    ranked_candidates = []

    for filepath in discover_source_files():
        if filepath.resolve() in full_candidate_set:
            continue

        try:
            lines = read_source(filepath)
        except Exception:
            continue

        score, features, hints = score_source(lines)

        if score < 12 or len(features) < 2:
            continue

        ranked_candidates.append(
            {
                "filepath": filepath,
                "lines": lines,
                "score": score,
                "features": features,
                "hints": hints,
            }
        )

    ranked_candidates.sort(
        key=lambda item: (
            item["score"],
            len(item["features"]),
        ),
        reverse=True,
    )

    ranked_candidates = ranked_candidates[:12]

    report.extend(
        [
            "",
            "=" * 88,
            "HIGH-SCORING COLLECTION CANDIDATES",
            "=" * 88,
        ]
    )

    for candidate_number, candidate in enumerate(
        ranked_candidates,
        start=1,
    ):
        filepath = candidate["filepath"]
        lines = candidate["lines"]
        relative_path = filepath.relative_to(
            PROJECT_ROOT
        )

        report.extend(
            [
                "",
                "-" * 88,
                (
                    f"CANDIDATE {candidate_number}: "
                    f"{relative_path}"
                ),
                f"RELEVANCE SCORE: {candidate['score']}",
                (
                    "FEATURE TERMS: "
                    + ", ".join(candidate["features"])
                ),
                (
                    "COLLECTION HINTS: "
                    + ", ".join(candidate["hints"])
                ),
                "-" * 88,
            ]
        )

        ranges = merge_context_ranges(
            matching_line_numbers(lines),
            len(lines),
        )

        for context_number, (start, end) in enumerate(
            ranges,
            start=1,
        ):
            report.extend(
                [
                    "",
                    (
                        f"Context {context_number}: "
                        f"lines {start}-{end}"
                    ),
                    "",
                    format_numbered_lines(
                        lines,
                        start,
                        end,
                    ),
                ]
            )

    OUTPUT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nFull files exported:")

    for filepath in exported_full_files:
        print(f"- {filepath}")

    print("\nAdditional collection candidates:")

    for candidate in ranked_candidates:
        print(
            "- "
            f"{candidate['filepath'].relative_to(PROJECT_ROOT)} "
            f"(score {candidate['score']})"
        )

    print("\nExport saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()