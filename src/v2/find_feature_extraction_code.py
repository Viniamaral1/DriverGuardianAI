"""
Search DriverGuardianAI source files for feature-extraction code.

Scans Python and Jupyter files for:
EAR, yawn score, head tilt, hands, low light, face confidence,
blink count, and MediaPipe usage.

Output:
results/v2/feature_code_search/feature_code_search_report.txt
results/v2/feature_code_search/matching_files.csv
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "feature_code_search"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "feature_code_search_report.txt"
)

MATCHING_FILES_PATH = (
    OUTPUT_DIRECTORY
    / "matching_files.csv"
)

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

SEARCH_GROUPS = {
    "ear": [
        "eye_aspect_ratio",
        "ear_left",
        "ear_right",
        "ear =",
        "ear_threshold",
    ],
    "yawn": [
        "yawn_score",
        "mouth opening",
        "mouth_aspect",
        "interocular",
    ],
    "head_tilt": [
        "head_tilt",
        "left_mid_y",
        "right_mid_y",
        "head tilt",
    ],
    "hands": [
        "hands_detected",
        "hands_on_wheel",
        "wrist.y",
        "wrist",
    ],
    "low_light": [
        "low_light",
        "mean_light",
        "brightness",
        "cvtcolor",
    ],
    "face_confidence": [
        "face_confidence",
        "min_detection_confidence",
        "detection_confidence",
    ],
    "blink": [
        "blink_count",
        "closed_frames",
        "blink_window",
        "minimum_closed_frames",
    ],
    "mediapipe": [
        "mediapipe",
        "face_mesh",
        "hands.process",
    ],
}


def should_skip(filepath: Path) -> bool:
    """
    Skip generated files, dependencies, data, logs, and models.
    """

    relative_parts = filepath.relative_to(
        PROJECT_ROOT
    ).parts[:-1]

    return any(
        part.lower() in EXCLUDED_DIRECTORIES
        for part in relative_parts
    )


def discover_source_files():
    """
    Find supported source files recursively.
    """

    files = []

    for filepath in PROJECT_ROOT.rglob("*"):

        if not filepath.is_file():
            continue

        if filepath.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if should_skip(
            filepath
        ):
            continue

        files.append(
            filepath
        )

    return sorted(
        files
    )


def read_python(filepath: Path):
    """
    Read Python source lines.
    """

    return filepath.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()


def read_notebook(filepath: Path):
    """
    Extract source lines from notebook code cells.
    """

    with filepath.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        notebook = json.load(
            file
        )

    lines = []

    for cell_index, cell in enumerate(
        notebook.get(
            "cells",
            []
        )
    ):

        if cell.get(
            "cell_type"
        ) != "code":
            continue

        lines.append(
            f"# Notebook code cell {cell_index}"
        )

        source = cell.get(
            "source",
            []
        )

        if isinstance(
            source,
            str
        ):
            lines.extend(
                source.splitlines()
            )

        else:
            for item in source:
                lines.extend(
                    str(
                        item
                    ).splitlines()
                )

    return lines


def read_source(filepath: Path):
    """
    Read a supported source file.
    """

    if filepath.suffix.lower() == ".py":
        return read_python(
            filepath
        )

    return read_notebook(
        filepath
    )


def find_matches(lines):
    """
    Find line numbers matching each feature group.
    """

    lower_lines = [
        line.lower()
        for line in lines
    ]

    matches = {}

    for group, terms in SEARCH_GROUPS.items():

        line_numbers = []

        for line_number, line in enumerate(
            lower_lines,
            start=1,
        ):

            if any(
                term.lower() in line
                for term in terms
            ):
                line_numbers.append(
                    line_number
                )

        if line_numbers:
            matches[group] = line_numbers

    return matches


def create_ranges(
    line_numbers,
    line_count,
    context=12,
):
    """
    Merge overlapping source-code contexts.
    """

    ranges = []

    for line_number in sorted(
        set(
            line_numbers
        )
    ):

        start = max(
            1,
            line_number - context,
        )

        end = min(
            line_count,
            line_number + context,
        )

        if (
            ranges
            and start <= ranges[-1][1] + 2
        ):
            ranges[-1] = (
                ranges[-1][0],
                max(
                    ranges[-1][1],
                    end,
                ),
            )

        else:
            ranges.append(
                (
                    start,
                    end,
                )
            )

    return ranges


def format_context(
    lines,
    start,
    end,
):
    """
    Format numbered source lines.
    """

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
    """
    Run the source-code search.
    """

    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Feature Extraction Code Search")
    print("=" * 72)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_files = discover_source_files()

    print(
        f"\nSource files discovered: {len(source_files)}"
    )

    report = [
        "DriverGuardianAI V2",
        "Feature Extraction Code Search Report",
        "=" * 72,
        "",
    ]

    matching_rows = []

    for filepath in source_files:

        try:
            lines = read_source(
                filepath
            )

        except Exception as error:
            print(
                f"Could not read {filepath}: {error}"
            )
            continue

        matches = find_matches(
            lines
        )

        if not matches:
            continue

        relative_path = filepath.relative_to(
            PROJECT_ROOT
        )

        groups = sorted(
            matches.keys()
        )

        all_line_numbers = []

        for numbers in matches.values():
            all_line_numbers.extend(
                numbers
            )

        matching_rows.append(
            {
                "source_path": str(
                    relative_path
                ),
                "matched_groups": " | ".join(
                    groups
                ),
                "matching_line_count": len(
                    set(
                        all_line_numbers
                    )
                ),
            }
        )

        report.extend(
            [
                "",
                "=" * 72,
                f"FILE: {relative_path}",
                (
                    "MATCHED GROUPS: "
                    + ", ".join(
                        groups
                    )
                ),
                "=" * 72,
            ]
        )

        ranges = create_ranges(
            all_line_numbers,
            len(
                lines
            ),
        )

        for index, (
            start,
            end,
        ) in enumerate(
            ranges,
            start=1,
        ):
            report.extend(
                [
                    "",
                    (
                        f"Context {index}: "
                        f"lines {start}-{end}"
                    ),
                    "-" * 72,
                    format_context(
                        lines,
                        start,
                        end,
                    ),
                ]
            )

    REPORT_PATH.write_text(
        "\n".join(
            report
        ),
        encoding="utf-8",
    )

    with MATCHING_FILES_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_path",
                "matched_groups",
                "matching_line_count",
            ],
        )

        writer.writeheader()
        writer.writerows(
            matching_rows
        )

    print(
        f"Matching source files: {len(matching_rows)}"
    )

    print(
        "\nMatching files:"
    )

    for row in matching_rows:
        print(
            f"- {row['source_path']} "
            f"[{row['matched_groups']}]"
        )

    print(
        "\nReport saved to:"
    )
    print(
        REPORT_PATH
    )

    print(
        "\nFile index saved to:"
    )
    print(
        MATCHING_FILES_PATH
    )


if __name__ == "__main__":
    main()