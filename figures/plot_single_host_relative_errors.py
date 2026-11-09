#!/usr/bin/env python3
"""Plot relative mean errors and paired-bootstrap intervals from JSON.

Input is the corresponding JSON produced by single_host_results_generation.py.
Only Matplotlib and NumPy are required; no statistics are recomputed here.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_plot_data(
    input_file: Path,
    group_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read this plot's JSON and select a group without silently discarding others."""
    with input_file.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("The input JSON must contain an object.")
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported or missing JSON schema_version (expected 1).")
    if data.get("plot_type") != "relative_errors":
        raise ValueError("Expected plot_type 'relative_errors' in the input JSON.")

    groups = data.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("The input JSON must contain a nonempty 'groups' object.")
    available = ", ".join(groups)
    if group_name is None:
        if len(groups) != 1:
            raise ValueError(
                f"This JSON contains multiple groups; choose --group from: {available}"
            )
        group_name = next(iter(groups))
    if group_name not in groups:
        raise ValueError(f"Unknown group {group_name!r}. Available groups: {available}")

    group = groups[group_name]
    if not isinstance(group, dict):
        raise ValueError(f"Group {group_name!r} must contain an object.")
    results = group.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"Group {group_name!r} must contain a nonempty 'results' list.")
    if not all(isinstance(result, dict) for result in results):
        raise ValueError("Every result must be an object.")
    return data, group


def save_figure(fig: plt.Figure, output_file: Path) -> None:
    """Save a PDF using the original layout, then close the figure."""
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {output_file}")


def plot_relative_errors_vs_reference(
    results: Sequence[dict[str, Any]],
    *,
    reference_label: str,
    output_file: Path,
) -> None:
    """Render stored paired-bootstrap estimates and intervals in percent."""
    if not results:
        raise ValueError("No results to plot.")
    results = sorted(results, key=lambda result: result["estimate_percent"])
    algorithm_names = [result["label"] for result in results]
    estimates = np.array([result["estimate_percent"] for result in results], dtype=float)
    lower_bounds = np.array([result["ci_lower_percent"] for result in results], dtype=float)
    upper_bounds = np.array([result["ci_upper_percent"] for result in results], dtype=float)
    if not np.isfinite(np.vstack([estimates, lower_bounds, upper_bounds])).all():
        raise ValueError("Relative errors and confidence bounds must be finite.")
    error_bars = np.vstack([estimates - lower_bounds, upper_bounds - estimates])
    if np.any(error_bars < 0):
        raise ValueError(
            "An estimate lies outside its confidence interval; the original "
            "centered error-bar rendering cannot represent this interval."
        )
    y_positions = np.arange(len(algorithm_names))

    figure_height = max(4.0, 0.55 * len(algorithm_names))
    fig, ax = plt.subplots(figsize=(10, figure_height))
    ax.errorbar(
        estimates,
        y_positions,
        xerr=error_bars,
        fmt="o",
        capsize=6,
        linewidth=3,
    )

    # Print text labels
    for x,y in zip(estimates, y_positions):
        ax.text(x-3, y-0.25,f"{x:.2f}%", fontsize=16)

    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(algorithm_names, fontsize=18)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"Increase in mean error relative to {reference_label} (%)", fontsize=18
    )
    ax.tick_params(axis='x', labelsize=16)
    ax.set_ylabel("Heuristic", fontsize=18)
    # ax.set_title(
    #     f"Relative mean error compared with {reference_label} "
    #     f"({100.0 * confidence:g}% paired-bootstrap intervals)"
    # )
    ax.grid(axis="x", alpha=0.3)

    # Adjust the limits for prettiness
    ax.set_ylim(ymax=ax.get_ylim()[1] - 0.5)
    ax.set_xlim(xmin=-2)

    save_figure(fig, output_file)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot relative mean errors and paired-bootstrap intervals from JSON.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--json-result-file", required=True, type=Path, metavar="PATH",
        help="Input relative-errors JSON file from single_host_results_generation.py.",
    )
    parser.add_argument(
        "--output-pdf", required=True, type=Path, metavar="PATH",
        help="Output PDF path; parent directories are created as needed.",
    )
    parser.add_argument(
        "--group", metavar="NAME",
        help="Failure-multiplier group to plot (automatic when there is only one).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.json_result_file.resolve() == args.output_pdf.resolve():
        parser.error("The input JSON and output PDF paths must be distinct.")
    try:
        data, group = load_plot_data(args.json_result_file, args.group)
        plot_relative_errors_vs_reference(
            group["results"],
            reference_label=data["reference_label"],
            output_file=args.output_pdf,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
