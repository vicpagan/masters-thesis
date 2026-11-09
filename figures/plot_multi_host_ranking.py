#!/usr/bin/env python3
"""
Plot multi-PE scheduler degradation data from JSON, using a fixed
algorithm order across N-groups and a colorblind-friendly palette.

Expected JSON shape:
{
  "2":  {"algorithm_name": 0.0, "other_algorithm": 0.16, ...},
  "4":  {...},
  "8":  {...},
  "16": {...}
}

By default, values are interpreted as fractional degradations:
0.16 -> 16%.

Use --broken-begin and --broken-end to omit an interval of the linear
x-axis. Bounds are in displayed percentage units (70 means 70%), even
when the JSON contains fractions. No bar may end inside the open interval.
Use --broken-gap to set the actual spacing between the two panels as a
fraction of their average width (default: 0.06). Zero removes the spacing;
small positive values create a narrow break. Bounds and data are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import ScaledTranslation


#DEFAULT_LABELS: dict[str, str] = {
#    "dynamic__dependent__aggressive": "DYN / DEP / AGG",
#    "greedy_foresighted_expected_error__dependent__variant":
#        "G-FS-X / DEP / VAR",
#    "greedy_foresighted_success_error_ratio__dependent__aggressive":
#        "G-FS-R / DEP / AGG",
#    "greedy_foresighted_expected_error__dependent__off":
#        "G-FS-X / DEP / OFF",
#}

DEFAULT_LABELS: dict[str, str] = {
    "dynamic__dependent__aggressive": "DYN",
    "greedy_foresighted_expected_error__dependent__aggressive": "FS-X",
    "greedy_foresighted_error_level__dependent__aggressive": "FS-E",
    "greedy_foresighted_success_error_ratio__dependent__aggressive": "FS-R",
}


# 15-color, colorblind-friendly palette.
DEFAULT_COLORBLIND_PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
    "#332288",  # dark blue
    "#88CCEE",  # light blue
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#AA4499",  # magenta
)


def _load_ranking_data(
    source: str | Path | Mapping[str | int, Mapping[str, Any]],
) -> dict[int, dict[str, float]]:
    """Load and validate ranking data from a JSON path or mapping."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except OSError as exc:
            raise OSError(f"Could not read JSON file {path!s}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path!s}: {exc}") from exc
    elif isinstance(source, Mapping):
        raw = source
    else:
        raise TypeError("source must be a JSON path or a mapping")

    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("The JSON root must be a non-empty object")

    parsed: dict[int, dict[str, float]] = {}
    for n_key, algorithms in raw.items():
        try:
            n = int(n_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid processor-count key: {n_key!r}") from exc
        if n <= 0:
            raise ValueError(f"Processor count must be positive; got {n}")

        if not isinstance(algorithms, Mapping) or not algorithms:
            raise ValueError(f"Entry for N={n} must be a non-empty object")

        parsed[n] = {}
        for algorithm, value in algorithms.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric value for {algorithm!r} at N={n}: {value!r}"
                ) from exc
            if not math.isfinite(numeric_value) or numeric_value < 0:
                raise ValueError(
                    f"Value for {algorithm!r} at N={n} must be finite "
                    f"and nonnegative; got {numeric_value}"
                )
            parsed[n][str(algorithm)] = numeric_value

    return dict(sorted(parsed.items()))


def _default_short_label(name: str) -> str:
    """Create a compact fallback label for an algorithm name."""
    if name in DEFAULT_LABELS:
        return DEFAULT_LABELS[name]

    parts = name.split("__")
    base_map = {
        "dynamic": "DYN",
        "random": "RAND",
        "greedy_foresighted_expected_error": "G-FS-X",
        "greedy_nearsighted_expected_error": "G-NS-X",
        "greedy_foresighted_probability_success": "G-FS-P",
        "greedy_nearsighted_probability_success": "G-NS-P",
        "greedy_foresighted_success_error_ratio": "G-FS-R",
        "greedy_nearsighted_success_error_ratio": "G-NS-R",
        "greedy_foresighted_error_level": "G-FS-E",
        "greedy_nearsighted_error_level": "G-NS-E",
    }
    option_map = {
        "independent": "IND",
        "dependent": "DEP",
        "aggressive": "AGG",
        "variant": "VAR",
        "off": "OFF",
    }

    base = base_map.get(parts[0], parts[0].replace("_", "-"))
    suffix = [option_map.get(part, part.replace("_", "-")) for part in parts[1:]]
    return " / ".join([base, *suffix])


def _ticks_in_range(start: float, end: float, step: float) -> list[float]:
    """Return regularly spaced ticks in the closed interval [start, end]."""
    if step <= 0 or not math.isfinite(step):
        raise ValueError("tick step must be finite and positive")

    epsilon = step * 1e-9
    first = math.ceil((start - epsilon) / step) * step
    ticks: list[float] = []
    tick = first
    while tick <= end + epsilon:
        snapped = round(tick / step) * step
        ticks.append(0.0 if math.isclose(snapped, 0.0, abs_tol=epsilon) else snapped)
        tick += step
    return ticks


def plot_multi_host_ranking(
    source: str | Path | Mapping[str | int, Mapping[str, Any]],
    output_path: str | Path | None = None,
    *,
    label_map: Mapping[str, str] | None = None,
    palette: list[str] | tuple[str, ...] | None = None,
    values_are_fractions: bool = True,
    sort_within_group: bool = False,
    algorithm_order: list[str] | tuple[str, ...] | None = None,
    figure_width: float = 3.45,
    row_height: float = 0.285,
    bar_height: float = 0.68,
    vertical_scale: float = 1.0,
    header_gap: float = 0.55,
    row_gap: float = 0.92,
    group_gap: float = 0.22,
    y_margin: float = 0.015,
    xscale: Literal["linear", "symlog"] = "linear",
    symlog_linthresh: float = 1.0,
    value_decimals: int = 1,
    dpi: int = 300,
    broken_begin: float | None = None,
    broken_end: float | None = None,
    broken_gap: float = 0.06,
    xtick_step: float | None = None,
) -> tuple[Figure, Axes]:
    """
    Create a tall, one-column horizontal bar chart grouped by processor count.

    When both broken_begin and broken_end are supplied, omit that open
    interval from the linear x-axis. Bounds use displayed percentage units.
    The two panels have identical data-unit widths; crossing bars are
    clipped at the break, and each value label appears only once.

    broken_gap is the actual gap as a fraction of the average panel width
    (default 0.06): gap = broken_gap * (left_width + right_width) / 2.
    No automatic minimum padding is added. Zero makes the panels touch;
    small positive values retain a visible break. Boundary tick labels
    extend into their own panels rather than into the gap.
    It has no effect without an axis break.

    xtick_step optionally requests regularly spaced major ticks in displayed
    percentage units. With a broken axis, the same global tick sequence is
    used on both panels and any ticks that fall strictly inside the omitted
    interval are simply not shown.

    Returns (fig, ax), as before. With a break, ax is the left panel and
    fig.axes[1] is the right panel. Broken axes cannot be combined with symlog.
    """
    data = _load_ranking_data(source)
    labels = dict(DEFAULT_LABELS)
    if label_map:
        labels.update(label_map)

    if palette is None:
        palette = list(DEFAULT_COLORBLIND_PALETTE)
    else:
        palette = list(palette)
    if not palette:
        raise ValueError("palette must contain at least one color")

    scale = 100.0 if values_are_fractions else 1.0

    if bar_height <= 0:
        raise ValueError("bar_height must be positive")
    if vertical_scale <= 0:
        raise ValueError("vertical_scale must be positive")
    if header_gap <= 0 or row_gap <= 0 or group_gap < 0:
        raise ValueError(
            "header_gap and row_gap must be positive; group_gap must be nonnegative"
        )
    if y_margin < 0:
        raise ValueError("y_margin must be nonnegative")

    if xscale not in ("linear", "symlog"):
        raise ValueError("xscale must be either 'linear' or 'symlog'")
    if not math.isfinite(broken_gap) or broken_gap < 0:
        raise ValueError("--broken-gap must be finite and nonnegative")
    if xtick_step is not None and (not math.isfinite(xtick_step) or xtick_step <= 0):
        raise ValueError("--xtick-step must be finite and positive")
    if (broken_begin is None) != (broken_end is None):
        raise ValueError("Supply both --broken-begin and --broken-end, or neither")
    has_break = broken_begin is not None
    if has_break:
        assert broken_begin is not None and broken_end is not None
        if not (
            math.isfinite(broken_begin)
            and math.isfinite(broken_end)
            and 0 < broken_begin < broken_end
        ):
            raise ValueError("Break bounds must be finite and satisfy 0 < begin < end")
        if xscale != "linear":
            raise ValueError("A broken x-axis cannot be combined with --symlog")
    elif xtick_step is not None and xscale == "symlog":
        raise ValueError("--xtick-step is supported only on a linear x-axis")

    # Fixed global order by default so the eye can scan consistently.
    all_algorithms = list(
        dict.fromkeys(
            algorithm
            for _, algorithms in data.items()
            for algorithm in algorithms.keys()
        )
    )
    if algorithm_order is None:
        global_order = all_algorithms
    else:
        seen = set()
        global_order = []
        for algorithm in algorithm_order:
            if algorithm in all_algorithms and algorithm not in seen:
                global_order.append(algorithm)
                seen.add(algorithm)
        for algorithm in all_algorithms:
            if algorithm not in seen:
                global_order.append(algorithm)
                seen.add(algorithm)

    color_map = {
        algorithm: palette[index % len(palette)]
        for index, algorithm in enumerate(global_order)
    }

    # Build rows.
    records: list[tuple[float, int, str, float]] = []
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    header_tick_indices: set[int] = set()

    y = 0.0
    for group_index, (n, algorithms) in enumerate(data.items()):
        header_tick_indices.add(len(tick_positions))
        tick_positions.append(y)
        tick_labels.append(f"N = {n}")
        y += header_gap * vertical_scale

        if sort_within_group:
            group_items = list(algorithms.items())
            group_items.sort(key=lambda item: (item[1], item[0]))
            ordered_algorithms = [name for name, _ in group_items]
        else:
            ordered_algorithms = [name for name in global_order if name in algorithms]

        for algorithm in ordered_algorithms:
            value = algorithms[algorithm] * scale
            records.append((y, n, algorithm, value))
            tick_positions.append(y)
            tick_labels.append(labels.get(algorithm, _default_short_label(algorithm)))
            y += row_gap * vertical_scale

        if group_index != len(data) - 1:
            y += group_gap * vertical_scale

    figure_height = max(
        2.2,
        row_height * vertical_scale * len(tick_positions) + 0.35,
    )

    max_value = max((value for _, _, _, value in records), default=0.0)
    xmax = max_value * 1.17 if max_value > 0 else 1.0
    if has_break:
        assert broken_begin is not None and broken_end is not None
        hidden_endpoints = [
            f"N={n}, {algorithm}: {value:g}%"
            for _, n, algorithm, value in records
            if broken_begin < value < broken_end
        ]
        if hidden_endpoints:
            raise ValueError(
                f"The break ({broken_begin:g}%, {broken_end:g}%) contains "
                "bar endpoints: " + "; ".join(hidden_endpoints)
            )
        if broken_end > max_value:
            raise ValueError(
                f"--broken-end must not exceed the largest plotted value "
                f"({max_value:g}%)"
            )
        # Base label headroom on the retained range, not the omitted interval.
        visible_span = broken_begin + (max_value - broken_end)
        xmax = max_value + 0.17 * visible_span

    with plt.rc_context(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
        }
    ):
        if has_break:
            assert broken_begin is not None and broken_end is not None
            fig, (ax, ax_right) = plt.subplots(
                1, 2,
                sharey=True,
                figsize=(figure_width, figure_height),
                constrained_layout=True,
                gridspec_kw={
                    # Equal physical length per percentage point on both sides.
                    "width_ratios": [broken_begin, xmax - broken_end],
                    # Used only to determine the outer margins. The actual
                    # gap is set explicitly after that initial layout.
                    "wspace": 0.0,
                },
            )
            axes = [ax, ax_right]
        else:
            fig, ax = plt.subplots(
                figsize=(figure_width, figure_height),
                constrained_layout=True,
            )
            axes = [ax]

        for algorithm in global_order:
            algorithm_rows = [
                (row_y, value)
                for row_y, _, row_algorithm, value in records
                if row_algorithm == algorithm
            ]
            if not algorithm_rows:
                continue
            # Draw the original bars on both panels; the x-limits clip out
            # the omitted interval without modifying the actual data.
            for panel in axes:
                panel.barh(
                    [row_y for row_y, _ in algorithm_rows],
                    [value for _, value in algorithm_rows],
                    height=bar_height * vertical_scale,
                    color=color_map.get(algorithm),
                    edgecolor="black",
                    linewidth=0.35,
                )

        left_annotations = []
        for row_y, _, _, value in records:
            text = "← best (0%)" if math.isclose(value, 0.0, abs_tol=1e-12) else f"{value:.1f}%"
            # Keep each annotation next to its true endpoint, not at the cut.
            label_ax = ax_right if has_break and value >= broken_end else ax
            annotation = label_ax.annotate(
                text,
                xy=(value, row_y),
                xytext=(2, -0.5),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6.2,
                clip_on=False,
            )
            if has_break and label_ax is ax:
                # First lay out a narrow gap. Labels near the cut are fitted
                # inside their bars below, rather than widening that gap.
                annotation.set_in_layout(False)
                left_annotations.append(annotation)

        ax.set_yticks(tick_positions, tick_labels)
        ax.tick_params(axis="y", length=0)

        for index, tick_label in enumerate(ax.get_yticklabels()):
            if index in header_tick_indices:
                tick_label.set_fontweight("bold")

        ax.invert_yaxis()
        if tick_positions:
            ymin = min(tick_positions) - y_margin
            bottom_margin = y_margin
            if has_break:
                # Keep the last row clear of the bottom spine and break marks.
                bottom_margin = max(
                    y_margin, (bar_height / 2 + 0.12 * row_gap) * vertical_scale
                )
            ymax = max(tick_positions) + bottom_margin
            ax.set_ylim(ymax, ymin)
        ax.margins(y=0)

        for panel in axes:
            panel.grid(axis="x", linestyle=":", linewidth=0.55)
            panel.set_axisbelow(True)
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)
            panel.xaxis.set_major_formatter("{x:g}%")

        if has_break:
            assert broken_begin is not None and broken_end is not None
            ax.set_xlim(0, broken_begin)
            ax_right.set_xlim(broken_end, xmax)
            ax_right.spines["left"].set_visible(False)
            ax_right.tick_params(axis="y", left=False, labelleft=False)

            # Slashes at the two ends of the broken bottom spine. Using marker
            # coordinates keeps their angle and size fixed when the plot resizes.
            break_style = dict(
                marker=[(-0.5, -1), (0.5, 1)],
                markersize=5,
                linestyle="none",
                color=plt.rcParams["axes.edgecolor"],
                markeredgewidth=0.7,
                clip_on=False,
                zorder=4,
            )
            ax.plot([1], [0], transform=ax.transAxes, **break_style)
            ax_right.plot([0], [0], transform=ax_right.transAxes, **break_style)

            # Mark both cut ends of every bar that resumes beyond the gap.
            crossing_rows = [
                row_y for row_y, _, _, value in records if value > broken_end
            ]
            if crossing_rows:
                ax.plot(
                    [1] * len(crossing_rows), crossing_rows,
                    transform=ax.get_yaxis_transform(), **break_style,
                )
                ax_right.plot(
                    [0] * len(crossing_rows), crossing_rows,
                    transform=ax_right.get_yaxis_transform(), **break_style,
                )
            xlabel = fig.supxlabel(
                "Degradation from best mean error",
                fontsize=plt.rcParams["axes.labelsize"],
            )
            fig.canvas.draw()

            # Keep the automatically computed OUTER margins, but not the
            # automatic inter-panel padding. Constrained layout reserves
            # space for tick labels even with wspace=0, which otherwise
            # prevents --broken-gap from producing a genuinely narrow cut.
            left_box = ax.get_position().frozen()
            right_box = ax_right.get_position().frozen()
            if hasattr(fig, "set_layout_engine"):
                fig.set_layout_engine("none")
            else:  # Matplotlib versions predating the layout-engine API.
                fig.set_constrained_layout(False)
                fig.set_tight_layout(False)

            # W = left_width + gap + right_width, with
            # gap = broken_gap * (left_width + right_width) / 2.
            # Preserve the data-unit scale on both panels.
            total_width = right_box.x1 - left_box.x0
            panel_width = total_width / (1.0 + broken_gap / 2.0)
            gap_width = panel_width * broken_gap / 2.0
            left_width = panel_width * broken_begin / (
                broken_begin + xmax - broken_end
            )
            right_width = panel_width - left_width
            ax.set_position([
                left_box.x0, left_box.y0, left_width, left_box.height,
            ])
            ax_right.set_position([
                left_box.x0 + left_width + gap_width,
                left_box.y0, right_width, left_box.height,
            ])

            if xtick_step is not None:
                ax.set_xticks(_ticks_in_range(0.0, broken_begin, xtick_step))
                ax_right.set_xticks(_ticks_in_range(broken_end, xmax, xtick_step))
            else:
                # Explicitly label both break boundaries; omit a neighboring
                # automatic tick when it would be too close to the boundary.
                for panel, boundary in ((ax, broken_begin), (ax_right, broken_end)):
                    automatic_ticks = list(panel.get_xticks())
                    step = (
                        automatic_ticks[1] - automatic_ticks[0]
                        if len(automatic_ticks) > 1 else 0.0
                    )
                    lo, hi = panel.get_xlim()
                    ticks = [
                        tick for tick in automatic_ticks
                        if lo <= tick <= hi and abs(tick - boundary) > 0.6 * step
                    ]
                    panel.set_xticks(sorted([*ticks, boundary]))
                    panel.set_xlim(lo, hi)

                    # Keep boundary labels out of the gap. A small inward
                    # offset also separates them when the requested gap is zero.
                    alignment = "right" if panel is ax else "left"
                    offset_points = -1.5 if panel is ax else 1.5
                    for tick_value, tick_label in zip(
                        panel.get_xticks(), panel.get_xticklabels()
                    ):
                        if tick_value == boundary:
                            tick_label.set_horizontalalignment(alignment)
                            tick_label.set_transform(
                                tick_label.get_transform() + ScaledTranslation(
                                    offset_points / 72.0, 0, fig.dpi_scale_trans
                                )
                            )

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for annotation in left_annotations:
                bbox = annotation.get_window_extent(renderer)
                endpoint_x = ax.transData.transform(annotation.xy)[0]
                bar_width = endpoint_x - ax.bbox.x0
                if bbox.x1 > ax.bbox.x1 - 3 and bar_width > bbox.width + 6:
                    annotation.set_horizontalalignment("right")
                    annotation.set_position((-2, -0.5))
                    annotation.set_bbox(dict(
                        facecolor=ax.get_facecolor(), edgecolor="none",
                        alpha=0.9, pad=0.2,
                    ))
                annotation.set_in_layout(True)

            # Center the common label under the plotting panels, excluding
            # the y-label column. Redraw after fitting the value labels.
            fig.canvas.draw()
            xlabel.set_x((ax.get_position().x0 + ax_right.get_position().x1) / 2)
        else:
            ax.set_xlabel("Degradation from best mean error")
            if xscale == "symlog":
                ax.set_xticks([0, 1, 10, 100], ["0", "1", "10", "100"])
                ax.set_xscale("symlog", linthresh=symlog_linthresh)
            else:
                if xtick_step is not None:
                    ax.set_xticks(_ticks_in_range(0.0, xmax, xtick_step))
            ax.set_xlim(0, xmax)

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=dpi, bbox_inches="tight",pad_inches=0)

    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot multi-PE scheduler degradation rankings."
    )
    parser.add_argument("json_file", type=Path, help="Input JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("multi_host_ranking.pdf"),
        help="Output plot path (default: multi_host_ranking.pdf)",
    )
    parser.add_argument(
        "--already-percent",
        action="store_true",
        help="Treat JSON values as percentages instead of fractions",
    )
    parser.add_argument(
        "--symlog",
        action="store_true",
        help="Use a symmetric-log x-axis to reveal small degradations",
    )
    parser.add_argument(
        "--broken-begin",
        type=float,
        help=(
            "Start of the omitted x-axis interval, in displayed percentage "
            "units (e.g. 70 means 70%%); requires --broken-end and a linear axis"
        ),
    )
    parser.add_argument(
        "--broken-end",
        type=float,
        help=(
            "End of the omitted x-axis interval, in displayed percentage "
            "units (e.g. 200 means 200%%); requires --broken-begin. "
            "No bar may end strictly between the two bounds"
        ),
    )
    parser.add_argument(
        "--broken-gap",
        type=float,
        default=0.06,
        help=(
            "Actual gap as a fraction of the average panel width "
            "(default: %(default)s); e.g. 0.02 is a narrow gap and 0 makes "
            "the panels touch. No automatic minimum padding is added. "
            "Must be finite and nonnegative. Used only with an axis break"
        ),
    )
    parser.add_argument(
        "--xtick-step",
        type=float,
        help=(
            "Regular major-tick spacing in displayed percentage units; "
            "e.g. --xtick-step 50 shows 0%%, 50%%, 100%%, ... . "
            "With a broken axis, ticks inside the omitted interval are hidden"
        ),
    )
    parser.add_argument(
        "--bar-height",
        type=float,
        default=0.68,
        help="Base horizontal-bar thickness (default: 0.68)",
    )
    parser.add_argument(
        "--vertical-scale",
        type=float,
        default=1.0,
        help=(
            "Scale all vertical dimensions; e.g. 0.8 makes the figure "
            "about 20%% shorter (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--header-gap",
        type=float,
        default=0.55,
        help="Gap from each N-label row to the first bar in that group (default: 0.55)",
    )
    parser.add_argument(
        "--row-gap",
        type=float,
        default=0.92,
        help="Gap between neighboring algorithm rows (default: 0.92)",
    )
    parser.add_argument(
        "--group-gap",
        type=float,
        default=0.22,
        help="Extra gap between N-groups (default: 0.22)",
    )
    parser.add_argument(
        "--y-margin",
        type=float,
        default=0.015,
        help="Fractional vertical outer margin (default: 0.015)",
    )
    args = parser.parse_args()

    try:
        plot_multi_host_ranking(
            args.json_file,
            output_path=args.output,
            values_are_fractions=not args.already_percent,
            bar_height=args.bar_height,
            vertical_scale=args.vertical_scale,
            header_gap=args.header_gap,
            row_gap=args.row_gap,
            group_gap=args.group_gap,
            y_margin=args.y_margin,
            xscale="symlog" if args.symlog else "linear",
            broken_begin=args.broken_begin,
            broken_end=args.broken_end,
            broken_gap=args.broken_gap,
            xtick_step=args.xtick_step,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
