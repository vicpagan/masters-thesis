#!/usr/bin/env python3

import argparse
import json
import subprocess
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re
import tempfile
import os
import math

def plot_results(all_results, upper_bound_simulation_results, output_file):
    """Create visualization of delta_t analysis"""

    fontsize = 26

    fig, ax1 = plt.subplots(figsize=(14, 7))

    configs = {
        'default': {
            'color_u': 'tab:red',
            'color_l': 'tab:blue',
            'label_prefix': '',
            'sim_color': 'purple'
        },
        'opt_exec': {
            'color_u': 'indianred',
            'color_l': 'lightblue',
            'label_prefix': 'Opt Exec',
            'sim_color': 'orchid'
        },
        'opt_sched': {
            'color_u': 'darkred',
            'color_l': 'darkblue',
            'label_prefix': 'Opt Sched',
            'sim_color': 'mediumorchid'
        },
        'opt_both': {
            'color_u': 'orangered',
            'color_l': 'steelblue',
            'label_prefix': 'Opt Both',
            'sim_color': 'darkviolet'
        }
    }

    ax1.set_xlabel(r'Discrete time-step $\delta$ (sec)', fontsize=fontsize)
    ax1.set_ylabel('Error', color='k', fontsize=fontsize)

    all_lines = []

    for config_key, data in all_results.items():
        if data is None:
            continue
        cfg = configs[config_key]
        delta_t = np.array(data["delta_t"])
        upper_bound = np.array(data["upper_bound"])
        lower_bound = np.array(data["lower_bound"])
        sim_results = data["simulation_avg_error"]


        # Upper bound line
        line_u = ax1.plot(
            delta_t, upper_bound,
            color=cfg['color_u'],
            linewidth=4,
            marker='o',
            label=f"{cfg['label_prefix']} Upper Bound",
            alpha=0.8
        )

        # Lower bound line
        line_l = ax1.plot(
            delta_t, lower_bound,
            color=cfg['color_l'],
            linewidth=4,
            marker='o',
            label=f"{cfg['label_prefix']} Lower Bound",
            alpha=0.8
        )

        plot_up_to = 15
        all_lines.extend(line_u + line_l)
        ax1.set_xlim(1, plot_up_to)
        ax1.set_xticks([1, 5, 10, 15])
        ax1.set_ylim(0.999*min(lower_bound[0:plot_up_to]), 1.005*max(upper_bound[0:plot_up_to]))

        # # Fill between bounds
        # ax1.fill_between(
        #     delta_t,
        #     lower_bound,
        #     upper_bound,
        #     alpha=0.05,
        #     color=cfg['color_u']
        # )

        # Simulation results (as lines instead of markers)
        if upper_bound_simulation_results:
            plot_simulation_results(upper_bound_simulation_results, delta_t, ax1, cfg, all_lines, cfg['color_u'], "Simulation (UB)")

        if sim_results:
            plot_simulation_results(sim_results, delta_t, ax1, cfg, all_lines, cfg['color_l'], "Simulation (LB)")



    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)

    # Computation time (no markers)
    if 'default' in all_results and all_results['default'] is not None:
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        comp_time = np.array(all_results['default']["computation_time_ms"]) / 1000.0
        delta_t = np.array(all_results['default']["delta_t"])

        color3 = 'tab:green'
        ax2.set_ylabel('Compute time (sec)', fontsize=fontsize)

        ax2.set_yticks([1,2, 10, 100])
        ax2.set_ylim(min(comp_time), max(comp_time))
        ax2.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, position: f'{value:g}')
        )
        ax2.yaxis.set_minor_formatter(ticker.NullFormatter())

        line_time = ax2.plot(
            delta_t,
            comp_time,
            color=color3,
            linewidth=4,
            linestyle='--',
            label='Bound compute time',
            alpha=0.7
        )

        ax2.tick_params(axis='y', labelsize=fontsize)

        time_leg = ax2.legend(
            line_time,
            ["Bound compute time"],
            loc='lower left',
            fontsize=fontsize,
            ncol=1,
            frameon=True,
            facecolor='white',
            edgecolor='black',
            framealpha=0.8
        )

    labels = [l.get_label() for l in all_lines]

    legend_ax = ax2 if ax2 is not None else ax1

    leg = legend_ax.legend(
    all_lines,
    labels,
    loc='upper left',
    bbox_to_anchor=(0.09, 1.0),
    fontsize=fontsize,
    ncol=2,
    frameon=True,
    facecolor='white',
    edgecolor='black',
    framealpha=0.8
    )
    legend_ax.add_artist(time_leg)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Plot saved to {output_file}")

    return fig

def plot_simulation_results(simulation_results, delta_t, ax1, cfg, all_lines, plot_color, plot_label):
    confidence_interval_delta = 0
    sim_delta_t = delta_t
    sim_avg_error = []
    for [avg, error_counts] in simulation_results:
        if avg is not None:
            sim_avg_error.append(avg)
            s_square = 0
            num_repeats = 0
            for error, count in error_counts.items():
                s_square += count * (float(error) - avg) * (float(error) - avg)
                num_repeats += count
            s_square = s_square / (num_repeats - 1)
            confidence_interval_delta = 1.96 * math.sqrt(s_square) / math.sqrt(num_repeats)
    #                    print(f"Error = {avg} +/- {confidence_interval_delta}")

    line_sim = ax1.plot(
        delta_t,
        sim_avg_error,
        color=plot_color,
        linewidth=2,
        linestyle='--',
        label=f"{cfg['label_prefix']} {plot_label}",
        alpha=0.9
    )
    all_lines.extend(line_sim)

    # Confidence interval half-width
    lower = [x - confidence_interval_delta for x in sim_avg_error]
    upper = [x + confidence_interval_delta for x in sim_avg_error]
    ax1.fill_between(sim_delta_t, lower, upper, alpha=0.3, color=plot_color, label='Confidence interval')


def compute_statistics(all_results, upper_bound_simulation_results):
    data = all_results["default"]
    delta_t = np.array(data["delta_t"])
    upper_bound = np.array(data["upper_bound"])
    lower_bound = np.array(data["lower_bound"])
    sim_results = data["simulation_avg_error"]
    comp_time = np.array(data["computation_time_ms"]) / 1000.0

    for idx in range(0, len(delta_t)):
        d = delta_t[idx]
        ct = comp_time[idx]
        bound_diff = upper_bound[idx] - lower_bound[idx]
        sim_diff = abs(sim_results[idx][0] - upper_bound_simulation_results[idx][0])
        perct_bound = 100.0 * bound_diff / upper_bound[idx]
        perct_sim = 100.0 * sim_diff / sim_results[idx][0]
        print(f"delta_t: {d}   %bound-diff = {perct_bound:.3}%  time: {ct:.2}  %sim-diff = {perct_sim:.3}%")


def main():
    parser = argparse.ArgumentParser(
        description="Plot delta_t results from lower-bound and optional upper-bound scheduling simulations."
    )
    parser.add_argument(
        "--lower-bound-result-file",
        required=True,
        metavar="PATH",
        help="JSON results file for lower-bound scheduling.",
    )
    parser.add_argument(
        "--upper-bound-result-file",
        metavar="PATH",
        help="Optional JSON results file for upper-bound scheduling.",
    )
    parser.add_argument(
        "--output-pdf",
        required=True,
        metavar="PATH",
        help="Path of the output PDF figure.",
    )
    args = parser.parse_args()

    with open(args.lower_bound_result_file, 'r') as f:
        all_results = json.load(f)

    if args.upper_bound_result_file is not None:
        with open(args.upper_bound_result_file, 'r') as f:
            upper_bound_simulation_results = json.load(f)["default"]["simulation_avg_error"]
    else:
        upper_bound_simulation_results = None

    # Generate plot
    plot_results(all_results, upper_bound_simulation_results, output_file=args.output_pdf)

    # Comparison statistics require upper-bound simulation results.
    if upper_bound_simulation_results is not None:
        compute_statistics(all_results, upper_bound_simulation_results)

if __name__ == "__main__":
    main()
