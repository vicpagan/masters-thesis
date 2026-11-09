#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import math
import sys
import bisect
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

fontsize = 26
markersize = 12
linewidth=5

def plot_curves(data):
    max_num_num_procs = len(data[list(data.keys())[0]])
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(data.keys())))


    # Paint tiles
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlabel('Number of compute nodes ($n$)', fontsize=fontsize)
    ax.set_ylabel('Approx. optimal expected error', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)

    max_min_error = 0.0
    color_idx = 0
    max_num_procs = 0
    ax.set_xscale('log', base=2)
    for deadline in data.keys():
        color = colors[color_idx]
        num_nodes_values = [x["num_nodes"] for x in data[deadline]]
        max_num_procs = max(max_num_procs, num_nodes_values[-1])
        error_values = [x["expected_error"] for x in data[deadline]]
        if deadline < 3600:
            deadline_string = str(int(deadline/60)) + " min"
        else:
            deadline_string = str(int(deadline/3600)) + " hour"
        ax.plot(num_nodes_values, error_values, '.-', label="deadline:" + deadline_string, color=color, linewidth=linewidth, markersize=markersize)
        min_error = min(error_values)
        best_num_procs = num_nodes_values[error_values.index(min_error)]
        #ax.plot([best_num_procs], [min_error], 'o', color=color, markersize=6)
        max_min_error = max(min_error, max_min_error)
        color_idx += 1

    # ax.set_xticks(num_nodes_values)  # Set major tick positions
    ax.xaxis.set_major_locator(ticker.LogLocator(base=2))
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=2))
    powers_of_two = [2**x for x in range(0, int(math.log2(max_num_procs)+1))]
    ax.set_xticks(powers_of_two)  # whatever range you need
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'$2^{{{int(np.log2(x))}}}$'
    ))
    ax.minorticks_off()
    ax.grid()

    # Compute ok limits
    ax.set_ylim(1, 103)
    ax.set_yticks([1,20,40,60,80,100])
    ax.legend(bbox_to_anchor=(0.36, 0.92), fontsize=fontsize)
    plt.tight_layout()
    output_filename = "./data_parallel_results.pdf"
    sys.stderr.write(f"Figure saved in {output_filename}\n")
    plt.savefig(output_filename, bbox_inches="tight",pad_inches=0)

def main():
    # Read JSON from stdin or file
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            json_object = json.load(f)
    else:
        json_object = json.load(sys.stdin)

    data = {}
    for key, values in json_object.items():
        data[float(key)] = values

    plot_curves(data)


if __name__ == '__main__':
    main()

