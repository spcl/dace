import matplotlib.pyplot as plt
import os
import json
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import seaborn as sns

from utils.plot import set_general_plot_style, save_plot
from utils.paths import get_vert_loops_dir
from utils.vertical_loops import get_dataframe
from scripts import Script


# TODO: Adapt to df changes
def key_program_sort(programs):
    """
    Sort by program name/index/version
    """
    programs = programs.str.extract("cloudsc_vert_loop_([0-9])_?([0-9])?")
    programs.fillna(0, inplace=True)
    programs[0] = pd.to_numeric(programs[0])
    programs[1] = pd.to_numeric(programs[1])
    programs = programs[0].apply(lambda x: 10*x) + programs[1]
    return programs


def check_for_common_node_gpu(node_df: pd.DataFrame) -> Optional[Tuple[str]]:
    """
    Check if all measurements of the given list where done on the same node and returns the gpu of the node.

    :param node_df: The nodes of the data
    :type node_df: pd.DataFrame
    :return: Tuple with node name in first position and string of the name of the gpu used in second or None if not all
    on same node
    :rtype: Optional[Tuple[str]]
    """
    hardware_filename = 'nodes.json'
    with open(hardware_filename) as node_file:
        node_data = json.load(node_file)
        nodes = node_df['node'].unique()
        if len(nodes) == 1:
            return (nodes[0], node_data['ault_nodes'][nodes[0]]['GPU'])
        else:
            return None


def get_speedups(data: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """
    Compute the "speedups" for all "list based" metrics. This means to divide the baseline by the value for all values
    in all programs. Removes any "non list" values except for program and size

    :param data: The complete data
    :type data: pd.DataFrame
    :param baseline: The name of the baseline program
    :type baseline: str
    :return: The speedup data
    :rtype: pd.DataFrame
    """
    # Groupd by program and size, remove any other non-list columns
    indices = ['program', 'size', 'short_desc']
    avg_data = data.groupby(indices).mean()
    speedup_data = avg_data.copy()

    def compute_speedup(col: pd.Series) -> pd.Series:
        return col.div(col[baseline, :]).apply(np.reciprocal)

    return speedup_data.apply(compute_speedup, axis='index')


class PlotVertLoop(Script):
    name = 'plot-vert'
    description = 'Plot the memory transfers and runtime of the verical loops'

    @staticmethod
    def action(args):
        set_general_plot_style()

        # Create subplots
        figure = plt.figure()
        ax_memory = figure.add_subplot(2, 2, 1)
        ax_runtime = figure.add_subplot(2, 2, 2)
        ax_speedups_memory = figure.add_subplot(2, 2, 3)
        ax_speedups_runtime = figure.add_subplot(2, 2, 4, sharey=ax_speedups_memory)

        # Get data
        data, descriptions, nodes = get_dataframe('vert_loop_[0-4,6-9]')
        data.sort_values('program', key=key_program_sort, inplace=True)
        speedups = get_speedups(data, 'cloudsc_vert_loop_4')
        speedups.sort_values('program', key=key_program_sort, inplace=True)
        node, gpu = check_for_common_node_gpu(nodes)
        data.reset_index(inplace=True)
        speedups.reset_index(inplace=True)
        run_counts = data.groupby(['program', 'size', 'short_desc']).count()['run number']
        if run_counts.min() == run_counts.max():
            run_count_str = run_counts.min()
        else:
            run_count_str = f"between {run_counts.min()} and {run_counts.max()}"

        # Set title
        figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")

        # Plot memory
        ax_memory.set_title('Measured Transferred Bytes')
        hue_order = [int(1e5), int(2e5), int(5e5)]
        sns.lineplot(data=data, x='program', y='measured bytes', hue='size', ax=ax_memory,
                     errorbar=('ci', 95), hue_order=hue_order, style='short_desc', palette='pastel')
        ax_memory.set_xlabel('')
        ax_memory.set_ylabel('Transferred to/from global memory [byte]')

        # Plot runtime
        ax_runtime.set_title('Kernel Runtime')
        sns.lineplot(data=data, x='program', y='runtime', hue='size', ax=ax_runtime, errorbar=('ci', 95),
                     hue_order=hue_order, style='short_desc', palette='pastel')
        ax_runtime.set_xlabel('')
        ax_runtime.set_ylabel('Runtime [s]')

        # Plot memory decrease
        ax_speedups_memory.set_title('Measured Transferred Bytes - Decrease')
        sns.lineplot(data=speedups, x='program', y='measured bytes', hue='size', ax=ax_speedups_memory,
                     hue_order=hue_order, legend=False, style='short_desc', palette='pastel')
        ax_speedups_memory.set_xlabel('')
        ax_speedups_memory.set_ylabel('Less memory transfers')

        # Plot speedup
        ax_speedups_runtime.set_title('Speedup')
        sns.lineplot(data=speedups, x='program', y='runtime', hue='size', ax=ax_speedups_runtime,
                     hue_order=hue_order, legend=False, style='short_desc', palette='pastel')
        ax_speedups_runtime.set_xlabel('')
        ax_speedups_runtime.set_ylabel('Speedup')

        # Move legend
        ax_runtime.get_legend().remove()
        handles, labels = ax_memory.get_legend_handles_labels()
        labels[labels.index('size')] = 'NBLOCKS'
        labels[labels.index('short_desc')] = 'Temporary array allocation'
        ax_memory.legend(handles, labels)
        sns.move_legend(ax_memory, 'center', bbox_to_anchor=(1., -1.35), ncol=7, frameon=False)
        # plt.figlegend(loc='lower right',bbox_to_anchor=(0.85,0.25))

        save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'plot.png'))
