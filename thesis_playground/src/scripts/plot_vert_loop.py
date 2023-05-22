import matplotlib.pyplot as plt
import os
import json
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import seaborn as sns

from utils.paths import get_vert_loops_dir
from utils.vertical_loops import get_dataframe, get_non_list_indices
from scripts import Script


# TODO: Adapt to df changes
def key_program_sort(programs):
    """
    Sort by program name/index/version
    """
    programs = programs.str.extract("cloudsc vert loop ([0-9]) ?([0-9])?")
    programs.fillna(0, inplace=True)
    programs[0] = pd.to_numeric(programs[0])
    programs[1] = pd.to_numeric(programs[1])
    programs = programs[0].apply(lambda x: 10*x) + programs[1]
    return programs


def check_for_common_node_gpu(data: pd.DataFrame) -> Optional[Tuple[str]]:
    """
    Check if all measurements of the given list where done on the same node and returns the gpu of the node.

    :param data: The measured data
    :type data: pd.DataFrame
    :return: Tuple with node name in first position and string of the name of the gpu used in second or None if not all
    on same node
    :rtype: Optional[Tuple[str]]
    """
    hardware_filename = 'nodes.json'
    with open(hardware_filename) as node_file:
        node_data = json.load(node_file)
        nodes = data['node'].unique()
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
    indices = ['program', 'size']
    scalar_cols = [c for c in get_non_list_indices() if c not in indices]
    avg_data = data.drop(scalar_cols, axis='columns').groupby(indices).mean()
    speedup_data = avg_data.copy()

    def compute_speedup(col: pd.Series) -> pd.Series:
        return col.div(col[baseline, :]).apply(np.reciprocal)

    return speedup_data.apply(compute_speedup, axis='index')


class PlotVertLoop(Script):
    name = 'plot-vert'
    description = 'Plot the memory transfers and runtime of the verical loops'

    @staticmethod
    def action(args):
        plt.rcParams.update({'figure.figsize': (19, 10)})
        plt.rcParams.update({'font.size': 12})
        sns.set_style('whitegrid')
        figure = plt.figure()
        ax_memory = figure.add_subplot(2, 2, 1)
        ax_runtime = figure.add_subplot(2, 2, 2)
        ax_speedups = figure.add_subplot(2, 2, 3)

        data = get_dataframe('vert_loop_[0-4,6-9]')
        data.sort_values('program', key=key_program_sort, inplace=True)
        speedups = get_speedups(data, 'cloudsc vert loop 4')

        data.reset_index(inplace=True)
        speedups.reset_index(inplace=True)
        node, gpu = check_for_common_node_gpu(data)
        figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu}")

        ax_memory.set_title('Measured Transferred Bytes')
        hue_order = ['1E+05', '2E+05', '5E+05']
        sns.pointplot(data=data, x='program', y='measured bytes', hue='size', ax=ax_memory, join=False,
                      errorbar=('ci', 95), scale=1.0, markers='o', errwidth=2, hue_order=hue_order)
        ax_memory.set_xlabel('')
        ax_memory.set_ylabel('Transferred to/from global memory [byte]')

        ax_runtime.set_title('Kernel Runtime')
        sns.pointplot(data=data, x='program', y='runtime', hue='size', ax=ax_runtime, join=False, errorbar=('ci', 95),
                      scale=1.0, markers='^', errwidth=2, hue_order=hue_order)
        ax_runtime.set_xlabel('')
        ax_runtime.set_ylabel('Runtime [s]')

        ax_speedups.set_title('Measured Transferred Bytes - Decrease')
        sns.lineplot(data=speedups, x='program', y='measured bytes', hue='size', ax=ax_speedups,  marker='o',
                     hue_order=hue_order, linestyle=':', legend=False)
        sns.lineplot(data=speedups, x='program', y='runtime', hue='size', ax=ax_speedups,  marker='^',
                     hue_order=hue_order, legend=False)
        ax_speedups.set_xlabel('')
        ax_speedups.set_ylabel('Speedup / less memory transfers')

        ax_runtime.get_legend().remove()
        sns.move_legend(ax_memory, 'center', bbox_to_anchor=(1.75, -.5), ncol=3, frameon=False, title='NBLOCKS')
        # plt.figlegend(loc='lower right',bbox_to_anchor=(0.85,0.25))

        plot_dir = os.path.join(get_vert_loops_dir(), 'plots')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        plt.savefig(os.path.join(plot_dir, "plot.png"))
