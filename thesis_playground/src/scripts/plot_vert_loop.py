from argparse import ArgumentParser
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import os
import json
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import seaborn as sns

from utils.plot import set_general_plot_style, save_plot
from utils.paths import get_vert_loops_dir
from utils.vertical_loops import get_dataframe
from scripts import Script


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


def get_speedups(data: pd.DataFrame, baseline_program: str, baseline_short_desc: Optional[str] = None) -> pd.DataFrame:
    """
    Compute the "speedups" for all "list based" metrics. This means to divide the baseline by the value for all values
    in all programs. Removes any "non list" values except for program and size

    :param data: The complete data
    :type data: pd.DataFrame
    :param baseline_program: The name of the baseline program
    :type baseline_program: str
    :param baseline_short_desc: The name of the baseline short_desc. If None will use respective short_desc for each
    measurement. Defaults to None
    :type baseline_short_desc: Optional[str]
    :return: The speedup data
    :rtype: pd.DataFrame
    """
    # Groupd by program and size, remove any other non-list columns
    indices = ['program', 'size', 'short_desc']
    avg_data = data.groupby(indices).mean()
    speedup_data = avg_data.copy()

    def compute_speedup(col: pd.Series) -> pd.Series:
        if baseline_short_desc is None:
            return col.div(col[baseline_program, :, :]).apply(np.reciprocal)
        else:
            return col.div(col[baseline_program, :, baseline_short_desc]).apply(np.reciprocal)

    return speedup_data.apply(compute_speedup, axis='index')


def better_program_names(legend: matplotlib.legend.Legend):
    """
    Replace the program names in the legend by more discriptive ones

    :param ax: The legend object where the labels should be changed
    :type ax: matplotlib.legend.Leged
    """
    # print([t for t in ax.get_legend().get_texts()])
    names_map = {
        'cloudsc_vert_loop_4': 'Non flipped (original)',
        'cloudsc_vert_loop_6': 'Flipped array structure',
        'cloudsc_vert_loop_6_1': 'Flipped array structure with fixed temporary array',
        'cloudsc_vert_loop_7': 'Flipped array structure with temporary & k-caching',
        'short_desc': 'temporary array allocation',
        'specialised': 'stack'
    }
    for text in legend.get_texts():
        if text.get_text() in names_map:
            text.set(text=names_map[text.get_text()])


def create_runtime_memory_plots(data: pd.DataFrame, run_count_str: str, node: str, gpu: str):
    """
    Creates and saves a plot depicting the runtime and bytes transferred

    :param data: The data to plto
    :type data: pd.DataFrame
    :param run_count_str: String describing how many runs were used to get data
    :type run_count_str: str
    :param node: The node used to get data
    :type node: str
    :param gpu: The gpu used to get data
    :type gpu: str
    """
    # Create subplots
    figure = plt.figure()
    ax_memory = figure.add_subplot(1, 2, 1)
    ax_runtime_low = figure.add_subplot(2, 2, 4)
    ax_runtime_high = figure.add_subplot(2, 2, 2, sharex=ax_runtime_low)
    figure.subplots_adjust(hspace=0.05)

    # hide the spines between ax_runtime_low and ax_runtime_high
    ax_runtime_low.spines['top'].set_visible(False)
    ax_runtime_high.spines['bottom'].set_visible(False)
    ax_runtime_high.tick_params(labeltop=False, labelbottom=False, bottom=False, top=False, axis='x')
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")

    # Plot memory
    ax_memory.set_title('Measured Transferred Bytes')
    hue_order = ['cloudsc_vert_loop_4', 'cloudsc_vert_loop_6', 'cloudsc_vert_loop_6_1', 'cloudsc_vert_loop_7']
    sns.lineplot(data=data, x='size', y='measured bytes', hue='program', ax=ax_memory, legend=True,
                 errorbar=('ci', 95), style='short_desc', palette='pastel', markers=True, hue_order=hue_order,
                 err_style='bars')
    ax_memory.set_xlabel('NBLOCKS')
    ax_memory.set_ylabel('Transferred to/from global memory [byte]')
    sns.lineplot(data=data[['size', 'theoretical bytes']].drop_duplicates(), ax=ax_memory, x='size',
                 y='theoretical bytes', linestyle='--', color='gray', label='theoretical bytes/size')

    # Plot runtime
    ax_runtime_high.set_title('Kernel Runtime')
    for ax in [ax_runtime_low, ax_runtime_high]:
        sns.lineplot(data=data, x='size', y='runtime', hue='program', ax=ax, errorbar=('ci', 95), legend=False,
                     style='short_desc', palette='pastel', markers=True, hue_order=hue_order, err_style='bars')
    ax_runtime_low.set_xlabel('NBLOCKS')
    ax_runtime_low.set_ylabel('Runtime [s]')
    ax_runtime_low.yaxis.set_label_coords(-.06, 1.0)
    ax_runtime_high.set_xlabel('')
    ax_runtime_high.set_ylabel('')

    ax_runtime_high.set_yticks(np.arange(2, 4, 0.2))
    ax_runtime_low.set_yticks(np.arange(0, 1, 0.2))
    ax_runtime_high.set_ylim(2.3, 3.2)
    ax_runtime_low.set_ylim(0, .9)

    sizes = data['size'].unique()
    size_formatter = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    ax_runtime_low.xaxis.set_major_formatter(size_formatter)
    ax_runtime_low.set_xticks(sizes)
    ax_memory.xaxis.set_major_formatter(size_formatter)
    ax_memory.set_xticks(sizes)
    bytes_formatter = EngFormatter(places=0, sep="\N{THIN SPACE}", unit='B')  # U+2009
    ax_memory.yaxis.set_major_formatter(bytes_formatter)

    # Add cut-out slanted line
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='gray', mec='gray', mew=1, clip_on=False)
    ax_runtime_high.plot([0, 1], [0, 0], transform=ax_runtime_high.transAxes, **kwargs)
    ax_runtime_low.plot([0, 1], [1, 1], transform=ax_runtime_low.transAxes, **kwargs)

    # Move legend
    handles, labels = ax_memory.get_legend_handles_labels()
    ax_memory.get_legend().remove()
    print(labels)
    # TODO: Fix label order
    new_order = [0, 5, 1, 6, 2, 7, 3, 4, 8]
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]
    better_program_names(figure.legend(handles, labels, loc='outside lower center', ncols=5, frameon=False))
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'runtime.pdf'))


def create_speedup_plots(data: pd.DataFrame, run_count_str: str, node: str, gpu: str):

    # Create subplots
    figure = plt.figure()
    ax1 = figure.add_subplot(1, 2, 1)
    ax2 = figure.add_subplot(1, 2, 2)
    ax1.axhline(y=1, color='gray', linestyle='--')
    ax2.axhline(y=1, color='gray', linestyle='--')
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")

    hue_order = ['cloudsc_vert_loop_4', 'cloudsc_vert_loop_6', 'cloudsc_vert_loop_6_1', 'cloudsc_vert_loop_7']
    # Plot speedup
    speedups = get_speedups(data, 'cloudsc_vert_loop_4')\
        .sort_values('program', key=key_program_sort)\
        .drop(index='specialised', level='short_desc')\
        .drop(index='cloudsc_vert_loop_4', level='program')\
        .reset_index()
    ax1.set_title('Speedup of heap allocation compared to non flipped')
    sns.lineplot(data=speedups, x='size', y='runtime', hue='program', ax=ax1, legend=False, markers=True,
                 hue_order=hue_order)
    ax1.set_xlabel('NBLOCKS')
    ax1.set_ylabel('Speedup')

    speedups = get_speedups(data, 'cloudsc_vert_loop_4', 'heap')\
        .sort_values('program', key=key_program_sort)\
        .drop(index='heap', level='short_desc')\
        .reset_index()
    sns.lineplot(data=speedups, x='size', y='runtime', hue='program', ax=ax2, legend=True, markers=True,
                 hue_order=hue_order)
    ax2.set_title('Speedup of non-flipped stack allocation compared to heap')
    ax2.set_xlabel('NBLOCKS')
    ax2.set_ylabel('Speedup')

    sizes = speedups['size'].unique()
    size_formatter = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    ax1.xaxis.set_major_formatter(size_formatter)
    ax1.set_xticks(sizes)
    ax2.xaxis.set_major_formatter(size_formatter)
    ax2.set_xticks(sizes)

    better_program_names(ax2.get_legend())

    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'speedups.pdf'))


class PlotVertLoop(Script):
    name = 'plot-vert'
    description = 'Plot the memory transfers and runtime of the verical loops'

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('--python-data', type=str, default=None, help='Path to python data to include as well')

    @staticmethod
    def action(args):
        set_general_plot_style()

        # Get data
        data, descriptions, nodes = get_dataframe('vert_loop_[0-4,6-9]')
        data.sort_values('program', key=key_program_sort, inplace=True)
        node, gpu = check_for_common_node_gpu(nodes)
        data.reset_index(inplace=True)
        run_counts = data.groupby(['program', 'size', 'short_desc']).count()['run number']
        if run_counts.min() == run_counts.max():
            run_count_str = run_counts.min()
        else:
            run_count_str = f"between {run_counts.min()} and {run_counts.max()}"

        if args.python_data is not None:
            py_data = pd.read_csv(args.python_data)
            py_data.rename(columns={'NBLOCKS': 'size', 'specialised': 'short_desc', 'D': 'measured bytes',
                                    'T': 'runtime'},
                           inplace=True)
            desc_map = {True: 'py specialised', False: 'py heap'}
            py_data['short_desc'] = py_data['short_desc'].apply(lambda specialised, map: desc_map[specialised],
                                                                map=desc_map)
            py_data['program'] = py_data['program'].apply(lambda program: f"cloudsc_{program}")
            data.info()
            data = pd.concat([data, py_data])
            print(data)

        create_runtime_memory_plots(data, run_count_str, node, gpu)
        create_speedup_plots(data, run_count_str, node, gpu)
