from argparse import ArgumentParser
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import os
import json
from typing import Optional, Tuple, List, Dict, Union
from numbers import Number
import numpy as np
import pandas as pd
import seaborn as sns
import copy

from utils.plot import save_plot, get_new_figure
from utils.paths import get_vert_loops_dir
from utils.vertical_loops import get_dataframe
from scripts import Script
from run_playground_vert_loop import df_index_cols


program_names_map = {
    'cloudsc_vert_loop_4': 'Non flipped (original)',
    'cloudsc_vert_loop_6': 'Flipped array structure',
    'cloudsc_vert_loop_6_1': 'Flipped array structure with fixed temporary array',
    'cloudsc_vert_loop_7': 'Flipped array structure with temporary & k-caching',
    'short_desc': 'temporary array allocation',
    'specialised': 'stack'
}
hue_order = ['cloudsc_vert_loop_4', 'cloudsc_vert_loop_6', 'cloudsc_vert_loop_6_1', 'cloudsc_vert_loop_7']


def get_arrowprops(update: Dict[str, Union[str, Number]]) -> Dict[str, Union[str, Number]]:
    props = dict(width=2.0, headwidth=7.0, headlength=7.0, shrink=0.01)
    props.update(update)
    return props


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


def get_speedups(data: pd.DataFrame, baseline_program: Optional[str] = None,
                 baseline_short_desc: Optional[str] = None) -> pd.DataFrame:
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
        if baseline_short_desc is None and baseline_program is not None:
            return col.div(col[baseline_program, :, :]).apply(np.reciprocal)
        elif baseline_short_desc is not None and baseline_program is not None:
            return col.div(col[baseline_program, :, baseline_short_desc]).apply(np.reciprocal)
        elif baseline_short_desc is not None and baseline_program is None:
            return col.div(col[:, :, baseline_short_desc]).apply(np.reciprocal)

    return speedup_data.apply(compute_speedup, axis='index')


def better_program_names(legend: matplotlib.legend.Legend, names_map: Optional[Dict[str, str]] = None):
    """
    Replace the program names in the legend by more discriptive ones

    :param legend: The legend object where the labels should be changed
    :type legend: matplotlib.legend.Leged
    :param names_map: Dictionay mapping the names/labels to change. Optional, defaults to None. If None will use the
    default description for cloudsc_vert_loop_[4-7]
    :type names_map: Optional[Dict[str, str]]
    """
    if names_map is None:
        names_map = program_names_map
    for text in legend.get_texts():
        if text.get_text() in names_map:
            text.set(text=names_map[text.get_text()])


def create_runtime_plot(data: pd.DataFrame, ax_low: matplotlib.axis.Axis, ax_high: matplotlib.axis.Axis,
                        hue_order: List[str], legend=False, annotate_runtimes=False):

    # hide the spines between ax_low and ax_high
    ax_low.spines['top'].set_visible(False)
    ax_high.spines['bottom'].set_visible(False)
    ax_high.tick_params(labeltop=False, labelbottom=False, bottom=False, top=False, axis='x')

    # Plot runtime
    ax_high.set_title('Kernel Runtime')
    for ax in [ax_low, ax_high]:
        sns.lineplot(data=data.reset_index(), x='size', y='runtime', hue='program', ax=ax, errorbar=('ci', 95),
                     style='short_desc', markers=True, hue_order=hue_order, err_style='bars',
                     legend=legend)
    ax_low.set_xlabel('NBLOCKS')
    ax_low.set_ylabel('Runtime [s]')
    ax_low.yaxis.set_label_coords(-.06, 1.0)
    ax_high.set_xlabel('')
    ax_high.set_ylabel('')

    ax_high.set_yticks(np.arange(2, 4, 0.2))
    ax_low.set_yticks(np.arange(0, 1, 0.2))
    ax_high.set_ylim(2.3, 3.3)
    ax_low.set_ylim(-.1, .9)

    sizes = data.reset_index()['size'].unique()
    size_formatter = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    ax_low.xaxis.set_major_formatter(size_formatter)
    ax_low.set_xticks(sizes)

    # Add cut-out slanted line
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='gray', mec='gray', mew=1, clip_on=False)
    ax_high.plot([0, 1], [0, 0], transform=ax_high.transAxes, **kwargs)
    ax_low.plot([0, 1], [1, 1], transform=ax_low.transAxes, **kwargs)

    # Add values of runtimes
    runtimes = data.reset_index()[['program', 'size', 'runtime', 'short_desc']]\
        .groupby(['program', 'size', 'short_desc'])\
        .mean().reset_index()

    if annotate_runtimes:
        runtime_annotation_offsets_500 = {
                ('cloudsc_vert_loop_4', 'specialised'): (0, 0.01),
                ('cloudsc_vert_loop_6', 'specialised'): (-33000, 0.02),
                ('cloudsc_vert_loop_6_1', 'specialised'): (33000, 0.02),
                ('cloudsc_vert_loop_7', 'specialised'): (0, -0.03),
                ('cloudsc_vert_loop_4', 'heap'): (0, 0.05),
                ('cloudsc_vert_loop_6', 'heap'): (0, -0.05),
                ('cloudsc_vert_loop_6_1', 'heap'): (0, -0.04),
                ('cloudsc_vert_loop_7', 'heap'): (0, 0.04),
                }

        runtime_annotation_offsets_100 = {
                ('cloudsc_vert_loop_4', 'specialised'): (0, 0.05),
                ('cloudsc_vert_loop_6', 'specialised'): (-33000, 0.02),
                ('cloudsc_vert_loop_6_1', 'specialised'): (33000, 0.03),
                ('cloudsc_vert_loop_7', 'specialised'): (0, -0.03),
                ('cloudsc_vert_loop_4', 'heap'): (0, 0.03),
                ('cloudsc_vert_loop_6', 'heap'): (0, 0.01),
                ('cloudsc_vert_loop_6_1', 'heap'): (0, 0.04),
                ('cloudsc_vert_loop_7', 'heap'): (0, -0.04),
                }
        for index, row in runtimes[runtimes['size'] == 500000].iterrows():
            hue_index = hue_order.index(row['program'])
            for ax in [ax_low, ax_high]:
                if (row['program'], row['short_desc']) in runtime_annotation_offsets_500:
                    offsets = runtime_annotation_offsets_500[(row['program'], row['short_desc'])]
                else:
                    offsets = (0, 0)
                x = row['size'] + offsets[0]
                y = row['runtime'] + offsets[1]
                ax.text(x, y, f"{row['runtime']:.3e}s", color=sns.color_palette()[hue_index],
                        horizontalalignment='center')

        for index, row in runtimes[runtimes['size'] == 100000].iterrows():
            hue_index = hue_order.index(row['program'])
            for ax in [ax_low, ax_high]:
                if (row['program'], row['short_desc']) in runtime_annotation_offsets_100:
                    offsets = runtime_annotation_offsets_100[(row['program'], row['short_desc'])]
                else:
                    offsets = (0, 0)
                x = row['size'] + offsets[0]
                y = row['runtime'] + offsets[1]
                ax.text(x, y, f"{row['runtime']:.3e}s", color=sns.color_palette()[hue_index],
                        horizontalalignment='center')


def create_memory_plot(data: pd.DataFrame, ax: matplotlib.axis.Axis, hue_order: List[str]):
    ax.set_title('Measured Transferred Bytes')
    sns.lineplot(data=data, x='size', y='measured bytes', hue='program', ax=ax, legend=True,
                 errorbar=('ci', 95), style='short_desc', markers=True, hue_order=hue_order,
                 err_style='bars')
    ax.set_xlabel('NBLOCKS')
    ax.set_ylabel('Transferred to/from global memory [byte]')
    sns.lineplot(data=data.reset_index()[['size', 'theoretical bytes']].drop_duplicates(), ax=ax, x='size',
                 y='theoretical bytes', linestyle='--', color='gray', label='theoretical bytes/size')

    size_formatter = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    ax.xaxis.set_major_formatter(size_formatter)
    ax.set_xticks(data.reset_index()['size'].unique())
    bytes_formatter = EngFormatter(places=0, sep="\N{THIN SPACE}", unit='B')  # U+2009
    ax.yaxis.set_major_formatter(bytes_formatter)


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
    figure = get_new_figure(4)
    ax_memory = figure.add_subplot(1, 2, 1)
    ax_runtime_low = figure.add_subplot(2, 2, 4)
    ax_runtime_high = figure.add_subplot(2, 2, 2, sharex=ax_runtime_low)
    figure.subplots_adjust(hspace=0.05)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")

    # Plot runtime
    create_runtime_plot(data, ax_runtime_low, ax_runtime_high, hue_order)

    # Plot memory
    create_memory_plot(data, ax_memory, hue_order)

    # Move legend
    handles, labels = ax_memory.get_legend_handles_labels()
    ax_memory.get_legend().remove()
    new_order = [0, 5, 1, 6, 2, 7, 3, 4, 8]
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]

    better_program_names(figure.legend(handles, labels, loc='outside lower center', ncols=5, frameon=False))
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'runtime_memory.pdf'))


def create_memory_order_only_plot(data: pd.DataFrame, run_count_str: str, node: str, gpu: str):
    # Create subplots
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    create_memory_plot(data.drop(index='specialised', level='short_desc'), ax, hue_order)
    ax.get_legend().remove()
    ax.text(3e5, 8e10, program_names_map[hue_order[0]], color=sns.color_palette()[0], horizontalalignment='center')
    ax.text(3e5, 6e10, program_names_map[hue_order[1]], color=sns.color_palette()[1], horizontalalignment='center')
    ax.text(2e5, 1e10, program_names_map[hue_order[2]], color=sns.color_palette()[2], horizontalalignment='center',
            verticalalignment='center', rotation=4)
    ax.text(4e5, 2e10, program_names_map[hue_order[3]], color=sns.color_palette()[3], horizontalalignment='center')
    ax.text(4.5e5, 0, 'theoretical limit (sum of sizes of arrays)', color='gray', horizontalalignment='center',
            verticalalignment='center', rotation=2)
    ax.annotate('', xytext=(4.5e5, 2e10), xy=(4.5e5, 7.5e9),
                arrowprops=get_arrowprops({'color': sns.color_palette()[3]}))
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'memory_array_order.pdf'))


def create_runtime_with_speedup_plot(data: pd.DataFrame, run_count_str: str, node: str, gpu: str):
    # Create subplots
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    # ax_runtime_high = figure.add_subplot(2, 1, 1, sharex=ax_runtime_low)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    ax.set_title('Runtimes of original version to best improved')

    # Plot runtime
    data_slowest = data.xs(('cloudsc_vert_loop_4', 'heap'), level=('program', 'short_desc'))
    data_fastest = data.xs(('cloudsc_vert_loop_7', 'specialised'), level=('program', 'short_desc'))
    data_slowest['program'] = 'slowest'
    data_fastest['program'] = 'fastest'

    hue_order = ['slowest', 'fastest']
    program_names = {'slowest': 'Original Version', 'fastest': 'Fastest improved Version'}
    sns.lineplot(pd.concat([data_slowest, data_fastest]), x='size', y='runtime', hue='program', ax=ax, legend=False,
                 hue_order=hue_order, errorbar=('ci', 95), err_style='bars')
    ax.set_ylabel('Runtime [s]')
    ax.set_xlabel('NBLOCKS')
    sizes = data.reset_index()['size'].unique()
    size_formatter = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    ax.xaxis.set_major_formatter(size_formatter)
    ax.set_xticks(sizes)

    avg_data = data.groupby(['program', 'size', 'short_desc']).mean()
    for size in sizes:
        time1 = avg_data.xs(('cloudsc_vert_loop_4', size, 'heap'))['runtime']
        time2 = avg_data.xs(('cloudsc_vert_loop_7', size, 'specialised'))['runtime']
        speedup = time1 / time2
        ax.annotate('', xy=(size, time1), xytext=(size, time2), arrowprops=get_arrowprops({'facecolor': 'black'}))
        ax.text(size, time2 + 0.6, f"{speedup:.1f}x")

    ax.text(2e5, 3.1, program_names[hue_order[0]], horizontalalignment='center', color=sns.color_palette()[0],
            verticalalignment='center', rotation=10)
    ax.text(3e5, 0.1, program_names[hue_order[1]], horizontalalignment='center', color=sns.color_palette()[1])
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'runtime.pdf'))


def create_runtime_stack_plot(data: pd.DataFrame, run_count_str: str, node: str, gpu: str):
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    ax.set_title('Runtimes of stack allocated versions')
    sns.lineplot(data=data.drop(index='heap', level='short_desc'), x='size', y='runtime', hue='program',
                 ax=ax, hue_order=hue_order, errorbar=('ci', 95), err_style='bars', legend=False, style='program',
                 linewidth=3,
                 dashes={'cloudsc_vert_loop_4': '', 'cloudsc_vert_loop_6': '', 'cloudsc_vert_loop_6_1': (1, 2),
                         'cloudsc_vert_loop_7': ''})
    ax.set_ylabel('Runtime [s]')
    ax.set_xlabel('NBLOCKS')
    sizes = data.reset_index()['size'].unique()
    size_formatter = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    ax.xaxis.set_major_formatter(size_formatter)
    ax.set_xticks(sizes)
    ax.text(3e5, 0.084, program_names_map[hue_order[0]], horizontalalignment='center', color=sns.color_palette()[0],
            verticalalignment='center', rotation=25)
    ax.text(1.8e5, 0.009, program_names_map[hue_order[1]], horizontalalignment='center', color=sns.color_palette()[1],
            verticalalignment='center', rotation=3)
    ax.text(3.8e5, 0.015, program_names_map[hue_order[2]], horizontalalignment='center', color=sns.color_palette()[2],
            verticalalignment='center', rotation=3)
    ax.text(3.8e5, 0.005, program_names_map[hue_order[3]], horizontalalignment='center', color=sns.color_palette()[3],
            verticalalignment='center', rotation=3)
    # better_program_names(ax.get_legend())
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'runtime_stack.pdf'))


def create_speedup_plots(data: pd.DataFrame, run_count_str: str, node: str, gpu: str):

    size_formatter = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    # Create subplots
    figure = get_new_figure(4)
    ax1 = figure.add_subplot(1, 1, 1)
    ax1.axhline(y=1, color='gray', linestyle='--')
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")

    # Plot speedup
    speedups = get_speedups(data, baseline_program='cloudsc_vert_loop_4')\
        .sort_values('program', key=key_program_sort)\
        .drop(index='specialised', level='short_desc')\
        .drop(index='cloudsc_vert_loop_4', level='program')\
        .reset_index()
    ax1.set_title('Speedup achieved using different array layouts compared to original')
    sns.lineplot(data=speedups, x='size', y='runtime', hue='program', ax=ax1, legend=False, marker='o',
                 hue_order=hue_order)
    ax1.set_xlabel('NBLOCKS')
    ax1.set_ylabel('Speedup')
    ax1.xaxis.set_major_formatter(size_formatter)
    ax1.set_xticks(speedups['size'].unique())
    # better_program_names(ax1.get_legend())
    ax1.text(300000, 1.2, program_names_map['cloudsc_vert_loop_6'], color=sns.color_palette()[1],
             horizontalalignment='center')
    ax1.text(250000, 6.3, program_names_map['cloudsc_vert_loop_6_1'], color=sns.color_palette()[2],
             horizontalalignment='center', verticalalignment='center', rotation=-12)
    ax1.text(350000, 7.5, program_names_map['cloudsc_vert_loop_7'], color=sns.color_palette()[3],
             horizontalalignment='center', verticalalignment='center', rotation=-15)
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'speedup_array_order.pdf'))
    ymin, ymax = ax1.get_ylim()

    figure = get_new_figure(4)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=1, color='gray', linestyle='--')
    speedups = get_speedups(data, baseline_program='cloudsc_vert_loop_4')\
        .sort_values('program', key=key_program_sort)\
        .drop(index='specialised', level='short_desc')\
        .drop(index='cloudsc_vert_loop_4', level='program')\
        .reset_index()
    sns.lineplot(data=speedups[speedups['program'] == 'cloudsc_vert_loop_6'],
                 x='size', y='runtime', hue='program', ax=ax, legend=False, marker='o',
                 hue_order=hue_order)
    ax.set_title('Speedup')
    ax.set_xlabel('NBLOCKS')
    ax.set_ylabel('Speedup')
    ax.xaxis.set_major_formatter(size_formatter)
    ax.set_xticks(speedups['size'].unique())
    ax.set_ylim(ymin, ymax)
    ax.text(300000, 1.2, program_names_map['cloudsc_vert_loop_6'], color=sns.color_palette()[1],
            horizontalalignment='center')
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'speedup_array_order_v1.pdf'))

    figure = get_new_figure(4)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=1, color='gray', linestyle='--')
    speedups = get_speedups(data, baseline_short_desc='heap')\
        .sort_values('program', key=key_program_sort)\
        .drop(index='heap', level='short_desc')\
        .reset_index()
    sns.lineplot(data=speedups, hue='program',
                 x='size', y='runtime', ax=ax, marker='o', legend=False)
    ax.set_title('Speedup stack allocation vs heap allocation')
    ax.set_xlabel('NBLOCKS')
    ax.set_ylabel('Speedup')
    ax.xaxis.set_major_formatter(size_formatter)
    ax.set_xticks(speedups['size'].unique())
    ax.text(1.5e5, 30, program_names_map[hue_order[0]], color=sns.color_palette()[0], horizontalalignment='center',
            verticalalignment='center', rotation=-3)
    ax.text(4e5, 350, program_names_map[hue_order[1]], color=sns.color_palette()[1], horizontalalignment='center',
            verticalalignment='center', rotation=-10)
    ax.text(4e5, 120, program_names_map[hue_order[2]], color=sns.color_palette()[2], horizontalalignment='center',
            verticalalignment='center', rotation=0)
    ax.text(3e5, 200, program_names_map[hue_order[3]], color=sns.color_palette()[3], horizontalalignment='center',
            verticalalignment='center', rotation=0)
    ax.annotate('', xy=(2e5, 85), xytext=(2e5, 185),
                arrowprops=get_arrowprops({'color': sns.color_palette()[3]}))
    ax.annotate('', xy=(4.5e5, 45), xytext=(4.5e5, 105),
                arrowprops=get_arrowprops({'color': sns.color_palette()[2]}))
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'speedup_temp_location.pdf'))


def create_speedup_py_plot(data: pd.DataFrame,  run_count_str: str, node: str, gpu: str):
    size_formatter = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
    figure = get_new_figure(4)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=1, color='gray', linestyle='--')
    data.drop(index=['cloudsc_vert_loop_4', 'cloudsc_vert_loop_6', 'cloudsc_vert_loop_6_1'],
              level='program', inplace=True)
    speedups = get_speedups(data, baseline_program='cloudsc_vert_loop_7')\
        .sort_values('program', key=key_program_sort)\
        .drop(index='heap', level='short_desc')\
        .drop(index='cloudsc_vert_loop_7', level='program')\
        .reset_index()
    hue_order = ['runtime', 'measured bytes']
    sns.lineplot(data=pd.melt(speedups, id_vars=['program', 'size'], value_vars=['runtime', 'measured bytes']),
                 hue='variable', x='size', y='value', ax=ax, marker='o', legend=False, hue_order=hue_order)
    ax.set_title('Speedup of Python to Fortran version using caching and flipped arrays allocated on stack')
    ax.set_xlabel('NBLOCKS')
    ax.set_ylabel('Speedup / Bytes reduced by')
    ax.xaxis.set_major_formatter(size_formatter)
    ax.set_xticks(speedups['size'].unique())
    program_names = {'measured bytes': 'Bytes transferred', 'runtime': 'Runtime'}
    ax.text(2.5e5, 4.4, program_names[hue_order[0]], color=sns.color_palette()[0], horizontalalignment='center',
            rotation=1)
    ax.text(2.5e5, 3.6, program_names[hue_order[1]], color=sns.color_palette()[1], horizontalalignment='center',
            rotation=1)
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'speedup_python.pdf'))


class PlotVertLoop(Script):
    name = 'plot-vert'
    description = 'Plot the memory transfers and runtime of the verical loops'

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('--python-data', type=str, default=None, help='Path to python data to include as well')

    @staticmethod
    def action(args):
        # Get data
        data, descriptions, nodes = get_dataframe('vert_loop_[0-4,6-9]')
        data.sort_values('program', key=key_program_sort, inplace=True)
        node, gpu = check_for_common_node_gpu(nodes)
        run_counts = data.reset_index().groupby(['program', 'size', 'short_desc']).count()['run number']
        if run_counts.min() == run_counts.max():
            run_count_str = run_counts.min()
        else:
            run_count_str = f"between {run_counts.min()} and {run_counts.max()}"

        create_memory_order_only_plot(data, run_count_str, node, gpu)
        create_runtime_memory_plots(data, run_count_str, node, gpu)
        create_speedup_plots(data, run_count_str, node, gpu)
        create_runtime_with_speedup_plot(data, run_count_str, node, gpu)
        create_runtime_stack_plot(data, run_count_str, node, gpu)

        if args.python_data is not None:
            py_data = pd.read_csv(args.python_data, index_col=df_index_cols).reset_index()
            py_data.rename(columns={'NBLOCKS': 'size', 'specialised': 'short_desc', 'D': 'measured bytes',
                                    'T': 'runtime'},
                           inplace=True)

            desc_map = {True: 'specialised', False: 'heap'}
            py_data['short_desc'] = py_data['short_desc'].apply(lambda specialised, map: desc_map[specialised],
                                                                map=desc_map)
            py_data['program'] = py_data['program'].apply(lambda program: f"py_{program}")
            py_data.set_index(['program', 'size', 'run_number', 'short_desc'], inplace=True)
            data = pd.concat([data, py_data])

            run_counts = data.reset_index().groupby(['program', 'size', 'short_desc']).count()['run number']
            if run_counts.min() == run_counts.max():
                run_count_str = run_counts.min()
            else:
                run_count_str = f"between {run_counts.min()} and {run_counts.max()}"
            create_speedup_py_plot(data, run_count_str, node, gpu)
