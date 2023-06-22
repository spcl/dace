from argparse import ArgumentParser
import matplotlib
from matplotlib.ticker import EngFormatter
import os
import json
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import seaborn as sns

from utils.plot import save_plot, get_new_figure, size_vs_y_plot, get_bytes_formatter, legend_on_lines, \
                       get_arrowprops, replace_legend_names
from utils.paths import get_vert_loops_dir
from utils.vertical_loops import get_dataframe, get_speedups, key_program_sort, switch_to_zsloqa_versions, limit_to_size
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
        gpus = [node_data['ault_nodes'][nodes[0]]['GPU']]
        node_str = ' and '.join(nodes)
        if len(gpus) == 1:
            return (node_str, gpus[0])
        else:
            return None


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
    replace_legend_names(legend, names_map)


def create_runtime_plot(data: pd.DataFrame, ax_low: matplotlib.axis.Axis, ax_high: matplotlib.axis.Axis,
                        hue_order: List[str], legend=False):

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


def create_memory_plot(data: pd.DataFrame, ax: matplotlib.axis.Axis, hue_order: List[str]):
    sns.lineplot(data=data, x='size', y='measured bytes', hue='program', ax=ax, legend=True,
                 errorbar=('ci', 95), style='short_desc', markers=True, hue_order=hue_order,
                 err_style='bars')
    sns.lineplot(data=data.reset_index()[['size', 'theoretical bytes']].drop_duplicates(), ax=ax, x='size',
                 y='theoretical bytes', linestyle='--', color='gray', label='theoretical bytes/size')

    size_vs_y_plot(ax, 'Transferred to/from global memory [byte]', 'Measured Transferred Bytes', data)
    ax.yaxis.set_major_formatter(get_bytes_formatter())


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

    better_program_names(figure.legend(handles, labels, loc='outside lower center', ncols=3, frameon=False))
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'runtime_memory.pdf'))


def create_memory_order_only_plot(data: pd.DataFrame, run_count_str: str, node: str, gpu: str, legend_on_line: bool):
    # Create subplots
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    create_memory_plot(data.drop(index='specialised', level='short_desc'), ax, hue_order)
    if legend_on_line:
        legend_on_lines(ax, [(3e5, 8e10), (3e5, 6e10), (2e5, 1e10), ((4e5, 2e10), (4.5e5, 7.5e9))],
                        [program_names_map[p] for p in hue_order],
                        rotations=[0, 0, 4, 0])
        ax.text(4.5e5, 0, 'theoretical limit (sum of sizes of arrays)', color='gray', horizontalalignment='center',
                verticalalignment='center', rotation=2)
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'memory_array_order.pdf'))


def create_runtime_with_speedup_plot(data: pd.DataFrame, run_count_str: str, node: str, gpu: str, legend_on_line: bool):
    # Create subplots
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    # ax_runtime_high = figure.add_subplot(2, 1, 1, sharex=ax_runtime_low)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    size_vs_y_plot(ax, 'Runtime [s]', 'Runtimes of original version to best improved', data)

    # Plot runtime
    data_slowest = data.xs(('cloudsc_vert_loop_4', 'heap'), level=('program', 'short_desc'))
    data_fastest = data.xs(('cloudsc_vert_loop_7', 'specialised'), level=('program', 'short_desc'))
    data_slowest['program'] = 'slowest'
    data_fastest['program'] = 'fastest'

    hue_order = ['slowest', 'fastest']
    program_names = {'slowest': 'Original Version', 'fastest': 'Fastest improved Version'}
    sns.lineplot(pd.concat([data_slowest, data_fastest]), x='size', y='runtime', hue='program', ax=ax,
                 hue_order=hue_order, errorbar=('ci', 95), err_style='bars')

    avg_data = data.groupby(['program', 'size', 'short_desc']).mean()
    sizes = data.reset_index()['size'].unique()
    for size in sizes:
        time1 = avg_data.xs(('cloudsc_vert_loop_4', size, 'heap'))['runtime']
        time2 = avg_data.xs(('cloudsc_vert_loop_7', size, 'specialised'))['runtime']
        speedup = time1 / time2
        ax.annotate('', xy=(size, time1), xytext=(size, time2), arrowprops=get_arrowprops({'facecolor': 'black'}))
        ax.text(size, time2 + 0.6, f"{speedup:.1f}x")

    if legend_on_line:
        legend_on_line(ax, ((2e5, 3.1), (3e5, 0.1)), [program_names[p] for p in hue_order], rotations=[10, 0])
    else:
        better_program_names(ax.get_legend())
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'runtime.pdf'))


def create_runtime_stack_plot(data: pd.DataFrame, run_count_str: str, node: str, gpu: str, legend_on_line: bool):
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    size_vs_y_plot(ax, 'Runtime [s]', 'Runtimes of stack allocated versions', data)
    dashes = {program: '' for program in data.reset_index()['program'].unique()}
    dashes['cloudsc_vert_loop_6_1'] = (1, 2)
    sns.lineplot(data=data.drop(index='heap', level='short_desc'), x='size', y='runtime', hue='program',
                 ax=ax, hue_order=hue_order, errorbar=('ci', 95), err_style='bars', style='program',
                 linewidth=3, dashes=dashes)
    if legend_on_line:
        legend_on_line(ax, ((3e5, 0.084), (1.8e5, 0.009), (3.8e5, 0.015), (3.8e5, 0.005)),
                       [program_names_map[p] for p in hue_order], rotations=[25, 3, 3, 3])
    else:
        better_program_names(ax.get_legend())
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'runtime_stack.pdf'))


def create_speedup_plots(data: pd.DataFrame, run_count_str: str, node: str, gpu: str, legend_on_line: bool):

    # Create subplots
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=1, color='gray', linestyle='--')
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")

    # Plot speedup
    speedups = get_speedups(data, baseline_program='cloudsc_vert_loop_4')\
        .sort_values('program', key=key_program_sort)\
        .drop(index='specialised', level='short_desc')\
        .drop(index='cloudsc_vert_loop_4', level='program')\
        .reset_index()

    size_vs_y_plot(ax, 'Speedup', 'Speedup achieved using different array layouts compared to original', data)
    sns.lineplot(data=speedups, x='size', y='runtime', hue='program', ax=ax, marker='o', hue_order=hue_order)
    if legend_on_line:
        legend_on_line(ax, ((300000, 1.2), (250000, 6.3), (350000, 7.5)), [program_names_map[p] for p in hue_order[1:]],
                       rotations=[0, -12, -15], color_palette_offset=1)
    else:
        better_program_names(ax.get_legend())
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'speedup_array_order.pdf'))
    ymin, ymax = ax.get_ylim()

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
                 x='size', y='runtime', hue='program', ax=ax, marker='o', hue_order=hue_order)
    size_vs_y_plot(ax, 'Speedup', 'Speedup', data)
    ax.set_ylim(ymin, ymax)
    if legend_on_line:
        legend_on_line(ax, ((300000, 1.2)), [program_names_map['clouds_vert_loop_6']], color_palette_offset=1)
    else:
        better_program_names(ax.get_legend())
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'speedup_array_order_v1.pdf'))

    figure = get_new_figure(4)
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=1, color='gray', linestyle='--')
    speedups = get_speedups(data, baseline_short_desc='heap')\
        .sort_values('program', key=key_program_sort)\
        .drop(index='heap', level='short_desc')\
        .reset_index()
    sns.lineplot(data=speedups, hue='program', x='size', y='runtime', ax=ax, marker='o')
    size_vs_y_plot(ax, 'Speedup', 'Speedup stack allocation vs heap allocation', data)
    if legend_on_line:
        legend_on_line(ax, ((1.5e5, 30), (4e5, 350), ((4e5, 120), (4.5e5, 45)), ((3e5, 200), (2e5, 85))),
                       [program_names_map[p] for p in hue_order], rotations=[-3, -10, 0, 0])
    else:
        better_program_names(ax.legend())
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'speedup_temp_location.pdf'))


def create_speedup_py_plot(data: pd.DataFrame,  run_count_str: str, node: str, gpu: str, legen_on_line: bool):
    figure = get_new_figure()
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=1, color='gray', linestyle='--')
    data.drop(index=['cloudsc_vert_loop_4', 'cloudsc_vert_loop_6', 'cloudsc_vert_loop_6_1'],
              level='program', inplace=True)

    py_fortran_program_mapping = {
        'py_vert_loop_7_1': 'cloudsc_vert_loop_7',
        'py_vert_loop_7_2': 'cloudsc_vert_loop_7_2',
        'py_vert_loop_7_3': 'cloudsc_vert_loop_7_3',
    }

    def compute_speedup(col: pd.Series) -> pd.Series:
        speedup_col = col.copy()
        for this_program_name in col.reset_index()['program'].unique():
            if this_program_name in py_fortran_program_mapping:
                fortran_program_name = py_fortran_program_mapping[this_program_name]
                update_col = pd.concat([speedup_col.xs(this_program_name, level='program', drop_level=False),
                                        speedup_col.xs(fortran_program_name, level='program', drop_level=False)])\
                    .div(col.xs(fortran_program_name, level='program'))
                speedup_col.update(update_col)
        return speedup_col.apply(np.reciprocal)

    speedups = data\
        .xs('specialised', level='short_desc') \
        .drop('cloudsc_vert_loop_4_ZSOLQA', level='program') \
        .drop('cloudsc_vert_loop_6_ZSOLQA', level='program') \
        .drop('cloudsc_vert_loop_6_1_ZSOLQA', level='program') \
        .groupby(['program', 'size']).mean() \
        .apply(compute_speedup, axis='index') \
        .drop('cloudsc_vert_loop_7', level='program') \
        .drop('cloudsc_vert_loop_7_2', level='program') \
        .drop('cloudsc_vert_loop_7_3', level='program')
    hue_order = ['runtime', 'measured bytes']
    size_vs_y_plot(ax, 'Speedup / Bytes reduced by',
                   'Speedup of Python to Fortran version using caching and flipped arrays allocated on stack', speedups)
    sns.lineplot(data=pd.melt(speedups.reset_index(), id_vars=['program', 'size'],
                              value_vars=['runtime', 'measured bytes']),
                 hue='program', style='variable', x='size', y='value', ax=ax, marker='o', legend=True,
                 style_order=hue_order)
    program_names = {'py_vert_loop_7_1': 'With ZSOLQA but not returning it',
                     'py_vert_loop_7_2': 'Without ZSOLQA',
                     'py_vert_loop_7_3': 'With ZSOLQA and returning it'}
    better_program_names(ax.legend(), program_names)
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'speedup_python.pdf'))

    figure = get_new_figure()
    figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=1, color='gray', linestyle='--')
    sns.lineplot(data=pd.melt(speedups.drop('py_vert_loop_7_1', level='program').reset_index(),
                              id_vars=['program', 'size'], value_vars=['runtime', 'measured bytes']),
                 hue='program', style='variable', x='size', y='value', ax=ax, marker='o', legend=True,
                 style_order=hue_order)

    size_vs_y_plot(ax, 'Speedup / Bytes reduced by',
                   'Speedup of Python to Fortran version using caching and flipped arrays allocated on stack', speedups)
    program_names = {'py_vert_loop_7_2': 'Without ZSOLQA',
                     'py_vert_loop_7_3': 'With ZSOLQA and returning it'}
    better_program_names(ax.legend(), program_names)
    save_plot(os.path.join(get_vert_loops_dir(), 'plots', 'speedup_python_correct_versions.pdf'))


class PlotVertLoop(Script):
    name = 'plot-vert'
    description = 'Plot the memory transfers and runtime of the verical loops'

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('--python-data', type=str, default=None, help='Path to python data to include as well')
        parser.add_argument('--legend-on-line', action='store_true', default=False)

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

        limit_to_size(data, max_size=int(5e5))
        switch_to_zsloqa_versions(data)

        data_without_7_2 = data.drop('cloudsc_vert_loop_7_2', level='program')
        create_memory_order_only_plot(data_without_7_2, run_count_str, node, gpu, args.legend_on_line)
        create_runtime_memory_plots(data_without_7_2, run_count_str, node, gpu)
        create_speedup_plots(data_without_7_2, run_count_str, node, gpu, args.legend_on_line)
        create_runtime_with_speedup_plot(data_without_7_2, run_count_str, node, gpu, args.legend_on_line)
        create_runtime_stack_plot(data_without_7_2, run_count_str, node, gpu, args.legend_on_line)

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

            limit_to_size(data, max_size=int(4e5))

            run_counts = data.reset_index().groupby(['program', 'size', 'short_desc']).count()['run number']
            if run_counts.min() == run_counts.max():
                run_count_str = run_counts.min()
            else:
                run_count_str = f"between {run_counts.min()} and {run_counts.max()}"
            create_speedup_py_plot(data, run_count_str, node, gpu, args.legend_on_line)
