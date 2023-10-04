import os
from typing import Optional, Union, Dict, Tuple
import pandas as pd
import seaborn as sns
from matplotlib.ticker import EngFormatter

from utils.data_analysis import compute_speedups, compute_speedups_min_max
from utils.experiments2 import get_program_infos
from utils.plot import save_plot, get_new_figure, size_vs_y_plot, get_bytes_formatter, legend_on_lines, \
                       replace_legend_names, legend_on_lines_dict, rotate_xlabels

hue_order = ['cloudsc_vert_loop_4_ZSOLQA', 'cloudsc_vert_loop_6_ZSOLQA', 'cloudsc_vert_loop_6_1_ZSOLQA',
             'cloudsc_vert_loop_7_3']


def plot_speedup_array_order(avg_data: pd.DataFrame, folder: str, title: str, legend_on_line: bool = False):
    """
    Plots speedup of kernel runtime copmaring the different array layouts on heap allocation

    :param avg_data: Averaged data used to compute speedup
    :type avg_data: pd.DataFrame
    :param folder: The folder to store the plot in
    :type folder: str
    :param title: The title of the plot
    :type title: str
    :param legend_on_line: If legend should be put on lines, defaults to False
    :type legend_on_line: bool, optional
    """
    # Create subplots
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=1, color='gray', linestyle='--')

    speedups = compute_speedups(avg_data, ('cloudsc_vert_loop_4_ZSOLQA'), ('program')) \
        .drop(index='stack', level='temp allocation')\
        .drop(index='cloudsc_vert_loop_4_ZSOLQA', level='program')\
        .reset_index()

    size_vs_y_plot(ax, 'Speedup', 'Speedup of kernel runtime achieved using different array layouts compared to original', speedups,
                   size_var_name='NBLOCKS')
    sns.lineplot(data=speedups, x='NBLOCKS', y='runtime', hue='program', ax=ax, marker='o', hue_order=hue_order,
                 style='program')
    program_names_map = get_program_infos()['variant description'].to_dict()
    if legend_on_line:
        legend_on_lines(ax, ((4e5, 2), (4.5e5, 15), (4.5e5, 7)), [program_names_map[p] for p in hue_order[1:]],
                        rotations=[0, -8, -5], color_palette_offset=1)
    else:
        replace_legend_names(ax.get_legend(), program_names_map)
    figure.suptitle(title)
    save_plot(os.path.join(folder, 'speedup_array_order.pdf'))


def plot_speedup_temp_allocation(data: pd.DataFrame, avg_data: pd.DataFrame, folder: str, title: str, legend_on_line: bool = False):
    """
    Plots speedup of kernel runtime, comparing the different temporary array allocation

    :param avg_data: Averaged data used to compute the speedup
    :type avg_data: pd.DataFrame
    :param folder: The folder to store the plot in
    :type folder: str
    :param title: The title of the plot
    :type title: str
    :param legend_on_line: If legend should be put on lines, defaults to False
    :type legend_on_line: bool, optional
    """
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(title)
    avg_data = avg_data.drop('temp', level='scope').drop('Total', level='scope')\
            .reset_index().set_index(['program', 'NBLOCKS', 'temp allocation'])

    speedups = compute_speedups(avg_data, ('heap'), ('temp allocation')) \
        .drop(index='heap', level='temp allocation')['runtime'] \
        .reset_index()

    min_speedups, max_speedups = compute_speedups_min_max(data['runtime'].dropna(), ('heap'), ('temp allocation'), 'run number')
    errors = min_speedups.rename(columns={'runtime': 'min_runtime'}).drop(columns='run number')\
        .join(max_speedups.rename(columns={'runtime': 'max_runtime'}).drop(columns='run number'))
    errors = errors.drop(index='heap', level='temp allocation')\
            .reset_index().set_index(['NBLOCKS', 'program'])\
            .drop(columns='temp allocation')\
            .drop(columns='num_kernels')\
            .drop(columns='scope')

    # Change NBLOCKS to be formatted nicer. barplots treats x as non numeric
    speedups['NBLOCKS'] = speedups['NBLOCKS'].apply(lambda x: f"{int(x/1000)}K")
    sns.barplot(data=speedups, hue='program', x='NBLOCKS', y='runtime', ax=ax, hue_order=hue_order)
    for idx, p in enumerate(ax.patches):
        size = ((idx % 7) + 1) * 1e5
        program = hue_order[idx//7]
        x = p.get_x()
        w = p.get_width()
        h = p.get_height()
        values = errors.xs((size, program), level=('NBLOCKS', 'program'))
        min_y = values['min_runtime'].values[0]
        max_y = values['max_runtime'].values[0]
        ax.vlines(x+w/2, min_y, max_y, color='black')

    ax.set_ylabel('Speedup')
    program_names_map = get_program_infos()['variant description'].to_dict()
    ax.legend(loc='upper right')
    replace_legend_names(ax.get_legend(), program_names_map)
    figure.suptitle('Speedup of kernel runtime stack allocation vs heap allocation')
    save_plot(os.path.join(folder, 'speedup_temp_allocation.pdf'))


def plot_runtime_speedup_temp_allocation_bar(data: pd.DataFrame, avg_data: pd.DataFrame, folder: str):
    figure = get_new_figure(4)
    ax1 = figure.add_subplot(1, 2, 1)
    ax2 = figure.add_subplot(1, 2, 2)
    figure.suptitle("Runtimes and speedups for stack vs. heap allocation")

    speedups = compute_speedups(avg_data, ('heap'), ('temp allocation'))\
        .drop(index='heap', level='temp allocation')\
        .drop('Total', level='scope')\
        .drop('temp', level='scope')\
        .reset_index().set_index(['program', 'NBLOCKS'])

    sns.barplot(data=data.xs(2e5, level='NBLOCKS').reset_index(), hue='temp allocation', x='program', y='runtime',
                ax=ax1, order=hue_order)
    sns.barplot(data=data.xs(7e5, level='NBLOCKS').reset_index(), hue='temp allocation', x='program', y='runtime',
                ax=ax2, order=hue_order)
    program_names_map = get_program_infos()['variant description'].to_dict()
    ax1.set_ylabel('Runtime [s]')
    ax2.set_ylabel('Runtime [s]')
    ax2.legend().remove()
    ax1.bar_label(ax1.containers[0], fmt='{:.1e}')
    ax1.bar_label(ax1.containers[1], fmt='{:.1e}')
    ax2.bar_label(ax2.containers[0], fmt='{:.1e}')
    ax2.bar_label(ax2.containers[1], fmt='{:.1e}')
    for idx, version in enumerate(hue_order):
        this_speedup = speedups.xs((version, 2e5), level=('program', 'NBLOCKS'))['runtime'].values[0]
        ax1.text(idx+0.25, 0.4, f"x{this_speedup:.0f}", rotation=90, horizontalalignment='center')
        this_speedup = speedups.xs((version, 7e5), level=('program', 'NBLOCKS'))['runtime'].values[0]
        ax2.text(idx+0.25, 1.4, f"x{this_speedup:.0f}", rotation=90, horizontalalignment='center')

    rotate_xlabels(ax1, angle=25, replace_dict=program_names_map)
    rotate_xlabels(ax2, angle=25, replace_dict=program_names_map)
    figure.tight_layout()
    ax1.legend(bbox_to_anchor=(1.5, 0.5))
    save_plot(os.path.join(folder, 'runtime_speedup_temp_allocation_bar.pdf'))

def plot_runtime_bar_stack(data: pd.DataFrame, folder: str, title: str):
    """
    Plot the kernel runtime

    :param data: The data to plot
    :type data: pd.DataFrame
    :param folder: The folder to store the plot in
    :type folder: str
    :param title: The title of the plot
    :type title: str
    :param legend_dict: Dictionary to use to put legend on lines, if None legend will be placed normally, defaults to
    None.
    :type legend_dict: Optional[Dict[str, Dict[str, Union[float, int, Tuple[float]]]]], optional
    :param limit_temp_allocation_to: The temporary array allocation the data should be limited to (either 'heap',
            'stack' or None. If None does not limit it, defaults to None
    :type limit_temp_allocation_to: Optional[str], optional
    """
    data = data.xs('stack', level='temp allocation')
    data['runtime'] = data['runtime'].apply(lambda x: x*1000)
    program_names_map = get_program_infos()['variant description'].to_dict()

    figure = get_new_figure(4)
    figure.suptitle('Kernel runtimes of stack allocated versions')
    ax1 = figure.add_subplot(1, 2, 1)
    ax2 = figure.add_subplot(1, 2, 2)
    sns.barplot(data.xs(2e5, level='NBLOCKS').reset_index(), x='program', y='runtime', ax=ax1, order=hue_order)
    sns.barplot(data.xs(7e5, level='NBLOCKS').reset_index(), x='program', y='runtime', ax=ax2, order=hue_order)
    ax1.bar_label(ax1.containers[0], fmt='{:.1f}')
    ax2.bar_label(ax2.containers[0], fmt='{:.1f}')
    rotate_xlabels(ax1, angle=30, replace_dict=program_names_map)
    rotate_xlabels(ax2, angle=30, replace_dict=program_names_map)
    ax1.set_ylabel('Runtime [ms]')
    ax1.set_title(f"NBLOCKS = {int(2e5):,}")
    ax1.set_xlabel('')
    ax2.set_ylabel('Runtime [ms]')
    ax2.set_title(f"NBLOCKS = {int(7e5):,}")
    ax2.set_xlabel('')
    ax1.set_ylim((0, ax1.get_ylim()[1]*1.1))
    ax2.set_ylim((0, ax2.get_ylim()[1]*1.1))
    figure.tight_layout()

    save_plot(os.path.join(folder, 'runtime_stack_bar.pdf'))


def plot_runtime(data: pd.DataFrame, folder: str, title: str,
                 legend_dict: Optional[Dict[str, Dict[str, Union[float, int, Tuple[float]]]]] = None,
                 limit_temp_allocation_to: Optional[str] = None):
    """
    Plot the kernel runtime

    :param data: The data to plot
    :type data: pd.DataFrame
    :param folder: The folder to store the plot in
    :type folder: str
    :param title: The title of the plot
    :type title: str
    :param legend_dict: Dictionary to use to put legend on lines, if None legend will be placed normally, defaults to
    None.
    :type legend_dict: Optional[Dict[str, Dict[str, Union[float, int, Tuple[float]]]]], optional
    :param limit_temp_allocation_to: The temporary array allocation the data should be limited to (either 'heap',
            'stack' or None. If None does not limit it, defaults to None
    :type limit_temp_allocation_to: Optional[str], optional
    """
    temp_allocations = ['stack', 'heap']
    if limit_temp_allocation_to is not None and limit_temp_allocation_to in temp_allocations:
        sub_title = f"Kernel Runtimes of {limit_temp_allocation_to} allocated versions"
        data = data.xs(limit_temp_allocation_to, level='temp allocation')
        filename = f"runtime_{limit_temp_allocation_to}.pdf"
    else:
        sub_title = "Kernel Runtimes"
        filename = "runtime.pdf"

    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(title)
    size_vs_y_plot(ax, 'Runtime [ms]', sub_title, data, size_var_name='NBLOCKS')
    data['runtime'] = data['runtime'].apply(lambda x: x*1000)
    additional_args = {}
    if limit_temp_allocation_to == 'stack':
        dashes = {program: '' for program in data.reset_index()['program'].unique()}
        dashes['cloudsc_vert_loop_6_1_ZSOLQA'] = (2, 2)
        additional_args['linewidth'] = 3
        additional_args['dashes'] = dashes
        additional_args['style'] = 'program'
    elif limit_temp_allocation_to is None:
        additional_args['style'] = 'temp allocation'

    sns.lineplot(data, x='NBLOCKS', y='runtime', hue='program', ax=ax, hue_order=hue_order, errorbar=('ci', 95),
                 err_style='bars', **additional_args)
    program_names_map = get_program_infos()['full description'].to_dict()
    if legend_dict is not None:
        # legend_on_lines(ax, ((3e5, 0.084), (1.8e5, 0.009), (3.8e5, 0.015), (3.8e5, 0.005)),
        #                 [program_names_map[p] for p in hue_order], rotations=[25, 3, 3, 3])
        legend_on_lines_dict(ax, legend_dict)
    else:
        replace_legend_names(ax.get_legend(), program_names_map)
    save_plot(os.path.join(folder, filename))


def plot_memory_transfers(data: pd.DataFrame, folder: str, title: str, legend_on_line: bool = False,
                          limit_temp_allocation_to: Optional[str] = None):
    """
    Plot the memory transfers

    :param data: The data to plot
    :type data: pd.DataFrame
    :param folder: The folder to store the plot in
    :type folder: str
    :param title: The title of the plot
    :type title: str
    :param legend_on_line: If legend should be put on lines, defaults to False
    :type legend_on_line: bool, optional
    :param limit_temp_allocation_to: The temporary array allocation the data should be limited to (either 'heap',
            'stack' or None. If None does not limit it, defaults to None
    :type limit_temp_allocation_to: Optional[str], optional
    """
    temp_allocations = ['stack', 'heap']
    if limit_temp_allocation_to is not None and limit_temp_allocation_to in temp_allocations:
        sub_title = f"Measured Transferred Bytes of {limit_temp_allocation_to} allocated versions"
        data = data.xs(limit_temp_allocation_to, level='temp allocation')
        filename = f"memory_{limit_temp_allocation_to}.pdf"
    else:
        sub_title = "Measured Transferred Bytes"
        filename = "memory.pdf"

    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(title)
    size_vs_y_plot(ax, 'Transferred to/from global memory [byte]', sub_title, data, size_var_name='NBLOCKS')
    additional_args = {}
    if limit_temp_allocation_to is None:
        additional_args['style'] = 'temp allocation'

    sns.lineplot(data, x='NBLOCKS', y='measured bytes', hue='program', ax=ax, hue_order=hue_order, errorbar=('ci', 95),
                 err_style='bars', **additional_args)
    sns.lineplot(data['theoretical bytes total'].dropna().reset_index(),
                 ax=ax, x='NBLOCKS', y='theoretical bytes total', linestyle='--', color='gray',
                 label='theoretical bytes/size')
    ax.yaxis.set_major_formatter(get_bytes_formatter())
    program_names_map = get_program_infos()['full description'].to_dict()
    if legend_on_line:
        legend_on_lines(ax, ((3e5, 0.084), (1.8e5, 0.009), (3.8e5, 0.015), (3.8e5, 0.005)),
                        [program_names_map[p] for p in hue_order], rotations=[25, 3, 3, 3])
    else:
        replace_legend_names(ax.get_legend(), program_names_map)
    save_plot(os.path.join(folder, filename))
