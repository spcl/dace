import os
import pandas as pd
import seaborn as sns

from utils.data_analysis import compute_speedups
from utils.experiments2 import get_program_infos
from utils.plot import save_plot, get_new_figure, size_vs_y_plot, get_bytes_formatter, legend_on_lines, \
                       get_arrowprops, replace_legend_names

hue_order = ['cloudsc_vert_loop_4_ZSOLQA', 'cloudsc_vert_loop_6_ZSOLQA', 'cloudsc_vert_loop_6_1_ZSOLQA',
             'cloudsc_vert_loop_7_ZSOLQA']


def plot_speedup_array_order(avg_data: pd.DataFrame, folder: str, legend_on_line: bool = False):
    # Create subplots
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=1, color='gray', linestyle='--')
    # figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")

    speedups = compute_speedups(avg_data, ('cloudsc_vert_loop_4_ZSOLQA'), ('program')) \
        .drop(index='stack', level='temp allocation')\
        .drop(index='cloudsc_vert_loop_4_ZSOLQA', level='program')\
        .reset_index()

    size_vs_y_plot(ax, 'Speedup', 'Speedup achieved using different array layouts compared to original', speedups,
                   size_var_name='NBLOCKS')
    sns.lineplot(data=speedups, x='NBLOCKS', y='runtime', hue='program', ax=ax, marker='o', hue_order=hue_order)
    program_names_map = get_program_infos()['base description'].to_dict()
    if legend_on_line:
        legend_on_line(ax, ((300000, 1.2), (250000, 6.3), (350000, 7.5)), [program_names_map[p] for p in hue_order[1:]],
                       rotations=[0, -12, -15], color_palette_offset=1)
    else:
        replace_legend_names(ax.get_legend(), program_names_map)
    save_plot(os.path.join(folder, 'speedup_array_order.pdf'))


def plot_speedup_temp_allocation(avg_data: pd.DataFrame, folder: str, legend_on_line: bool = False):
    figure = get_new_figure(4)
    ax = figure.add_subplot(1, 1, 1)
    ax.axhline(y=1, color='gray', linestyle='--')
    # figure.suptitle(f"Vertical Loop Programs run on {node} using NVIDIA {gpu} averaging {run_count_str} runs")

    speedups = compute_speedups(avg_data, ('heap'), ('temp allocation')) \
        .drop(index='heap', level='temp allocation') \
        .reset_index()

    sns.lineplot(data=speedups, hue='program', x='NBLOCKS', y='runtime', ax=ax, marker='o', hue_order=hue_order)
    size_vs_y_plot(ax, 'Speedup', 'Speedup stack allocation vs heap allocation', speedups, size_var_name='NBLOCKS')
    program_names_map = get_program_infos()['base description'].to_dict()
    if legend_on_line:
        legend_on_line(ax, ((1.5e5, 30), (4e5, 350), ((4e5, 120), (4.5e5, 45)), ((3e5, 200), (2e5, 85))),
                       [program_names_map[p] for p in hue_order], rotations=[-3, -10, 0, 0])
    else:
        replace_legend_names(ax.get_legend(), program_names_map)
    save_plot(os.path.join(folder, 'speedup_temp_allocation.pdf'))
