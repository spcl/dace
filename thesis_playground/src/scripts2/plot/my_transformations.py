import os
from typing import Optional, Union, Dict, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_analysis import compute_speedups
from utils.experiments2 import get_program_infos
from utils.plot import save_plot, get_new_figure, size_vs_y_plot, get_bytes_formatter, legend_on_lines, \
                       replace_legend_names, legend_on_lines_dict, rotate_xlabels


def plot_runtime(data: pd.DataFrame, folder: str, title: str):
    figure = get_new_figure()
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(title)
    size_vs_y_plot(ax, 'Total time [s]', "Total Runtime", data, size_var_name='NBLOCKS')
    sns.lineplot(data, x='NBLOCKS', y='Total time', hue='k_caching', style='change_strides',
                 ax=ax, errorbar=('ci', 95), err_style='bars')
    save_plot(os.path.join(folder, 'runtime.pdf'))


def plot_kerneltime(data: pd.DataFrame, folder: str, title: str):
    figure = get_new_figure()
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(title)
    size_vs_y_plot(ax, 'time [s]', "Kernel time", data, size_var_name='NBLOCKS')
    sns.lineplot(data.xs('work', level='scope'), x='NBLOCKS', y='runtime', hue='k_caching', style='change_strides',
                 ax=ax, errorbar=('ci', 95), err_style='bars')
    save_plot(os.path.join(folder, 'kerneltime.pdf'))


def plot_change_stride_runtime_bar(data: pd.DataFrame, folder: str, title: str):
    figure = get_new_figure()
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(title)

    # Only use a specific NBLOCKS value
    nblocks = int(1e5)
    data = data.xs(nblocks, level='NBLOCKS')
    ax.set_title(f"Comparison of kernel runtimes @ NBLOCKS={nblocks:,}")
    # Add optimisation columns to merge k_caching and change_stride columns
    index_cols = list(data.index.names)
    index_cols.remove('num_kernels')
    data = data\
        .reset_index()\
        .replace({'k_caching': {True: 'K-caching', False: 'no K-caching'},
                  'change_strides': {True: 'changed strides', False: 'original strides'}})
    data['optimisation'] = data['k_caching'] + ' and ' + data['change_strides']
    index_cols.append('optimisation')
    data = data.set_index(index_cols)
    order = list(data.index.get_level_values('optimisation').drop_duplicates())
    order = ['no K-caching and original strides', 'K-caching and original strides',
             'no K-caching and changed strides', 'K-caching and changed strides']
    data = data.drop('Total', level='scope').drop('temp', level='scope')
    combined_data = data.xs('work', level='scope')['runtime'].add(data.xs('transpose', level='scope')['runtime'],
                                                                  fill_value=0.0)
    sns.barplot(combined_data.reset_index(), x='optimisation', y='runtime', ax=ax, order=order, errorbar=('ci', 95),
                label='Transpose', color=sns.color_palette()[0])
    sns.barplot(data.xs('work', level='scope').reset_index(), x='optimisation', y='runtime', ax=ax, order=order,
                errorbar=('ci', 95), label='Main work loop', color=sns.color_palette()[1])
    ax.legend()
    rotate_xlabels(ax, angle=20)
    plt.tight_layout()
    save_plot(os.path.join(folder, 'change_strides_runtime_bar_combined.pdf'))

    figure = get_new_figure()
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(title)
    sns.barplot(data.reset_index(), x='optimisation', y='runtime', hue='scope', ax=ax, order=order, errorbar=('ci', 95))
    ax.legend()
    rotate_xlabels(ax, angle=20)
    plt.tight_layout()
    save_plot(os.path.join(folder, 'change_strides_runtime_bar.pdf'))


def plot_speedup(data: pd.DataFrame, folder: str, title_total: str, title_ncu: str):
    speedups = compute_speedups(data, (False, False), ('k_caching', 'change_strides'))
    index_cols = list(speedups.index.names)
    speedups = speedups.reset_index()
    speedups = speedups.drop(speedups[(speedups['k_caching'] == False) &
                                      (speedups['change_strides'] == False)].index)
    index_cols.remove('k_caching')
    index_cols.remove('change_strides')
    speedups = speedups\
        .replace({'k_caching': {True: 'with K-caching', False: 'no K-caching'},
                  'change_strides': {True: 'with change strides', False: 'original strides'}})
    speedups['optimisation'] = speedups['k_caching'] + ' and ' + speedups['change_strides']
    index_cols.append('optimisation')
    speedups = speedups.set_index(index_cols)

    figure = get_new_figure()
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(title_ncu)
    ax.axhline(y=1, color='gray', linestyle='--')
    size_vs_y_plot(ax, 'Speedup', "Kernel Runtime - Speedup", speedups, size_var_name='NBLOCKS')
    sns.lineplot(speedups.xs('work', level='scope'), x='NBLOCKS', y='runtime', hue='optimisation', markers=True,
                 style='optimisation', dashes=False, markersize=15,
                 ax=ax, errorbar=('ci', 95), err_style='bars')
    save_plot(os.path.join(folder, 'speedup_kernel.pdf'))

    figure = get_new_figure()
    ax = figure.add_subplot(1, 1, 1)
    figure.suptitle(title_total)
    ax.axhline(y=1, color='gray', linestyle='--')
    size_vs_y_plot(ax, 'Speedup', "Total Runtime - Speedup", speedups, size_var_name='NBLOCKS')
    sns.lineplot(speedups.xs('Total', level='scope'), x='NBLOCKS', y='Total time', hue='optimisation',
                 ax=ax, errorbar=('ci', 95), err_style='bars')
    save_plot(os.path.join(folder, 'speedup_total.pdf'))
