import os
from typing import Optional, Union, Dict, Tuple
import pandas as pd
import seaborn as sns

from utils.data_analysis import compute_speedups
from utils.experiments2 import get_program_infos
from utils.plot import save_plot, get_new_figure, size_vs_y_plot, get_bytes_formatter, legend_on_lines, \
                       replace_legend_names, legend_on_lines_dict


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
    work_data = data.xs('work', level='scope')
    # transpose kernel data not there
    # transpose_data = data.xs('transpose', level='scope')
    size_vs_y_plot(ax, 'time [s]', "Kernel time", data, size_var_name='NBLOCKS')
    sns.lineplot(work_data, x='NBLOCKS', y='runtime', hue='k_caching', style='change_strides',
                 ax=ax, errorbar=('ci', 95), err_style='bars')
    # sns.lineplot(transpose_data, x='NBLOCKS', y='runtime', hue='k_caching', style='change_strides', ax=ax, errorbar=('ci', 95), err_style='bars')
    save_plot(os.path.join(folder, 'kerneltime.pdf'))


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
