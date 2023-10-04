import os
from typing import Optional, Union, Dict, Tuple
import pandas as pd
import seaborn as sns

from utils.data_analysis import compute_speedups
from utils.experiments2 import get_program_infos
from utils.plot import save_plot, get_new_figure, size_vs_y_plot, get_bytes_formatter, legend_on_lines, \
                       replace_legend_names, legend_on_lines_dict, rotate_xlabels


def plot_runtime_bar(data: pd.DataFrame, avg_data: pd.DataFrame, folder: str, title_total: str, title_ncu: str):
    speedups = compute_speedups(avg_data, (False, False), ('outside_first', 'move_assignments_outside'))\
            .xs((1, 'Total', 3e3, True, False),
                level=('class_number', 'scope', 'KLON', 'outside_first','move_assignments_outside'))\
            .reset_index()\
            .drop(columns=['num_kernels', 'KLEV'])\
            .set_index(['program'])

    figure = get_new_figure()
    programs = data.xs((1, False), level=('class_number', 'move_assignments_outside')).reset_index()['program'].unique()
    axes = figure.subplots(1, len(programs))
    for ax, program in zip(axes, programs):
        this_data = data.xs((1, 'Total', 3e3, program, False),
                            level=('class_number', 'scope', 'KLON', 'program', 'move_assignments_outside'))\
                                    .reset_index()
        sns.barplot(this_data,
                    x='outside_first', y='Total time', ax=ax)
        ax.bar_label(ax.containers[0], fmt='%.2f')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(f"Loops l. {program.split('_')[-1]}")
        rotate_xlabels(ax, angle=20, replace_dict={'False': 'default order', 'True': 'outside first'})

    axes[0].set_ylabel('Total time [s]')
    axes[0].annotate(f"{speedups.loc[programs[0]]['Total time']:.1f}x", xy=(False, 0.03), horizontalalignment='center')
    axes[1].annotate(f"{speedups.loc[programs[1]]['Total time']:.1f}x", xy=(False, 0.1), horizontalalignment='center')
    axes[2].annotate(f"{speedups.loc[programs[2]]['Total time']:.1f}x", xy=(False, 0.1), horizontalalignment='center')
    axes[3].annotate(f"{speedups.loc[programs[3]]['Total time']:.1f}x", xy=(False, 0.8), horizontalalignment='center')
    figure.tight_layout()
    save_plot(os.path.join(folder, 'runtime_bar_class_1.pdf'))

    figure = get_new_figure()
    programs = data.xs((2, True), level=('class_number', 'outside_first')).reset_index()['program'].unique()
    axes = figure.subplots(1, len(programs))
    # print(data.xs((2), level=('class_number')))
    for ax, program in zip(axes, programs):
        this_data = data.xs((2, 5e3, program, True),
                            level=('class_number', 'KLON', 'program', 'outside_first'))\
                                    .reset_index()
        this_data['runtime'] = this_data['runtime'].apply(lambda x: x*1000)
        sns.barplot(this_data,
                    x='move_assignments_outside', y='runtime', ax=ax)
        ax.bar_label(ax.containers[0], fmt='%.2f')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(f"Loops l. {program.split('_')[-1]}")
        rotate_xlabels(ax, angle=20, replace_dict={'False': 'original', 'True': 'move assignment outside'})

    axes[0].set_ylabel('Kernel time [ms]')
    figure.tight_layout()
    save_plot(os.path.join(folder, 'runtime_bar_class_2.pdf'))


def plot_speedup(data: pd.DataFrame, folder: str, title_total: str, title_ncu: str):
    speedups = compute_speedups(data, (False, False), ('outside_first', 'move_assignments_outside'))
    index_cols = list(speedups.index.names)
    speedups = speedups.reset_index()
    speedups = speedups.drop(speedups[(speedups['outside_first'] == False) &
                                      (speedups['move_assignments_outside'] == False)].index)
    index_cols.remove('outside_first')
    index_cols.remove('move_assignments_outside')
    speedups = speedups\
        .replace({'outside_first': {True: 'with outside loops first', False: ''},
                  'move_assignments_outside': {True: 'with move assignments outside', False: ''}})
    speedups['optimisation'] = speedups['outside_first'] + ' and ' + speedups['move_assignments_outside']
    index_cols.append('optimisation')
    speedups = speedups.set_index(index_cols)

    for class_number in range(1, 4):
        figure = get_new_figure()
        ax = figure.add_subplot(1, 1, 1)
        figure.suptitle(title_ncu)
        ax.axhline(y=1, color='gray', linestyle='--')
        size_vs_y_plot(ax, 'Speedup', "Total Runtime - Speedup", speedups, size_var_name='KLON')
        sns.lineplot(speedups.xs((class_number, 'Total'), level=('class_number', 'scope')), x='KLON', y='Total time',
                     hue='program', style='optimisation',  ax=ax)
        save_plot(os.path.join(folder, f"speedup_total_class{class_number}.pdf"))

    for class_number in range(2, 4):
        figure = get_new_figure()
        ax = figure.add_subplot(1, 1, 1)
        figure.suptitle(title_ncu)
        ax.axhline(y=1, color='gray', linestyle='--')
        size_vs_y_plot(ax, 'Speedup', "Total Kerneltime - Speedup", speedups, size_var_name='KLON')
        sns.lineplot(speedups.xs(class_number, level='class_number'), x='KLON', y='runtime', hue='program',
                     style='optimisation',  ax=ax)
        save_plot(os.path.join(folder, f"speedup_kernel_class{class_number}.pdf"))


def plot_roofline_kernel(data: pd.DataFrame, folder: str, title: str):
    figure = get_new_figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.set(xscale="log", yscale="log")
    figure.suptitle(title)
    data = data.assign(operational_intensity=lambda x: x['dW'] / x['measured bytes'])
    sns.scatterplot(data=data,
                    x='operational_intensity', y='measured performance',
                    hue='class_number', hue_order=[1, 2, 3],
                    palette={1: sns.color_palette()[0], 2: sns.color_palette()[1], 3: sns.color_palette()[2]},
                    style='class_number', s=140)
    peak_performance_min = data['peak performance'].min()
    peak_performance_max = data['peak performance'].max()
    peak_performance = data['peak performance'].mean()
    max_bandwidth = data['peak bandwidth'].mean()
    crosspoint_intensity = peak_performance / max_bandwidth
    color = 'black'
    min_intensity = data['operational_intensity'].min()
    max_intensity = data['operational_intensity'].max()
    ax.loglog([min_intensity, crosspoint_intensity], [max_bandwidth * (min_intensity), peak_performance], color=color)
    ax.loglog([crosspoint_intensity, 10], [peak_performance, peak_performance], color=color)
    ax.set_xlabel("Performance")
    ax.set_ylabel("Operational Intensity")
    ax.text(0.3, 5.4e11, f"{max_bandwidth/1e9:.0f} GByte/s", rotation=29)
    ax.text(5, 8e12, f"{peak_performance/1e9:.0f} GFlop/s")
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1]*1.1)

    print(f"min peak P: {peak_performance_min}, max peak P: {peak_performance_max}")
    save_plot(os.path.join(folder, "roofline_kernel.pdf"))


def plot_roofline_theoretical(data: pd.DataFrame, folder: str, title: str):
    figure = get_new_figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.set(xscale="log", yscale="log")
    figure.suptitle(title)
    data = data.assign(operational_intensity=lambda x: x['theoretical flops'] / x['theoretical bytes total'])
    data = data.assign(performance=lambda x: x['theoretical flops'] / x['Total time'])
    sns.scatterplot(data=data,
                    x='operational_intensity', y='measured performance',
                    size='KLON', hue='program', style='class_number')
    # TODO: Replace peak performance/bandwidth by values from other source
    peak_performance_min = data['peak performance'].min()
    peak_performance_max = data['peak performance'].max()
    peak_performance = data['peak performance'].mean()
    max_bandwidth = data['peak bandwidth'].mean()
    crosspoint_intensity = peak_performance / max_bandwidth
    color = 'black'
    min_intensity = data['operational_intensity'].min()
    max_intensity = data['operational_intensity'].max()
    ax.loglog([min_intensity, crosspoint_intensity], [max_bandwidth * (min_intensity), peak_performance], color=color)
    ax.loglog([crosspoint_intensity, max_intensity], [peak_performance, peak_performance], color=color)

    print(f"min peak P: {peak_performance_min}, max peak P: {peak_performance_max}")
    save_plot(os.path.join(folder, "roofline_theoretical.pdf"))
