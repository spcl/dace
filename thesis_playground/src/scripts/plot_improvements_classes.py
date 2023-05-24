import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from scripts import Script
from utils.paths import get_complete_results_dir
from utils.plot import set_general_plot_style, save_plot, rotate_xlabels
from utils.complete_results import get_dataframe


def key_program_sort(programs):
    """
    Sort by program name/index/version
    """
    programs = programs.str.extract("cloudsc_class([0-9])_([0-9]{3,4})")
    programs[0] = pd.to_numeric(programs[0])
    programs[1] = pd.to_numeric(programs[1])
    programs = programs[0].apply(lambda x: 10000*x) + programs[1]
    return programs


class PlotImprovementsClasses(Script):
    name = 'plot-improvements-classes'
    description = "Plot the improvements made in the classes"

    @staticmethod
    def action(args):
        set_general_plot_style()

        # Create subplots
        figure = plt.figure()
        ax_runtime_speedup_1 = figure.add_subplot(1, 2, 1)
        ax_runtime_speedup_2 = figure.add_subplot(1, 2, 2)
        figure.subplots_adjust(hspace=0.05)

        # Get data
        data, classes, parameters = get_dataframe(['all_first_opt_no_pattern', 'class2_3_baseline_no_pattern',
                                                   'class1_baseline'])
        data = data.join(classes, on='program')

        run_numbers = data \
            .reset_index()\
            .drop(['unit', 'node', 'experiment name', 'run description', 'kernel name', 'class', 'value'],
                  axis='columns') \
            .groupby(['program', 'measurement name', 'auto_opt']) \
            .count()
        number_of_runs = run_numbers.min()[0]
        if run_numbers.max()[0] != number_of_runs:
            number_of_runs = None
            print(f"WARNING: Number of runs varies between {run_numbers.min()[0]} and {run_numbers.max()[0]}")
        avg_data = data \
            .reset_index() \
            .drop(['run number', 'unit', 'node', 'experiment name', 'run description', 'kernel name', 'class'],
                  axis='columns') \
            .groupby(['program', 'measurement name', 'auto_opt']) \
            .mean()
        improved_data = avg_data.xs('My', level=2)
        baseline_data = avg_data.xs('DaCe', level=2)
        speedup_data = improved_data.copy()
        speedup_data = baseline_data.div(improved_data).sort_values('program', key=key_program_sort)
        unique_params = parameters.reset_index() \
            .drop(['experiment name', 'run description'], axis='columns') \
            .groupby(['program']) \
            .nunique() \
            .max(axis='columns')

        if unique_params.max(axis='index') > 1:
            print("WARNING: Some programs don't have the same parameters")
            print(unique_params)

        data.reset_index(inplace=True)
        speedup_data = speedup_data.join(classes, on='program')
        speedup_data.reset_index(inplace=True)

        figure.suptitle(f"Speedup of my improvements averaged over {number_of_runs} runs")
        ax_runtime_speedup_1.axhline(y=1, color='gray')
        ax_runtime_speedup_2.axhline(y=1, color='gray')
        hue_order = ['Total Time', 'Kernel Time', 'Kernel Cycles']
        sns.pointplot(data=speedup_data[speedup_data['class'] == 1], x='program', y='value', ax=ax_runtime_speedup_1,
                      join=False, hue='measurement name')
        sns.pointplot(data=speedup_data[speedup_data['class'] > 1], x='program', y='value', ax=ax_runtime_speedup_2,
                      join=False, hue='measurement name', hue_order=hue_order)
        ax_runtime_speedup_1.set_ylabel('Speedup')
        ax_runtime_speedup_2.set_ylabel('')
        ax_runtime_speedup_1.set_xlabel('')
        ax_runtime_speedup_2.set_xlabel('')
        rotate_xlabels(ax_runtime_speedup_1)
        rotate_xlabels(ax_runtime_speedup_2)
        ax_runtime_speedup_1.legend([], [], frameon=False)

        # Save plot
        plt.tight_layout()
        save_plot(os.path.join(get_complete_results_dir(), 'plots', 'improvements.png'))
