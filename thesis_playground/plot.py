"""Program to plot results"""
from argparse import ArgumentParser
import json
import os
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable

from utils import get_results_dir
from measurement_data import MeasurementRun


def create_plot_grid(
        runs_data: List[MeasurementRun],
        filename: str,
        plot_title: str,
        subfigure_plot_function: Callable[[matplotlib.axis.Axis, MeasurementRun], None]):
    """
    Function to easily plot the same plot for several programs and runs in a 2D grid.
    Y axes are shared between the different plots in a row. X axes are shared in each column.

    :param runs_data: The list of all run data
    :type runs_data: List[MeasurementRun]
    :param filename: The filename where the plot should be saved after
    :type filename: str
    :param plot_title: The title of the whole plot
    :type plot_title: str
    :param subfigure_plot_function: The function which is called to create a subplot. It is given the axis to of the
    plot and the MeasurementRun object with the data to plot there
    :type subfigure_plot_function: Callable[[matplotlib.axis.Axis, MeasurementRun], None]
    """

    fig = plt.figure(constrained_layout=True)
    fig.suptitle(plot_title)

    subfigs = fig.subfigures(nrows=len(runs_data), ncols=1)
    if not isinstance(subfigs, np.ndarray):
        subfigs = [subfigs]

    first_row_axes = []
    for row, (run_data, subfig) in enumerate(zip(runs_data, subfigs)):
        subfig.suptitle(f"{run_data.description} at {run_data.date.strftime('%Y-%m-%d %H:%H')} "
                        f"and commit {run_data.git_hash} on {run_data.node}")
        for col, result in enumerate(run_data.data, start=1):
            if row == 0:
                ax = subfig.add_subplot(1, len(run_data.data), col)
                first_row_axes.append(ax)
            else:
                ax = subfig.add_subplot(1, len(run_data.data), col, sharex=first_row_axes[col-1])
            subfigure_plot_function(ax, result)

    plt.savefig(filename)


def create_qq_plot(runs_data: List[MeasurementRun], filename: str = "qqplot.png"):
    """
    Creates a QQ plot of the total runtime

    :param runs_data: The data to use
    :type runs_data: List[MeasurementRun]
    :param filename: The name of the file where the plot is saved, defaults to qqplot.png
    :type filename: str
    """

    def subfigure_plot_function(ax, result):
        measurement = result.measurements['Total time'][0]
        ax.set_title(f"{result.program} (#={measurement.amount()})")
        sm.qqplot(np.array(measurement.data), line='45', ax=ax)

    create_plot_grid(runs_data, filename, "QQ Plots of total runtime", subfigure_plot_function)


def create_histogram(runs_data: List[MeasurementRun], filename: str = "histogram.png"):
    """
    Creates a histogram plot of the total runtime

    :param runs_data: [TODO:description]
    :type runs_data: List[MeasurementRun]
    :param filename: The name of the file where the plot is saved, defaults to histogram.png
    :type filename: str
    """

    def subfigure_plot_function(ax, result):
        measurement = result.measurements['Total time'][0]
        ax.set_title(f"{result.program} (#={measurement.amount()})")
        bins = np.linspace(measurement.min(), measurement.max(), num=20)
        ax.hist(measurement.data, bins=bins)
        ax.set_xlabel(f'Time [{measurement.unit}]')

    create_plot_grid(runs_data, filename, "Total runtime histogram", subfigure_plot_function)


def main():
    parser = ArgumentParser(description="Plots the results it reads from the given paths")
    parser.add_argument(
            'file',
            type=str,
            nargs='+',
            help="Path to the json files to read")

    args = parser.parse_args()

    plt.rcParams.update({'figure.figsize': (19, 10)})
    plt.rcParams.update({'font.size': 12})

    run_data = []
    for file_path in args.file:
        with open(os.path.join(get_results_dir(), file_path)) as file:
            run_data.append(json.load(file, object_hook=MeasurementRun.from_json))

    create_qq_plot(run_data)
    create_histogram(run_data)


if __name__ == '__main__':
    main()
