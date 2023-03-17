from argparse import ArgumentParser
import json
import os
from typing import Dict, List
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

from utils import get_results_dir
from measurement_data import ProgramMeasurement


def create_qq_plot(results: Dict):
    fig, axes = plt.subplots(1, len(results), sharey=True, sharex=False, tight_layout=True)
    fig.suptitle('QQ Plots of total runtime')
    for ax, result in zip(axes, results):
        measurement = result.measurements['Total time']
        ax.set_title(f"{result.program} (#={measurement.amount()})")
        sm.qqplot(np.array(measurement.data), line='45', ax=ax)

    plt.savefig('qqplot.png')


def create_histogram(results: List[ProgramMeasurement]):
    fig, axes = plt.subplots(1, len(results), sharey=True, sharex=False, tight_layout=True)
    fig.suptitle('Total runtime historgram')
    for ax, result in zip(axes, results):
        measurement = result.measurements['Total time']
        ax.set_title(f"{result.program} (#={measurement.amount()})")
        bins = np.linspace(measurement.min(), measurement.max(), num=20)
        ax.hist(measurement.data, bins=bins)
        ax.set_xlabel(f'Time [{measurement.unit}]')

    plt.savefig('histogram.png')


def main():
    parser = ArgumentParser()
    parser.add_argument(
            'file',
            type=str,
            nargs='+',
            help="Path to the json files to read")

    args = parser.parse_args()

    plt.rcParams.update({'figure.figsize': (19, 10)})
    plt.rcParams.update({'font.size': 12})

    with open(os.path.join(get_results_dir(), args.file[0])) as file:
        results = json.load(file, object_hook=ProgramMeasurement.from_json)
        create_qq_plot(results)
        create_histogram(results)


if __name__ == '__main__':
    main()
