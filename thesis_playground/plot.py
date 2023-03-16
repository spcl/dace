from argparse import ArgumentParser
import json
import os
from typing import Dict
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils import get_results_dir


def create_qq_plot(results: Dict):
    programs = list(results.keys())
    # measurements = list(results[programs[0]].keys())
    fig, axes = plt.subplots(1, len(programs), sharey=True, sharex=True, tight_layout=True)
    for ax, program in zip(axes, programs):
        measurement = 'Total time [ms]'
        ax.set_title(program)
        data = np.array(results[program][measurement]['data'])
        sm.qqplot(data, line='45', ax=ax)

    plt.savefig('qqplot.png')


def create_histogram(results: Dict):
    programs = list(results.keys())
    fig, axes = plt.subplots(1, len(programs), sharey=True, sharex=False, tight_layout=True)
    for ax, program in zip(axes, programs):
        measurement = 'Total time [ms]'
        ax.set_title(program)
        data = np.array(results[program][measurement]['data'])
        bins = np.linspace(data.min(), data.max(), num=20)
        ax.hist(data, bins=bins)
        ax.set_xlabel('Time [ms]')

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

    with open(os.path.join(get_results_dir(), args.file)) as file:
        results = json.load(file)
        create_qq_plot(results)
        create_histogram(results)


if __name__ == '__main__':
    main()
