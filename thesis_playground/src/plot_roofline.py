from argparse import ArgumentParser
import json
from os import path
import matplotlib.pyplot as plt

from utils.paths import get_results_dir
from utils.plot import plot_roofline_cycles, plot_roofline_seconds
from measurement_data import MeasurementRun
from flop_computation import read_roofline_data


def main():
    parser = ArgumentParser(description="Creates a roofline plot")
    parser.add_argument(
            'files',
            type=str,
            nargs='+',
            help='Basename of the results and roofline file. Without file ending')
    parser.add_argument(
            '--output',
            type=str,
            help='Name of the file where to store the plot without ending')

    args = parser.parse_args()

    plt.rcParams.update({'figure.figsize': (19, 10)})
    plt.rcParams.update({'font.size': 12})
    hardware_filename = 'nodes.json'

    figure = plt.figure()
    ax_cycles = figure.add_subplot(2, 1, 1)
    ax_seconds = figure.add_subplot(2, 1, 2)
    for index, file in enumerate(args.files):
        print(file)
        results_filename = path.join(get_results_dir(), f"{file}.json")
        roofline_filename = path.join(get_results_dir(), f"{file}_roofline.json")
        with open(results_filename) as results_file:
            with open(hardware_filename) as node_file:
                run_data = json.load(results_file, object_hook=MeasurementRun.from_json)
                roofline_data = read_roofline_data(roofline_filename)
                node_data = json.load(node_file)
                gpu = node_data['ault_nodes'][run_data.node]['GPU']
                figure.suptitle(f"Roofline on {run_data.node} using a {gpu}")
                plot_roofline_cycles(run_data, roofline_data, node_data['GPUs'][gpu], ax_cycles, points_only=index > 0)
                plot_roofline_seconds(run_data, roofline_data, node_data['GPUs'][gpu], ax_seconds,
                                      points_only=index > 0)

    if args.output is None:
        plot_filename = f"{args.files[0]}_roofline.pdf"
    else:
        plot_filename = f"{args.output}.pdf"
    print(f"Save plot into {plot_filename}")
    plt.savefig(plot_filename)


if __name__ == '__main__':
    main()
