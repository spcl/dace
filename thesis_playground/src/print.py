from argparse import ArgumentParser
import json
import os

from utils.general import get_results_dir
from utils.print import print_results_v2, print_performance, print_flop_counts, print_memory_details
from measurements.data import MeasurementRun
from measurements.flop_computation import read_roofline_data


def main():
    parser = ArgumentParser()
    parser.add_argument('file', type=str, help="Path to the json files to read")
    parser.add_argument('--roofline', type=str, default=None, help="Path to roofline data file")
    parser.add_argument('--detail', default=False, action='store_true')
    hardware_filename = 'nodes.json'

    args = parser.parse_args()
    with open(os.path.join(get_results_dir(), args.file)) as file:
        run_data = json.load(file, object_hook=MeasurementRun.from_json)
        print_results_v2(run_data)

        if args.roofline is not None:
            with open(hardware_filename) as hardware_file:
                node_data = json.load(hardware_file)
                roofline_data = read_roofline_data(os.path.join(get_results_dir(), args.roofline))
                gpu = node_data['ault_nodes'][run_data.node]['GPU']
                print()
                print('Performance')
                print_performance(roofline_data, run_data, hardware_dict=node_data['GPUs'][gpu])
                if args.detail:
                    print()
                    print("Work details")
                    print_flop_counts(roofline_data)
                    print()
                    print("Memory details")
                    print_memory_details(run_data)


if __name__ == '__main__':
    main()
