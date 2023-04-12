from argparse import ArgumentParser
import json
import os

from utils import get_results_dir
from print_utils import print_results_v2, print_performance
from measurement_data import MeasurementRun
from flop_computation import read_roofline_data


def main():
    parser = ArgumentParser()
    parser.add_argument('file', type=str, help="Path to the json files to read")
    parser.add_argument('--roofline', type=str, default=None, help="Path to roofline data file")

    args = parser.parse_args()
    with open(os.path.join(get_results_dir(), args.file)) as file:
        run_data = json.load(file, object_hook=MeasurementRun.from_json)
        print_results_v2(run_data)

        if args.roofline is not None:
            roofline_data = read_roofline_data(os.path.join(get_results_dir(), args.roofline))
            print_performance(roofline_data, run_data)


if __name__ == '__main__':
    main()
