from argparse import ArgumentParser
import json
import os

from utils import print_results_v2, get_results_dir
from measurement_data import MeasurementRun


def main():
    parser = ArgumentParser()
    parser.add_argument('file', type=str, help="Path to the json files to read")

    args = parser.parse_args()
    with open(os.path.join(get_results_dir(), args.file)) as file:
        run_data = json.load(file, object_hook=MeasurementRun.from_json)
        print_results_v2(run_data)


if __name__ == '__main__':
    main()
