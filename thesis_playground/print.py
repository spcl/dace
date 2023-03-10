from argparse import ArgumentParser
import json
import os

from utils import print_results_v2, get_results_dir


def main():
    parser = ArgumentParser()
    parser.add_argument(
            'file',
            type=str,
            help="Path to the json files to read")

    args = parser.parse_args()
    with open(os.path.join(get_results_dir(), args.file)) as file:
        results = json.load(file)
        print_results_v2(results)


if __name__ == '__main__':
    main()
