import json
from argparse import ArgumentParser
from tabulate import tabulate
import re

from test import get_stats


def main():
    parser = ArgumentParser()
    parser.add_argument(
            'dace_file',
            type=str,
            help='File with the results from DaCe instrumentation')
    parser.add_argument(
            'ncu_file',
            type=str,
            help='File with the results from NCU')

    args = parser.parse_args()

    flat_data = []
    headers = ["program", "measurement", "min", "max", "avg", "median"]
    # dictionary linking the triplet of (sdfg_id, state_id, node_id) to the program name
    key_map = {}
    with open(args.dace_file) as file:
        data = json.load(file)

        for program in data:
            for measurement in data[program]:
                if measurement not in ['LIKWID keys', 'GPU keys']:
                    flat_data.append([program, measurement])
                    for key in headers[2:]:
                        flat_data[-1].append(data[program][measurement][key])
                else:
                    key_map[tuple(data[program][measurement][0])] = program

    with open(args.ncu_file) as file:
        data = json.load(file)
        for kernel_name in data:
            match = re.match(r"[a-z_0-9]*_([0-9]*_[0-9]*_[0-9]*)\(", kernel_name)
            id_triplet = tuple([int(id) for id in match.group(1).split('_')])
            program = key_map[id_triplet]
            for measurement in ['durations', 'cycles']:
                stats = get_stats(data[kernel_name][measurement])
                unit = data[kernel_name][f"{measurement}_unit"]
                flat_data.append([program, f"{measurement} [{unit}] (#={len(data[kernel_name][measurement])})"])
                for key in headers[2:]:
                    flat_data[-1].append(stats[key])

    # print(key_map)
    flat_data.sort(key=lambda row: row[0])
    print(tabulate(flat_data, headers=headers))


if __name__ == '__main__':
    main()
