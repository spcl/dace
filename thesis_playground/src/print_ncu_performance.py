from argparse import ArgumentParser
from subprocess import run
import csv
import os
from typing import Dict
from tabulate import tabulate
import sys

from utils import convert_to_bytes
from print_utils import sort_by_program_number
from ncu_utils import get_action, get_achieved_performance, get_achieved_work, get_achieved_bytes, get_peak_performance

sys.path.insert(0, '/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.8.0-fjdnxm6yggxxp75sb62xrxxmeg4s24ml/nsight-compute-2022.3.0/extras/python/')
import ncu_report


def main():
    parser = ArgumentParser(description='TODO')
    parser.add_argument('basename')
    # parser.add_argument('--detail', action='store_true', default=False)

    args = parser.parse_args()

    folder = 'ncu-reports'
    flat_data = []
    headers = ['program', 'Q [byte]', 'W [flop]', 'I [Flop/byte]', 'P [flop/s]', 'beta [byte/s]',
               'peak P [flop/s]', 'peak beta [byte/s]', 'Performance [%]', 'Bandwidth [%]']
    for filename in os.listdir(folder):
        if filename.startswith(f"report_{args.basename}"):
            path = os.path.join(folder, filename)
            if len(filename.split(f"report_{args.basename}_")) == 1:
                program = args.basename
            else:
                program = filename.split(f"report_{args.basename}_")[1].split(".")[0]
            action = get_action(path)
            achieved_work = get_achieved_work(action)
            Q = get_achieved_bytes(action)
            peak = get_peak_performance(action)
            achieved_performance = get_achieved_performance(action)
            flat_data.append([program, int(Q), int(achieved_work['dW']), achieved_work['dW'] / Q,
                              achieved_performance[0], achieved_performance[1],
                              peak[0], peak[1],
                              achieved_performance[0] / peak[0] * 100, achieved_performance[1] / peak[1] * 100])

    sort_by_program_number(flat_data)
    print(tabulate(flat_data, headers=headers, intfmt=',',
                   floatfmt=(None, None, None, ".3f", ".1E", ".1E", ".1E", ".1E", ".1f", ".1f")))


if __name__ == '__main__':
    main()
