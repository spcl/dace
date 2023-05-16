from argparse import ArgumentParser
from tabulate import tabulate
import json
import os
import numpy as np

from utils.vertical_loops import get_data
from scripts import Script


class PrintMUEVertLoop(Script):
    name = "print-mue-vert"
    description = "Print MUE for the vertical loop results"

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('--mwe', action='store_true', default=False)

    @staticmethod
    def action(args):
        if args.mwe:
            data = get_data('_mwe_')
        else:
            data = get_data()

        tabulate_data = []
        for program in data:
            ncuQ_avg = int(np.average(data[program]['measured bytes']))
            ncuQ_range = int(np.max(data[program]['measured bytes']) - np.min(data[program]['measured bytes']))
            myQ = data[program]['theoretical bytes']
            myQ_temp = data[program]['theoretical bytes temp']
            io_efficiency = myQ / ncuQ_avg
            io_efficiency_temp = myQ_temp / ncuQ_avg
            bw_efficiency = np.average(data[program]['achieved bandwidth'] / data[program]['peak bandwidth'])
            mue = io_efficiency * bw_efficiency
            mue_temp = io_efficiency_temp * bw_efficiency
            runtime = np.average(data[program]['runtime'])
            tabulate_data.append([program, ncuQ_avg, ncuQ_range, myQ, myQ_temp, io_efficiency, io_efficiency_temp,
                                  bw_efficiency, mue, mue_temp, runtime])

        tabulate_data.sort()
        print(tabulate(tabulate_data,
                       headers=['program', 'measured bytes', 'measured bytes range', 'theoretical bytes', 'theo. bytes with temp',
                                'I/O eff.', 'I/O eff. w/ temp', 'BW eff.', 'MUE', 'MUE with temp',
                                'runtime [s]'],
                       intfmt=',', floatfmt=(None, None, None, None, '.2f', '.2f', '.2f', '.2f', '.2e', '.2e', '.2e')))
