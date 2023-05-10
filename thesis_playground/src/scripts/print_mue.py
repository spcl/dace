from argparse import ArgumentParser
from tabulate import tabulate

from utils.ncu import get_achieved_bytes, get_achieved_performance, get_peak_performance
from utils.complete_results import get_actions, get_program_measurements
from measurements.flop_computation import get_number_of_bytes_2
from scripts import Script


class PrintMUE(Script):
    name = "print-mue"
    description = "Prints the MUE for the data in complete_results/all_frist_opt"

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('--tablefmt', type=str, default="simple", help="Table format for tabulate to use")

    @staticmethod
    def action(args):
        data = {}
        for program, action in get_actions('all_first_opt').items():
            data[program] = {}
            data[program]['measured bytes'] = get_achieved_bytes(action)
            data[program]['achieved bandwidth'] = get_achieved_performance(action)[1]
            data[program]['peak bandwidth'] = get_peak_performance(action)[1]

        for program_measurement in get_program_measurements('all_first_opt'):
            program = program_measurement.program
            myQ = get_number_of_bytes_2(program_measurement.parameters, program)[0]
            if program not in data:
                data[program] = {}
            data[program]['theoretical bytes'] = myQ

        tabulate_data = []
        for program in data:
            ncuQ = data[program]['measured bytes']
            myQ = data[program]['theoretical bytes']
            io_efficiency = myQ / ncuQ
            bw_efficiency = data[program]['achieved bandwidth'] / data[program]['peak bandwidth']
            mue = io_efficiency * bw_efficiency
            tabulate_data.append([program, ncuQ, myQ, io_efficiency, bw_efficiency, mue])

        tabulate_data.sort(key=lambda row: int(row[0].split('_')[1][-1])*1000 + int(row[0].split('_')[2]))
        print(tabulate(tabulate_data,
                       headers=['program', 'measured bytes', 'theoretical bytes', 'I/O efficiency', 'BW efficiency',
                                'MUE'],
                       intfmt=',', floatfmt='.2f', tablefmt=args.tablefmt))
