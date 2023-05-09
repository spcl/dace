from argparse import ArgumentParser
from tabulate import tabulate

from utils.ncu import get_achieved_work, get_achieved_bytes, get_runtime, get_peak_performance, get_achieved_performance
from measurements.flop_computation import get_number_of_bytes_2
from utils.complete_results import get_actions, get_program_measurements, get_roofline_data
from scripts import Script


class PrintCompleteResults(Script):
    name = "print-complete-results"
    description = "Print results from complete results"

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('name', help="Name/folder in complete_results to take the data from")

    @staticmethod
    def action(args):

        columns = [
                    ('program', lambda msr, action, roof: msr.program),
                    ('W [flop]', lambda msr, action, roof: roof[0].get_total_flops(), ','),
                    ('W (ncu) [flop]', lambda msr, action, roof: get_achieved_work(action)['dW'], ','),
                    ('Q [bytes]', lambda msr, action, roof: get_number_of_bytes_2(msr.parameters, msr.program)[0], ','),
                    ('D (ncu) [bytes]', lambda msr, action, roof: get_achieved_bytes(action), ','),
                    ('Kernel time [s]', lambda msr, action, roof: get_runtime(action), '.3e'),
                    ('Bandwidth [%]', lambda msr, action, roof: float(get_achieved_performance(action)[1] /
                                                                get_peak_performance(action)[1]) * 100, '.2f'),
                ]

        ncu_actions = get_actions(args.name)
        roofline_data = get_roofline_data(args.name)
        tabulate_data = []
        for msr in get_program_measurements(args.name):
            action = ncu_actions[msr.program]
            roofline = roofline_data[msr.program]
            tabulate_data.append([col[1](msr, action, roofline) for col in columns])

        floatfmt = [c[2] if isinstance(val, float) else None for c, val in zip(columns, tabulate_data[0])]
        intfmt = [c[2] if isinstance(val, int) else None for c, val in zip(columns, tabulate_data[0])]
        print(tabulate(tabulate_data, headers=[col[0] for col in columns], intfmt=intfmt, floatfmt=floatfmt))
