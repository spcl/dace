from argparse import ArgumentParser
from tabulate import tabulate

from utils.ncu import get_achieved_work, get_achieved_bytes
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
                    ('W [flop]', lambda msr, action, roof: roof[0].get_total_flops()),
                    ('W (ncu) [flop]', lambda msr, action, roof: get_achieved_work(action)['dW']),
                    ('Q [bytes]', lambda msr, action, roof: get_number_of_bytes_2(msr.parameters, msr.program)[0]),
                    ('D (ncu) [bytes]', lambda msr, action, roof: get_achieved_bytes(action)),
                ]

        ncu_actions = get_actions(args.name)
        roofline_data = get_roofline_data(args.name)
        tabulate_data = []
        for msr in get_program_measurements(args.name):
            action = ncu_actions[msr.program]
            roofline = roofline_data[msr.program]
            tabulate_data.append([col[1](msr, action, roofline) for col in columns])

        print(tabulate(tabulate_data, headers=[col[0] for col in columns], intfmt=','))
