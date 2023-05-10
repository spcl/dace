from argparse import ArgumentParser
from tabulate import tabulate

from utils.ncu import get_achieved_work, get_achieved_bytes, get_runtime, get_peak_performance, get_achieved_performance
from measurements.flop_computation import get_number_of_bytes_2
from utils.complete_results import get_actions, get_program_measurements, get_roofline_data
from scripts import Script


class PrintDetailCompleteResults(Script):
    name = "print-detail-complete-results"
    description = "Print details of some programs from complete results"

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('name', help="Name/folder in complete_results to take the data from")
        parser.add_argument('--program', nargs='+', default=None, help="Programs to print")

    @staticmethod
    def action(args):

        rows = [
                    ('KLON', lambda msr, action, roof: msr.parameters['KLON'], ','),
                    ('KLEV', lambda msr, action, roof: msr.parameters['KLEV'], ','),
                    ('KIDIA', lambda msr, action, roof: msr.parameters['KIDIA'], ','),
                    ('KFDIA', lambda msr, action, roof: msr.parameters['KFDIA'], ','),
                    ('W [flop]', lambda msr, action, roof: roof[0].get_total_flops(), ','),
                    ('adds [flop]', lambda msr, action, roof: roof[0].adds, ','),
                    ('muls [flop]', lambda msr, action, roof: roof[0].muls, ','),
                    ('divs [flop]', lambda msr, action, roof: roof[0].divs, ','),
                    ('sqrt [flop]', lambda msr, action, roof: roof[0].roots, ','),
                    ('dW (ncu) [flop]', lambda msr, action, roof: get_achieved_work(action)['dW'], ','),
                    ('dadds (ncu) [flop]', lambda msr, action, roof: get_achieved_work(action)['dadds'], ','),
                    ('dmuls (ncu) [flop]', lambda msr, action, roof: get_achieved_work(action)['dmuls'], ','),
                    ('dfmas (ncu) [flop]', lambda msr, action, roof: get_achieved_work(action)['dfmas'], ','),
                    ('fW (ncu) [flop]', lambda msr, action, roof: get_achieved_work(action)['fW'], ','),
                    ('fadds (ncu) [flop]', lambda msr, action, roof: get_achieved_work(action)['fadds'], ','),
                    ('fmuls (ncu) [flop]', lambda msr, action, roof: get_achieved_work(action)['fmuls'], ','),
                    ('ffmas (ncu) [flop]', lambda msr, action, roof: get_achieved_work(action)['ffmas'], ','),
                    ('Q [bytes]', lambda msr, action, roof: get_number_of_bytes_2(msr.parameters, msr.program)[0], ','),
                    ('D (ncu) [bytes]', lambda msr, action, roof: get_achieved_bytes(action), ','),
                    ('Kernel time [s]', lambda msr, action, roof: get_runtime(action), '.3e'),
                    ('Bandwidth [%]', lambda msr, action, roof: float(get_achieved_performance(action)[1] /
                                                                get_peak_performance(action)[1]) * 100, '.2f'),
                ]

        ncu_actions = get_actions(args.name)
        roofline_data = get_roofline_data(args.name)
        tabulate_data = []
        measurements = get_program_measurements(args.name)
        programs = args.program
        if programs is None:
            programs = [msr.program for msr in measurements]
        for row in rows:
            tab_row = [row[0]]
            for msr in measurements:
                if msr.program in programs:
                    action = ncu_actions[msr.program]
                    roofline = roofline_data[msr.program]
                    format_str = "{0:"+row[2]+"}"
                    tab_row.append(format_str.format(row[1](msr, action, roofline)))
            tabulate_data.append(tab_row)

        colalign = ["left", *["right"] * len(programs)]
        print(tabulate(tabulate_data, headers=['Value', *programs], colalign=colalign))
