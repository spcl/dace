from argparse import ArgumentParser

from utils.print import print_dataframe
from utils.vertical_loops import get_dataframe
from scripts import Script


class PrintMUEVertLoop(Script):
    name = "print-mue-vert"
    description = "Print MUE for the vertical loop results"

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('--mwe', action='store_true', default=False)
        parser.add_argument('--file-regex', type=str, default=None,
                            help='Only incldue files which match the given regex (anywhere in the filename)')

    @staticmethod
    def action(args):
        if args.mwe:
            data, descriptions, nodes = get_dataframe('_mwe_')
        elif args.file_regex is not None:
            data, descriptions, nodes = get_dataframe(args.file_regex)
        else:
            data, descriptions, nodes = get_dataframe()

        indices = ['program', 'size', 'short_desc']
        grouped_data = data.groupby(by=indices).mean()
        grouped_data['measured bytes min'] = data['measured bytes'].groupby(by=indices).min()
        grouped_data['measured bytes max'] = data['measured bytes'].groupby(by=indices).max()
        grouped_data['measured bytes range'] = grouped_data['measured bytes max'] - grouped_data['measured bytes min']
        grouped_data['io efficiency'] = grouped_data['theoretical bytes'] / grouped_data['measured bytes']
        grouped_data['io efficiency temp'] = grouped_data['theoretical bytes temp'] / grouped_data['measured bytes']
        grouped_data['bw efficiency'] = grouped_data['achieved bandwidth'] / grouped_data['peak bandwidth']
        grouped_data['mue'] = grouped_data['io efficiency'] * grouped_data['bw efficiency']
        grouped_data['mue temp'] = grouped_data['io efficiency temp'] * grouped_data['bw efficiency']
        grouped_data['count'] = data['measured bytes'].groupby(by=indices).count()
        grouped_data = grouped_data.reset_index()

        columns = {
                'program': ('Program', None),
                'short_desc': ('desc', None),
                'size': ('NBLOCKS', '.1e'),
                'measured bytes': ('measured bytes', '.3e'),
                'measured bytes range': ('range (max-min)', '.3e'),
                'theoretical bytes': ('theo. bytes', '.3e'),
                'theoretical bytes temp': ('theo. bytes temp', '.3e'),
                'io efficiency': ('I/O eff.', '.2f'),
                'io efficiency temp': ('I/O eff. temp', '.2f'),
                'bw efficiency': ('bw eff', '.2f'),
                'mue': ('MUE', '.2f'),
                'mue temp': ('MUE temp', '.2f'),
                'runtime': ('T [s]', '.3e'),
                'count': ('#', ''),
                }
        print_dataframe(columns, grouped_data)
