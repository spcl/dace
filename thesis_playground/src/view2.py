from argparse import ArgumentParser
import os
import numpy as np
import shutil

from utils.paths import get_results_2_folder, get_experiments_2_file
from utils.experiments2 import get_experiment_list_df
from utils.print import print_dataframe
from measurements.data2 import get_data_wideformat, average_data, add_column_if_not_exist


def action_print(args):
    columns = {
            'experiment id': ('Exp ID', ','),
            'node': ('Node', None),
            'program': ('Program', None),
            'NBLOCKS': ('NBLOCKS', ','),
            'runtime': ('Kernel T [s]', '.3e'),
            'Total time': ('tot T [s]', '.3e'),
            'measured bytes': ('D [b]', '.3e'),
            'theoretical bytes total': ('Q [b]', '.3e'),
            'k_caching': ('K-caching', None),
            'change_strides': ('change strides', None),
    }

    df = get_data_wideformat(args.experiment_ids).dropna()
    df = average_data(df).reset_index().join(get_experiment_list_df(), on='experiment id')
    add_column_if_not_exist(df, [('runtime', -1), ('Total time', -1), ('measured bytes', -1)])
    print_dataframe(columns, df.reset_index(), args.tablefmt)


def action_list_experiments(args):
    columns = {
            'experiment id': ('Exp ID', ','),
            'description': ('Description', None),
            'node': ('Node', None),
            'datetime': ('Date', None)
            }
    print_dataframe(columns, get_experiment_list_df().reset_index(), args.tablefmt)


def action_remove_experiment(args):
    experiment_ids = []
    if args.all_to is None:
        experiment_ids = [args.experiment_id]
    else:
        experiment_ids = np.arange(args.experiment_id, args.all_to+1)
    for experiment_id in experiment_ids:
        for program_dir in os.listdir(get_results_2_folder()):
            program_dir = os.path.join(get_results_2_folder(), program_dir)
            exp_dir = os.path.join(program_dir, str(experiment_id))
            if os.path.exists(exp_dir):
                print(f"Remove {exp_dir}")
                shutil.rmtree(exp_dir)
        experiments = get_experiment_list_df().drop(int(experiment_id), axis='index')
        experiments.to_csv(get_experiments_2_file())


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Commands",
        help="See the help of the respective command")

    print_parser = subparsers.add_parser('print', description='Print selected data')
    print_parser.add_argument('experiment_ids', nargs='+', help='Ids of experiments to print')
    print_parser.add_argument('--tablefmt', default='plain', help='Table format for tabulate')
    print_parser.set_defaults(func=action_print)

    list_experiments_parser = subparsers.add_parser('list-experiments', description='List experiments')
    list_experiments_parser.add_argument('--tablefmt', default='plain', help='Table format for tabulate')
    list_experiments_parser.set_defaults(func=action_list_experiments)

    remove_experiment_parser = subparsers.add_parser('remove-experiment', description='Remove experiment')
    remove_experiment_parser.add_argument('experiment_id', type=int)
    remove_experiment_parser.add_argument('--all-to', type=int, default=None,
                                          help="Delete all experiment ids from the first given to (inlcuding) this")
    remove_experiment_parser.set_defaults(func=action_remove_experiment)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
