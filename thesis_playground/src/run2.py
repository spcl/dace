from argparse import ArgumentParser
import os
import shutil
from subprocess import check_output
from numbers import Number
from typing import Optional, Union, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import json
from subprocess import run

from utils.print import print_dataframe
from utils.execute_dace import RunConfig, gen_ncu_report
from utils.paths import get_results_2_folder, get_thesis_playground_root_dir
from execute.parameters import ParametersProvider
from measurements.profile_config import ProfileConfig
from measurements.data2 import get_data_wideformat

experiment_list_df_path = os.path.join(get_results_2_folder(), 'experiments.csv')


def do_vertical_loops(additional_desc: Optional[str] = None, selected_program: Optional[str] = None):
    programs = [
            'cloudsc_vert_loop_4_ZSOLQA',
            'cloudsc_vert_loop_6_ZSOLQA',
            'cloudsc_vert_loop_6_1_ZSOLQA',
            'cloudsc_vert_loop_7_3'
            ]
    if selected_program is None:
        for program in programs:
            function_args = {'selected_program': program}
            if additional_desc is not None:
                function_args['additional_desc'] = additional_desc
            # can try start_new_session=True
            print(f"Starting a new process calling itself with {function_args}")
            run(['python3', __file__, 'profile', 'vert-loop', '--args', json.dumps(function_args)])
    else:
        print()
        print(f" *** Profile only program {selected_program} ***")
        print()
        programs = [selected_program]
        profile_configs = []
        for program in programs:
            params_list = []
            # for nblock in [4e5]:
            for nblock in np.arange(6e5, 1e4, -1e5):
            # for nblock in np.arange(3e3, 1e3, -1e3):
                params = ParametersProvider(program,
                                            update={'NBLOCKS': int(nblock), 'KLEV': 137, 'KFDIA': 1, 'KIDIA': 1, 'KLON': 1})
                params_list.append(params)
            profile_configs.append(ProfileConfig(program, params_list, ['NBLOCKS'], ncu_repetitions=0,
                tot_time_repetitions=1))

        experiment_desc = "Vertical loops with ZSOLQA"
        if additional_desc is not None:
            experiment_desc += f" with {additional_desc}"
        profile(profile_configs, RunConfig(), experiment_desc, [('temp allocation', 'stack')], ncu_report=False)
        for profile_config in profile_configs:
            profile_config.set_heap_limit = True
            profile_config.heap_limit_str = "(KLON * (NCLV - 1)) + KLON * NCLV * (NCLV - 1) + KLON * (NCLV - 1) +" + \
                                                "KLON * (KLEV - 1) + 4 * KLON"
        profile(profile_configs, RunConfig(specialise_symbols=False), experiment_desc, [('temp allocation', 'heap')],
                ncu_report=False)


base_experiments = {
    'vert-loop': do_vertical_loops
}


def get_experiment_list_df() -> pd.DataFrame():
    if not os.path.exists(experiment_list_df_path):
        os.makedirs(os.path.dirname(experiment_list_df_path), exist_ok=True)
        return pd.DataFrame()
    else:
        return pd.read_csv(experiment_list_df_path, index_col=['experiment id'])


def profile(program_configs: List[ProfileConfig], run_config: RunConfig, experiment_description: str,
            additional_columns: List[Tuple[str, Union[Number, str]]] = [], ncu_report: bool = True):
    """
    Profile the given programs with the given configurations

    :param program_configs: The profile configurations to use
    :type program_configs: List[ProfileConfig]
    :param run_config: The run configuration to use. Same configuration for every profiling run
    :type run_config: RunConfig
    :param experiment_description: The description of this experiment
    :type experiment_description: str
    :param additional_columns: List of any additional columns to add to the experiments database. Contains a tuple for
    each column. First is key, then value, defaults to []
    :type additional_columns: List[Tuple[str, Union[Number, str]]], optional
    :param ncu_report: If a full ncu report should be created and stored, defaults to True
    :type ncu_report: bool
    """
    experiment_list_df = get_experiment_list_df()
    if 'experiment id' in experiment_list_df.reset_index().columns and len(experiment_list_df.index) > 0:
        new_experiment_id = experiment_list_df.reset_index()['experiment id'].max() + 1
    else:
        new_experiment_id = 0


    git_hash = check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=get_thesis_playground_root_dir())\
        .decode('UTF-8').replace('\n', '')
    node = check_output(['uname', '-a']).decode('UTF-8').split(' ')[1].split('.')[0]
    this_experiment_data = {
        'experiment id': new_experiment_id,
        'description': experiment_description,
        'git hash': git_hash, 'node': node,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    for key, value in additional_columns:
        this_experiment_data.update({key: value})
    experiment_list_df = pd.concat([experiment_list_df,
                                   pd.DataFrame([this_experiment_data]).set_index(['experiment id'])])
    experiment_list_df.to_csv(experiment_list_df_path)

    for program_config in program_configs:
        experiment_folder = os.path.join(get_results_2_folder(), program_config.program, str(new_experiment_id))
        os.makedirs(experiment_folder, exist_ok=True)
        if ncu_report:
            program_config.compile(program_config.sizes[0], run_config)
            additional_columns_values = [col[1] for col in additional_columns]
            report_name = f"{program_config.program}_{new_experiment_id}" + '_'.join(additional_columns_values) + \
                          ".ncu-rep"
            gen_ncu_report(program_config.program,
                           os.path.join(experiment_folder, report_name),
                           run_config,
                           ncu_args=['--set', 'full'],
                           program_args=program_config.get_program_command_line_arguments(program_config.sizes[0],
                                                                                          run_config))
        program_config.profile(run_config).to_csv(os.path.join(experiment_folder, 'results.csv'))
        for index, params in enumerate(program_config.sizes):
            with open(os.path.join(experiment_folder, f"{index}_params.json"), 'w') as file:
                json.dump(params.get_dict(), file)


def action_profile(args):
    function_args = json.loads(args.args)
    base_experiments[args.base_exp](**function_args)


def action_print(args):
    columns = {
            'experiment id': ('Exp ID', ','),
            'node': ('Node', None),
            'program': ('Program', None),
            'NBLOCKS': ('NBLOCKS', ','),
            'runtime': ('T [s]', '.3e'),
            'measured bytes': ('D [b]', '.3e'),
            'theoretical bytes total': ('Q [b]', '.3e')
    }

    df = get_data_wideformat(args.experiment_ids).dropna()
    index_cols = list(df.index.names)
    index_cols.remove('run number')
    df = df.reset_index().groupby(index_cols).mean()
    df = df.join(get_experiment_list_df(), on='experiment id')
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
        experiment_ids = args.experiment_id
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
        experiments.to_csv(experiment_list_df_path)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Commands",
        help="See the help of the respective command")

    profile_parser = subparsers.add_parser('profile', description='Execute a profiling run')
    profile_parser.add_argument('base_exp', choices=base_experiments.keys(),
                                help='Name of the base experiment to use')
    profile_parser.add_argument('--args', type=str, default='{}',
                                help='Additional arguments passed to the base experiment function as a json-dictionary')
    profile_parser.set_defaults(func=action_profile)

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
