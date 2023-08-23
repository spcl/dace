from argparse import ArgumentParser
import os
from subprocess import check_output
from numbers import Number
from typing import Optional, Union, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import json
import sympy
import shutil
import logging

from utils.log import setup_logging, close_filehandlers
from utils.execute_dace import RunConfig, test_program
from utils.paths import get_results_2_folder, get_thesis_playground_root_dir, get_experiments_2_file, \
                        create_if_not_exist, get_results_2_logdir
from utils.experiments2 import get_experiment_list_df
from execute.parameters import ParametersProvider
from measurements.data2 import read_data_from_result_file
from measurements.profile_config import ProfileConfig

logger = logging.getLogger("run2")


def do_cloudsc() -> List[int]:
    program = 'cloudsc_vert_loop_10'
    params = []
    experiment_ids = []
    params.append(ParametersProvider(program, update={'NBLOCKS': 1000}))
    params.append(ParametersProvider(program, update={'NBLOCKS': 2000}))
    params.append(ParametersProvider(program, update={'NBLOCKS': 3000}))
    profile_config = ProfileConfig(program, params, ['NBLOCKS'], ncu_repetitions=0, tot_time_repetitions=5)
    # experiment_desc = 'Full cloudsc'
    experiment_desc = 'vert_loop_10'
    experiment_ids.append(profile(profile_config, RunConfig(k_caching=False, change_stride=False),
                                  experiment_desc + ' baseline', [('k_caching', "False"), ('change_strides', 'False')],
                                  ncu_report=False))
    experiment_ids.append(profile(profile_config, RunConfig(k_caching=True, change_stride=True),
                                  experiment_desc + ' my optimisations', [('k_caching', "True"), ('change_strides', 'True')],
                                  ncu_report=False))
    return experiment_ids


def do_test() -> List[int]:
    program = 'mwe_array_order'
    profile_configs = []
    params1 = [ParametersProvider(program, update={'NBLOCKS': 100, 'KLEV': 137, 'KFDIA': 1, 'KIDIA': 1, 'KLON': 1})]
    params2 = [ParametersProvider(program, update={'NBLOCKS': 200, 'KLEV': 137, 'KFDIA': 1, 'KIDIA': 1, 'KLON': 1})]
    profile_configs.append(ProfileConfig(program, params1, ['NBLOCKS'], ncu_repetitions=1, tot_time_repetitions=5))
    profile_configs.append(ProfileConfig(program, params2, ['NBLOCKS'], ncu_repetitions=0, tot_time_repetitions=5))
    experiment_desc = "test run"
    experiment_ids = []
    experiment_ids.append(profile(profile_configs, RunConfig(k_caching=False, change_stride=False), experiment_desc,
                          [('k_caching', "False"), ('change_strides', 'False')], ncu_report=False,
                          debug_mode=False))
    return experiment_ids


def do_classes(additional_desc: Optional[str] = None) -> List[int]:
    class1 = ['cloudsc_class1_2783', 'cloudsc_class1_2857', 'cloudsc_class1_658', 'cloudsc_class1_670']
    class2 = ['cloudsc_class2_1516', 'cloudsc_class2_1762', 'cloudsc_class2_781']
    class3 = ['cloudsc_class3_1985', 'cloudsc_class3_2120', 'cloudsc_class3_691', 'cloudsc_class3_965']
    profile_configs = []
    klev_value = 1000
    klon_start = 1000
    klon_end = 3000
    klon_step = 1000
    for program in class1:
        params = []
        for klon_value in np.arange(klon_start, klon_end+1, klon_step):
            params.append(ParametersProvider(program, update={'KLON': int(klon_value), 'KLEV': int(klev_value)}))
        profile_configs.append(ProfileConfig(program, params, ['KLON', 'KLEV'], tot_time_repetitions=10,
                               ncu_repetitions=0))

    for program in class2:
        params = []
        for klon_value in np.arange(klon_start, klon_end+1, klon_step):
            params.append(ParametersProvider(program, update={'KLON': int(klon_value), 'KLEV': int(klev_value)}))
        profile_configs.append(ProfileConfig(program, params, ['KLON', 'KLEV'], tot_time_repetitions=10,
                                             ncu_repetitions=2))
    for program in class3:
        params = []
        for klon_value in np.arange(klon_start, klon_end+1, klon_step):
            params.append(ParametersProvider(program, update={'KLON': int(klon_value), 'KLEV': int(klev_value)}))
        profile_configs.append(ProfileConfig(program, params, ['KLON', 'KLEV'], tot_time_repetitions=10,
                                             ncu_repetitions=2))

    experiment_ids = []
    experiment_ids.append(profile(profile_configs, RunConfig(k_caching=False, change_stride=False,
                                                             outside_loop_first=False, move_assignment_outside=False),
                                  "Class 1-3 Baseline",
                                  [('outside_first', 'False'), ('move_assignments_outside', 'False')],
                                  ncu_report=False))
    # Add ncu data for class 1 once outside first is active
    for config in profile_configs:
        if config.program in class1:
            config.ncu_repetitions = 2
    experiment_ids.append(profile(profile_configs, RunConfig(k_caching=False, change_stride=False,
                                                             outside_loop_first=True, move_assignment_outside=False),
                                  "Class 1-3 outside map first",
                                  [('outside_first', 'True'), ('move_assignments_outside', 'False')],
                                  ncu_report=False))
    experiment_ids.append(profile(profile_configs, RunConfig(k_caching=False, change_stride=False,
                                                             outside_loop_first=True, move_assignment_outside=True),
                                  "Class 1-3 both improvements",
                                  [('outside_first', 'True'), ('move_assignments_outside', 'True')], ncu_report=False))
    return experiment_ids


def do_k_caching(additional_desc: Optional[str] = None, nblock_min: Number = 1.0e5-2, nblock_max: Number = 4.5e5,
                 nblock_step: Number = 5e4, debug_mode: bool = False, nblock_split: Number = 1.5e5) -> List[int]:
    program = 'cloudsc_vert_loop_10'
    test_program(program, RunConfig(k_caching=True, change_stride=True))
    test_program(program, RunConfig(k_caching=True, change_stride=False))
    test_program(program, RunConfig(k_caching=False, change_stride=True))
    test_program(program, RunConfig(k_caching=False, change_stride=False))
    params_list_small = []
    params_list_big = []
    profile_configs = []
    for nblock in np.arange(nblock_split, nblock_min, -nblock_step):
        params = ParametersProvider(program, update={'NBLOCKS': int(nblock), 'KLEV': 137, 'KFDIA': 1, 'KIDIA': 1,
                                                     'KLON': 1})
        params_list_small.append(params)
    for nblock in np.arange(nblock_max, nblock_split, -nblock_step):
        params = ParametersProvider(program, update={'NBLOCKS': int(nblock), 'KLEV': 137, 'KFDIA': 1, 'KIDIA': 1,
                                                     'KLON': 1})
        params_list_big.append(params)
    profile_configs.append(ProfileConfig(program, params_list_small, ['NBLOCKS'], ncu_repetitions=2,
                           tot_time_repetitions=10,
                           ncu_kernels={'work': r'stateinner_loops_[\w_0-9]*', 'transpose': r'transpose_[\w_0-9]*'}))
    profile_configs.append(ProfileConfig(program, params_list_big, ['NBLOCKS'], ncu_repetitions=0,
                                         tot_time_repetitions=10))
    experiment_desc = "Vertical loop example"
    experiment_ids = []
    experiment_ids.append(profile(profile_configs, RunConfig(k_caching=False, change_stride=False),
                                  experiment_desc+" baseline",
                                  [('k_caching', "False"), ('change_strides', 'False')], ncu_report=True,
                                  debug_mode=debug_mode))
    experiment_ids.append(profile(profile_configs, RunConfig(k_caching=True, change_stride=False),
                                  experiment_desc+" k_caching",
                                  [('k_caching', "True"), ('change_strides', 'False')], ncu_report=True,
                                  debug_mode=debug_mode))
    experiment_ids.append(profile(profile_configs, RunConfig(k_caching=True, change_stride=True),
                          experiment_desc+" both",
                          [('k_caching', "True"), ('change_strides', 'True')], ncu_report=True,
                          debug_mode=debug_mode))
    experiment_ids.append(profile(profile_configs, RunConfig(k_caching=False, change_stride=True),
                                  experiment_desc+" change stride",
                                  [('k_caching', "False"), ('change_strides', 'True')], ncu_report=False,
                                  debug_mode=debug_mode))
    return experiment_ids


def do_vertical_loops(additional_desc: Optional[str] = None, nblock_min: Number = 1e5-2, nblock_max: Number = 7e5,
                      nblock_step: Number = 1e5, debug_mode: bool = False) -> List[int]:
    programs = [
            'cloudsc_vert_loop_4_ZSOLQA',
            'cloudsc_vert_loop_6_ZSOLQA',
            'cloudsc_vert_loop_6_1_ZSOLQA',
            'cloudsc_vert_loop_7_3'
            ]
    profile_configs = []
    for program in programs:
        test_program(program, RunConfig())
        params_list = []
        for nblock in np.arange(nblock_max, nblock_min, -nblock_step):
            params = ParametersProvider(program,
                                        update={'NBLOCKS': int(nblock), 'KLEV': 137, 'KFDIA': 1, 'KIDIA': 1, 'KLON': 1})
            params_list.append(params)
        profile_configs.append(ProfileConfig(program, params_list, ['NBLOCKS'], ncu_repetitions=3,
                                             tot_time_repetitions=3))

    experiment_ids = []
    experiment_desc = "Vertical loops with ZSOLQA"
    if additional_desc is not None:
        experiment_desc += f" with {additional_desc}"
    logger.info("run stack profile")
    experiment_ids.append(profile(profile_configs, RunConfig(), experiment_desc, [('temp allocation', 'stack')],
                          ncu_report=True, debug_mode=debug_mode))
    for profile_config in profile_configs:
        profile_config.set_heap_limit = True
        KLON, NCLV, KLEV = sympy.symbols("KLON NCLV KLEV")
        profile_config.heap_limit_expr = (KLON * (NCLV - 1)) + KLON * NCLV * (NCLV - 1) + KLON * (NCLV - 1) + \
            KLON * (KLEV - 1) + 4 * KLON
    logger.info("run heap profile")
    experiment_ids.append(profile(profile_configs, RunConfig(specialise_symbols=False), experiment_desc,
                          [('temp allocation', 'heap')], ncu_report=True, debug_mode=debug_mode))
    return experiment_ids


base_experiments = {
    'vert-loop': do_vertical_loops,
    'k_caching': do_k_caching,
    'classes': do_classes,
    'test': do_test
}


def profile(program_configs: List[ProfileConfig], run_config: RunConfig, experiment_description: str,
            additional_columns: List[Tuple[str, Union[Number, str]]] = [], ncu_report: bool = True,
            append_to_last_experiment: bool = False, debug_mode: bool = False) -> int:
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
    :param append_to_last_experiment: Set to true the experiment id should be the same as the last one. Use carefully.
    Should only be used when calling this function twice consecutively. Will not write to experiments.csv is set to
    True. defaults to False.
    :type append_to_last_experiment: bool
    :param debug_mode: Set to true if compile for debugging, defaults to False
    :type debug_mode: bool
    :return: Experiment Id
    :rtype: int
    """
    experiment_list_df = get_experiment_list_df()
    if 'experiment id' in experiment_list_df.reset_index().columns and len(experiment_list_df.index) > 0:
        if append_to_last_experiment:
            new_experiment_id = experiment_list_df.reset_index()['experiment id'].max()
        else:
            new_experiment_id = experiment_list_df.reset_index()['experiment id'].max() + 1
    else:
        new_experiment_id = 0
    logger.info(f"Profile for {experiment_description} using experiment id: {new_experiment_id}")

    if not append_to_last_experiment:
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
        experiment_list_df.to_csv(get_experiments_2_file())

    for program_config in program_configs:
        logger.info(f"Run {program_config.program} with experiment id {new_experiment_id}")
        experiment_folder = os.path.join(get_results_2_folder(), program_config.program, str(new_experiment_id))
        os.makedirs(experiment_folder, exist_ok=True)
        additional_columns_values = [col[1] for col in additional_columns]

        # Generate SDFG only once -> save runtime
        # sdfg_name = f"{program_config.program}_{new_experiment_id}_" + '_'.join(additional_columns_values) + ".sdfg"
        # sdfg_path = os.path.join(experiment_folder, sdfg_name)
        # print_with_time(f"[run2::profile] Generate SDFG and save it into {sdfg_path}")
        # sdfg = program_config.compile(program_config.sizes[0], run_config, specialise_changing_sizes=False)
        # sdfg.save(sdfg_path)
        additional_args = {}
        # additional_args['sdfg_path'] = sdfg_path
        additional_args['debug_mode'] = debug_mode
        if ncu_report:
            create_if_not_exist(os.path.join(get_results_2_folder(), "ncu_reports"))
            additional_args['ncu_report_path'] = os.path.join(get_results_2_folder(), "ncu_reports",
                                                              f"{program_config.program}_{new_experiment_id}_" +
                                                              '_'.join(additional_columns_values) +
                                                              ".ncu-rep")
        additional_args['sdfg_path'] = os.path.join(experiment_folder, 'graph.sdfg')
        df = program_config.profile(run_config, **additional_args)
        results_file = os.path.join(experiment_folder, 'results.csv')
        if os.path.exists(results_file):
            existing_df = read_data_from_result_file(program_config.program, new_experiment_id)\
                          .reset_index(level='experiment id').drop('experiment id', axis='columns')
            indices = df.index.names
            # bring indices into the same order, otherwise concat does not work as expected
            df = df.reset_index().set_index(indices)
            existing_df = existing_df.reset_index().set_index(indices)
            df = pd.concat([df, existing_df])
        df.to_csv(results_file)
        for index, params in enumerate(program_config.sizes):
            with open(os.path.join(experiment_folder, f"{index}_params.json"), 'w') as file:
                json.dump(params.get_dict(), file)
        return new_experiment_id


def action_profile(args):
    node = check_output(['uname', '-a']).decode('UTF-8').split(' ')[1].split('.')[0]
    logdir = get_results_2_logdir(node=node, profile_name=args.base_exp)
    create_if_not_exist(logdir)

    if args.logfile is None:
        logfile = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    else:
        logfile = args.logfile

    logfile_path = os.path.join(logdir, logfile)
    file_handlers = setup_logging(logfile_path, f"{logfile_path}.all")
    logger.info(f"Use logfile {logfile}")
    function_args = json.loads(args.args)
    experiment_ids = []
    try:
        experiment_ids = base_experiments[args.base_exp](**function_args)
        experiment_ids = [str(id) for id in experiment_ids]
    except:
        logger.error("An error occured while profiling")
    finally:
        if len(experiment_ids) > 0:
            new_logfile = os.path.join(logdir, '-'.join(experiment_ids)+'-date-'+logfile)
        else:
            new_logfile = os.path.join('FAILING-'.join(experiment_ids)+'-date-'+logfile)
        logger.info(f"Move logfile from {logfile} to {new_logfile}")
        close_filehandlers(file_handlers)
        shutil.move(logfile_path, new_logfile)
        shutil.move(f"{logfile_path}.all", f"{new_logfile}.all")


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
    profile_parser.add_argument('--logfile', type=str, default=None, help='Name of logfile')
    profile_parser.set_defaults(func=action_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
