from argparse import ArgumentParser
from subprocess import run
import copy
import os
from typing import Dict
import dace
import numpy as np
import cupy as cp
import pandas as pd

from python_programs import vert_loop_7, vert_loop_7_1, vert_loop_7_1_no_klon, vert_loop_7_1_no_temp, \
                            vert_loop_wip
from execute.parameters import ParametersProvider
from utils.general import optimize_sdfg, copy_to_device, use_cache, enable_debug_flags, get_programs_data, \
                          read_source, get_fortran, get_inputs, get_outputs
from utils.ncu import get_all_actions_filtered, get_achieved_performance, get_peak_performance, get_achieved_bytes, \
                      get_runtime
from utils.paths import get_thesis_playground_root_dir, get_playground_results_dir
from utils.print import print_dataframe
from utils.python import gen_arguments, get_size_of_parameters


# symbols = {'KLEV': 137, 'NCLV': 10, 'KLON': 1, 'KIDIA': 0, 'KFDIA': 1, 'NCLDQI': 2, 'NCLDQL': 3,
#            'NCLDQS': 5, 'NCLDTOP': 0}
symbols = ParametersProvider('cloudsc_vert_loop_7').get_dict()
df_index_cols = ['program', 'NBLOCKS', 'specialised', 'run_number']


kernels = {
    'vert_loop_7': vert_loop_7,
    'vert_loop_7_1': vert_loop_7_1,
    'vert_loop_7_1_no_klon': vert_loop_7_1_no_klon,
    'vert_loop_7_1_no_temp': vert_loop_7_1_no_temp,
    'vert_loop_wip': vert_loop_wip,
}


def run_function_dace(f: dace.frontend.python.parser.DaceProgram, symbols: Dict[str, int], save_graphs: bool = False,
                      define_symbols: bool = False) -> Dict:
    sdfg = f.to_sdfg(validate=True, simplify=True)
    additional_args = {}
    if save_graphs:
        additional_args['verbose_name'] = f"py_{f.name}"
    if define_symbols:
        additional_args['symbols'] = copy.deepcopy(symbols)
        # del additional_args['symbols']['NBLOCKS']

    optimize_sdfg(sdfg, device=dace.DeviceType.GPU, **additional_args)
    csdfg = sdfg.compile()
    arguments = gen_arguments(f, symbols)
    arguments_device = copy_to_device(arguments)
    csdfg(**arguments_device, **symbols)
    return arguments_device


def print_data(data: pd.DataFrame):
    """
    Print the data

    :param data: The data
    :type data: pd.DataFrame
    """
    columns = {
        'program': ('Program', None),
        'NBLOCKS': ('NBLOCKS', '.1e'),
        'specialised': ('Specialised Symbols', None),
        'D': ('measured bytes', '.3e'),
        'bw_ratio': ('bw eff', '.2f'),
        'T': ('T [s]', '.3e'),
        'number_runs': ('#', ','), }
    avg_data = data.groupby(['program', 'NBLOCKS', 'specialised']).mean()
    avg_data['number_runs'] = data.groupby(['program', 'NBLOCKS', 'specialised']).count()['D']
    print_dataframe(columns, avg_data.reset_index(), tablefmt='pipe')


def action_run(args):
    print(f"Run {args.program} with NBLOCKS={args.NBLOCKS:,}", end="")
    if args.debug:
        print(" in DEBUG mode", end="")
        enable_debug_flags()
    if args.cache:
        use_cache(dacecache_folder=args.program)
    symbols.update({'NBLOCKS': args.NBLOCKS})
    run_function_dace(kernels[args.program], symbols, args.save_graphs, args.define_symbols)


def action_profile(args):
    data = []
    symbols.update({'NBLOCKS': args.NBLOCKS})

    if args.export:
        datafile = os.path.join(get_playground_results_dir(), 'python', args.export)
        if not args.export.endswith('.csv'):
            datafile = f"{datafile}.csv"
    # Read any existing data -> increase run number from there onwards
    # But is only relevant if we want to save it later again
    existing_df = None
    if args.export and os.path.exists(datafile):
        df = pd.read_csv(datafile, index_col=df_index_cols)

    # collect data
    for program in args.programs:
        for run_number in range(args.repetitions):
            report_path = '/tmp/profile.ncu-rep'
            if args.ncu_report and run_number == 0:
                report_path = os.path.join(get_thesis_playground_root_dir(), 'ncu-reports',
                                           f"report_py_{program}_{args.NBLOCKS:.1e}.ncu-rep")
                print(f"Save ncu report into {report_path}")
            cmds = ['ncu', '--set', 'full', '-f', '--export', report_path, 'python3', __file__, 'run',
                    program, '--NBLOCKS', str(args.NBLOCKS)]
            if args.cache:
                cmds.append('--cache')
            if not args.define_symbols:
                cmds.append('--not-define-symbols')
            run(cmds)
            actions = get_all_actions_filtered(report_path, '_numpy_full_')
            action = actions[0]
            if len(actions) > 1:
                print(f"WARNING: More than one action, taking first {action}")
            upper_Q = get_size_of_parameters(kernels[program], symbols)
            D = get_achieved_bytes(action)
            bw = get_achieved_performance(action)[1] / get_peak_performance(action)[1]
            T = get_runtime(action)
            if existing_df is not None:
                existing_runs = existing_df.xs((program, args.NBLOCKS, args.define_symbols),
                                               level=('program', 'NBLOCKS', 'specialised'))
                max_run = existing_runs['run_number'].max()
            else:
                max_run = 0
            data.append([program, D, bw, upper_Q, T, run_number + max_run])

    for row in data:
        row.append(args.NBLOCKS)
        row.append(args.define_symbols)

    # create dataframe
    this_df = pd.DataFrame(data, columns=['program', 'D', 'bw_ratio', 'upper_Q', 'T', 'run_number', 'NBLOCKS',
                                          'specialised'])
    this_df.set_index(df_index_cols, inplace=True)

    print_data(this_df)

    # save data
    if args.export is not None:
        print(f"Save data into {datafile}")

        # create folder if neccessary
        if not os.path.exists(os.path.dirname(datafile)):
            os.makedirs(os.path.dirname(datafile))

        # read any existing data, append and write again
        if os.path.exists(datafile):
            df = pd.read_csv(datafile, index_col=df_index_cols)
            df = pd.concat([df, this_df])
        else:
            df = this_df
        df.to_csv(datafile)


def action_test(args):
    symbols.update({'NBLOCKS': 4, 'KLEV': 7})
    for program in args.programs:
        if args.cache:
            use_cache(dacecache_folder=program)
        dace_f = kernels[program]
        arguments = gen_arguments(dace_f, symbols)
        # Does not work because it can not convert the symbol NBLOCKS
        # vert_loop_symbol_wrapper(**symbols, func=dace_f.f, func_args=arguments)
        # dace_f.f(**arguments, **symbols)
        globals = {}
        globals['arguments'] = arguments
        globals['dace_f'] = dace_f
        for k, v in symbols.items():
            globals[k] = v
        # eval('dace_f.f(**arguments)', globals)

        arguments_device = copy_to_device(copy.deepcopy(arguments))
        sdfg = dace_f.to_sdfg(validate=True, simplify=True)
        optimize_sdfg(sdfg, device=dace.DeviceType.GPU)
        csdfg = sdfg.compile()
        csdfg(**arguments, **symbols)

        assert cp.allclose(cp.asarray(arguments), arguments_device)


def action_print(args):
    joined_df = pd.DataFrame()
    for file in args.files:
        if not file.endswith('.csv'):
            file = f"{file}.csv"
        path = os.path.join(get_playground_results_dir(), 'python', file)
        df = pd.read_csv(path, index_col=df_index_cols)
        joined_df = pd.concat([joined_df, df])

    print_data(joined_df)


def action_compare_fortran(args):
    program_py = 'vert_loop_7_1'
    program_f = 'cloudsc_vert_loop_7'
    params = ParametersProvider(program_f, testing=True)
    symbols = params.get_dict()
    print(symbols)
    if args.cache:
        use_cache(dacecache_folder=f"python_programs_{program_py}_{program_py}")

    # arguments_py_naive = gen_arguments(kernels[program_py], symbols)
    # vert_loop_7(**arguments_py_naive)
    # print(arguments_py_naive['PLUDE_NF'])
    # return 0
    outputs_py = run_function_dace(kernels[program_py], symbols, define_symbols=True)

    programs_data = get_programs_data()
    fsource = read_source(program_f)
    program_name = programs_data['programs'][program_f]
    routine_name = f'{program_name}_routine'
    ffunc = get_fortran(fsource, program_name, routine_name)
    arguments_f = gen_arguments(kernels[program_py], symbols)
    symbols_f = {}
    for param in programs_data['program_parameters'][program_f]:
        symbols_f[param] = symbols[param]
    arguments_f.update(symbols_f)
    for key in arguments_f:
        if isinstance(arguments_f[key], np.ndarray):
            arguments_f[key] = np.asfortranarray(arguments_f[key].transpose())
    ffunc(**{k.lower(): v for k, v in arguments_f.items()})

    for param in outputs_py:
        if isinstance(outputs_py[param], cp.ndarray):
            outputs_py[param] = np.asfortranarray(outputs_py[param].get().transpose())

    for param in outputs_py:
        if isinstance(outputs_py[param], np.ndarray):
            if not np.allclose(outputs_py[param], arguments_f[param]):
                print(f"ERROR: {param} is not the same")
                print("Python")
                print(outputs_py[param])
                print("Fortran")
                print(arguments_f[param])


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Commands",
        help="See the help of the respective command")

    run_parser = subparsers.add_parser('run', description='Run a given kernel')
    run_parser.add_argument('program', type=str)
    run_parser.add_argument('--NBLOCKS', type=int, default=int(1e5))
    run_parser.add_argument('--cache', action='store_true', default=False)
    run_parser.add_argument('--debug', action='store_true', default=False)
    run_parser.add_argument('--save-graphs', action='store_true', default=False)
    run_parser.add_argument('--not-define-symbols', dest='define_symbols', action='store_false', default=True)
    run_parser.set_defaults(func=action_run)

    profile_parser = subparsers.add_parser('profile', description='Profile given kernels')
    profile_parser.add_argument('programs', type=str, nargs='+')
    profile_parser.add_argument('--cache', action='store_true', default=False)
    profile_parser.add_argument('--NBLOCKS', type=int, default=int(1e5))
    profile_parser.add_argument('--ncu-report', action='store_true', default=False)
    profile_parser.add_argument('--export', default=None, type=str,
                                help=f"Save into the given filename in {get_playground_results_dir()}/python."
                                     f"Will append if File exists. Will store in csv format")
    profile_parser.add_argument('--not-define-symbols', dest='define_symbols', action='store_false', default=True)
    profile_parser.add_argument('--repetitions', type=int, default=1)
    profile_parser.set_defaults(func=action_profile)

    test_parser = subparsers.add_parser('test', description='Test given kernels')
    test_parser.add_argument('programs', type=str, nargs='+')
    test_parser.add_argument('--cache', action='store_true', default=False)
    test_parser.set_defaults(func=action_test)

    compare_fortran_parser = subparsers.add_parser('compare-fortran', description='Compare output to fortran version')
    compare_fortran_parser.add_argument('--cache', action='store_true', default=False)
    compare_fortran_parser.set_defaults(func=action_compare_fortran)

    print_parser = subparsers.add_parser('print', description='Print saved results')
    print_parser.add_argument('files', type=str, nargs='+', help=f"Files in {get_playground_results_dir()}/python")
    print_parser.set_defaults(func=action_print)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
