from argparse import ArgumentParser
from subprocess import run
import copy
import os
from typing import Dict
from tabulate import tabulate
import dace
import cupy as cp
import pandas as pd

from python_programs import vert_loop_7, vert_loop_7_1, vert_loop_7_1_no_klon, vert_loop_7_1_no_temp, \
                            vert_loop_wip
from utils.general import optimize_sdfg, copy_to_device, use_cache, enable_debug_flags
from utils.ncu import get_all_actions_filtered, get_achieved_performance, get_peak_performance, get_achieved_bytes, \
                      get_runtime
from utils.paths import get_thesis_playground_root_dir, get_playground_results_dir
from utils.print import print_dataframe
from utils.python import gen_arguments, get_size_of_parameters


symbols = {'KLEV': 137, 'NCLV': 10, 'KLON': 1, 'KIDIA': 0, 'KFDIA': 1, 'NCLDQI': 2, 'NCLDQL': 3,
           'NCLDQS': 5, 'NCLDTOP': 0}
df_index_cols = ['program', 'NBLOCKS', 'description', 'specialised']


kernels = {
    'vert_loop_7': vert_loop_7,
    'vert_loop_7_1': vert_loop_7_1,
    'vert_loop_7_1_no_klon': vert_loop_7_1_no_klon,
    'vert_loop_7_1_no_temp': vert_loop_7_1_no_temp,
    'vert_loop_wip': vert_loop_wip,
}


def run_function_dace(f: dace.frontend.python.parser.DaceProgram, symbols: Dict[str, int], save_graphs: bool = False,
                      define_symbols: bool = False):
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


def action_run(args):
    print(f"Run {args.program} with NBLOCKS={args.NBLOCKS:,}", end="")
    if args.debug:
        print(" in DEBUG mode", end="")
        enable_debug_flags()
    if args.cache:
        use_cache(dacecache_folder=args.program)
    print()
    symbols.update({'NBLOCKS': args.NBLOCKS})
    run_function_dace(kernels[args.program], symbols, args.save_graphs, args.define_symbols)


def action_profile(args):
    data = []
    symbols.update({'NBLOCKS': args.NBLOCKS})

    # collect data
    for program in args.programs:
        report_path = '/tmp/profile.ncu-rep'
        if args.ncu_report:
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
        data.append([program, D, bw, upper_Q, T])

    # Print data
    print(tabulate(data, headers=['program', 'D', 'BW [ratio]', 'upper limit Q', 'T [s]'], intfmt=',', floatfmt='.3e'))

    # save data
    if args.export is not None:
        datafile = os.path.join(get_playground_results_dir(), 'python', args.export)
        if not args.export.endswith('.csv'):
            datafile = f"{datafile}.csv"
        print(f"Save data into {datafile}")
        for row in data:
            row.append(args.NBLOCKS)
            if args.description is None:
                print("WARNING: No description set. It is recommended to do so when saving the results")
                row.append('N.A.')
            else:
                row.append(args.description)
            row.append(args.define_symbols)

        # create dataframe
        this_df = pd.DataFrame(data, columns=['program', 'D', 'bw_ratio', 'upper_Q', 'T', 'NBLOCKS', 'description',
                                              'specialised'])
        this_df.set_index(df_index_cols, inplace=True)

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
        eval('dace_f.f(**arguments)', globals)

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

    columns = {
        'program': ('Program', None),
        'NBLOCKS': ('NBLOCKS', '.1e'),
        'specialised': ('Specialised Symbols', None),
        'D': ('measured bytes', '.3e'),
        'bw_ratio': ('bw eff', '.2f'),
        'T': ('T [s]', '.3e'),
    }

    print_dataframe(columns, joined_df.reset_index(), tablefmt='pipe')


# TODO: Add multiple runs
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
    profile_parser.add_argument('--description', default=None, type=str, help='Description of the specific run')
    profile_parser.add_argument('--not-define-symbols', dest='define_symbols', action='store_false', default=True)
    profile_parser.set_defaults(func=action_profile)

    test_parser = subparsers.add_parser('test', description='Test given kernels')
    test_parser.add_argument('programs', type=str, nargs='+')
    test_parser.add_argument('--cache', action='store_true', default=False)
    test_parser.set_defaults(func=action_test)

    print_parser = subparsers.add_parser('print', description='Print saved results')
    print_parser.add_argument('files', type=str, nargs='+', help=f"Files in {get_playground_results_dir()}/python")
    print_parser.set_defaults(func=action_print)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
