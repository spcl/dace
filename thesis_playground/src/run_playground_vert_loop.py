from argparse import ArgumentParser
from subprocess import run
import copy
import os
from typing import Dict
import dace
import numpy as np
import cupy as cp
import pandas as pd
import seaborn as sns

from python_programs import vert_loop_7, vert_loop_7_1, vert_loop_7_1_no_klon, vert_loop_7_1_no_temp, \
                            vert_loop_wip, vert_loop_7_2, vert_loop_7_3, vert_loop_wip_scalar_offset, small_wip
from execute.parameters import ParametersProvider
from utils.general import optimize_sdfg, copy_to_device, use_cache, enable_debug_flags, get_programs_data, \
                          read_source, get_fortran, remove_build_folder, reset_graph_files, print_compare_matrix
from utils.ncu import get_all_actions_filtered, get_achieved_performance, get_peak_performance, get_achieved_bytes, \
                      get_runtime
from utils.paths import get_thesis_playground_root_dir, get_playground_results_dir
from utils.print import print_dataframe
from utils.python import gen_arguments, get_size_of_parameters, get_dacecache_folder, df_index_cols, \
                         description_df_path, get_joined_df, convert_to_plain_python
from utils.plot import get_new_figure, save_plot


# symbols = {'KLEV': 137, 'NCLV': 10, 'KLON': 1, 'KIDIA': 0, 'KFDIA': 1, 'NCLDQI': 2, 'NCLDQL': 3,
#            'NCLDQS': 5, 'NCLDTOP': 0}
symbols = ParametersProvider('cloudsc_vert_loop_7').get_dict()
# Symbols to decrement by one for python version due to 1 indexing for fortran
decrement_symbols = ['NCLDQI', 'NCLDQL', 'NCLDQR', 'NCLDQS']

kernels = {
    'vert_loop_7': vert_loop_7,
    'vert_loop_7_1': vert_loop_7_1,
    'vert_loop_7_2': vert_loop_7_2,
    'vert_loop_7_3': vert_loop_7_3,
    'vert_loop_7_1_no_klon': vert_loop_7_1_no_klon,
    'vert_loop_7_1_no_temp': vert_loop_7_1_no_temp,
    'vert_loop_wip': vert_loop_wip,
    'vert_loop_wip_scalar_offset': vert_loop_wip_scalar_offset,
    'small_wip': small_wip,
}


def generate_sdfg(f: dace.frontend.python.parser.DaceProgram, symbols: Dict[str, int], save_graphs: bool = False,
                  define_symbols: bool = False, use_dace_auto_opt: bool = False) -> dace.SDFG:
    """
    Generates the SDFG for the given dace program

    :param f: The dace program for which to generate the SDFG
    :type f: dace.frontend.python.parser.DaceProgram
    :param symbols: The symbols used to specialise the SDFG
    :type symbols: Dict[str, int]
    :param save_graphs: If intermediate SDFGs should be stored, defaults to False
    :type save_graphs: bool, optional
    :param define_symbols: If symbols should be specialised, defaults to False
    :type define_symbols: bool, optional
    :param use_dace_auto_opt: Set to true if DaCe default auto opt should be used, defaults to False
    :type use_dace_auto_opt: bool
    :return: The generated SDFG
    :rtype: dace.SDFG
    """
    print(symbols)
    sdfg = f.to_sdfg(validate=True, simplify=True)
    additional_args = {}
    if save_graphs:
        program_name = f"py_{f.name}"
        reset_graph_files(program_name)
        additional_args['verbose_name'] = program_name
    if define_symbols:
        print(f"[run_function_dace] symbols: {symbols}")
        additional_args['symbols'] = copy.deepcopy(symbols)
        for symbol in decrement_symbols:
            additional_args['symbols'][symbol] -= 1
    if use_dace_auto_opt:
        additional_args['use_my_auto_opt'] = False

    optimize_sdfg(sdfg, device=dace.DeviceType.GPU, **additional_args)
    return sdfg


def run_function_dace(f: dace.frontend.python.parser.DaceProgram, symbols: Dict[str, int], save_graphs: bool = False,
                      define_symbols: bool = False) -> Dict:

    csdfg = generate_sdfg(f, symbols, save_graphs, define_symbols).compile()
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
    description_df = pd.read_csv(description_df_path, index_col=['experiment_id'])
    columns = {
        'program': ('Program', None),
        'description': ('Description', None),
        'NBLOCKS': ('NBLOCKS', '.1e'),
        'specialised': ('Specialised Symbols', None),
        'D': ('measured bytes', '.3e'),
        'bw_ratio': ('bw eff', '.2f'),
        'T': ('T [s]', '.3e'),
        'number_runs': ('#', ','),
        }
    avg_data = data.groupby(['program', 'NBLOCKS', 'specialised', 'experiment_id']).mean()
    avg_data['number_runs'] = data.groupby(['program', 'NBLOCKS', 'specialised']).count()['D']
    print_dataframe(columns, avg_data.join(description_df, on='experiment_id').reset_index(), tablefmt='pipe')


def action_run(args):
    print(f"Run {args.program} with NBLOCKS={args.NBLOCKS:,}")
    if args.debug:
        print("Enable debug mode")
        enable_debug_flags()
    if args.cache:
        use_cache(dacecache_folder=get_dacecache_folder(args.program))
    else:
        remove_build_folder(dacecache_folder=get_dacecache_folder(args.program))
    symbols.update({'NBLOCKS': args.NBLOCKS})
    run_function_dace(kernels[args.program], symbols, args.save_graphs, args.define_symbols)


def action_profile(args):
    data = []
    sizes = [args.NBLOCKS]
    if args.size_range:
        sizes = [int(5e5), int(4e5), int(3e5), int(2e5), int(1e5)]
        # sizes = np.arange(int(5e5), int(1e6), int(1e5))

    if args.export:
        datafile = os.path.join(get_playground_results_dir(), 'python', args.export)
        if not args.export.endswith('.csv'):
            datafile = f"{datafile}.csv"
    existing_df = None
    experiment_id = 0
    if args.export and os.path.exists(datafile):
        existing_df = pd.read_csv(datafile, index_col=df_index_cols)
        experiment_id = int(existing_df.reset_index()['experiment_id'].max()) + 1

    for size in sizes:
        symbols.update({'NBLOCKS': size})

        # collect data
        for program in args.programs:
            for run_number in range(args.repetitions):
                report_path = '/tmp/profile.ncu-rep'
                if args.ncu_report and run_number == 0:
                    report_path = os.path.join(get_thesis_playground_root_dir(), 'ncu-reports',
                                               f"report_py_{program}_{size:.1e}.ncu-rep")
                    print(f"Save ncu report into {report_path}")
                cmds = ['ncu', '--set', 'full', '-f', '--export', report_path, 'python3', __file__, 'run',
                        program, '--NBLOCKS', str(size)]
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
                data.append([program, D, bw, upper_Q, T, run_number, size])

    for row in data:
        row.append(args.define_symbols)
        row.append(experiment_id)

    # create dataframe
    this_df = pd.DataFrame(data, columns=['program', 'D', 'bw_ratio', 'upper_Q', 'T', 'run_number', 'NBLOCKS',
                                          'specialised', 'experiment_id'])
    this_df.set_index(df_index_cols, inplace=True)

    # save data
    if args.export is not None:
        print(f"Save data into {datafile}")

        # create folder if neccessary
        if not os.path.exists(os.path.dirname(datafile)):
            os.makedirs(os.path.dirname(datafile))

        # read any existing data, append and write again
        if os.path.exists(datafile):
            df = pd.concat([existing_df, this_df])
        else:
            df = this_df
        df.to_csv(datafile)

        if os.path.exists(description_df_path):
            description_df = pd.read_csv(description_df_path, index_col=['experiment_id'])
        else:
            description_df = pd.DataFrame(columns=['experiment_id', 'description']).set_index('experiment_id')
        description = args.description
        if description is None:
            description = 'N.A.'
        description_df.loc[experiment_id] = description
        description_df.to_csv(description_df_path)

    print_data(this_df)


def action_test(args):
    symbols.update({'NBLOCKS': 4, 'KLEV': 7})
    for program in args.programs:
        if args.cache:
            use_cache(dacecache_folder=get_dacecache_folder(args.program))
        else:
            remove_build_folder(dacecache_folder=get_dacecache_folder(program))
        dace_f = kernels[program]
        arguments = gen_arguments(dace_f, symbols)

        py_code = convert_to_plain_python(dace_f, symbols)
        py_code += f"{dace_f.f.__name__}({', '.join([k for k in arguments])})"

        arguments_device = copy_to_device(copy.deepcopy(arguments))
        exec(py_code, arguments)
        arguments = {k: arguments[k] for k in arguments_device}

        sdfg = dace_f.to_sdfg(validate=True, simplify=True)
        optimize_sdfg(sdfg, device=dace.DeviceType.GPU, use_my_auto_opt=not args.use_dace_auto_opt)
        csdfg = sdfg.compile()
        csdfg(**arguments_device, **symbols)

        arguments_py = copy_to_device(arguments)
        for key in arguments_py:
            if not cp.allclose(arguments_py[key], arguments_device[key]):
                print(f"{key} are not the same")
                print("Python")
                print(arguments_py[key])
                print("DaCe")
                print(arguments_device[key])


def action_print(args):
    print_data(get_joined_df(args.files))


def action_plot(args):
    joined_df = get_joined_df(args.files)
    plot_dir = os.path.join(get_playground_results_dir(), 'plots')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    fig = get_new_figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    sns.lineplot(joined_df, x='NBLOCKS', y='T', hue='program', ax=ax1)
    sns.lineplot(joined_df, x='NBLOCKS', y='D', hue='program', ax=ax2)
    save_plot(os.path.join(plot_dir, "py_results.pdf"))


def action_compare_fortran(args):
    # Programs to test, first py kernel, then fortran program name
    programs = [
            ('vert_loop_7_1', 'cloudsc_vert_loop_7'),
            ('vert_loop_7_2', 'cloudsc_vert_loop_7_2'),
            ('vert_loop_7_3', 'cloudsc_vert_loop_7_3')
                ]

    for program_py, program_f in programs:
        print(f"Compare python {program_py} to fortran {program_f}")
        params = ParametersProvider(program_f, testing=True)
        symbols = params.get_dict()
        if args.cache:
            use_cache(dacecache_folder=get_dacecache_folder(program_py))
        else:
            remove_build_folder(dacecache_folder=get_dacecache_folder(program_py))

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

        np.set_printoptions(precision=2)
        for param in outputs_py:
            if isinstance(outputs_py[param], np.ndarray):
                if not np.allclose(outputs_py[param], arguments_f[param]):
                    print(f"ERROR: {param} is not the same")
                    shape = outputs_py[param].shape
                    print_compare_matrix(outputs_py[param], arguments_f[param], [slice(end) for end in shape])


def action_gen_graph(args):
    symbols.update({'NBLOCKS': 100000})
    sdfg = generate_sdfg(kernels[args.program], symbols, True, True, args.use_dace_auto_opt)
    if args.gen_code:
        sdfg.compile()
        print(f"Generated code is at: {sdfg.build_folder}")


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
    profile_parser.add_argument('--description', type=str, default=None)
    profile_parser.add_argument('programs', type=str, nargs='+')
    profile_parser.add_argument('--cache', action='store_true', default=False)
    profile_parser.add_argument('--NBLOCKS', type=int, default=int(1e5))
    profile_parser.add_argument('--size-range', action='store_true', help="Run for NBLOCKS=100K...500K.")
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
    test_parser.add_argument('--use-dace-auto-opt', action='store_true', help='Use default DaCe auto opt')
    test_parser.set_defaults(func=action_test)

    compare_fortran_parser = subparsers.add_parser('compare-fortran', description='Compare output to fortran version')
    compare_fortran_parser.add_argument('--cache', action='store_true', default=False)
    compare_fortran_parser.set_defaults(func=action_compare_fortran)

    print_parser = subparsers.add_parser('print', description='Print saved results')
    print_parser.add_argument('files', type=str, nargs='+', help=f"Files in {get_playground_results_dir()}/python")
    print_parser.set_defaults(func=action_print)

    plot_parser = subparsers.add_parser('plot')
    plot_parser.add_argument('files', type=str, nargs='+', help=f"Files in {get_playground_results_dir()}/python")
    plot_parser.set_defaults(func=action_plot)

    gen_graph_parser = subparsers.add_parser('gen-graphs')
    gen_graph_parser.add_argument('program', type=str)
    gen_graph_parser.add_argument('--gen-code', action='store_true', help='Generate CUDA code')
    gen_graph_parser.add_argument('--use-dace-auto-opt', action='store_true', help='Use default DaCe auto opt')
    gen_graph_parser.set_defaults(func=action_gen_graph)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
