from argparse import ArgumentParser
import logging
from datetime import datetime
import os
import dace

from utils.generate_sdfg import optimise_basic_sdfg, get_basic_sdfg, get_path_of_basic_sdfg
from utils.general import remove_build_folder, enable_debug_flags, reset_graph_files, replace_symbols_by_values
from utils.log import setup_logging
from utils.paths import get_full_cloudsc_log_dir
from utils.full_cloudsc import add_synchronize, instrument_sdfg, compile_sdfg, get_sdfg, get_program_name, opt_levels
from execute.parameters import ParametersProvider

logger = logging.getLogger(__name__)


def action_compile(args):
    if args.sdfg_file is None:
        sdfg = get_sdfg(args.opt_level, args.device, args.version)
    else:
        sdfg_file = args.sdfg_file
        sdfg = dace.sdfg.sdfg.SDFG.from_file(sdfg_file)
    logger.info("Load SDFG from %s", sdfg_file)
    compile_sdfg(sdfg, args.NBLOCKS, args.version, args.opt_level, args.device, instrument=args.instrument,
                 debug=args.debug_mode, build_dir=args.build_dir)


def action_gen_graph(args):
    device_map = {'GPU': dace.DeviceType.GPU, 'CPU': dace.DeviceType.CPU}
    device = device_map[args.device]
    program = get_program_name(args.version)
    verbose_name = f"{program}_{opt_levels[args.opt_level]['name']}_{args.device.lower()}"
    logfile = os.path.join(
        get_full_cloudsc_log_dir(),
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{verbose_name}.log")
    setup_logging(level=args.log_level.upper(), logfile=logfile, full_logfile=f"{logfile}.all")
    logger.info("Use program: %s", program)
    reset_graph_files(verbose_name)

    params = ParametersProvider(program)
    # params = ParametersProvider(program, update={'NBLOCKS': 16384})
    run_config = opt_levels[args.opt_level]["run_config"]
    run_config.device = device
    logger.debug(run_config)
    sdfg = get_basic_sdfg(program, run_config, params, ['NBLOCKS'])
    # NCLDQR got forgotten in the basic sdfg
    replace_symbols_by_values(sdfg, {'NCLDQR': str(params['NCLDQR'])})
    sdfg = optimise_basic_sdfg(sdfg,
                               run_config,
                               params, ['NBLOCKS'],
                               verbose_name=verbose_name,
                               instrument=False,
                               storage_on_gpu=False)
    logger.info("Generated SDFG")
    sdfg_path = os.path.join(get_full_cloudsc_log_dir(), f"{verbose_name}.sdfg")
    logger.info("Save SDFG into %s", sdfg_path)
    sdfg.save(sdfg_path)


def action_change_ncldtop(args):
    setup_logging(level="DEBUG")
    program = 'cloudscexp4'
    run_config = opt_levels[args.opt_level]["run_config"]
    logger.debug(run_config)
    params = ParametersProvider(program, update={'NBLOCKS': 16384})
    basic_sdfg = get_basic_sdfg(program, run_config, params, ['NBLOCKS'])

    # Change value of NCLDTOP
    print(basic_sdfg.constants['NCLDTOP'])
    for nsdfg in basic_sdfg.sdfg_list:
        nsdfg.add_constant('NCLDTOP', 15)
    print(basic_sdfg.constants['NCLDTOP'])
    basic_sdfg.save(get_path_of_basic_sdfg(program, run_config, ['NBLOCKS']))


def action_profile(args):
    program = get_program_name(args.version)
    remove_build_folder(dacecache_folder=program.upper())
    verbose_name = f"{program}_{opt_levels[args.opt_level]['name']}"
    sdfg_file = os.path.join(get_full_cloudsc_log_dir(), f"{verbose_name}_{args.device.lower()}.sdfg")
    logger.info("Load SDFG from %s", sdfg_file)
    sdfg = dace.sdfg.sdfg.SDFG.from_file(sdfg_file)
    reports = sdfg.get_instrumentation_reports()
    # data = []
    for report in reports:
        print(report.durations)
        # data.append({'measurement': '})

    if args.clear:
        sdfg.clear_instrumentation_reports()


def main():
    parser = ArgumentParser(description="Generate SDFG or code of the full cloudsc code")
    parser.add_argument('--log-level', default='info')
    parser.add_argument('--log-file', default=None)
    subparsers = parser.add_subparsers(
            title="Commands",
            help="See the help of the respective command")

    gen_parser = subparsers.add_parser('gen', description="Generate SDFG")
    gen_parser.add_argument('opt_level')
    gen_parser.add_argument('--version', default=4, type=int)
    gen_parser.set_defaults(func=action_gen_graph)
    gen_parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')

    compile_parser = subparsers.add_parser('compile', description="Compile code from SDFG")
    compile_parser.add_argument('opt_level')
    compile_parser.add_argument('--version', default=4, type=int)
    compile_parser.add_argument('--debug-build', action='store_true', default=False)
    compile_parser.add_argument('--sdfg-file', default=None, help="Take non default SDFG file from here")
    compile_parser.add_argument('--build-dir', default=None, help="Folder to build & generate the DaCe code into")
    compile_parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    compile_parser.add_argument('--instrument', action='store_true', default=False, help='Instrument SDFG')
    compile_parser.add_argument('--NBLOCKS', type=int, help="Change NBLOCKS to new value", default=16384)
    compile_parser.set_defaults(func=action_compile)

    change_parser = subparsers.add_parser('change', description="Change NCLDTOP in basic SDFG")
    change_parser.add_argument('opt_level')
    change_parser.add_argument('--version', default=4, type=int)
    change_parser.set_defaults(func=action_change_ncldtop)

    profile_parser = subparsers.add_parser('profile', description="Read instrumentation reports")
    profile_parser.add_argument('opt_level')
    profile_parser.add_argument('--version', default=4, type=int)
    profile_parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    profile_parser.add_argument('--clear', action='store_true', default=False, help='Clear instrumentation reports')
    profile_parser.set_defaults(func=action_profile)

    args = parser.parse_args()
    add_args = {}
    if args.log_file is not None:
        add_args['full_logfile'] = args.log_file
    setup_logging(level=args.log_level.upper())
    args.func(args)


if __name__ == '__main__':
    main()
