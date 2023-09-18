from argparse import ArgumentParser
import logging
from datetime import datetime
import os
import dace

from utils.generate_sdfg import optimise_basic_sdfg, get_basic_sdfg
from utils.general import remove_build_folder, enable_debug_flags, reset_graph_files, replace_symbols_by_values
from utils.log import setup_logging
from utils.run_config import RunConfig
from utils.paths import get_full_cloudsc_log_dir, get_verbose_graphs_dir
from execute.parameters import ParametersProvider
from print_memlets import find_all_strange_memlets

logger = logging.getLogger(__name__)

opt_levels = {
    "baseline": {
        "run_config": RunConfig(k_caching=False, change_stride=False, outside_loop_first=False,
                                move_assignment_outside=False, full_cloudsc_fixes=True),
        "name": "baseline"
        },
    "all": {
        "run_config": RunConfig(k_caching=True, change_stride=True, outside_loop_first=True, full_cloudsc_fixes=True),
        "name": "all_opt"
        },
    "all-custom": {
        "run_config": RunConfig(k_caching=True, change_stride=True, outside_loop_first=True, full_cloudsc_fixes=True),
        "name": "all_opt_custom"
        }
}


def get_program_name(args) -> str:
    if args.version == 3:
        program = 'cloudscexp3'
    elif args.version == 4:
        program = 'cloudscexp4'
    else:
        program = 'cloudscexp2'
    return program


def action_compile(args):
    program = get_program_name(args)
    remove_build_folder(program)
    verbose_name = f"{program}_{opt_levels[args.opt_level]['name']}"
    sdfg_file = os.path.join(get_full_cloudsc_log_dir(), f"{verbose_name}.sdfg")
    logger.info("Load SDFG from %s", sdfg_file)
    sdfg = dace.sdfg.sdfg.SDFG.from_file(sdfg_file)
    if args.debug_build:
        logger.info("Enable Debug Flags")
        enable_debug_flags()
    logger.info("Build into %s", sdfg.build_folder)
    sdfg.compile()
    signature_file = os.path.join(get_full_cloudsc_log_dir(), f"signature_dace_{program}.txt")
    logger.info("Write signature file into %s", signature_file)
    with open(signature_file, 'w') as file:
        file.write(sdfg.signature())


def action_gen_graph(args):
    program = get_program_name(args)
    verbose_name = f"{program}_{opt_levels[args.opt_level]['name']}"
    logfile = os.path.join(
        get_full_cloudsc_log_dir(),
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{verbose_name}.log")
    setup_logging(level=args.log_level.upper(), logfile=logfile, full_logfile=f"{logfile}.all")
    logger.info("Use program: %s", program)
    reset_graph_files(verbose_name)

    params = ParametersProvider(program, update={'NBLOCKS': 16384, 'NCLDTOP': 25})
    run_config = opt_levels[args.opt_level]["run_config"]
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

    compile_parser = subparsers.add_parser('compile', description="Compile code from SDFG")
    compile_parser.add_argument('opt_level')
    compile_parser.add_argument('--version', default=4, type=int)
    compile_parser.add_argument('--debug-build', action='store_true', default=False)
    compile_parser.set_defaults(func=action_compile)

    args = parser.parse_args()
    add_args = {}
    if args.log_file is not None:
        add_args['full_logfile'] = args.log_file
    setup_logging(level=args.log_level.upper())
    args.func(args)


def load_basic_sdfg():
    setup_logging(level="DEBUG")
    program = 'cloudscexp4'
    run_config = RunConfig(k_caching=True, change_stride=True, outside_loop_first=False)
    params = ParametersProvider(program, update={'NBLOCKS': 16384})
    basic_sdfg = get_basic_sdfg(program, run_config, params, ['NBLOCKS'])

if __name__ == '__main__':
    main()
    # load_basic_sdfg()
