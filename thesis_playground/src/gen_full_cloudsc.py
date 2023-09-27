from argparse import ArgumentParser
import logging
from datetime import datetime
import os
import dace
from subprocess import run
from dace.sdfg import SDFG
from dace import nodes

from utils.generate_sdfg import optimise_basic_sdfg, get_basic_sdfg, get_path_of_basic_sdfg
from utils.general import remove_build_folder, enable_debug_flags, reset_graph_files, replace_symbols_by_values
from utils.log import setup_logging
from utils.run_config import RunConfig
from utils.paths import get_full_cloudsc_log_dir, get_dacecache
from execute.parameters import ParametersProvider

logger = logging.getLogger(__name__)

opt_levels = {
    "baseline": {
        "run_config": RunConfig(k_caching=False, change_stride=False, outside_loop_first=False,
                                move_assignment_outside=False, full_cloudsc_fixes=True),
        "name": "baseline"
        },
    "k-caching": {
        "run_config": RunConfig(k_caching=True, change_stride=False, outside_loop_first=False,
                                move_assignment_outside=False, full_cloudsc_fixes=True),
        "name": "k_caching"
        },
    "change-strides": {
        "run_config": RunConfig(k_caching=False, change_stride=True, outside_loop_first=False,
                                move_assignment_outside=False, full_cloudsc_fixes=True),
        "name": "change_strides"
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


def add_synchronize(dacecache_folder: str):
    src_dir = os.path.join(get_dacecache(), dacecache_folder, 'src', 'cpu')
    src_file = os.path.join(src_dir, os.listdir(src_dir)[0])
    if len(os.listdir(src_dir)) > 1:
        logger.warning(f"More than one files in {src_dir}")
    lines = run(['grep', '-rn', 'std::chrono::high_resolution_clock::now', src_file], capture_output=True).stdout.decode('UTF-8')
    logger.debug("Add synchronizes to %s", src_file)
    line_numbers_to_insert = []

    for line in lines.split('\n'):
        if len(line) > 0:
            if line.split()[2].startswith('__dace_t'):
                line_numbers_to_insert.append(int(line.split(':')[0])-1)

    logger.debug("Insert synchronize into line numbers: %s", line_numbers_to_insert)
    with open(src_file, 'r') as f:
        contents = f.readlines()
        for offset, line_number in enumerate(line_numbers_to_insert):
            contents.insert(line_number+offset,
                            "        DACE_GPU_CHECK(cudaDeviceynchronize());\n")

    with open(src_file, 'w') as f:
        contents = "".join(contents)
        f.write(contents)


def get_program_name(args) -> str:
    if args.version == 3:
        program = 'cloudscexp3'
    elif args.version == 4:
        program = 'cloudscexp4'
    else:
        program = 'cloudscexp2'
    return program


def instrument_sdfg(sdfg: SDFG, opt_level: str, device: str):
    if opt_level in ['k-caching', 'baseline']:
        if device == 'CPU':
            cloudsc_state = sdfg.find_state('stateCLOUDSC')
        elif device == 'GPU':
            cloudsc_state = sdfg.find_state('CLOUDSCOUTER4_copyin')
        nblocks_maps = [n for n in cloudsc_state.nodes() if isinstance(n, nodes.MapEntry) and n.label ==
                        'stateCLOUDSC_map']
    if opt_level in ['all', 'change-strides']:
        changed_strides_state = sdfg.find_state('with_changed_strides')
        nsdfg_changed_strides = [n for n in changed_strides_state.nodes() if isinstance(n, nodes.NestedSDFG)][0]
        nblocks_maps = [n for n in nsdfg_changed_strides.sdfg.find_state('CLOUDSCOUTER4_copyin').nodes()
                        if isinstance(n, nodes.MapEntry) and n.label == 'stateCLOUDSC_map']
        sdfg.find_state('transform_data').instrument = dace.InstrumentationType.Timer
        sdfg.find_state('transform_data_back').instrument = dace.InstrumentationType.Timer

    if len(nblocks_maps):
        logger.warning("There is more than one map to instrument: %s", nblocks_maps)
    nblocks_maps[0].instrument = dace.InstrumentationType.Timer


def action_compile(args):
    program = get_program_name(args)
    remove_build_folder(dacecache_folder=program.upper())
    if args.sdfg_file is None:
        verbose_name = f"{program}_{opt_levels[args.opt_level]['name']}"
        sdfg_file = os.path.join(get_full_cloudsc_log_dir(), f"{verbose_name}_{args.device.lower()}.sdfg")
    else:
        sdfg_file = args.sdfg_file
    logger.info("Load SDFG from %s", sdfg_file)
    sdfg = dace.sdfg.sdfg.SDFG.from_file(sdfg_file)
    if args.debug_build:
        logger.info("Enable Debug Flags")
        enable_debug_flags()
    logger.info("Build into %s", sdfg.build_folder)
    if args.build_dir is not None:
        sdfg.build_folder = args.build_dir
    else:
        sdfg.build_folder = os.path.abspath(sdfg.build_folder)
    if args.instrument:
        instrument_sdfg(sdfg, args.opt_level, args.device)
    sdfg.compile()
    if args.instrument:
        add_synchronize(f"CLOUDSCOUTER{args.version}")
    signature_file = os.path.join(get_full_cloudsc_log_dir(), f"signature_dace_{program}.txt")
    logger.info("Write signature file into %s", signature_file)
    with open(signature_file, 'w') as file:
        file.write(sdfg.signature())


def action_gen_graph(args):
    device_map = {'GPU': dace.DeviceType.GPU, 'CPU': dace.DeviceType.CPU}
    device = device_map[args.device]
    program = get_program_name(args)
    verbose_name = f"{program}_{opt_levels[args.opt_level]['name']}_{args.device.lower()}"
    logfile = os.path.join(
        get_full_cloudsc_log_dir(),
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{verbose_name}.log")
    setup_logging(level=args.log_level.upper(), logfile=logfile, full_logfile=f"{logfile}.all")
    logger.info("Use program: %s", program)
    reset_graph_files(verbose_name)

    params = ParametersProvider(program, update={'NBLOCKS': 16384})
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
    program = get_program_name(args)
    remove_build_folder(dacecache_folder=program.upper())
    verbose_name = f"{program}_{opt_levels[args.opt_level]['name']}"
    sdfg_file = os.path.join(get_full_cloudsc_log_dir(), f"{verbose_name}_{args.device.lower()}.sdfg")
    logger.info("Load SDFG from %s", sdfg_file)
    sdfg = dace.sdfg.sdfg.SDFG.from_file(sdfg_file)
    reports = sdfg.get_instrumentation_reports()
    # data = []
    for report in reports:
        print(report)
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
