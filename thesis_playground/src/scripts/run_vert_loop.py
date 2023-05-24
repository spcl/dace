from argparse import ArgumentParser
import os
import json

from utils.execute_dace import RunConfig, gen_ncu_report, profile_program, compile_for_profile
from utils.paths import get_vert_loops_dir
from utils.general import use_cache, disable_cache, insert_heap_size_limit, get_programs_data
from utils.print import print_with_time
from execute.data import ParametersProvider
from measurements.data import MeasurementRun
from scripts import Script

# sizes = [1e5, 2e5, 5e5]
vert_sizes = [5e5, 2e5, 1e5]
# vert_sizes = [5e5]
# vert_versions = ['cloudsc_vert_loop_6', 'cloudsc_vert_loop_5']
# vert_versions = ['cloudsc_vert_loop_4', 'cloudsc_vert_loop_5', 'cloudsc_vert_loop_6', 'cloudsc_vert_loop_6_1',
#                  'cloudsc_vert_loop_7']
vert_versions = ['cloudsc_vert_loop_4']
mwe_versions = ['cloudsc_vert_loop_orig_mwe_no_klon', 'cloudsc_vert_loop_mwe_no_klon']
mwe_sizes = [5e4]


class RunVertLoop(Script):
    name = "run-vert"
    description = "Run the vertical loops"

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('--use-dace-auto-opt', default=False, action='store_true',
                            help='Use DaCes auto_opt instead of mine')
        parser.add_argument('--repetitions', default=3, type=int)
        parser.add_argument('--mwe', action='store_true', default=False)
        parser.add_argument('--versions', nargs='+', default=None)
        parser.add_argument('--size', default=None)
        parser.add_argument('--microbenchmark', action='store_true', default=False)
        parser.add_argument('--only', choices=['specialise', 'heap', 'both'], default='both')
        # parser.add_argument('--no-cache', action='store_true', default=False)

    @staticmethod
    def action(args):

        run_config = RunConfig(use_dace_auto_opt=args.use_dace_auto_opt)

        if args.mwe:
            versions = mwe_versions
            sizes = mwe_sizes
        else:
            versions = vert_versions
            sizes = vert_sizes

        if args.versions is not None:
            versions = args.versions
        if args.size is not None:
            sizes = [int(args.size)]

        if args.microbenchmark:
            versions = ['microbenchmark_v1', 'microbenchmark_v3']
            sizes = [6553*32]

        for version in versions:

            version_dir_specialised = os.path.join(get_vert_loops_dir(), version, 'specialised')
            version_dir_heap = os.path.join(get_vert_loops_dir(), version, 'heap')
            os.makedirs(version_dir_heap, exist_ok=True)
            os.makedirs(version_dir_specialised, exist_ok=True)

            if args.only == 'specialise':
                specialise_arr = [True]
                dir_arr = [version_dir_specialised]
            elif args.only == 'heap':
                specialise_arr = [False]
                dir_arr = [version_dir_heap]
            else:
                specialise_arr = [False, True]
                dir_arr = [version_dir_heap, version_dir_specialised]

            for specialise, res_dir in zip(specialise_arr, dir_arr):
                if not specialise:
                    run_config.specialise_symbols = False
                    compile_for_profile(version, ParametersProvider(version), run_config)
                    programs = get_programs_data()['programs']
                    insert_heap_size_limit(f"{programs[version]}_routine",
                                           "(KLON * (NCLV - 1)) + KLON * NCLV * (NCLV - 1) + KLON * (NCLV - 1) +"
                                           "KLON * (KLEV - 1) + 4 * KLON")
                else:
                    run_config.specialise_symbols = True

                if not args.microbenchmark and not specialise:
                    use_cache(version)
                else:
                    disable_cache()

                for size in sizes:
                    description = f"With {size:.0E} NBLOCKS"
                    if specialise:
                        description += " and specialised symbols"

                    run_data = MeasurementRun(description)
                    params = ParametersProvider(version, update={'NBLOCKS': int(size), 'KLEV': 137, 'KFDIA': 1, 'KIDIA': 1,
                                                                 'KLON': 1})
                    print(f"Run {version} with KLEV: {params['KLEV']} NBLOCKS: {params['NBLOCKS']:,} KLON: {params['KLON']} "
                          f"KFDIA: {params['KFDIA']} KIDIA: {params['KIDIA']} and specialise: {specialise}")
                    program_data = profile_program(version, run_config, params, repetitions=1)
                    run_data.add_program_data(program_data)

                    filename = os.path.join(res_dir, f"result_{version}_{size:.0E}.json")
                    print_with_time(f"Save results into {filename}")
                    with open(filename, 'w') as file:
                        json.dump(run_data, file, default=MeasurementRun.to_json)

                    for index in range(args.repetitions):
                        report_filename = os.path.join(res_dir, f"report_{version}_{size:.0E}_{index}.ncu-rep")
                        program_args = ['--NBLOCKS', str(int(size))]
                        if not specialise:
                            program_args.append('--not-specialise-symbols')
                        gen_ncu_report(version, report_filename, run_config, ['--set', 'full'],
                                       program_args=program_args)
