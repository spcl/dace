from argparse import ArgumentParser
import os
import json

from utils.execute_dace import RunConfig, test_program, gen_ncu_report, profile_program
from utils.general import use_cache
from utils.print import print_with_time
from execute.data import ParametersProvider
from measurements.data import MeasurementRun
from scripts import Script

# sizes = [1e5, 2e5, 5e5]
vert_sizes = [5e5, 2e5, 1e5]
# sizes = [5e5]
# versions = ['cloudsc_vert_loop_4', 'cloudsc_vert_loop_5', 'cloudsc_vert_loop_6']
vert_versions = ['cloudsc_vert_loop_5', 'cloudsc_vert_loop_6']

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

    @staticmethod
    def action(args):

        results_folder = 'vert_loop_results'
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)
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

        for version in versions:
            use_cache(version)
            # if not test_program(version, run_config):
            #     continue

            for size in sizes:
                run_data = MeasurementRun(f"With {size:.0E} NBLOCKS")
                params = ParametersProvider(version, update={'NBLOCKS': int(size), 'KLEV': 137, 'KFDIA': 1, 'KIDIA': 1,
                    'KLON': 1})
                print(f"Run {version} with KLEV: {params['KLEV']} NBLOCKS: {params['NBLOCKS']:,} KLON: {params['KLON0']} "
                      f"KFDIA: {params['KFDIA']} KIDIA: {params['KIDIA']}")
                program_data = profile_program(version, run_config, params, repetitions=1)
                run_data.add_program_data(program_data)

                filename = os.path.join(results_folder, f"result_{version}_{size:.0E}.json")
                print_with_time(f"Save results into {filename}")
                with open(filename, 'w') as file:
                    json.dump(run_data, file, default=MeasurementRun.to_json)

                for index in range(args.repetitions):
                    report_filename = os.path.join(results_folder, f"report_{version}_{size:.0E}_{index}.ncu-rep")
                    gen_ncu_report(version, report_filename, run_config, ['--set', 'full'],
                                   program_args=['--NBLOCKS', str(int(size))])
