from argparse import ArgumentParser
import os
import json

from utils.execute_dace import RunConfig, test_program, gen_ncu_report, profile_program
from utils.general import use_cache
from utils.print import print_with_time
from data import ParametersProvider
from measurement_data import MeasurementRun

sizes = [1e5, 2e5, 5e5]
# versions = ['cloudsc_vert_loop_4', 'cloudsc_vert_loop_5']
versions = ['cloudsc_vert_loop_5']


def main():
    parser = ArgumentParser()
    parser.add_argument('--use-dace-auto-opt', default=False, action='store_true',
                        help='Use DaCes auto_opt instead of mine')
    parser.add_argument('--repetitions', default=3)
    args = parser.parse_args()

    results_folder = 'vert_loop_results'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    run_config = RunConfig(use_dace_auto_opt=args.use_dace_auto_opt)

    for version in versions:
        use_cache(version)
        if not test_program(version, run_config):
            continue

        run_data = MeasurementRun("With {size:.0E} NBLOCKS")
        for size in sizes:
            params = ParametersProvider(version, update={'NLOCKS': int(size)})
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


if __name__ == '__main__':
    main()
