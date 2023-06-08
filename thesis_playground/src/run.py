from argparse import ArgumentParser
import os
import re
import json


from utils.paths import get_results_dir
from utils.general import get_programs_data, use_cache, remove_build_folder
from utils.print import print_with_time, print_results_v2, print_performance
from utils.execute_dace import RunConfig, test_program, profile_program, get_roofline_data, gen_ncu_report, \
                               gen_nsys_report
from utils.ncu import get_all_actions_matching_re, action_list_to_dict, get_runtime, get_cycles
from measurements.data import ProgramMeasurement, MeasurementRun
from measurements.flop_computation import save_roofline_data
from execute.parameters import ParametersProvider


def main():
    parser = ArgumentParser(
        description='Test and profiles the given programs. Results are saved into the results folder')
    parser.add_argument('-p',
                        '--programs',
                        type=str,
                        nargs='+',
                        help='Names of the programs to use. Can be several separated by space')
    parser.add_argument('-c',
                        '--class',
                        type=int,
                        choices=[1, 2, 3],
                        default=None,
                        dest='kernel_class',
                        help="Run all programs of a given class")
    parser.add_argument('--cache', default=False,
                        action='store_true',
                        help='Use the cache, does not regenerate and rebuild the code')
    parser.add_argument('-r', '--repetitions', type=int, default=5, help='Number of repetitions')
    parser.add_argument('--nsys', default=False, action='store_true', help='Also run nsys profile to generate a report')
    parser.add_argument('--no-ncu', default=False, action='store_true', help='Do not run ncu')
    parser.add_argument('-o', '--output', type=str, default=None, help='Also run nsys profile to generate a report')
    parser.add_argument('-d',
                        '--description',
                        type=str,
                        default='Unknown',
                        help='Description to be saved into the json file')
    parser.add_argument('--no-total', default=False, action='store_true',
                        help='Do not run measurement of total runtime')
    parser.add_argument('--ncu-repetitions', type=int, default=5,
                        help='Number of times ncu is run to measure kernel runtime')
    parser.add_argument('--skip-test', default=False, action='store_true',
                        help='Dont compare output to fortran output before profiling')
    parser.add_argument('--use-dace-auto-opt', default=False, action='store_true',
                        help='Use DaCes auto_opt instead of mine')
    parser.add_argument('--roofline', default=False, action='store_true',
                        help='Stores roofline data. Filename generated using value of --output flag')
    parser.add_argument('--pattern', choices=['const', 'formula', 'worst'], type=str, default=None,
                        help='Pattern for in and output')
    parser.add_argument('--ncu-report', default=False, action='store_true',
                        help='Create a full ncu report and save it')
    parser.add_argument('--ncu-report-folder', default='ncu-reports',
                        help='Folder where the ncu report is stored (default is "ncu-reports")')
    parser.add_argument('--results-folder', default='results',
                        help='Folder where the results json file are stored (default is "results")')

    args = parser.parse_args()
    test_program_path = os.path.join(os.path.dirname(__file__), 'run_program.py')
    run_config = RunConfig()
    run_config.set_from_args(args)

    programs = get_programs_data()['programs']
    selected_programs = [] if args.programs is None else args.programs
    if args.kernel_class is not None:
        selected_programs = [p for p in programs if p.startswith(f"cloudsc_class{args.kernel_class}")]

    if len(selected_programs) == 0:
        print("ERRROR: Need to specify programs either with --programs or --class")
        return 1

    run_data = MeasurementRun(args.description)

    run_data.properties['auto_opt'] = 'DaCe' if args.use_dace_auto_opt else 'My'
    roofline_data = {}

    for program in selected_programs:
        print_with_time(f"Run program {program}")
        if args.cache:
            if not use_cache(program):
                return 1
        else:
            remove_build_folder(program)

        if not args.skip_test:
            if not test_program(program, run_config):
                continue
        params = ParametersProvider(program)
        if not args.no_total:
            program_data = profile_program(program, run_config, params, repetitions=args.repetitions)
        else:
            program_data = ProgramMeasurement(program, params)
        command_program = ['python3', test_program_path, program, '--repetitions', '1']
        if args.use_dace_auto_opt:
            command_program.append('--use-dace-auto-opt')
        if args.pattern is not None:
            command_program.extend(['--pattern', args.pattern])

        if not args.no_ncu:
            print_with_time("Measure kernel runtime")
            for _ in range(args.ncu_repetitions):
                if gen_ncu_report(program, '/tmp/profile.ncu-rep', run_config):
                    regex_str = r"[a-z_0-9]*_([0-9]*_[0-9]*_[0-9]*)"
                    all_actions = action_list_to_dict(get_all_actions_matching_re('/tmp/profile.ncu-rep', regex_str))
                    for name, actions in all_actions.items():
                        if len(actions) > 1:
                            print(f"WARNING: Multiple actions found, taking only the frist: "
                                  f"{[a.name for a in actions]}")
                        action = actions[0]
                        match = re.match(regex_str, name)
                        if match is not None:
                            id_triplet = tuple([int(id) for id in match.group(1).split('_')])
                            time_measurement = program_data.get_measurement('Kernel Time', kernel=str(id_triplet))
                            cycles_measurement = program_data.get_measurement('Kernel Cycles', kernel=str(id_triplet))
                            if time_measurement is None:
                                program_data.add_measurement('Kernel Time', 'seconds', kernel_name=str(id_triplet))
                                time_measurement = program_data.get_measurement('Kernel Time', kernel=str(id_triplet))
                            if cycles_measurement is None:
                                program_data.add_measurement('Kernel Cycles', 'cycles', kernel_name=str(id_triplet))
                                cycles_measurement = program_data.get_measurement('Kernel Cycles',
                                                                                  kernel=str(id_triplet))
                            time_measurement.add_value(get_runtime(action))
                            cycles_measurement.add_value(get_cycles(action))

        if args.nsys:
            gen_nsys_report(program, f"report_{program}.nsys-rep", run_config)

        if args.ncu_report:
            filename = f"report_{program}.ncu-rep"
            if args.output is not None:
                filename = f"report_{''.join(args.output.split('.')[:-1])}_{program}.ncu-rep"
            filename = os.path.join(get_results_dir(args.ncu_report_folder), filename)
            gen_ncu_report(program, filename, run_config, ['--set', 'full'])

        run_data.add_program_data(program_data)

        if args.roofline:
            print_with_time("Compute roofline data")
            roofline_data[program] = get_roofline_data(program, params, pattern=args.pattern)

    if args.output is not None:
        filename = os.path.join(get_results_dir(args.results_folder), args.output)
        if not (args.no_total and args.no_ncu):
            print_with_time(f"Save results into {filename}")
            with open(filename, 'w') as file:
                json.dump(run_data, file, default=MeasurementRun.to_json)
        if args.roofline:
            roofline_filename = f"{''.join(filename.split('.')[:-1])}_roofline.json"
            print(f"Save roofline data into {roofline_filename}")
            save_roofline_data(roofline_data, roofline_filename)

    print_with_time("Print results")
    print_results_v2(run_data)
    if args.roofline:
        with open('nodes.json') as node_file:
            node_data = json.load(node_file)
            gpu = node_data['ault_nodes'][run_data.node]['GPU']
            print_performance(roofline_data, run_data, node_data['GPUs'][gpu])


if __name__ == '__main__':
    main()
