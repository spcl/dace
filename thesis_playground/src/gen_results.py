from argparse import ArgumentParser
import os
import json
from subprocess import run
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from numbers import Number
from measurement_data import MeasurementRun, ProgramMeasurement
from flop_computation import FlopCount, read_roofline_data

from utils.plot import plot_roofline_seconds


RESULTS_DIR = 'complete_results'


class CompleteData:
    classical_data: Dict[str, MeasurementRun]
    classical_data_program: Dict[str, ProgramMeasurement]
    ncu_files: Dict[str, str]
    my_roofline_data: Dict[str, Tuple[FlopCount, Number]]

    def __init__(self, name: str):
        self.classical_data = {}
        self.ncu_files = {}
        self.roofline_files = {}
        programs_in_class = {}

        this_dir = os.path.join(RESULTS_DIR, name)
        raw_data_dir = os.path.join(this_dir, 'raw_data')
        with open(os.path.join(this_dir, "info.json")) as info_file:
            info_data = json.load(info_file)

            for class_number in info_data['classes']:
                results_file_path = os.join(raw_data_dir, f"class_{class_number}.json")
                roofline_file_path = os.path.join(raw_data_dir, f"class_{class_number}_roofline.json")
                if os.path.exists(results_file_path):
                    run_data = json.load(results_file_path, object_hook=MeasurementRun.from_json)
                    programs_in_class[class_number] = []
                    for program_measurement in run_data.data:
                        self.classical_data[program_measurement.program] = run_data
                        self.classical_data_program[program_measurement.program] = program_measurement
                        programs_in_class.append(program_measurement.program)

                if os.path.exits(roofline_file_path):
                    self.my_roofline_data.update(read_roofline_data(roofline_file_path))

            for class_number in info_data['classes']:
                for program in programs_in_class[class_number]:
                    ncu_file_path = os.join(raw_data_dir, f"report_class_{class_number}_{program}.ncu-rep")
                    if os.path.exists(ncu_file_path):
                        self.ncu_files[program] = ncu_file_path


# def plot_roofline(data: CompleteData):
#     plt.rcParams.update({'figure.figsize': (19, 10)})
#     plt.rcParams.update({'font.size': 12})
#     figure = plt.figure()
#     ax_seconds = figure.add_subplot(1, 1, 2)

#     hardware_filename = 'nodes.json'
#     with open(hardware_filename) as node_file:
#         node_data = json.load(node_file)
#         for run_data in data.classical_data:
#                     roofline_data = read_roofline_data(roofline_filename)
#                     gpu = node_data['ault_nodes'][run_data.node]['GPU']
#                     figure.suptitle(f"Roofline on {run_data.node} using a {gpu}")
#                     plot_roofline_seconds(run_data, roofline_data, node_data['GPUs'][gpu], ax_seconds,
#                                           points_only=index > 0)


def run_programs(info_data: Dict, raw_data_dir: str):
    """
    Runs all programs/classes defined by the info_data dict

    :param info_data: Dictionary with information about this set of results
    :type info_data: Dict
    :param raw_data_dir: Path to the directory containig all raw data
    :type raw_data_dir: str
    """
    run_path = os.path.join(os.path.dirname(__file__), 'run.py')
    for class_number in info_data['classes']:
        run(['python3', run_path, '--class', str(class_number), '--ncu-report', '--roofline',
             '--ncu-report-folder', raw_data_dir, '--results-folder', raw_data_dir,
             '--output', f"class_{class_number}.json",
             *info_data['additional_run_flags']])


def main():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--no-running', action='store_true', default=False, help="Don't rerun")
    args = parser.parse_args()

    this_dir = os.path.join(RESULTS_DIR, args.name)

    with open(os.path.join(this_dir, "info.json")) as info_file:
        info_data = json.load(info_file)
        raw_data_dir = os.path.join(this_dir, 'raw_data')
        if not os.path.exists(raw_data_dir):
            os.mkdir(raw_data_dir)

        run_programs(info_data, raw_data_dir)


if __name__ == '__main__':
    main()
