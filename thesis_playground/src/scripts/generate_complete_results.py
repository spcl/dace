from argparse import ArgumentParser
import os
import json
from subprocess import run
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from numbers import Number

from measurements.data import MeasurementRun, ProgramMeasurement
from measurements.flop_computation import FlopCount, read_roofline_data
from utils.paths import get_complete_results_dir
from utils.plot import update_min_max, draw_program_points, draw_roofline, draw_ncu_points
from utils.ncu import get_all_actions_filtered
from scripts import Script


class CompleteData:
    classical_data: Dict[str, MeasurementRun]
    classical_data_program: Dict[str, ProgramMeasurement]
    ncu_files: Dict[str, str]
    my_roofline_data: Dict[str, Tuple[FlopCount, Number]]

    def __init__(self, name: str):
        self.classical_data = {}
        self.classical_data_program = {}
        self.ncu_files = {}
        self.my_roofline_data = {}
        programs_in_class = {}

        this_dir = os.path.join(get_complete_results_dir(), name)
        raw_data_dir = os.path.join(this_dir, 'raw_data')
        with open(os.path.join(this_dir, "info.json")) as info_file:
            info_data = json.load(info_file)

            for class_number in info_data['classes']:
                results_file_path = os.path.join(raw_data_dir, f"class_{class_number}.json")
                roofline_file_path = os.path.join(raw_data_dir, f"class_{class_number}_roofline.json")
                if os.path.exists(results_file_path):
                    with open(results_file_path) as file:
                        run_data = json.load(file, object_hook=MeasurementRun.from_json)
                        programs_in_class[class_number] = []
                        for program_measurement in run_data.data:
                            self.classical_data[program_measurement.program] = run_data
                            self.classical_data_program[program_measurement.program] = program_measurement
                            programs_in_class[class_number].append(program_measurement.program)

                if os.path.exists(roofline_file_path):
                    self.my_roofline_data.update(read_roofline_data(roofline_file_path))

            for class_number in info_data['classes']:
                for program in programs_in_class[class_number]:
                    ncu_file_path = os.path.join(raw_data_dir, f"report_class_{class_number}_{program}.ncu-rep")
                    if os.path.exists(ncu_file_path):
                        self.ncu_files[program] = ncu_file_path


def plot_roofline(data: CompleteData, dir: str):
    plt.rcParams.update({'figure.figsize': (19, 10)})
    plt.rcParams.update({'font.size': 12})
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)

    hardware_filename = 'nodes.json'
    with open(hardware_filename) as node_file:
        node_data = json.load(node_file)
        min_intensity = None
        max_intensity = None
        node = None
        for run_data in data.classical_data.values():
            if node is None:
                node = run_data.node
                gpu = node_data['ault_nodes'][run_data.node]['GPU']
            elif node != run_data.node:
                print(f"WARNING: Encountered another node {run_data.node} but will use hardware from {node}")

        figure.suptitle(f"Roofline on {node} using a {gpu}")
        for program, program_data in data.classical_data_program.items():
            roofline_data = data.my_roofline_data[program]
            actions = get_all_actions_filtered(data.ncu_files[program], "cupy_")
            action = actions[0]
            if len(actions) > 1:
                print(f"WARNING: More than one action found, taking first {actions[0]} of {len(actions)}")
            for measurement in program_data.measurements['Kernel Time']:
                label = f"{program} using {run_data.description}"
                min_intensity, max_intensity = update_min_max(
                        draw_program_points(measurement, label, ax, *roofline_data), min_intensity, max_intensity)
                min_intensity, max_intensity = update_min_max(draw_ncu_points(action, label, ax), min_intensity,
                                                              max_intensity)

        hardware_data = node_data['GPUs'][gpu]
        peak_performance = hardware_data['flop_per_second']['theoretical']
        draw_roofline(ax, peak_performance,
                      hardware_data['bytes_per_second']['global']['measured'],
                      'bytes / second',
                      min_intensity, max_intensity, color='black',
                      bandwidth_label="Global, Measured")

        ax.legend()
        plt.savefig(os.path.join(dir, "roofline.png"))


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


class GenerateCompleteResults(Script):
    name = "gen-results"
    description = "Generate a set of complete results as defined by the corresponding info.json file"

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('name')
        parser.add_argument('--no-running', action='store_true', default=False, help="Don't rerun")

    @staticmethod
    def action(args):
        this_dir = os.path.join(get_complete_results_dir(), args.name)

        with open(os.path.join(this_dir, "info.json")) as info_file:
            info_data = json.load(info_file)
            raw_data_dir = os.path.join(this_dir, 'raw_data')
            if not os.path.exists(raw_data_dir):
                os.mkdir(raw_data_dir)
            plot_dir = os.path.join(this_dir, 'plots')
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)

            if not args.no_running:
                run_programs(info_data, raw_data_dir)

            data = CompleteData(args.name)
            plot_roofline(data, plot_dir)
