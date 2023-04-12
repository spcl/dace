from argparse import ArgumentParser
import json
from os import path
from typing import Dict, Tuple, Optional
from numbers import Number
import matplotlib.pyplot as plt
import matplotlib
import math

from utils import get_results_dir, convert_to_seconds
from measurement_data import MeasurementRun
from flop_computation import FlopCount, read_roofline_data


def draw_roofline(ax: matplotlib.axis.Axis, peak_performance: float, max_bandwidth: float, max_bandwidth_unit: str,
                  min_intensity: float, max_intensity: float, color: str, bandwidth_label: Optional[str]):
    crosspoint_intensity = peak_performance / max_bandwidth
    if max_intensity < crosspoint_intensity:
        max_intensity = crosspoint_intensity*2.0
    ax.loglog([min_intensity, crosspoint_intensity], [max_bandwidth * (min_intensity), peak_performance], color=color)
    ax.loglog([crosspoint_intensity, max_intensity], [peak_performance, peak_performance], color=color)
    dx = crosspoint_intensity-min_intensity
    dy = peak_performance-max_bandwidth*min_intensity
    angle = math.atan(dy/dx)*180/math.pi
    text = f"beta={max_bandwidth:.3e} [{max_bandwidth_unit}]"
    if bandwidth_label is not None:
        text += f" ({bandwidth_label})"
    ax.text(min_intensity, max_bandwidth*min_intensity, text, rotation=angle,
            rotation_mode='anchor', transform_rotates_text=True)


def plot_roofline_cycles(run_data: MeasurementRun, roofline_data: Dict[str, Tuple[FlopCount, Number]],
                         hardware_data: Dict, ax: matplotlib.axis.Axis):
    peak_performance = hardware_data['flop_per_second']['theoretical'] / hardware_data['graphics_clock']

    ax.set_xlabel("Operational Intensity")
    ax.set_ylabel("Performance [flop/cycle]")

    min_intensity = None
    max_intensity = None
    for program_measurement in run_data.data:
        program = program_measurement.program
        for measurement in program_measurement.measurements['Kernel Cycles']:

            cycles = measurement.average()
            flops = roofline_data[program][0].get_total_flops()
            bytes = roofline_data[program][1]
            performance = flops / cycles
            intensity = flops / bytes
            ax.scatter(intensity, performance, label=program)
            if min_intensity is None:
                min_intensity = intensity
            min_intensity = min(intensity, min_intensity)
            if max_intensity is None:
                max_intensity = intensity
            max_intensity = max(intensity, max_intensity)

    # Draw rooflines
    draw_roofline(ax, peak_performance,
                  hardware_data['bytes_per_second']['global']['measured'] / hardware_data['global_memory_clock'],
                  'bytes / cycle',
                  min_intensity, max_intensity, color='black',
                  bandwidth_label="Global, Measured")
    # Using here the higher max graphics clock. Don't quite know if this is correct
    draw_roofline(ax, peak_performance,
                  hardware_data['bytes_per_second']['shared']['measured'] / hardware_data['graphics_clock'],
                  'bytes / cycle',
                  min_intensity, max_intensity, color='black',
                  bandwidth_label="Shared, Measured")
    ax.legend()
    ax.grid()


def plot_roofline_seconds(run_data: MeasurementRun, roofline_data: Dict[str, Tuple[FlopCount, Number]],
                          hardware_data: Dict, ax: matplotlib.axis.Axis):
    peak_performance = hardware_data['flop_per_second']['theoretical']

    ax.set_xlabel("Operational Intensity")
    ax.set_ylabel("Performance [flop/seconds]")

    min_intensity = None
    max_intensity = None
    for program_measurement in run_data.data:
        program = program_measurement.program
        for measurement in program_measurement.measurements['Kernel Time']:

            runtime = convert_to_seconds(measurement.average(), measurement.unit)
            flops = roofline_data[program][0].get_total_flops()
            bytes = roofline_data[program][1]
            performance = flops / runtime
            intensity = flops / bytes
            ax.scatter(intensity, performance, label=program)
            if min_intensity is None:
                min_intensity = intensity
            min_intensity = min(intensity, min_intensity)
            if max_intensity is None:
                max_intensity = intensity
            max_intensity = max(intensity, max_intensity)

    # Draw rooflines
    draw_roofline(ax, peak_performance,
                  hardware_data['bytes_per_second']['global']['measured'],
                  'bytes / second',
                  min_intensity, max_intensity, color='black',
                  bandwidth_label="Global, Measured")
    # Using here the higher max graphics clock. Don't quite know if this is correct
    draw_roofline(ax, peak_performance,
                  hardware_data['bytes_per_second']['shared']['measured'],
                  'bytes / second',
                  min_intensity, max_intensity, color='black',
                  bandwidth_label="Shared, Measured")
    ax.legend()
    ax.grid()


def main():
    parser = ArgumentParser(description="Creates a roofline plot")
    parser.add_argument(
            'files',
            type=str,
            nargs='+',
            help='Basename of the results and roofline file. Without file ending')

    args = parser.parse_args()

    plt.rcParams.update({'figure.figsize': (19, 10)})
    plt.rcParams.update({'font.size': 12})
    hardware_filename = 'nodes.json'

    for file in args.files:
        results_filename = path.join(get_results_dir(), f"{file}.json")
        roofline_filename = path.join(get_results_dir(), f"{file}_roofline_rough.json")
        plot_filename = f"{file}_roofline.pdf"
        with open(results_filename) as results_file:
            with open(hardware_filename) as node_file:
                run_data = json.load(results_file, object_hook=MeasurementRun.from_json)
                roofline_data = read_roofline_data(roofline_filename)
                node_data = json.load(node_file)
                gpu = node_data['ault_nodes'][run_data.node]['GPU']
                figure = plt.figure()
                figure.suptitle("Roofline on {run_data.node} using a {gpu}")
                ax = figure.add_subplot(2, 1, 1)
                plot_roofline_cycles(run_data, roofline_data, node_data['GPUs'][gpu], ax)
                ax = figure.add_subplot(2, 1, 2)
                plot_roofline_seconds(run_data, roofline_data, node_data['GPUs'][gpu], ax)
                print(f"Save plot into {plot_filename}")
                plt.savefig(plot_filename)


if __name__ == '__main__':
    main()
