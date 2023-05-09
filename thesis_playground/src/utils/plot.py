from typing import Dict, Tuple, Optional
from numbers import Number
import matplotlib
import matplotlib.pyplot as plt
import math

from utils.general import convert_to_seconds
from measurements.data import MeasurementRun, Measurement
from measurements.flop_computation import FlopCount
from utils.ncu import get_achieved_bytes, get_achieved_work, get_runtime
from utils.ncu_report import IAction
from utils.general import get_programs_data


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


def draw_ncu_points(action: IAction, label: str, ax: matplotlib.axis.Axis, program: Optional[str] = None,
                    marker: str = "x"):
    """
    Draws points onto the given axis. Uses the Q, W and T data from the given ncu_report action object

    :param action: The ncu action object
    :type action: Action
    :param label: The label of the measurement
    :type label: str
    :param ax: The axis
    :type ax: matplotlib.axis.Axis
    :param program: Name of the program of the measurement. If given will color all points of the same program in the
    same color
    :type program: Optional[str]
    :param marker: Marker for the points as passed to matplotlib, defaults to "x"
    :type marker: str
    """
    bytes_count = get_achieved_bytes(action)
    flop = get_achieved_work(action)['dW']
    time = get_runtime(action)
    performance = flop / time
    intensity = flop / bytes_count
    if program is None:
        ax.scatter(intensity, performance, label=label, marker=marker)
    else:
        ax.scatter(intensity, performance, label=label, marker=marker, color=get_color_from_program_name(program))
    return intensity


def draw_program_points(measurement: Measurement, label: str, ax: matplotlib.axis.Axis,
                        flop_count: FlopCount, byte_count: Number, program: Optional[str] = None,
                        marker: str = ".") -> Number:
    """
    Draws point of the given measurement onto the given axis

    :param measurement: The measurement
    :type measurement: Measurement
    :param label: The label of the measurement
    :type label: str
    :param ax: The axis
    :type ax: matplotlib.axis.Axis
    :param flop_count: The FlopCount object
    :type flop_count: FlopCount
    :param byte_count: The number of bytes transferred
    :type byte_count: Number
    :param program: Name of the program of the measurement. If given will color all points of the same program in the
    same color
    :type program: Optional[str]
    :param marker: Marker for the points as passed to matplotlib, defaults to "."
    :type marker: str
    :return: The operational intensity
    :rtype: Number
    """
    time = measurement.average()
    if measurement.unit.endswith('second'):
        time = convert_to_seconds(time, measurement.unit)
    flops = flop_count.get_total_flops()
    performance = flops / time
    intensity = flops / byte_count
    if program is None:
        ax.scatter(intensity, performance, label=label, marker=marker)
    else:
        ax.scatter(intensity, performance, label=label, marker=marker, color=get_color_from_program_name(program))
    return intensity


def get_color_from_program_name(program: str):
    """
    Gets the color as used by matplotlib given the program name

    :param program: The name of the program
    :type program: str
    """
    programs = list(get_programs_data()['programs'].keys())
    index = programs.index(program)
    # normalized_index = float(index) / float(len(programs))
    cmap = plt.get_cmap('tab10')
    normalized_index = float(index % cmap.N) / float(cmap.N)
    return cmap(normalized_index)


def update_min_max(value: Number, min_value: Optional[Number] = None,
                   max_value: Optional[Number] = None) -> Tuple[Number, Number]:
    """
    Updates the given min, max values with the given value. If the given min, max values are None, will use the given
    value instead

    :param value: The value
    :type value: Number
    :param min_value: The min value to update, defaults to None
    :type min_value: Optional[Number], optional
    :param max_value: The max value to update, defaults to None
    :type max_value: Optional[Number], optional
    :return: The updated min and max value (in this order in the tuple)
    :rtype: Tuple[Number, Number]
    """
    if min_value is None:
        min_value = value
    min_value = min(value, min_value)
    if max_value is None:
        max_value = value
    max_value = max(value, max_value)
    return (min_value, max_value)


def plot_roofline_cycles(run_data: MeasurementRun, roofline_data: Dict[str, Tuple[FlopCount, Number]],
                         hardware_data: Dict, ax: matplotlib.axis.Axis, points_only: bool = False):
    peak_performance = hardware_data['flop_per_second']['theoretical'] / hardware_data['graphics_clock']

    ax.set_xlabel("Operational Intensity")
    ax.set_ylabel("Performance [flop/cycle]")

    min_intensity = None
    max_intensity = None
    for program_measurement in run_data.data:
        program = program_measurement.program
        label = f"{program} using {run_data.description}"
        for measurement in program_measurement.measurements['Kernel Cycles']:
            min_intensity, max_intensity = update_min_max(
                    draw_program_points(measurement, label, ax, *roofline_data[program]), min_intensity,
                    max_intensity)

    if not points_only:
        # Draw rooflines
        # Need to divide by the graphics clock to get the bytes/cycle in graphics cycles
        draw_roofline(ax, peak_performance,
                      hardware_data['bytes_per_second']['global']['measured'] / hardware_data['graphics_clock'],
                      'bytes / cycle',
                      min_intensity, max_intensity, color='black',
                      bandwidth_label="Global, Measured")
        # Using here the higher max graphics clock. Don't quite know if this is correct
        draw_roofline(ax, peak_performance,
                      hardware_data['bytes_per_second']['shared']['measured'] / hardware_data['graphics_clock'],
                      'bytes / cycle',
                      min_intensity, max_intensity, color='black',
                      bandwidth_label="Shared, Measured")
        draw_roofline(ax, peak_performance,
                      hardware_data['bytes_per_second']['l2']['measured'] / hardware_data['graphics_clock'],
                      'bytes / cylce',
                      min_intensity, max_intensity, color='black',
                      bandwidth_label="L2, Measured")
        draw_roofline(ax, peak_performance,
                      hardware_data['bytes_per_cycle']['l1']['measured'],
                      'bytes / cylce',
                      min_intensity, max_intensity, color='black',
                      bandwidth_label="L1, Measured")
    ax.legend()
    ax.grid()


def draw_rooflines_seconds(ax: matplotlib.axis.Axis, hardware_data: Dict, min_intensity: float, max_intensity: float):
    """
    Draw the rooflines using units based on seconds

    :param ax: The axis to plot it on
    :type ax: matplotlib.axis.Axis
    :param hardware_data: The dicionary with the hardware data
    :type hardware_data: Dict
    :param min_intensity: The min intensity, defines length on x-axis
    :type min_intensity: float
    :param max_intensity: The max intensity, defines length on x-axis
    :type max_intensity: float
    """
    peak_performance = hardware_data['flop_per_second']['theoretical']
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
    draw_roofline(ax, peak_performance,
                  hardware_data['bytes_per_second']['l2']['measured'],
                  'bytes / second',
                  min_intensity, max_intensity, color='black',
                  bandwidth_label="L2, Measured")
    draw_roofline(ax, peak_performance,
                  hardware_data['bytes_per_cycle']['l1']['measured'] * hardware_data['graphics_clock'],
                  'bytes / second',
                  min_intensity, max_intensity, color='black',
                  bandwidth_label="L1, Measured")


def plot_roofline_seconds(run_data: MeasurementRun, roofline_data: Dict[str, Tuple[FlopCount, Number]],
                          hardware_data: Dict, ax: matplotlib.axis.Axis, points_only: bool = False,
                          actions: Optional[Dict[str, IAction]] = None):

    ax.set_xlabel("Operational Intensity")
    ax.set_ylabel("Performance [flop/seconds]")

    min_intensity = None
    max_intensity = None
    for program_measurement in run_data.data:
        program = program_measurement.program
        label = f"{program} using {run_data.description}"
        for measurement in program_measurement.measurements['Kernel Time']:
            min_intensity, max_intensity = update_min_max(
                    draw_program_points(measurement, label, ax, *roofline_data[program]), min_intensity,
                    max_intensity)
            if actions is not None:
                min_intensity, max_intensity = update_min_max(draw_ncu_points(actions[program], label, ax),
                                                              min_intensity, max_intensity)

    if not points_only:
        draw_rooflines_seconds()

    ax.legend()
    ax.grid()
