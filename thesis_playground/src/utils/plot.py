from typing import Dict, Tuple, Optional, List, Union
from numbers import Number
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import seaborn as sns
import math
import os
import json
import pandas as pd

from utils.general import convert_to_seconds
from measurements.data import MeasurementRun, Measurement
from measurements.flop_computation import FlopCount


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
    # Hardcoded values because formula with transform_rotates_text=True does not work as expected
    angle = 30
    ax.text(min_intensity, max_bandwidth*min_intensity, text, rotation=angle,
            rotation_mode='anchor', transform_rotates_text=False, color=color)
    ax.grid()


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
    # programs = list(get_programs_data()['programs'].keys())
    # index = programs.index(program)
    # # normalized_index = float(index) / float(len(programs))
    # cmap = plt.get_cmap('tab10')
    # normalized_index = float(index % cmap.N) / float(cmap.N)
    # return cmap(normalized_index)

    # TODO: Are these better colors?
    colors = {
            'cloudsc_class1_658': 'xkcd:purple',
            'cloudsc_class1_670': 'xkcd:lavender',
            'cloudsc_class1_2783': 'xkcd:blue',
            'cloudsc_class1_2857': 'xkcd:light blue',
            'cloudsc_class2_781': 'xkcd:red',
            'cloudsc_class2_1516': 'xkcd:orange',
            'cloudsc_class2_1762': 'xkcd:pink',
            'cloudsc_class3_691': 'xkcd:green',
            'cloudsc_class3_965': 'xkcd:lime green',
            'cloudsc_class3_1985': 'xkcd:olive drab',
            'cloudsc_class3_2120': 'xkcd:mustard',
            }
    return colors[program]


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


def save_plot(path: str):
    """
    Saves the current plot into the given filepath. Makes sure that all folders are created and prints a message where
    it is saved to.

    :param path: Path to the file to save it to
    :type path: str
    """
    os.makedirs(os.path.basename(path), exist_ok=True)
    print(f"Store plot into {path}")
    plt.savefig(path)


def rotate_xlabels(ax: matplotlib.axis.Axis, angle: int = 45):
    """
    Rotates the x labels/ticks of the given axis by the given label

    :param ax: The axis
    :type ax: matplotlib.axis.Axis
    :param angle: The angle to rotate by, defaults to 45
    :type angle: int, optional
    """
    ax.set_xticklabels(ax.get_xticklabels(), rotation=angle, horizontalalignment='right')


def get_new_figure(number_of_colors: Optional[int] = None) -> matplotlib.figure.Figure:
    """
    Clean figure and axis and return a new figure

    :param number_of_colors: Number of colors, if None will use a palette which does not require this information,
    defaults to None
    :type number_of_colors: Optional[int]
    :return: New figure
    :rtype: matplotlib.figure.Figure
    """
    plt.rcParams.update({'figure.figsize': (19, 10)})
    plt.rcParams.update({'font.size': 12})
    plt.cla()
    plt.clf()
    fig = plt.figure()
    sns.set_style('whitegrid')
    sns.set(font_scale=2)
    if number_of_colors is None:
        sns.set_palette('pastel')
    else:
        sns.set_palette('husl', number_of_colors)
    return fig


def size_vs_y_plot(ax: matplotlib.axis.Axis, ylabel: str, title: str, data: pd.DataFrame, size_var_name: str = 'size'):
    """
    Adds x and y labels and x ticks based on data for a plot with size on x axis

    :param ax: The axis to act on
    :type ax: matplotlib.axis.Axis
    :param ylabel: Label for y axis
    :type ylabel: str
    :param title: Title of the axis plot
    :type title: str
    :param data: The data, used to get x ticks
    :type data: pd.DataFrame
    :param size_var_name: Name of the size variable/column. Defaults to 'size'
    :type size_var_name: str
    """
    ax.set_xlabel('NBLOCKS')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(EngFormatter(places=0, sep="\N{THIN SPACE}"))
    ax.set_xticks(data.reset_index()[size_var_name].unique())


def get_bytes_formatter() -> matplotlib.ticker.EngFormatter:
    """
    Returns formatter for bytes

    :return: Formatter
    :rtype: matplotlib.ticker.EngFormatter
    """
    return EngFormatter(places=0, sep="\N{THIN SPACE}", unit='B')


def get_arrowprops(update: Dict[str, Union[str, Number]]) -> Dict[str, Union[str, Number]]:
    """
    Return properties dict for a simple arrow.

    :param update: Dict with values to change
    :type update: Dict[str, Union[str, Number]]
    :return: The properties dict
    :rtype: Dict[str, Union[str, Number]]
    """
    props = dict(width=2.0, headwidth=7.0, headlength=7.0, shrink=0.01)
    props.update(update)
    return props


def legend_on_lines(ax: matplotlib.axis.Axis,
                    positions: List[Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]],
                    program_names: List[str],
                    rotations: Optional[Union[List[float], Dict[str, float]]] = None,
                    color_palette_offset: int = 0):
    """
    Put legend labels in the plot on given positions. Removes any pre existing legend

    :param ax: The axis ot act on
    :type ax: matplotlib.axis.Axis
    :param positions: The positions. Each tuple is a xy position. If there is a tuple with two tuples inside, the first
    gives the position of the text from which onwards an arrow will be drawn to the position given by the 2nd tuple
    :type positions: List[Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]]
    :param program_names: List of the legend names, needs to have same length as positions
    :type program_names: List[str]
    :param rotations: Rotation angles. Can either be a list, then it has to be the same length as positions, or a
    dictionary mapping program name to angle. If none all angles are 0, defaults to None
    :type rotations: Optional[Union[List[float], Dict[str, float]]], optional
    :param color_palette_offset: Set to a non zero value if colors of lines do not start at index 0 for the color
    palette, defaults to 0
    :type color_palette_offset: int, optional
    """
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    for index, (pos, program) in enumerate(zip(positions, program_names)):
        angle = 0
        if rotations is not None:
            if isinstance(rotations, List):
                angle = rotations[index]
            elif isinstance(rotations, Dict):
                angle = rotations[program]
        text_pos = pos
        if isinstance(pos[0], Tuple):
            text_pos = pos[0]
            ax.annotate('', xytext=text_pos, xy=pos[1],
                        arrowprops=get_arrowprops({'color': sns.color_palette()[index + color_palette_offset]}))

        ax.text(text_pos[0], text_pos[1], program, color=sns.color_palette()[index + color_palette_offset],
                horizontalalignment='center', verticalalignment='center', rotation=angle)


def legend_on_lines_dict( ax: matplotlib.axis.Axis, positions: Dict[str, Dict[str, Union[int, float]]]):
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    for program, pos_data in positions.items():
        angle = 0
        if rotations is not None:
            if isinstance(rotations, List):
                angle = rotations[index]
            elif isinstance(rotations, Dict):
                angle = rotations[program]
        text_pos = pos
        if isinstance(pos[0], Tuple):
            text_pos = pos[0]
            ax.annotate('', xytext=text_pos, xy=pos[1],
                        arrowprops=get_arrowprops({'color': sns.color_palette()[index + color_palette_offset]}))

        ax.text(text_pos[0], text_pos[1], program, color=sns.color_palette()[index + color_palette_offset],
                horizontalalignment='center', verticalalignment='center', rotation=angle)




def replace_legend_names(legend: matplotlib.legend.Legend, names_map: Optional[Dict[str, str]] = None):
    """
    Replace the program names in the legend by more discriptive ones

    :param legend: The legend object where the labels should be changed
    :type legend: matplotlib.legend.Leged
    :param names_map: Dictionay mapping the names/labels to change.
    :type names_map: Dict[str, str]
    """
    for text in legend.get_texts():
        if text.get_text() in names_map:
            text.set(text=names_map[text.get_text()])


def get_node_gpu_map() -> Dict[str, str]:
    """
    Returns mapping node -> GPU

    :return: Node -> GPU map
    :rtype: Dict[str, str]
    """
    hardware_filename = 'nodes.json'
    with open(hardware_filename) as node_file:
        node_data = json.load(node_file)
        return {node: node_data['ault_nodes'][node]['GPU'] for node in node_data['ault_nodes']}
