from typing import Dict, Tuple, Optional, List, Union
from numbers import Number
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import seaborn as sns
import os
import json
import pandas as pd


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


def rotate_xlabels(ax: matplotlib.axis.Axis, angle: int = 45, replace_dict: Dict[str, str] = {}):
    """
    Rotates the x labels/ticks of the given axis by the given label. Optionally also changes the texts

    :param ax: The axis
    :type ax: matplotlib.axis.Axis
    :param angle: The angle to rotate by, defaults to 45
    :type angle: int, optional
    :param replace_dict: Dictonary with tick names and their replacement, optional
    :type replace_dict: Dict[str, str]
    """
    labels = ax.get_xticklabels()
    for label in labels:
        if label.get_text() in replace_dict:
            label.set(text=replace_dict[label.get_text()])
    ax.set_xticklabels(labels, rotation=angle, horizontalalignment='right')


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


def legend_on_lines_dict(ax: matplotlib.axis.Axis, positions: Dict[str, Dict[str, Union[int, float, Tuple[float]]]]):
    """
    Put legend on labels in the plot in given positions. Remvoes any pre existing legend.

    The given dictionary is expected to have an entry per line to put a legend on. Key is the legend text. Value is a
    dictionary with the following enties:
        - position: Tuple[float] = Position of the text in data coordinates
        - color_index: Union[int] = If int, index in the color palette, gives the color to use, if string will use that
          string as a color
        - rotation: float = Optional, rotation of the text
        - text_position: Tuple[floaŧ] = Optional, gives position of text with an arrow pointing to position

    :param ax: The axis to act on
    :type ax: matplotlib.axis.Axis
    :param positions: Dictionary with the positions and optionally other information.
    :type positions: Dict[str, Dict[str, Union[int, float, Tuple[float]]]]
    """
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    for program, pos_data in positions.items():
        verticalalignment='center'
        angle = 0
        if 'rotation' in pos_data:
            angle = pos_data['rotation']
        pos = pos_data['position']
        if isinstance(pos_data['color_index'], str):
            color = pos_data['color_index']
        else:
            color = sns.color_palette()[pos_data['color_index']]
        if 'text_position' in pos_data:
            text_pos = pos_data['text_position']
            ax.annotate('', xytext=text_pos, xy=pos,
                        arrowprops=get_arrowprops({'color': color}))
            pos = text_pos
            verticalalignment='bottom'

        ax.text(pos[0], pos[1], program, color=color,
                horizontalalignment='center', verticalalignment=verticalalignment, rotation=angle)


def replace_legend_names(legend: matplotlib.legend.Legend, names_map: Dict[str, str]):
    """
    Replace the program names in the legend by more discriptive ones

    :param legend: The legend object where the labels should be changed
    :type legend: matplotlib.legend.Leged
    :param names_map: Dictionay mapping the names/labels to change.
    :type names_map: Dict[str, str]
    """
    if legend is not None:
        for text in legend.get_texts():
            if text is not None and text.get_text() in names_map:
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
