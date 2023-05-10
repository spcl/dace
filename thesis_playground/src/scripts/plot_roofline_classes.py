import os
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from typing import Optional, List, Tuple

from utils.paths import get_complete_results_dir
from utils.plot import update_min_max, draw_program_points, draw_roofline, draw_ncu_points
from utils.ncu import get_all_actions_filtered
from scripts import Script
from scripts.generate_complete_results import CompleteData


def add_points(ax: matplotlib.axis.Axis, data: CompleteData, desc: str, marker: str) -> Tuple[float]:
    min_intensity = None
    max_intensity = None
    for program, program_data in data.classical_data_program.items():
        roofline_data = data.my_roofline_data[program]
        actions = get_all_actions_filtered(data.ncu_files[program], "cupy_")
        action = actions[0]
        if len(actions) > 1:
            print(f"WARNING: More than one action found, taking first {actions[0]} of {len(actions)}")
        for measurement in program_data.measurements['Kernel Time']:
            label = f"{program} {desc}"
            min_intensity, max_intensity = update_min_max(
                    draw_program_points(measurement, label, ax, *roofline_data, program=program, marker=marker),
                    min_intensity, max_intensity)
            # min_intensity, max_intensity = update_min_max(draw_ncu_points(action, label, ax, program=program),
            #                                               min_intensity, max_intensity)
    return min_intensity, max_intensity


def check_for_common_node_gpu(data_list: List[CompleteData]) -> Optional[Tuple[str]]:
    """
    Check if all measurements of the given list where done on the same node and returns the gpu of the node.

    :param data_list: List of data
    :type data_list: List[CompleteData]
    :return: Tuple with node name in first position and string of the name of the gpu used in second or None if not all
    on same node
    :rtype: Optional[Tuple[str]]
    """
    hardware_filename = 'nodes.json'
    with open(hardware_filename) as node_file:
        node_data = json.load(node_file)
        node = None
        for data in data_list:
            for run_data in data.classical_data.values():
                if node is None:
                    node = run_data.node
                    gpu = node_data['ault_nodes'][run_data.node]['GPU']
                elif node != run_data.node:
                    print(f"WARNING: Encountered another node {run_data.node} but will use hardware from {node}")
                    return None
    return (node, gpu)


class PlotRooflineClasses(Script):
    name = "plot-roofline-classes"
    description = "Plot the roofline of all the classes in one plot"

    @staticmethod
    def action(args):
        # TODO: Make legend more compact -> one entry per program and differene ncu/my and baseline/my
        baseline_data = CompleteData('class2_3_baseline')
        my_data = CompleteData('all_first_opt')

        plt.rcParams.update({'figure.figsize': (19, 10)})
        plt.rcParams.update({'font.size': 12})
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)

        node, gpu = check_for_common_node_gpu([baseline_data, my_data])
        figure.suptitle(f"Roofline on {node} using a {gpu}")

        hardware_filename = 'nodes.json'
        with open(hardware_filename) as node_file:
            node_data = json.load(node_file)
            hardware_data = node_data['GPUs'][gpu]
            peak_performance = hardware_data['flop_per_second']['theoretical']
            min_i, max_i = add_points(ax, baseline_data, "Baseline", marker=".")
            min_i, max_i = add_points(ax, my_data, "My Improvements", marker="x")

            draw_roofline(ax, peak_performance,
                          hardware_data['bytes_per_second']['global']['measured'],
                          'bytes / second',
                          min_i, max_i, color='black',
                          bandwidth_label="Global, Measured")

            # Only use one legend entry per program
            new_handles, new_labels = [], []
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                program = label.split(' ')[0]
                if program not in new_labels:
                    color = handle.get_edgecolor()
                    new_handles.append(mpatches.Patch(color=color, label=program))
                    new_labels.append(program)

            # sort by class and program number
            new_handles.sort(key=lambda handle: int(handle.get_label().split('_')[1][-1])*10000 +
                             int(handle.get_label().split('_')[2]))
            # Add baseline and my improvements distinction
            new_handles.append(mlines.Line2D([], [], marker='.', label="Baseline", markersize=15, color='w',
                               markerfacecolor='k'))
            new_handles.append(mlines.Line2D([], [], marker='x', label="My Improvements", markersize=10, color='k',
                               lw=0))

            ax.legend(handles=new_handles)
            dir = os.path.join(get_complete_results_dir(), "plots")
            if not os.path.exists(dir):
                os.mkdir(dir)
            filename = os.path.join(dir, "roofline_all.png")
            print(f"Save plot into {filename}")
            plt.savefig(filename)
