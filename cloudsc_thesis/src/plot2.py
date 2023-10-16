from argparse import ArgumentParser
import os
from typing import List
import json
import pandas as pd

from utils.experiments2 import get_experiment_list_df, get_program_infos
from measurements.data2 import get_data_wideformat, average_data
import scripts2.plot.vert_loop as vert_loop
import scripts2.plot.my_transformations as my_trans
import scripts2.plot.classes as classes
from utils.paths import get_plots_2_folder
from utils.plot import get_node_gpu_map


def generate_run_count_str(run_counts: pd.Series) -> str:
    if run_counts.min() == run_counts.max():
        return str(run_counts.min())
    else:
        return f"between {run_counts.min()} and {run_counts.max()}"


def plot_classes(experiment_ids: List[int], folder_name: str = 'classes'):
    folder = os.path.join(get_plots_2_folder(), folder_name)
    os.makedirs(folder, exist_ok=True)
    measurement_data = get_data_wideformat(experiment_ids)
    # Add class information
    measurement_index_cols = measurement_data.index.names
    measurement_data = measurement_data.reset_index().assign(class_number=lambda x: pd.to_numeric(x['program'].str.get(13)))
    measurement_data = measurement_data.set_index([*measurement_index_cols, 'class_number'])
    data = measurement_data.join(get_experiment_list_df(), on='experiment id')
    nodes = data['node'].unique()
    gpus = set([get_node_gpu_map()[node] for node in nodes])
    node = ' and '.join(nodes)
    gpu = ' and '.join(gpus)

    index_cols = list(data.index.names)
    data = data.reset_index().assign(class_number=lambda x: pd.to_numeric(x['program'].str.get(13)))
    index_cols.remove('experiment id')
    index_cols.append('move_assignments_outside')
    index_cols.append('outside_first')
    data = data.set_index(index_cols)
    index_cols.remove('run number')
    # TODO: Average data doesn't seem to work
    avg_data = average_data(measurement_data).join(get_experiment_list_df(), on='experiment id')\
        .reset_index().set_index(index_cols).drop('experiment id', axis='columns')

    index_cols.remove('scope')
    run_counts_total = data.xs('Total', level='scope').reset_index().groupby(index_cols).count()['run number']
    run_counts_ncu = data.drop('Total', level='scope').drop('temp', level='scope')\
        .reset_index().groupby(index_cols).count()['run number']

    title_ncu = f"Vertical Loop Program 10 on {node} using NVIDIA {gpu} averaging " + \
                f"{generate_run_count_str(run_counts_ncu)} runs"
    title_total = f"Vertical Loop Program 10 on {node} using NVIDIA {gpu} averaging " + \
                  f"{generate_run_count_str(run_counts_total)} runs"
    print(f"Node: {node}, GPU: {gpu}, total time runs: {generate_run_count_str(run_counts_total)}, ncu runs: "
          f"{generate_run_count_str(run_counts_ncu)}")
    # classes.plot_speedup(avg_data, folder, title_total, title_ncu)
    classes.plot_roofline_kernel(avg_data.xs((True, True), level=('move_assignments_outside', 'outside_first')), folder,
                                 'Roofline plot of the three optimized classes')
    # classes.plot_roofline_theoretical(avg_data.xs((True, True), level=('move_assignments_outside', 'outside_first')),
    #                                   folder, title_total)
    classes.plot_runtime_bar(data, avg_data, folder, title_total, title_ncu)


def plot_my_transformations(experiment_ids: List[int], folder_name: str = 'my-transformations'):
    folder = os.path.join(get_plots_2_folder(), folder_name)
    os.makedirs(folder, exist_ok=True)

    measurement_data = get_data_wideformat(experiment_ids)
    data = measurement_data.join(get_experiment_list_df(), on='experiment id')
    index_cols = list(data.index.names)
    index_cols.append('k_caching')
    index_cols.append('change_strides')
    index_cols.remove('experiment id')
    data = data.reset_index().set_index(index_cols).drop('experiment id', axis='columns')
    index_cols.remove('run number')
    avg_data = average_data(measurement_data).join(get_experiment_list_df(), on='experiment id')\
        .reset_index().set_index(index_cols).drop('experiment id', axis='columns')

    nodes = data['node'].unique()
    gpus = set([get_node_gpu_map()[node] for node in nodes])
    node = ' and '.join(nodes)
    gpu = ' and '.join(gpus)
    run_counts_total = data.xs('Total', level='scope').reset_index().groupby(['program', 'NBLOCKS', 'change_strides', 'k_caching']).count()['run number']
    run_counts_ncu = data.xs('work', level='scope').reset_index().groupby(['program', 'NBLOCKS', 'change_strides', 'k_caching']).count()['run number']

    title_ncu = f"Vertical Loop Program 10 on {node} using NVIDIA {gpu} averaging " + \
                f"{generate_run_count_str(run_counts_ncu)} runs"
    title_total = f"Vertical Loop Program 10 on {node} using NVIDIA {gpu} averaging " + \
                  f"{generate_run_count_str(run_counts_total)} runs"
    my_trans.plot_runtime(data, folder, title_total)
    my_trans.plot_kerneltime(data, folder, title_ncu)
    my_trans.plot_speedup(avg_data, folder, title_total, title_ncu)
    my_trans.plot_change_stride_runtime_bar(data, folder, title_ncu)


def plot_vert_loop(experiment_ids: List[int], folder_name: str = 'vert-loop', legend_on_line: bool = False):
    folder = os.path.join(get_plots_2_folder(), folder_name)
    os.makedirs(folder, exist_ok=True)

    measurement_data = get_data_wideformat(experiment_ids)
    data = measurement_data.join(get_experiment_list_df(), on='experiment id')
    index_cols = list(data.index.names)
    index_cols.append('temp allocation')
    index_cols.remove('experiment id')
    data = data.reset_index().set_index(index_cols).drop('experiment id', axis='columns')
    index_cols.remove('run number')
    avg_data = average_data(measurement_data).join(get_experiment_list_df(), on='experiment id')\
        .reset_index().set_index(index_cols).drop('experiment id', axis='columns')
    # print(measurement_data.xs(('cloudsc_vert_loop_4_ZSOLQA', 100000, 60),
    #                           level=('program', 'NBLOCKS', 'experiment id'))['Total time'])
    # print("Averaged")
    # print(average_data(measurement_data).xs(('cloudsc_vert_loop_4_ZSOLQA', 100000, 60),
    #                                         level=('program', 'NBLOCKS', 'experiment id'))['Total time'])
    # print(average_data(measurement_data).join(get_experiment_list_df(), on='experiment id').xs(
    #         ('cloudsc_vert_loop_4_ZSOLQA', 100000), level=('program', 'NBLOCKS'))['Total time'])
    # print(avg_data.xs(('cloudsc_vert_loop_4_ZSOLQA', 100000), level=('program', 'NBLOCKS'))['Total time'])

    # Create speedup plots
    nodes = data['node'].unique()
    gpus = set([get_node_gpu_map()[node] for node in nodes])
    node = ' and '.join(nodes)
    gpu = ' and '.join(gpus)
    index_cols.remove('scope')
    run_counts_total = data.xs('Total', level='scope').reset_index().groupby(index_cols).count()['run number']
    run_counts_ncu = data.drop('Total', level='scope').drop('temp', level='scope')\
        .reset_index().groupby(index_cols).count()['run number']
    title_total = f"Vertical Loop Programs on {node} using NVIDIA {gpu} averaging "\
                  f"{generate_run_count_str(run_counts_total)} runs"
    title_ncu = f"Vertical Loop Programs on {node} using NVIDIA {gpu} averaging "\
                f"{generate_run_count_str(run_counts_ncu)} runs"
    print(f"Node: {node}, GPU: {gpu}, total time runs: {generate_run_count_str(run_counts_total)}, ncu runs: "
          f"{generate_run_count_str(run_counts_ncu)}")

    program_names_map = get_program_infos()['variant description'].to_dict()
    vert_loop.plot_speedup_array_order(avg_data.copy(), folder, title_ncu, legend_on_line=legend_on_line)
    vert_loop.plot_speedup_temp_allocation(data.copy(), avg_data.copy(), folder, title_ncu, legend_on_line=legend_on_line)
    vert_loop.plot_runtime(data.copy(), folder, title_ncu, limit_temp_allocation_to='stack', legend_dict={
        program_names_map['cloudsc_vert_loop_4_ZSOLQA']:
            {'position': (3e5, 28), 'rotation': 25, 'color_index': 0},
        program_names_map['cloudsc_vert_loop_6_ZSOLQA']:
            {'position': (2.4e5, 4), 'rotation': 3, 'color_index': 1},
        program_names_map['cloudsc_vert_loop_6_1_ZSOLQA']:
            {'position': (5.6e5, 7), 'rotation': 3, 'color_index': 2},
        program_names_map['cloudsc_vert_loop_7_3']:
            {'position': (3.8e5, 1), 'rotation': 3, 'color_index': 3}
        })
    vert_loop.plot_runtime(data.copy(), folder, title_ncu, limit_temp_allocation_to='heap', legend_dict={
        program_names_map['cloudsc_vert_loop_4_ZSOLQA']:
            {'position': (2e5, 1.35), 'rotation': 7, 'color_index': 0},
        program_names_map['cloudsc_vert_loop_6_ZSOLQA']:
            {'position': (5.5e5, 1.6), 'rotation': 5, 'color_index': 1},
        program_names_map['cloudsc_vert_loop_6_1_ZSOLQA']:
            {'position': (5.5e5, 0.09), 'rotation': 3, 'color_index': 2},
        program_names_map['cloudsc_vert_loop_7_3']:
            {'position': (3.0e5, 0.32), 'rotation': 5, 'color_index': 3}
        })
    vert_loop.plot_runtime(data.copy(), folder, title_ncu)
    vert_loop.plot_memory_transfers(data.copy(), folder, title_ncu, limit_temp_allocation_to='stack', legend_on_line=legend_on_line)
    vert_loop.plot_memory_transfers(data.copy(), folder, title_ncu, limit_temp_allocation_to='heap', legend_on_line=legend_on_line)
    vert_loop.plot_memory_transfers(data.copy(), folder, title_ncu, legend_on_line=legend_on_line)
    vert_loop.plot_runtime_bar_stack(data.copy(), folder, title_ncu)
    vert_loop.plot_runtime_speedup_temp_allocation_bar(data.copy(), avg_data.copy(), folder)


def action_script(args):
    scripts = {
            'vert-loop': (plot_vert_loop, {'experiment_ids': [11, 12],
                                           'folder_name': 'vert-loop'}),
            'vert-loop-trivial-elimination': (plot_vert_loop, {'experiment_ids': [13, 14],
                                                               'folder_name': 'vert-loop-trivial-elimination'}),

            'vert-loop-ampere': (plot_vert_loop, {'experiment_ids': [292, 293],
                                                  'folder_name': 'vert-loop-ampere',
                                                  'legend_on_line': True}),
            'my-transformations': (plot_my_transformations, {'experiment_ids': [176, 177, 188, 189]}),
            'my-transformations-ampere': (plot_my_transformations, {'experiment_ids': [218, 219, 220, 221],
                                                                    'folder_name': 'my-transformations-ampere'}),
            'classes': (plot_classes, {'experiment_ids': [321, 322, 323]})
    }
    function, func_args = scripts[args.script_name]
    additional_args = json.loads(args.args)
    if len(additional_args) > 0:
        func_args.update(json.loads(args.args))
    function(**func_args)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Commands",
        help="See the help of the respective command")

    script_parser = subparsers.add_parser('script', description='Run predefined script to create a set of pltos')
    script_parser.add_argument('script_name', type=str, help='Name of the script to execute')
    script_parser.add_argument('--args', type=str, default='{}',
                               help='Additional arguments passed to the plot script function as a json-dictionary')
    script_parser.set_defaults(func=action_script)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
