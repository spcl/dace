from tabulate import tabulate
import json
import os
import re

from utils.ncu import get_achieved_bytes, get_all_actions, get_achieved_performance, get_peak_performance, \
                      action_list_to_dict
from flop_computation import get_number_of_bytes_2
from measurement_data import MeasurementRun

ncu_folder = 'ncu-reports'
results_folder = 'results'
ncu_files = [
        'report_vert_loop_4_10k_cloudsc_vert_loop_4.ncu-rep',
        'report_vert_loop_4_100k_cloudsc_vert_loop_4.ncu-rep',
        'report_vert_loop_4_200k_cloudsc_vert_loop_4.ncu-rep',
        'report_vert_loop_4_300k_cloudsc_vert_loop_4.ncu-rep',
        'report_vert_loop_4_500k_cloudsc_vert_loop_4.ncu-rep',
        'report_vert_loop_4_1M_cloudsc_vert_loop_4.ncu-rep',
        'report_vert_loop_5_300k_cloudsc_vert_loop_5.ncu-rep',
        'report_vert_loop_5_500k_cloudsc_vert_loop_5.ncu-rep',
        'report_vert_loop_5_1M_cloudsc_vert_loop_5.ncu-rep'
        ]
result_files = [
        'vert_loop_4_10k.json',
        'vert_loop_4_100k.json',
        'vert_loop_4_200k.json',
        'vert_loop_4_300k.json',
        'vert_loop_4_500k.json',
        'vert_loop_4_1M.json',
        'vert_loop_5_300k.json',
        'vert_loop_5_500k.json',
        'vert_loop_5_1M.json'
        ]


def main():
    data = {}
    for ncu_file in ncu_files:
        label = '_'.join(ncu_file.split('_')[1:5])
        ncu_file = os.path.join(ncu_folder, ncu_file)
        if not os.path.exists(ncu_file):
            print(f"File {ncu_file} not found")
            continue
        actions = get_all_actions(ncu_file)
        if len(actions) > 1:
            actions = action_list_to_dict(actions)
            actions_to_consider = []
            for name, action in actions.items():
                if re.match("[a-z_]*_map", name) is not None:
                    actions_to_consider.append(*action)
            if len(actions_to_consider) > 1:
                print(f"Found multiple possible actions, taking first only with name: {actions_to_consider[0].name()}")
            if len(actions_to_consider) == 0:
                print(f"No possible action found, actions are: {actions}")
            action = actions_to_consider[0]
        else:
            action = actions[0]

        data[label] = {}
        data[label]['measured bytes'] = get_achieved_bytes(action)
        data[label]['achieved bandwidth'] = get_achieved_performance(action)[1]
        data[label]['peak bandwidth'] = get_peak_performance(action)[1]

    for result_file in result_files:
        label = result_file.split('.')[0]
        result_file = os.path.join(results_folder, result_file)
        if not os.path.exists(result_file):
            print(f"File {result_file} not found")
            continue
        with open(result_file) as file:
            run_data = json.load(file, object_hook=MeasurementRun.from_json)
            for program_measurement in run_data.data:
                myQ = get_number_of_bytes_2(program_measurement.parameters, program_measurement.program)[0]
                myQ_temp = get_number_of_bytes_2(program_measurement.parameters, program_measurement.program,
                                                 temp_arrays=True)[0]
                if label not in data:
                    data[label] = {}
                data[label]['theoretical bytes'] = myQ
                data[label]['theoretical bytes temp'] = myQ_temp

    tabulate_data = []
    for program in data:
        ncuQ = data[program]['measured bytes']
        myQ = data[program]['theoretical bytes']
        myQ_temp = data[program]['theoretical bytes temp']
        io_efficiency = myQ / ncuQ
        io_efficiency_temp = myQ_temp / ncuQ
        bw_efficiency = data[program]['achieved bandwidth'] / data[program]['peak bandwidth']
        mue = io_efficiency * bw_efficiency
        mue_temp = io_efficiency_temp * bw_efficiency
        tabulate_data.append([program, ncuQ, myQ, myQ_temp, io_efficiency, io_efficiency_temp, bw_efficiency, mue,
                              mue_temp])

    print(tabulate(tabulate_data,
                   headers=['program', 'measured bytes', 'theoretical bytes', 'theoretical bytes with temp',
                            'I/O efficiency', 'I/O efficiency with temp', 'BW efficiency', 'MUE', 'MUE with temp'],
                   intfmt=',', floatfmt='.2f'))


if __name__ == '__main__':
    main()
