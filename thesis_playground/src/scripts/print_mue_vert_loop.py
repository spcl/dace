from argparse import ArgumentParser
from tabulate import tabulate
import json
import os
import re
import numpy as np

from utils.ncu import get_achieved_bytes, get_all_actions, get_achieved_performance, get_peak_performance, \
                      action_list_to_dict, get_runtime
from measurements.flop_computation import get_number_of_bytes_2
from measurements.data import MeasurementRun
from scripts import Script

folder = 'vert_loop_results'


class PrintMUEVertLoop(Script):
    name = "print-mue-vert"
    description = "Print MUE for the vertical loop results"

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('--mwe', action='store_true', default=False)

    @staticmethod
    def action(args):
        data = {}
        files = os.listdir(folder)
        if args.mwe:
            files = [f for f in files if re.search('_mwe_', f)]
        for file in files:
            if file.split('.')[-1] == 'ncu-rep':
                label = '_'.join(file.split('.')[0].split('_')[1:-1])
                actions = get_all_actions(os.path.join(folder, file))
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

                if label not in data:
                    data[label] = {}
                if 'measured bytes' not in data[label]:
                    data[label]['measured bytes'] = np.ndarray((0), dtype=np.int32)
                    data[label]['achieved bandwidth'] = np.ndarray((0), dtype=np.int32)
                    data[label]['peak bandwidth'] = np.ndarray((0), dtype=np.int32)
                    data[label]['runtime'] = np.ndarray((0), dtype=np.float64)
                data[label]['measured bytes'] = np.append(data[label]['measured bytes'], get_achieved_bytes(action))
                data[label]['achieved bandwidth'] = np.append(data[label]['achieved bandwidth'], get_achieved_performance(action)[1])
                data[label]['peak bandwidth'] = np.append(data[label]['peak bandwidth'], get_peak_performance(action)[1])
                data[label]['runtime'] = np.append(data[label]['runtime'], get_runtime(action))
                if get_achieved_bytes(action) is None:
                    print(f"WARNING: measured bytes is None in {file}")
                if get_achieved_performance(action)[1] is None:
                    print(f"WARNING: achieved bandwidth is None in {file}")
                if get_peak_performance(action)[1] is None:
                    print(f"WARNING: peak bandwidth is None in {file}")
            elif file.split('.')[-1] == 'json':
                label = '_'.join(file.split('.')[0].split('_')[1:])
                with open(os.path.join(folder, file)) as f:
                    run_data = json.load(f, object_hook=MeasurementRun.from_json)
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
            ncuQ_avg = int(np.average(data[program]['measured bytes']))
            ncuQ_range = int(np.max(data[program]['measured bytes']) - np.min(data[program]['measured bytes']))
            myQ = data[program]['theoretical bytes']
            myQ_temp = data[program]['theoretical bytes temp']
            io_efficiency = myQ / ncuQ_avg
            io_efficiency_temp = myQ_temp / ncuQ_avg
            bw_efficiency = np.average(data[program]['achieved bandwidth'] / data[program]['peak bandwidth'])
            mue = io_efficiency * bw_efficiency
            mue_temp = io_efficiency_temp * bw_efficiency
            runtime = np.average(data[program]['runtime'])
            tabulate_data.append([program, ncuQ_avg, ncuQ_range, myQ, myQ_temp, io_efficiency, io_efficiency_temp,
                                  bw_efficiency, mue, mue_temp, runtime])

        tabulate_data.sort()
        print(tabulate(tabulate_data,
                       headers=['program', 'measured bytes', 'measured bytes range', 'theoretical bytes', 'theo. bytes with temp',
                                'I/O eff.', 'I/O eff. w/ temp', 'BW eff.', 'MUE', 'MUE with temp',
                                'runtime [s]'],
                       intfmt=',', floatfmt=(None, None, None, None, '.2f', '.2f', '.2f', '.2f', '.2e', '.2e', '.2e')))
