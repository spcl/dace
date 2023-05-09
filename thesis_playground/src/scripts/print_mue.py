from tabulate import tabulate
import json
import os
import re

from utils.ncu import get_achieved_bytes, get_all_actions, get_achieved_performance, get_peak_performance, \
                      action_list_to_dict
from utils.paths import get_complete_results_dir
from measurements.flop_computation import get_number_of_bytes_2
from measurements.data import MeasurementRun
from scripts import Script


class PrintMUE(Script):
    name = "print-mue"
    description = "Prints the MUE for the data in complete_results/all_frist_opt"

    @staticmethod
    def action(args):
        ncu_folder = os.path.join(get_complete_results_dir(), 'all_first_opt', 'raw_data')
        results_folder = ncu_folder
        result_files = [f for f in os.listdir(results_folder)
                        if re.match(r"class_[1-3].json", f)]
        ncu_files = [f for f in os.listdir(ncu_folder)
                     if re.match(r"report_class_[1-3]_cloudsc_class[1-3]_[0-9]{3,4}.ncu-rep", f)]
        data = {}
        for ncu_file in ncu_files:
            ncu_file = os.path.join(ncu_folder, ncu_file)
            program = re.match("[a-z_0-9]*(cloudsc_class[1-3]_[0-9]{3,4})", os.path.basename(ncu_file)).group(1)
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

            if program not in data:
                data[program] = {}
            data[program]['measured bytes'] = get_achieved_bytes(action)
            data[program]['achieved bandwidth'] = get_achieved_performance(action)[1]
            data[program]['peak bandwidth'] = get_peak_performance(action)[1]

        for result_file in result_files:
            result_file = os.path.join(results_folder, result_file)
            with open(result_file) as file:
                run_data = json.load(file, object_hook=MeasurementRun.from_json)
                for program_measurement in run_data.data:
                    program = program_measurement.program
                    myQ = get_number_of_bytes_2(program_measurement.parameters, program)[0]
                    if program not in data:
                        data[program] = {}
                    data[program]['theoretical bytes'] = myQ

        tabulate_data = []
        for program in data:
            ncuQ = data[program]['measured bytes']
            myQ = data[program]['theoretical bytes']
            io_efficiency = myQ / ncuQ
            bw_efficiency = data[program]['achieved bandwidth'] / data[program]['peak bandwidth']
            mue = io_efficiency * bw_efficiency
            tabulate_data.append([program, ncuQ, myQ, io_efficiency, bw_efficiency, mue])

        print(tabulate(tabulate_data,
                       headers=['program', 'measured bytes', 'theoretical bytes', 'I/O efficiency', 'BW efficiency', 'MUE'],
                       intfmt=',', floatfmt='.2f'))
