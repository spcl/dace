from argparse import ArgumentParser
from subprocess import run
import csv
import os
from typing import Dict
from tabulate import tabulate
import sys

from utils import convert_to_bytes
from print_utils import sort_by_program_number
from ncu_utils import get_action

sys.path.insert(0, '/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.8.0-fjdnxm6yggxxp75sb62xrxxmeg4s24ml/nsight-compute-2022.3.0/extras/python/')
import ncu_report


def extract_roofline_data_from_ncu(filename: str) -> Dict:
    csv_stdout = run(['ncu', '--import', filename, '--csv', '--section', 'MemoryWorkloadAnalysis_Tables',
                      '--section', 'SpeedOfLight_RooflineChart', '--section', 'SpeedOfLight', '--page',
                      'details', '--details-all'], capture_output=True)
    reader = csv.reader(csv_stdout.stdout.decode('UTF-8').split('\n')[:-1])

    # create dict where key is header name and value the index/column of it
    header = {}
    for index, key in enumerate(next(reader)):
        header[key] = index

    Q = 0
    W = 0.0
    cycles = 0
    for row in reader:
        if row[header['Metric Name']] == 'dram__bytes_write.sum':
            Q += convert_to_bytes(row[header['Metric Value']], row[header['Metric Unit']])
        if row[header['Metric Name']] == 'dram__bytes_read.sum':
            Q += convert_to_bytes(row[header['Metric Value']], row[header['Metric Unit']])
        if row[header['Metric Name']] == 'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed':
            W += float(row[header['Metric Value']])
        if row[header['Metric Name']] == 'smsp__sass_thread_inst_executed_op_dmul.sum.per_cycle_elapsed':
            W += float(row[header['Metric Value']])
        if row[header['Metric Name']] == 'smsp__sass_thread_inst_executed_op_dfma.sum.per_cycle_elapsed':
            W += 2*float(row[header['Metric Value']])
        if row[header['Metric Name']] == 'Elapsed Cycles':
            cycles = int(row[header['Metric Value']].replace(',', ''))

    return {"Q": Q, "W": W*cycles}


def extract_roofline_data_from_ncu_2(filename: str) -> Dict:
    my_action = get_action(filename)

    Q = my_action.metric_by_name('dram__bytes_write.sum').as_uint64()
    Q += my_action.metric_by_name('dram__bytes_read.sum').as_uint64()
    cycles = my_action.metric_by_name('gpc__cycles_elapsed.max').as_double()
    dadds = my_action.metric_by_name('smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed').as_double()
    dmuls = my_action.metric_by_name('smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed').as_double()
    dfmas = my_action.metric_by_name('smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed').as_double()
    fadds = my_action.metric_by_name('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed').as_double()
    fmuls = my_action.metric_by_name('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed').as_double()
    ffmas = my_action.metric_by_name('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed').as_double()
    dadds *= cycles
    dmuls *= cycles
    dfmas *= cycles
    fadds *= cycles
    fmuls *= cycles
    ffmas *= cycles
    W = dadds + dmuls + 2*dfmas
    beta_pct = my_action.metric_by_name('dram__bytes_read.sum.pct_of_peak_sustained_elapsed').as_double()
    beta_pct += my_action.metric_by_name('dram__bytes_write.sum.pct_of_peak_sustained_elapsed').as_double()
    return {"Q": Q, "W": int(W), "I": W/Q, "dadds": int(dadds), "dmuls": int(dmuls), "dfmas": int(dfmas),
            "fadds": int(fadds), "fmuls": int(fmuls), "ffmas": int(ffmas), "cycles": cycles, "beta_pct": beta_pct}


def extract_peak_data_from_ncu(filename: str) -> Dict:
    my_action = get_action(filename)

    pi = 2*my_action.metric_by_name('sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained').as_double()
    beta_global = my_action.metric_by_name('dram__bytes.sum.peak_sustained').as_double()

    return {"pi": pi, "beta_global": beta_global}


def main():
    parser = ArgumentParser(description='TODO')
    parser.add_argument('basename')
    parser.add_argument('--detail', action='store_true', default=False)

    args = parser.parse_args()
    print(args)

    folder = 'ncu-reports'
    data = {}
    data_peak = None
    for filename in os.listdir(folder):
        if filename.startswith(f"report_{args.basename}"):
            path = os.path.join(folder, filename)
            if data_peak is None:
                print(f"Read peak data from {path}")
                data_peak = extract_peak_data_from_ncu(path)
            print(f"Reading report from {path}")
            if len(filename.split(f"report_{args.basename}_")) == 1:
                program = args.basename
            else:
                program = filename.split(f"report_{args.basename}_")[1].split(".")[0]
            if os.path.isfile(path):
                data[program] = extract_roofline_data_from_ncu_2(path)

    flat_data = []
    headers = ['program', 'Q [byte]', 'W [flop]', 'I [Flop/byte]', 'P [flop/cycle]', 'b [byte/cycle]',
               'Pi [flop/cycle]', 'max b [byte/cycle]', 'Performance [%]', 'Bandwidth [%]', 'Bandwidth ncu [%]']
    if args.detail:
        headers.extend(['dadds', 'dmuls', 'dfmas', 'fadds', 'fmuls', 'ffmas'])
    programs = list(data.keys())
    for program in programs:
        row = [program, data[program]['Q'], data[program]['W'], data[program]['I'],
               data[program]['W'] / data[program]['cycles'], data[program]['Q'] / data[program]['cycles'],
               int(data_peak['pi']), int(data_peak['beta_global']),
               data[program]['W'] / (data[program]['cycles'] * data_peak['pi']) * 100,
               data[program]['Q'] / (data[program]['cycles'] * data_peak['beta_global']) * 100,
               data[program]['beta_pct']]
        if args.detail:
            row.extend([data[program]['dadds'], data[program]['dmuls'], data[program]['dfmas']])
            row.extend([data[program]['fadds'], data[program]['fmuls'], data[program]['ffmas']])
        flat_data.append(row)

    sort_by_program_number(flat_data)
    print(tabulate(flat_data, headers=headers, intfmt=',',
                   floatfmt=(None, None, None, ".3f", ".1f", ".1f", None, None, ".1f", ".1f", ".1f")))


if __name__ == '__main__':
    main()
