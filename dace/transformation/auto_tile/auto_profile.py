import os
import random
from typing import Any, Dict, Type
import dace
from dace.sdfg.sdfg import SDFG
import sympy

from dace.transformation.auto_tile import peak_flops_and_badwidth_nvidia
from dace.transformation.auto_tile import auto_tile_util

import os
from pathlib import Path
import json

outdir = "individual_kernels"

arch_str = peak_flops_and_badwidth_nvidia.get_arch(0)


def auto_profile(sdfg: SDFG,
                 defined_symbols: Dict[Type[str], Any],
                 verbose: bool = False,
                 re_profile : bool = False,
                 save_individual_kernels : bool = False,
                 no_profile : bool = False):
    sdfg_name = sdfg._name
    kernel_sdfgs = dict()
    perf_results = dict()
    peak_flops, peak_bandwidth = peak_flops_and_badwidth_nvidia.get_peak_flops_and_mem_bandwidth(
        0)

    # If not all symbols are provided create random values for the symbols
    #all_symbols = sdfg.free_symbols
    #undefined_symbols = all_symbols - set(defined_symbols.keys())
    #for sym in undefined_symbols:
    #    defined_symbols[sym] = random.randint(1, 4096)
    #auto_tile_util.generate_random_data()

    file_name = f"{sdfg_name}_perf_results.json"
    if not re_profile and Path(file_name).is_file():
        with open(file_name, "r") as json_file:
            data_dict = json.load(json_file)
            for k, v in data_dict.items():
                filename, mem_str, flop_str, time, perc = v
                nv = (filename, sympy.sympify(mem_str), sympy.sympify(flop_str), time, perc)
                data_dict[k] = nv
            return data_dict

    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and \
                    node.map.schedule == dace.ScheduleType.GPU_Device:
                guid = node.guid
                kernel_sdfg_name = f"{guid}_aopt.sdfg"

                #try:
                if verbose:
                    print("Extract kernel: ", node, "(", node.map, ")")
                kernel_sdfg = auto_tile_util._copy_sub_scope(state, node)
                #auto_tile_util.convert_inputs_to_gpu_storage(kernel_sdfg)

                auto_tile_util.set_transient(kernel_sdfg)
                kernel_sdfgs[guid] = kernel_sdfg
                assert (len(kernel_sdfg.nodes()) == 1)

                if save_individual_kernels:
                    kernel_sdfg.save(os.path.join(outdir, kernel_sdfg_name))

                kernel_state = kernel_sdfg.nodes()[0]
                kernel_entry = auto_tile_util.find_node_in_state_by_cond(kernel_state,
                                                                            lambda n: isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_Device)

                flops, mem_access = auto_tile_util.get_flops_and_mem_access(
                    kernel_sdfg, kernel_state, kernel_entry)
                if verbose:
                    print("Flops and Mem Accesses in the kernel:",
                            flops, " | ", mem_access)
                if not no_profile:
                    inputs = auto_tile_util.generate_random_data(
                        kernel_sdfg, defined_symbols)
                    solved_flops = auto_tile_util.solve(flops, inputs)
                    solved_mem_access = auto_tile_util.solve(
                        mem_access, inputs)
                    if verbose:
                        print("Flops and Mem Accesses in the kernel after instantiating the symbols:",
                            solved_flops, " | ", solved_mem_access)

                if not no_profile:
                    time = auto_tile_util.run_and_measure_time(
                        kernel_sdfg, inputs)
                else:
                    time = 0.0

                perc_peak = 0.0
                """
                if not no_profile and solved_flops == 0:
                    perc_peak = auto_tile_util.percentage_bandwidth(
                        time, solved_mem_access, peak_bandwidth * 1e9)
                else:
                    perc_peak = auto_tile_util.percentage_peak(
                        time, solved_flops, solved_mem_access, peak_flops * 1e9, peak_bandwidth * 1e9)
                """

                if verbose:
                    print("Percentage of peak: ",
                            perc_peak, ", time", time)
                perf_results[guid] = (
                    kernel_sdfg_name, str(flops), str(mem_access), float(time), float(perc_peak))
                """
                except Exception as e:
                    print("Error in extracting and profiling kernel:",
                          node, "(", node.map, ")")
                    print("Error:", e)
                    perf_results[guid] = (
                        kernel_sdfg_name, "0", "0", 0.0, float("nan"))
                """

    if verbose:
        print("Profiled performance results:")
        print("{")
        for k, v in perf_results.items():
            print(f"\t{k}: {v}")
        print("}")

    with open(f"{sdfg_name}_perf_results.json", "w") as json_file:
        json.dump(perf_results, json_file, indent=2)
    return perf_results
