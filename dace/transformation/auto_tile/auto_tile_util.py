import copy
from ctypes import sizeof
import random
import statistics
from typing import Any, Dict, Type
import dace
import cupy
from dace.sdfg.analysis.cutout import SDFGCutout
from dace.sdfg.sdfg import SDFG, SDFGState
import re

def _copy_sub_scope(state: dace.sdfg.SDFGState, scope_entry: dace.nodes.MapEntry):
    nn = []
    for n in state.bfs_nodes(scope_entry):
        if n == state.exit_node(scope_entry):
            break
        nn.append(n)

    cut_sdfg = SDFGCutout.singlestate_cutout(state, *nn)
    return cut_sdfg


def get_flops_and_mem_access(sdfg, state, device_map_entry):
    mem_access = 0
    for e in state.in_edges(device_map_entry) + state.out_edges(state.exit_node(device_map_entry)):
        u, uc, v, vc, memlet = e
        acc = 1
        for (beg, end, step) in memlet.subset:
            acc *= (end+1-beg)/step
        mem_access += acc * sizeof(sdfg.arrays[memlet.data].dtype.as_ctypes())

    s = [device_map_entry]
    visited_guids = set()
    iter_length = 1
    flops = 0
    while s:
        n = s.pop(0)
        if n.guid in visited_guids:
            continue
        visited_guids.add(n.guid)
        if isinstance(n, dace.nodes.MapEntry):
            for (beg, end, step) in n.map.range:
                iter_length *= (end+1-beg)/step
        if isinstance(n, dace.nodes.Tasklet):
            #if has_floating_arithmetic_operation(n.code.as_string):
            #    flops += iter_length
            if not "assign" in n.label and not "_full" in n.label: # TODO: analyze tasklet
                flops += iter_length
        if isinstance(n, dace.nodes.MapExit):
            for (beg, end, step) in n.map.range:
                iter_length /= (end+1-beg)/step
        if n != state.exit_node(device_map_entry):
            for _, _, v, _, _ in state.out_edges(n):
                if not v.guid in visited_guids:
                    s.append(v)

    return (flops, mem_access)


def find_node_by_cond(state, start_map_entry, cond):
    s = set([start_map_entry])
    while s:
        n = s.pop()
        if n != start_map_entry and cond(n):
            return n
        if n != state.exit_node(start_map_entry):
            s = s.union([v for _, _, v, _, _ in state.out_edges(n)])
    return None

def find_node_in_state_by_cond(state, cond):
    for n in state.nodes():
        if cond(n):
            return n
    return None

def generate_random_data(kernel_sdfg : SDFG , defined_symbols : Dict[Type[str], Any]):
    randomly_generated_data = dict()
    kernel_sdfg_args = kernel_sdfg.arglist()
    for argname, arr in kernel_sdfg_args.items():
        if argname in defined_symbols or argname in randomly_generated_data:
            continue
        if isinstance(arr, dace.data.Array):
            shape = arr.shape
            ns = []
            for s in shape:
                if str(s) in defined_symbols:
                    ns.append(defined_symbols[str(s)])
                else:
                    ns.append(s)
            shape = tuple(ns)
            np_dtype = dace.dtypes.typeclass.as_numpy_dtype(arr.dtype)
            new_input = cupy.random.rand(*shape).astype(np_dtype)
            randomly_generated_data[argname] = new_input
        elif isinstance(arr, dace.data.Scalar):
            shape = (1, )
            np_dtype = dace.dtypes.typeclass.as_numpy_dtype(arr.dtype)
            new_input = cupy.empty(1, dtype=np_dtype)
            new_input[0] = random.random()
            randomly_generated_data[argname] = new_input[0]
        else:
            raise Exception(f"Input type, {type(arr)} is not dace.data.Array or dace.data.Scalar (not supported)")


    randomly_generated_data.update(defined_symbols)
    return randomly_generated_data

def solve(expr, defined_symbols):
    if isinstance(expr, int):
        return expr
    free_symbols = expr.free_symbols
    for sym in free_symbols:
        if str(sym) in defined_symbols:
            expr = expr.subs(sym, defined_symbols[str(sym)])
    return dace.symbolic.simplify(expr)

def run_and_measure_time(kernel_sdfg : SDFG, inputs):
    assert(len(kernel_sdfg.nodes())==1)
    kernel_state = kernel_sdfg.nodes()[0]

    kernel_sdfg.instrument = dace.InstrumentationType.Timer
    for node in kernel_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            node.instrument = dace.InstrumentationType.GPU_Events

    time = 0.0
    for _ in range(5):
        kernel_sdfg(**inputs)
        report = kernel_sdfg.get_latest_report()
        report.process_events()

        final_list = next(iter(report.durations.values()))
        final_list = next(iter(final_list.values()))
        final_list = next(iter(final_list.values()))

        time += statistics.median(final_list)
    time /= 5.0
    return time

def percentage_peak(time, flops, mem_accessed, peak_flops, peak_bandwidh):
    op_intensity = flops / mem_accessed
    theo_max_perf = min(op_intensity*peak_bandwidh, peak_flops)
    my_perf = flops / (time/1e3)
    return (my_perf*100) / theo_max_perf

def percentage_bandwidth(time, mem_accessed, peak_bandwidh):
    my_bandwidth = mem_accessed / (time/1e3)
    return (my_bandwidth*100) / peak_bandwidh

def convert_inputs_to_gpu_storage(kernel_sdfg : SDFG):
    for state in kernel_sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and \
                node.map.schedule == dace.ScheduleType.GPU_Device:
                for in_edge in state.in_edges(node):
                    in_node,_,_,_,_ = in_edge
                    if isinstance(in_node, dace.nodes.AccessNode) and \
                        kernel_sdfg.arrays[in_node.data].storage != dace.StorageType.GPU_Global and \
                        not kernel_sdfg.arrays[in_node.data].transient:
                        kernel_sdfg.arrays[in_node.data].storage = dace.StorageType.GPU_Global

def set_transient(kernel_sdfg : SDFG):
    input_output_arrs = []
    for state in kernel_sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and \
                node.map.schedule == dace.ScheduleType.GPU_Device:
                for in_edge in state.in_edges(node):
                    in_node,_,_,_,_ = in_edge
                    if isinstance(in_node, dace.nodes.AccessNode):
                        input_output_arrs.append(in_node.data)
            elif isinstance(node, dace.nodes.MapExit) and \
                node.map.schedule == dace.ScheduleType.GPU_Device:
                for out_edge in state.out_edges(node):
                    _,_,out_node,_,_ = out_edge
                    if isinstance(out_node, dace.nodes.AccessNode):
                        input_output_arrs.append(out_node.data)
    for arr_name, arr in kernel_sdfg.arrays.items():
        if not arr_name in input_output_arrs:
            arr.transient = True