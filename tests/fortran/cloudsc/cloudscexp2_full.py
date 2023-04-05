# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import ast
import copy
import cupy as cp
import dace
from dace.frontend.fortran import fortran_parser
from dace.frontend.python import astutils
from dace.sdfg import utils
from dace.transformation import helpers
from dace.transformation.auto.auto_optimize import auto_optimize, greedy_fuse, tile_wcrs
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes import RemoveUnusedSymbols, ScalarToSymbolPromotion, ScalarFission
from importlib import import_module
import numpy as np
from numbers import Integral, Number
from numpy import f2py
import os
import pytest
import sys
import tempfile
from typing import Dict, Union

# Transformations
from dace.transformation.dataflow import MapCollapse, TrivialMapElimination, MapFusion, ReduceExpansion
from dace.transformation.interstate import LoopToMap, RefineNestedAccess, MoveLoopIntoMap, LoopUnroll, TrivialLoopElimination, GPUTransformSDFG
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.subgraph import helpers as xfsh
from dace.transformation import helpers as xfh


from dace.transformation.dataflow import MapFission, RemoveIntermediateWrite
from dace.transformation.interstate import InlineSDFG
from dace.transformation.interstate.move_loop_into_map import MoveMapIntoLoop, MoveMapIntoIf
from typing import List, Set, Tuple


from cupyx.profiler import benchmark


def find_defined_symbols(sdfg: dace.SDFG) -> Set[str]:
    return sdfg.symbols.keys() - sdfg.free_symbols


def fix_sdfg_symbols(sdfg: dace.SDFG):
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                fix_sdfg_symbols(node.sdfg)
    for s in find_defined_symbols(sdfg):
        del sdfg.symbols[s]
        if sdfg.parent is not None and s in sdfg.parent_nsdfg_node.symbol_mapping:
            del sdfg.parent_nsdfg_node.symbol_mapping[s]


def fix_arrays(sdfg: dace.SDFG):
    repl_dict = {'_for_it_0': 1}
    for sd in sdfg.all_sdfgs_recursive():
        sd.replace_dict(repl_dict, replace_in_graph=False, replace_keys=False)


def fix_sdfg_parents(sdfg: dace.SDFG):
    for sd in sdfg.all_sdfgs_recursive():
        if sd.parent is not None:
            if sd.parent.parent != sd.parent_sdfg:
                print(f"Fixing parent of {sd.label}")
                sd.parent_sdfg = sd.parent.parent


def find_toplevel_maps(sdfg: dace.SDFG) -> List[Tuple[dace.SDFG, dace.SDFGState, dace.nodes.MapEntry]]:
    map_entries = []
    for state in sdfg.nodes():
        scope_dict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and scope_dict[node] is None:
                map_entries.append((sdfg, state, node))
            elif isinstance(node, dace.nodes.NestedSDFG) and scope_dict[node] is None:
                map_entries.extend(find_toplevel_maps(node.sdfg))
    return map_entries


def count_sdfg_transient_memory(sdfg: dace.SDFG) -> int:
    memory = 0
    # for _, desc in sdfg.arrays.items():
    #     if desc.transient:
    #         memory += desc.total_size
    for sd in sdfg.all_sdfgs_recursive():
        for _, desc in sd.arrays.items():
            if desc.transient:
                memory += desc.total_size
    return memory


def count_map_transient_memory(sdfg: dace.SDFG, state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> int:
    memory = 0
    scope_children = state.scope_children()[map_entry]
    for node in scope_children:
        # TODO: This is too broad.
        if isinstance(node, dace.nodes.AccessNode) and node.desc(sdfg).transient:
            memory += sdfg.arrays[node.data].total_size
        elif isinstance(node, dace.nodes.MapEntry):
            memory += count_map_transient_memory(sdfg, state, node)
        elif isinstance(node, dace.nodes.NestedSDFG):
            memory += count_sdfg_transient_memory(node.sdfg)
    return memory


def fission_sdfg(sdfg: dace.SDFG, name: str = None, iteration: int = 0) -> int:

    if name:
        name = f"{name}_"
    else:
        name = ""
    top_level_maps = find_toplevel_maps(sdfg)
    count = 0
    for sd, state, map_entry in top_level_maps:
        mem = count_map_transient_memory(sd, state, map_entry)
        if dace.symbolic.issymbolic(mem) or mem > 100:
            print(f"[{sd.label}, {state}, {map_entry}]: {mem} {'(TO FISSION)' if dace.symbolic.issymbolic(mem) else ''}")
            scope_children = state.scope_children()[map_entry]
            if len(scope_children) == 2 and isinstance(scope_children[0], dace.nodes.NestedSDFG):
                try:
                    MapFission.apply_to(sd, expr_index=1, map_entry=map_entry, nested_sdfg=scope_children[0])
                except ValueError:
                    try:
                        MoveMapIntoLoop.apply_to(sd, map_entry=map_entry, nested_sdfg=scope_children[0], map_exit=scope_children[1])
                    except ValueError:
                        try:
                            MoveMapIntoIf.apply_to(sd, map_entry=map_entry, nested_sdfg=scope_children[0], map_exit=scope_children[1])
                        except ValueError:
                            print("Map cannot be moved into loop")
                            continue
            else:
                try:
                    MapFission.apply_to(sd, expr_index=0, map_entry=map_entry)
                except ValueError:
                    print("Map cannot be fissioned (expr_index=0)")
                    continue
            count += 1
            sdfg.save(f'{name}interim_step_{iteration}.sdfg')
            # sdfg.validate()
            sdfg.simplify()
            sdfg.apply_transformations_repeated(RemoveIntermediateWrite)
            sd.apply_transformations_repeated(MapCollapse)
            sdfg.simplify()
            sdfg.apply_transformations_repeated(RemoveIntermediateWrite)
            # fix_sdfg_symbols(sdfg)
            # fix_arrays(sdfg)
            sdfg.save(f'{name}fission_step_{iteration}.sdfg')
    return count


def read_source(filename: str, extension: str = 'f90') -> str:
    source = None
    with open(os.path.join(os.path.dirname(__file__), f'{filename}.{extension}'), 'r') as file:
        source = file.read()
    assert source
    return source


def get_fortran(source: str, program_name: str, subroutine_name: str, fortran_extension: str = '.f90'):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        f2py.compile(source, modulename=program_name, extra_args=["--opt='-fdefault-real-8'"], verbose=True, extension=fortran_extension)
        sys.path.append(tmp_dir)
        module = import_module(program_name)
        function = getattr(module, subroutine_name.lower())
        os.chdir(cwd)
        return function


def get_sdfg(source: str, program_name: str, normalize_offsets: bool = False) -> dace.SDFG:

    source_fixed=source.replace("_JPRB","")
    intial_sdfg = fortran_parser.create_sdfg_from_string(source_fixed, program_name)
    
    # Find first NestedSDFG
    sdfg = None
    for state in intial_sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                sdfg = node.sdfg
                break
    if not sdfg:
        raise ValueError("SDFG not found.")

    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()

    # if normalize_offsets:
    #     my_simplify = Pipeline([RemoveUnusedSymbols(), ScalarToSymbolPromotion()])
    # else:
    #     my_simplify = Pipeline([RemoveUnusedSymbols()])
    # my_simplify.apply_pass(sdfg, {})

    # if normalize_offsets:
    #     utils.normalize_offsets(sdfg)

    # for sd in sdfg.all_sdfgs_recursive():
    #     sd.replace('NCLV', '5')
    #     sd.replace('NCLDQL', '1')
    #     sd.replace('NCLDQI', '2')
    #     sd.replace('NCLDQR', '3')
    #     sd.replace('NCLDQS', '4')
    #     sd.replace('NCLDQV', '5')
    #     #sdfg.add_constant('NCLDTOP', 15)

    #     for state in sd.nodes():
    #         for edge in list(state.edges()):
    #             if not edge.data.is_empty() and edge.data.data == 'NSSOPT':
    #                 if edge in state.edges():
    #                     state.remove_memlet_path(edge, remove_orphans=True)
    #     sd.replace('NSSOPT', '1')
    #     for state in sd.nodes():
    #         for edge in list(state.edges()):
    #             if not edge.data.is_empty() and edge.data.data == 'NCLDTOP':
    #                 if edge in state.edges():
    #                     state.remove_memlet_path(edge, remove_orphans=True)
    #     sd.replace('NCLDTOP', '15')

    repl_dict = {'KLON': 'NPROMA',
                 'NCLV': '5',
                 'NCLDQL': '1',
                 'NCLDQI': '2',
                 'NCLDQR': '3',
                 'NCLDQS': '4',
                 'NCLDQV': '5',
                 'NSSOPT': '1',
                 'NCLDTOP': '15',
                 'KLEV':'137',
                 'KFLDX': '1',}

    # Verify and fix NSSOPT and NCLDTOP
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.nodes():
            for edge in list(state.edges()):
                if not edge.data.is_empty() and edge.data.data in ('NSSOPT', 'NCLDTOP'):
                    print(f"Found {edge.data.data} in {sd.name}, {state.name}, {edge}, and fixing ...")
                    if edge in state.edges():
                        state.remove_memlet_path(edge, remove_orphans=True)
    # Promote/replace symbols
    for sd in sdfg.all_sdfgs_recursive():
        sd.replace_dict(repl_dict)
        if sd.parent_nsdfg_node is not None:
            for k in repl_dict.keys():
                if k in sd.parent_nsdfg_node.symbol_mapping:
                    del sd.parent_nsdfg_node.symbol_mapping[k]
        for k in repl_dict.keys():
            if k in sd.symbols:
                del sd.symbols[k]
    # Verify promotion/replacement
    for sd in sdfg.all_sdfgs_recursive():
        assert not any(k in sd.symbols for k in repl_dict.keys())
        assert not any(k in str(s) for s in sd.free_symbols for k in repl_dict.keys())
        for state in sd.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    assert not any(k in node.code.as_string for k in repl_dict.keys())
                elif isinstance(node, dace.nodes.NestedSDFG):
                    for s, m in node.symbol_mapping.items():
                        assert not any(k in str(s) for k in repl_dict.keys())
                        assert not any(k in str(s) for s in m.free_symbols for k in repl_dict.keys())
    
    for sd in sdfg.all_sdfgs_recursive():
        promoted = ScalarToSymbolPromotion().apply_pass(sd, {})
        print(f"Promoted the following scalars: {promoted}")
    from dace.sdfg import utils
    utils.normalize_offsets(sdfg)
    sdfg.simplify(verbose=True)

    repl_dict = {'NPROMA': '1'}
    # Promote/replace symbols
    for sd in sdfg.all_sdfgs_recursive():
        sd.replace_dict(repl_dict)
        if sd.parent_nsdfg_node is not None:
            for k in repl_dict.keys():
                if k in sd.parent_nsdfg_node.symbol_mapping:
                    del sd.parent_nsdfg_node.symbol_mapping[k]
        for k in repl_dict.keys():
            if k in sd.symbols:
                del sd.symbols[k]
    # Verify promotion/replacement
    for sd in sdfg.all_sdfgs_recursive():
        assert not any(k in sd.symbols for k in repl_dict.keys())
        assert not any(k in str(s) for s in sd.free_symbols for k in repl_dict.keys())
        for state in sd.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.Tasklet):
                    assert not any(k in node.code.as_string for k in repl_dict.keys())
                elif isinstance(node, dace.nodes.NestedSDFG):
                    for s, m in node.symbol_mapping.items():
                        assert not any(k in str(s) for k in repl_dict.keys())
                        assert not any(k in str(s) for s in m.free_symbols for k in repl_dict.keys())

    sdfg.simplify(verbose=True)

    sdfg.save('CLOUDSCOUTER_before_loop_elimination.sdfg')

    helpers.split_interstate_edges(sdfg)
    sdfg.apply_transformations_repeated(TrivialLoopElimination, validate=False)
    sdfg.save('CLOUDSCOUTER_loops_eliminated_internal.sdfg')

    pipeline = Pipeline([ScalarFission()])
    for sd in sdfg.all_sdfgs_recursive():
        results = pipeline.apply_pass(sd, {})[ScalarFission.__name__]
    
    sdfg.simplify(verbose=True)

    return sdfg


def validate_sdfg(sdfg, inputs, outputs, outputs_f, permutation=None) -> bool:
    outputs_d = copy.deepcopy(outputs)
    sdfg(**inputs, **outputs_d)

    if permutation is not None:
        outputs_d = unpermute_output(outputs_d, permutation)

    success = True
    for k in outputs_f.keys():
        farr = outputs_f[k]
        darr = outputs_d[k]
        if np.allclose(farr, darr):
            print(f"{k}: OK!")
        else:
            print(f"{k}: relative error is {np.linalg.norm(farr - darr) / np.linalg.norm(farr)}")
            success = False
    
    return success
        # assert np.allclose(farr, darr)


def debug_auto_optimize(sdfg: dace.SDFG, inputs, outputs, ref_outputs):

    auto_optimize(sdfg, dace.DeviceType.Generic)

    if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
        return

    fix_sdfg_symbols(sdfg)
    count = 1
    iteration = 0
    while count > 0:
        count = fission_sdfg(sdfg, sdfg.name, iteration)
        print(f"Fissioned {count} maps")
        if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
            return
        iteration += 1
    sdfg.simplify()
    sdfg.save('CLOUDSC_final.sdfg')

    return validate_sdfg(sdfg, inputs, outputs, ref_outputs)


    device = dace.DeviceType.Generic
    validate = False
    validate_all = False

    sdfg.save(f"{sdfg.name}_autoopt_0.sdfg")
    if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
        return


    # Simplification and loop parallelization
    transformed = True
    sdfg.apply_transformations_repeated(TrivialMapElimination, validate=validate, validate_all=validate_all)

    sdfg.save(f"{sdfg.name}_autoopt_1.sdfg")
    if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
        return

    i = 2
    while transformed:
        sdfg.simplify(validate=False, validate_all=validate_all)

        sdfg.save(f"{sdfg.name}_autoopt_{i}.sdfg")
        i += 1
        if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
            return

        for s in sdfg.sdfg_list:
            xfh.split_interstate_edges(s)
        l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                   validate=False,
                                                   validate_all=validate_all,
                                                   func=validate_sdfg, args=[inputs, outputs, ref_outputs])
        transformed = l2ms > 0

        sdfg.save(f"{sdfg.name}_autoopt_{i}.sdfg")
        i += 1
        if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
            return

    # Collapse maps and eliminate trivial dimensions
    sdfg.simplify()

    sdfg.save(f"{sdfg.name}_autoopt_{i}.sdfg")
    i += 1
    if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
        return

    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)
    sdfg.save(f"{sdfg.name}_autoopt_{i}.sdfg")
    i += 1
    if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
        return

    # fuse subgraphs greedily
    sdfg.simplify()

    sdfg.save(f"{sdfg.name}_autoopt_{i}.sdfg")
    i += 1
    if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
        return

    greedy_fuse(sdfg, device=device, validate_all=validate_all)

    sdfg.save(f"{sdfg.name}_autoopt_{i}.sdfg")
    i += 1
    if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
        return

    # fuse stencils greedily
    greedy_fuse(sdfg, device=device, validate_all=validate_all, recursive=False, stencil=True)

    sdfg.save(f"{sdfg.name}_autoopt_{i}.sdfg")
    i += 1
    if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
        return

    # Move Loops inside Maps when possible
    from dace.transformation.interstate import MoveLoopIntoMap
    # sdfg.apply_transformations_repeated([MoveLoopIntoMap])

    # # Apply GPU transformations and set library node implementations
    # if device == dtypes.DeviceType.GPU:
    #     sdfg.apply_gpu_transformations()
    #     sdfg.simplify()

    # if device == dtypes.DeviceType.FPGA:
    #     # apply FPGA Transformations
    #     sdfg.apply_fpga_transformations()
    #     fpga_auto_opt.fpga_global_to_local(sdfg)
    #     fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

    #     # Set all library nodes to expand to fast library calls
    #     set_fast_implementations(sdfg, device)
    #     return sdfg

    # Tiled WCR and streams
    for nsdfg in list(sdfg.all_sdfgs_recursive()):
        tile_wcrs(nsdfg, validate_all)
    
    sdfg.save(f"{sdfg.name}_autoopt_{i}.sdfg")
    i += 1
    if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
        return

    # Collapse maps
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)
    for node, _ in sdfg.all_nodes_recursive():
        # Set OMP collapse property to map length
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            # FORNOW: Leave out
            # node.map.collapse = len(node.map.range)
            pass
    
    sdfg.save(f"{sdfg.name}_autoopt_{i}.sdfg")
    i += 1
    if not validate_sdfg(sdfg, inputs, outputs, ref_outputs):
        return

    # if device == dtypes.DeviceType.Generic:
    #     # Validate at the end
    #     if validate or validate_all:
    #         sdfg.validate()

    #     return sdfg
    return sdfg


def change_data_layout(sdfg: dace.SDFG):

    def _permute(before, perm):
        return type(before)([before[i] for i in perm])
    
    perms = dict()
    for name, desc in sdfg.arrays.items():

        # We target arrays that have NBLOCKS and KLEV or KLEV + 1 in their shape
        shape_str = [str(s) for s in desc.shape]
        try:
            nblocks_idx = shape_str.index('NBLOCKS')
        except ValueError:
            continue

        if '137' in shape_str:
            klev_idx = shape_str.index('137')
        elif '138' in shape_str:
            klev_idx = shape_str.index('138')
        else:
            continue

        # Change the layout
        dl_perm = list(range(len(desc.shape)))
        dl_perm[klev_idx] = nblocks_idx
        dl_perm[nblocks_idx] = klev_idx
        desc.shape = _permute(desc.shape, dl_perm)
        # desc.strides = _permute(desc.strides, dl_perm)
        strides = [1]
        for i in range(1, len(desc.shape)):
            strides.append(strides[i-1] * desc.shape[i-1])
        desc.strides = tuple(strides)

        perms[name] = dl_perm

    # Change memlets and recurse in nested SDFGs
    for state in sdfg.nodes():
        visited = set()
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.AccessNode) and node.data in perms:
                for e0 in state.all_edges(node):
                    if e0 in visited:
                        continue
                    for e1 in state.memlet_tree(e0):
                        if e1 in visited:
                            continue
                        visited.add(e1)
                        if e1.data.data == node.data:
                            e1.data.subset.ranges = _permute(e1.data.subset.ranges, perms[node.data])
                        else:
                            e1.data.other_subset.ranges = _permute(e1.data.other_subset.ranges, perms[node.data])
                    visited.add(e0)
            # elif isinstance(node, dace.sdfg.nodes.NestedSDFG):
            #     change_data_layout(node.sdfg)
    
    # Change InterstateEdges
    for edge in sdfg.edges():
        memlets = edge.data.get_read_memlets(sdfg.arrays)
        for m in memlets:
            if m.data in perms:
                m.subset.ranges = _permute(m.subset.ranges, perms[m.data])
        for node in ast.walk(edge.data.condition.code[0]):
            if isinstance(node, ast.Subscript):
                m = memlets.pop(0)
                subscript: ast.Subscript = ast.parse(str(m)).body[0].value
                assert isinstance(node.value, ast.Name) and node.value.id == m.data
                node.slice = ast.copy_location(subscript.slice, node.slice)
        edge.data._cond_sympy = None
        for k, v in edge.data.assignments.items():
            vast = ast.parse(v)
            for node in ast.walk(vast):
                if isinstance(node, ast.Subscript):
                    m = memlets.pop(0)
                    subscript: ast.Subscript = ast.parse(str(m)).body[0].value
                    assert isinstance(node.value, ast.Name) and node.value.id == m.data
                    node.slice = ast.copy_location(subscript.slice, node.slice)
            newv = astutils.unparse(vast)
            edge.data.assignments[k] = newv
        assert not memlets
    
    for sd in sdfg.all_sdfgs_recursive():
        if sd is sdfg:
            continue
        for nname, ndesc in sd.arrays.items():
            if nname in perms:
                shape_str = [str(s) for s in ndesc.shape]
                if '137' in shape_str:
                    nklev_idx = shape_str.index('137')
                elif '138' in shape_str:
                    nklev_idx = shape_str.index('138')
                else:
                    continue
                odesc = sdfg.arrays[nname]
                shape_str = [str(s) for s in odesc.shape]
                if 'NBLOCKS' in shape_str:
                    onblocks_idx = shape_str.index('NBLOCKS')
                else:
                    continue
                if '137' in shape_str:
                    oklev_idx = shape_str.index('137')
                elif '138' in shape_str:
                    oklev_idx = shape_str.index('138')
                else:
                    continue
                
                if len(ndesc.strides) > 3:
                    print(f"{nname}: {ndesc.shape}, {ndesc.strides}")
                old_strides = list(ndesc.strides)
                old_strides[nklev_idx] = odesc.strides[oklev_idx]
                if len(ndesc.strides) > 3 and len(odesc.strides) > 3:
                    old_strides[nklev_idx + 1] = odesc.strides[oklev_idx + 1]
                ndesc.strides = tuple(old_strides)
                if len(ndesc.strides) > 3:
                    print(f"{nname}: {ndesc.shape}, {ndesc.strides}")
                    print()
                if 'NBLOCKS' not in sd.parent_nsdfg_node.symbol_mapping:
                    sd.parent_nsdfg_node.symbol_mapping['NBLOCKS'] = 'NBLOCKS'
                    sd.add_symbol('NBLOCKS', dace.int32)

    return perms


def unpermute_output(output: Dict[str, np.ndarray], perm: Dict[str, List[int]]) -> Dict[str, np.ndarray]:

    unpermuted = dict()
    for name, arr in output.items():
        if name in perm:
            unpermuted[name] = arr.transpose(perm[name]).reshape(arr.shape)
        else:
            unpermuted[name] = arr
    
    return unpermuted

nbvalue = 32768


parameters = {
    'KLON': 1,  # Should be equal to NPROMA
    # 'KLON': 128,
    'KLEV': 137,
    'KIDIA': 1,
    'KFDIA': 137,
    # 'KFDIA': 128,
    'KFLDX': 1,
    # 'KFLDX': 25,

    'NCLV': 5,
    'NCLDQI': 2,
    'NCLDQL': 1,
    'NCLDQR': 3,
    'NCLDQS': 4,
    'NCLDQV': 5,
    'NCLDTOP': 15,
    'NSSOPT': 1,
    'NAECLBC': 1,
    'NAECLDU': 1,
    'NAECLOM': 1,
    'NAECLSS': 1,
    'NAECLSU': 1,
    'NCLDDIAG': 1,
    'NAERCLD': 1,

    'NBLOCKS': nbvalue,
    # 'NBLOCKS': 512,
    # 'LDMAINCALL': np.bool_(True),  # boolean (LOGICAL)
    # 'LDSLPHY': np.bool_(True),  # boolean (LOGICAL),
    # 'LAERLIQAUTOCP': np.bool_(True),  # boolean (LOGICAL)
    # 'LAERLIQAUTOCPB': np.bool_(True),  # boolean (LOGICAL)
    # 'LAERLIQAUTOLSP': np.bool_(True),  # boolean (LOGICAL)
    # 'LAERLIQCOLL': np.bool_(True),  # boolean (LOGICAL)
    # 'LAERICESED': np.bool_(True),  # boolean (LOGICAL)
    # 'LAERICEAUTO': np.bool_(True),  # boolean (LOGICAL)
    # 'LCLDEXTRA': np.bool_(True),  # boolean (LOGICAL)
    # 'LCLDBUDGET': np.bool_(True),  # boolean (LOGICAL)
    'LDMAINCALL': np.int32(np.bool_(True)),  # boolean (LOGICAL)
    'LDSLPHY': np.int32(np.bool_(True)),  # boolean (LOGICAL),
    'LAERLIQAUTOCP': np.int32(np.bool_(True)),  # boolean (LOGICAL)
    'LAERLIQAUTOCPB': np.int32(np.bool_(True)),  # boolean (LOGICAL)
    'LAERLIQAUTOLSP': np.int32(np.bool_(True)),  # boolean (LOGICAL)
    'LAERLIQCOLL': np.int32(np.bool_(True)),  # boolean (LOGICAL)
    'LAERICESED': np.int32(np.bool_(True)),  # boolean (LOGICAL)
    'LAERICEAUTO': np.int32(np.bool_(True)),  # boolean (LOGICAL)
    'LCLDEXTRA': np.int32(np.bool_(True)),  # boolean (LOGICAL)
    'LCLDBUDGET': np.int32(np.bool_(True)),  # boolean (LOGICAL)
    'NGPBLKS': 10,
    'NUMOMP': 10,
    'NGPTOT': nbvalue*1,
    # 'NGPTOT': 65536,
    'NGPTOTG': nbvalue*1,
    # 'NGPTOTG': 65536,
    'NPROMA': 1,
    # 'NPROMA': 128,
    'NBETA': nbvalue,
}


data = {
    'NSHAPEP': (0,),
    'NSHAPEQ': (0,),
    'PTSPHY': (0,),
    'R2ES': (0,),
    'R3IES': (0,),
    'R3LES': (0,),
    'R4IES': (0,),
    'R4LES': (0,),
    'R5ALSCP': (0,),
    'R5ALVCP': (0,),
    'R5IES': (0,),
    'R5LES': (0,),
    'RALFDCP': (0,),
    'RALSDCP': (0,),
    'RALVDCP': (0,),
    'RAMID': (0,),
    'RAMIN': (0,),
    'RBETA': (0,),
    'RBETAP1': (0,),
    'RCCN': (0,),
    'RCCNOM': (0,),
    'RCCNSS': (0,),
    'RCCNSU': (0,),
    'RCL_AI': (0,),
    'RCL_BI': (0,),
    'RCL_CI': (0,),
    'RCL_DI': (0,),
    'RCL_X1I': (0,),
    'RCLCRIT': (0,),
    'RCLCRIT_LAND': (0,),
    'RCLCRIT_SEA': (0,),
    'RCLDIFF': (0,),
    'RCLDIFF_CONVI': (0,),
    'RCLDMAX': (0,),
    'RCLDTOPCF': (0,),
    'RCLDTOPP': (0,),
    'RCOVPMIN': (0,),
    'RCPD': (0,),
    'RD': (0,),
    'RDEPLIQREFDEPTH': (0,),
    'RDEPLIQREFRATE': (0,),
    'RETV': (0,),
    'RG': (0,),
    'RICEHI1': (0,),
    'RICEHI2': (0,),
    'RICEINIT': (0,),
    'RKCONV': (0,),
    'RKOOP1': (0,),
    'RKOOP2': (0,),
    'RKOOPTAU': (0,),
    'RLCRITSNOW': (0,),
    'RLMIN': (0,),
    'RLMLT': (0,),
    'RLSTT': (0,),
    'RLVTT': (0,),
    'RNICE': (0,),
    'RPECONS': (0,),
    'RPRC1': (0,),
    'RPRC2': (0,),
    'RPRECRHMAX': (0,),
    'RSNOWLIN1': (0,),
    'RSNOWLIN2': (0,),
    'RTAUMEL': (0,),
    'RTHOMO': (0,),
    'RTICE': (0,),
    'RTICECU': (0,),
    'RTT': (0,),
    'RTWAT': (0,),
    'RTWAT_RTICE_R': (0,),
    'RTWAT_RTICECU_R': (0,),
    'RV': (0,),
    'RVICE': (0,),
    'RVRAIN': (0,),
    'RVRFACTOR': (0,),
    'RVSNOW': (0,),
    'ZEPSEC': (0,),
    'ZEPSILON': (0,),
    'ZRG_R': (0,),
    'ZRLDCP': (0,),
    'ZQTMST': (0,),
    'ZVPICE': (0,),
    'ZVPLIQ': (0,),
    'RCL_KKAac': (0,),
    'RCL_KKBac': (0,),
    'RCL_KKAau': (0,),
    'RCL_KKBauq': (0,),
    'RCL_KKBaun': (0,),
    'RCL_KK_CLOUD_NUM_SEA': (0,),
    'RCL_KK_CLOUD_NUM_LAND': (0,),
    'RCL_CONST1I': (0,),
    'RCL_CONST2I': (0,),
    'RCL_CONST3I': (0,),
    'RCL_CONST4I': (0,),
    'RCL_CONST5I': (0,),
    'RCL_CONST6I': (0,),
    'RCL_APB1': (0,),
    'RCL_APB2': (0,),
    'RCL_APB3': (0,),
    'RCL_CONST1S': (0,),
    'RCL_CONST2S': (0,),
    'RCL_CONST3S': (0,),
    'RCL_CONST4S': (0,),
    'RCL_CONST5S': (0,),
    'RCL_CONST6S': (0,),
    'RCL_CONST7S': (0,),
    'RCL_CONST8S': (0,),
    'RDENSREF': (0,),
    'RCL_KA273': (0,),
    'RCL_CDENOM1': (0,),
    'RCL_CDENOM2': (0,),
    'RCL_CDENOM3': (0,),
    'RCL_CONST1R': (0,),
    'RCL_CONST2R': (0,),
    'RCL_CONST3R': (0,),
    'RCL_CONST4R': (0,),
    'RCL_FAC1': (0,),
    'RCL_FAC2': (0,),
    'RCL_CONST5R': (0,),
    'RCL_CONST6R': (0,),
    'RCL_FZRAB': (0,),
    'RCL_X2I': (0,),
    'RCL_X3I': (0,),
    'RCL_X4I': (0,),
    'RCL_AS': (0,),
    'RCL_BS': (0,),
    'RCL_CS': (0,),
    'RCL_DS': (0,),
    'RCL_X1S': (0,),
    'RCL_X2S': (0,),
    'RCL_X3S': (0,),
    'RCL_X4S': (0,),
    'RDENSWAT': (0,),
    'RCL_AR': (0,),
    'RCL_BR': (0,),
    'RCL_CR': (0,),
    'RCL_DR': (0,),
    'RCL_X1R': (0,),
    'RCL_X2R': (0,),
    'RCL_X4R': (0,),
    'RCL_SCHMIDT': (0,),
    'RCL_DYNVISC': (0,),
    'RCL_FZRBB': (0,),
    'IPHASE': (parameters['NCLV'],),
    'KTYPE': [(parameters['KLON'], parameters['NBLOCKS']), np.int32],
    'LDCUM': [(parameters['KLON'], parameters['NBLOCKS']), np.bool_],
    'PA': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PAP': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PAPH': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PCCN': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PCLV': (parameters['KLON'], parameters['KLEV'], parameters['NCLV'], parameters['NBLOCKS']),
    'PCOVPTOT': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PDYNA': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PDYNI': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PDYNL': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PEXTRA': (parameters['KLON'], parameters['KLEV'], parameters['KFLDX'], parameters['NBLOCKS']),
    'PFCQLNG': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFCQNNG': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFCQRNG': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFCQSNG': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFHPSL': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFHPSN': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFPLSL': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFPLSN': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFSQIF': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFSQITUR': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFSQLF': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFSQLTUR': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFSQRF': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PFSQSF': (parameters['KLON'], parameters['KLEV']+1, parameters['NBLOCKS']),
    'PICRIT_AER': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PLCRIT_AER': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PHRLW': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PLSM': (parameters['KLON'], parameters['NBLOCKS']),
    'PHRSW': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PLU': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PLUDE': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PMFD': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PMFU': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PNICE': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PRAINFRAC_TOPRFZ': (parameters['KLON'], parameters['NBLOCKS']),
    'PQ': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PRE_ICE': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PSNDE': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PSUPSAT': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PT': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PVERVEL': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PVFA': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PVFI': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PVFL': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_a': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV'], parameters['NBLOCKS']),
    'tendency_cml_o3': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_q': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_T': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_u': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_v': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_a': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV'], parameters['NBLOCKS']),
    'tendency_loc_o3': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_q': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_T': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_u': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_v': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_a': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV'], parameters['NBLOCKS']),
    'tendency_tmp_o3': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_q': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_T': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_u': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_v': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'ZA': (parameters['KLON'], parameters['KLEV']),
    'ZAORIG': (parameters['KLON'], parameters['KLEV']),
    'ZCLDTOPDIST': (parameters['KLON'],),
    'ZCONVSINK': (parameters['KLON'], parameters['NCLV']),
    'ZCONVSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZCORQSICE': (parameters['KLON']),
    'ZCORQSLIQ': (parameters['KLON']),
    'ZCOVPTOT': (parameters['KLON'],),
    'ZDA': (parameters['KLON']),
    'ZCOVPCLR': (parameters['KLON'],),
    'ZCOVPMAX': (parameters['KLON'],),
    'ZDTGDP': (parameters['KLON'],),
    'ZICENUCLEI': (parameters['KLON'],),
    'ZRAINCLD': (parameters['KLON'],),
    'ZSNOWCLD': (parameters['KLON'],),
    'ZDA': (parameters['KLON'],),
    'ZDP': (parameters['KLON'],),
    'ZRHO': (parameters['KLON'],),
    'ZFALLSINK': (parameters['KLON'], parameters['NCLV']),
    'ZFALLSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZFOKOOP': (parameters['KLON'],),
    'ZFLUXQ': (parameters['KLON'], parameters['NCLV']),
    'ZFOEALFA': (parameters['KLON'], parameters['KLEV']+1),
    'ZICECLD': (parameters['KLON'],),
    'ZICEFRAC': (parameters['KLON'], parameters['KLEV']),
    'ZICETOT': (parameters['KLON'],),
    'ZLI': (parameters['KLON'], parameters['KLEV']),
    'ZLIQFRAC': (parameters['KLON'], parameters['KLEV']),
    'ZLNEG': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZPFPLSX': (parameters['KLON'], parameters['KLEV']+1, parameters['NCLV']),
    'ZPSUPSATSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZSOLQA': (parameters['KLON'], parameters['NCLV'], parameters['NCLV']),
    'ZMELTMAX': (parameters['KLON'],),
    'ZQPRETOT': (parameters['KLON'],),
    'ZQSLIQ': (parameters['KLON'], parameters['KLEV']),
    'ZQSICE': (parameters['KLON'], parameters['KLEV']),
    'ZQX': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZQX0': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZQXFG': (parameters['KLON'], parameters['NCLV']),
    'ZQXN': (parameters['KLON'], parameters['NCLV']),
    'ZQXN2D': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZSOLAC': (parameters['KLON'],),
    'ZSUPSAT': (parameters['KLON'],),
    'ZTP1': (parameters['KLON'], parameters['KLEV']),
    # 'DEBUG_EPSILON': (2,),
}



programs = {
    'cloudscexp2_full_20230324': ('CLOUDPROGRAM', 'CLOUDSCOUTER')
}


program_parameters = {
    'cloudscexp2_full_20230324': (
        'NBLOCKS', 'NGPBLKS', 'NUMOMP', 'NGPTOT', 'NGPTOTG', 'NPROMA',
        'KLON', 'KLEV', 'KFLDX', 'LDSLPHY',  'LDMAINCALL',
        'NCLV', 'NCLDQL','NCLDQI','NCLDQR','NCLDQS', 'NCLDQV',
        'LAERLIQAUTOLSP', 'LAERLIQAUTOCP', 'LAERLIQAUTOCPB','LAERLIQCOLL','LAERICESED','LAERICEAUTO',
        'LCLDEXTRA', 'LCLDBUDGET', 'NSSOPT', 'NCLDTOP',
        'NAECLBC', 'NAECLDU', 'NAECLOM', 'NAECLSS', 'NAECLSU', 'NCLDDIAG', 'NAERCLD', 'LAERLIQAUTOLSP',
        'LAERLIQAUTOCP', 'LAERLIQAUTOCPB','LAERLIQCOLL','LAERICESED','LAERICEAUTO', 'NBETA')
}


program_inputs = {
    'cloudscexp2_full_20230324': (
        'PTSPHY','PT', 'PQ',
        'tendency_cml_a', 'tendency_cml_cld', 'tendency_cml_o3', 'tendency_cml_q',
        'tendency_cml_T', 'tendency_cml_u', 'tendency_cml_v',
        'tendency_loc_a', 'tendency_loc_cld', 'tendency_loc_o3', 'tendency_loc_q',
        'tendency_loc_T', 'tendency_loc_u', 'tendency_loc_v',
        'tendency_tmp_a', 'tendency_tmp_cld', 'tendency_tmp_o3', 'tendency_tmp_q',
        'tendency_tmp_T', 'tendency_tmp_u', 'tendency_tmp_v',
        'PVFA', 'PVFL', 'PVFI', 'PDYNA', 'PDYNL', 'PDYNI',
        'PHRSW', 'PHRLW',
        'PVERVEL',  'PAP',      'PAPH',
        'PLSM',     'LDCUM',    'KTYPE',
        'PLU',    'PSNDE',    'PMFU',     'PMFD',
  #!---prognostic fields
        'PA',
        'PCLV',
        'PSUPSAT',
#!-- arrays for aerosol-cloud interactions
#!!! & PQAER,    KAER, &
        'PLCRIT_AER','PICRIT_AER',
        'PRE_ICE',
        'PCCN',     'PNICE',
        'RG', 'RD', 'RCPD', 'RETV', 'RLVTT', 'RLSTT', 'RLMLT', 'RTT', 'RV', 
        'R2ES', 'R3LES', 'R3IES', 'R4LES', 'R4IES', 'R5LES', 'R5IES',
        'R5ALVCP', 'R5ALSCP', 'RALVDCP', 'RALSDCP', 'RALFDCP', 'RTWAT', 'RTICE', 'RTICECU',
        'RTWAT_RTICE_R', 'RTWAT_RTICECU_R', 'RKOOP1', 'RKOOP2',
        'RAMID',
        'RCLDIFF', 'RCLDIFF_CONVI', 'RCLCRIT','RCLCRIT_SEA', 'RCLCRIT_LAND','RKCONV',
        'RPRC1', 'RPRC2','RCLDMAX', 'RPECONS','RVRFACTOR', 'RPRECRHMAX','RTAUMEL', 'RAMIN',
        'RLMIN','RKOOPTAU', 'RCLDTOPP','RLCRITSNOW','RSNOWLIN1', 'RSNOWLIN2','RICEHI1',
        'RICEHI2', 'RICEINIT', 'RVICE','RVRAIN','RVSNOW','RTHOMO','RCOVPMIN', 'RCCN','RNICE',
        'RCCNOM', 'RCCNSS', 'RCCNSU', 'RCLDTOPCF', 'RDEPLIQREFRATE','RDEPLIQREFDEPTH',
        'RCL_KKAac', 'RCL_KKBac', 'RCL_KKAau','RCL_KKBauq', 'RCL_KKBaun',
        'RCL_KK_CLOUD_NUM_SEA', 'RCL_KK_CLOUD_NUM_LAND', 'RCL_AI', 'RCL_BI', 'RCL_CI',
        'RCL_DI', 'RCL_X1I', 'RCL_X2I', 'RCL_X3I','RCL_X4I', 'RCL_CONST1I', 'RCL_CONST2I',
        'RCL_CONST3I', 'RCL_CONST4I','RCL_CONST5I', 'RCL_CONST6I','RCL_APB1','RCL_APB2',
        'RCL_APB3', 'RCL_AS', 'RCL_BS','RCL_CS', 'RCL_DS','RCL_X1S', 'RCL_X2S','RCL_X3S',
        'RCL_X4S', 'RCL_CONST1S', 'RCL_CONST2S', 'RCL_CONST3S', 'RCL_CONST4S',
        'RCL_CONST5S', 'RCL_CONST6S','RCL_CONST7S','RCL_CONST8S','RDENSWAT', 'RDENSREF',
        'RCL_AR','RCL_BR', 'RCL_CR', 'RCL_DR', 'RCL_X1R', 'RCL_X2R', 'RCL_X4R','RCL_KA273',
        'RCL_CDENOM1', 'RCL_CDENOM2','RCL_CDENOM3','RCL_SCHMIDT','RCL_DYNVISC','RCL_CONST1R',
        'RCL_CONST2R', 'RCL_CONST3R', 'RCL_CONST4R', 'RCL_FAC1','RCL_FAC2', 'RCL_CONST5R',
        'RCL_CONST6R', 'RCL_FZRAB', 'RCL_FZRBB',
        'NSHAPEP', 'NSHAPEQ','RBETA','RBETAP1'      
    ),
}


program_outputs = {
    'cloudscexp2_full_20230324': (
        'PLUDE',
    #  !---diagnostic output
        'PCOVPTOT', 'PRAINFRAC_TOPRFZ',
#  !---resulting fluxes
        'PFSQLF',   'PFSQIF' ,  'PFCQNNG',  'PFCQLNG',
        'PFSQRF',   'PFSQSF' ,  'PFCQRNG',  'PFCQSNG',
        'PFSQLTUR', 'PFSQITUR' ,
        'PFPLSL',   'PFPLSN',   'PFHPSL',   'PFHPSN',
        'PEXTRA',
        # 'DEBUG_EPSILON'
    ),
}


def get_inputs(program: str, rng: np.random.Generator) -> Dict[str, Union[Number, np.ndarray]]:
    inp_data = dict()
    for p in program_parameters[program]:
        inp_data[p] = parameters[p]
    for inp in program_inputs[program]:
        if inp not in data:
            print(inp)
            continue
        info = data[inp]
        if isinstance(info, list):
            shape, dtype = info
        else:
            shape = info
            dtype = np.float64
        method = lambda s, d: rng.random(s, d)
        if issubclass(dtype, Integral) or dtype is np.bool_:
            if dtype is np.bool_:
                method = lambda s, d: rng.integers(0, 2, s, d)
                dtype = np.int32
            else:
                method = lambda s, d: rng.integers(0, 10, s, d)
        if shape == (0,):  # Scalar
            inp_data[inp] = method(None, dtype)
        else:
            inp_data[inp] = np.asfortranarray(method(shape, dtype))
    return inp_data


def get_outputs(program: str, rng: np.random.Generator) -> Dict[str, Union[Number, np.ndarray]]:
    out_data = dict()
    for out in program_outputs[program]:
        info = data[out]
        if isinstance(info, list):
            shape, dtype = info
        else:
            shape = info
            dtype = np.float64
        method = lambda s, d: rng.random(s, d)
        if issubclass(dtype, Integral) or dtype is np.bool_:
            if dtype is np.bool_:
                method = lambda s, d: rng.integers(0, 2, s, d)
                dtype = np.int32
            else:
                method = lambda s, d: rng.integers(0, 10, s, d)
        if shape == (0,):  # Scalar
            raise NotImplementedError
        else:
            out_data[out] = np.asfortranarray(method(shape, dtype))
    return out_data


def force_maps(sdfg: dace.SDFG):
    itervars = (f'_for_it_{i}' for i in (27, 30, 43, 46, 47, 52, 63, 64))
    helpers.split_interstate_edges(sdfg)
    for itervar in itervars:
        num = sdfg.apply_transformations_repeated(LoopToMap, options={'itervar': itervar}, permissive=True)
        print(f'Applied {num} LoopToMap')
    sdfg.simplify()


def move_loops(sdfg: dace.SDFG):
    helpers.split_interstate_edges(sdfg)
    num = sdfg.apply_transformations_repeated(MoveLoopIntoMap)
    print(f'Applied {num} transformations')
    sdfg.simplify()
    num = sdfg.apply_transformations_repeated(MapCollapse)
    print(f'Applied {num} transformations')
    sdfg.simplify()
    num = sdfg.apply_transformations_repeated(MapCollapse)
    print(f'Applied {num} transformations')
    sdfg.simplify()


@pytest.mark.skip
def test_program(program: str, device: dace.DeviceType, sdfg_id: int):

    fsource = read_source(program)
    program_name, routine_name = programs[program]
    ffunc = get_fortran(fsource, program_name, routine_name)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng)
    outputs_f = get_outputs(program, rng)
    outputs_d = copy.deepcopy(outputs_f)

    print("Running Fortran ...")
    ffunc(**{k.lower(): v for k, v in inputs.items()}, **{k.lower(): v for k, v in outputs_f.items()})

    if sdfg_id < 1:
        sdfg = get_sdfg(fsource, program_name, normalize_offsets=True)
        sdfg.save('CLOUDSCOUTER_simplify.sdfg')
        # sdfg.compile()
        print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f)}")
    
    if sdfg_id < 2:
        sdfg = dace.SDFG.from_file('CLOUDSCOUTER_simplify.sdfg')
        auto_optimize(sdfg, dace.DeviceType.Generic)
        sdfg.simplify()
        sdfg.save('CLOUDSCOUTER_autoopt.sdfg')
        # sdfg.compile()
        print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f)}")
    
    if sdfg_id < 3:
        sdfg = dace.SDFG.from_file('CLOUDSCOUTER_autoopt.sdfg')
        sdfg.apply_transformations_repeated(TrivialMapElimination)
        sdfg.simplify()
        sdfg.save('CLOUDSCOUTER_autoopt_map_elimination.sdfg')
        # sdfg.compile()
        print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f)}")
    
    if sdfg_id < 4:
        sdfg = dace.SDFG.from_file('CLOUDSCOUTER_autoopt_map_elimination.sdfg')
        force_maps(sdfg)
        sdfg.apply_transformations_repeated(TrivialMapElimination)
        sdfg.simplify()
        sdfg.save('CLOUDSCOUTER_autoopt_loops.sdfg')
        # sdfg.compile()
        print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f)}")
    
    if sdfg_id < 5:
        sdfg = dace.SDFG.from_file('CLOUDSCOUTER_autoopt_loops.sdfg')
        move_loops(sdfg)
        sdfg.simplify()
        sdfg.save('CLOUDSCOUTER_autoopt_loops_moved.sdfg')
        # sdfg.compile()
        print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f)}")

    if sdfg_id < 6:
        sdfg = dace.SDFG.from_file('CLOUDSCOUTER_autoopt_loops_moved.sdfg')
        for sd in sdfg.all_sdfgs_recursive():
            helpers.split_interstate_edges(sd)
        sdfg.apply_transformations_repeated(LoopUnroll)
        sdfg.simplify()
        sdfg.save('CLOUDSCOUTER_autoopt_loops_unrolled.sdfg')
        # sdfg.compile()
        print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f)}")

    
    sdfg = dace.SDFG.from_file('CLOUDSCOUTER_autoopt_loops_unrolled.sdfg')
    sdfg.simplify(verbose=True)
    print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f)}")

    greedy_fuse(sdfg, False)
    sdfg.simplify(verbose=True)
    sdfg.save('CLOUDSCOUTER_fuse.sdfg')
    # sdfg = dace.SDFG.from_file('CLOUDSCOUTER_fuse.sdfg')
    print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f)}")

    sdfg = dace.SDFG.from_file('CLOUDSCOUTER_fuse.sdfg')
    permutation = change_data_layout(sdfg)
    sdfg.save('CLOUDSCOUTER_layout.sdfg')
    print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f, permutation=permutation)}")

    exit(0)

    # inputs_dev = {k: cp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
    # outputs_dev = {k: cp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in outputs_d.items()}

    # def _to_gpu(sdfg: dace.SDFG):
    #     for _, desc in sdfg.arrays.items():
    #         if not desc.transient and isinstance(desc, dace.data.Array):
    #             desc.storage = dace.dtypes.StorageType.GPU_Global
    #     for state in sdfg.states():
    #         for node in state.nodes():
    #             if isinstance(node, dace.sdfg.nodes.MapEntry):
    #                 node.schedule = dace.ScheduleType.GPU_Device
    
    # def _func(exec, inputs, outputs):
    #     exec(**inputs, **outputs)
    
    # sdfg = dace.SDFG.from_file('CLOUDSCOUTER_fuse.sdfg')
    # _to_gpu(sdfg)
    # csdfg = sdfg.compile()
    # print(benchmark(_func, (csdfg, inputs_dev, outputs_dev), n_repeat=10, n_warmup=10))

    # sdfg = dace.SDFG.from_file('CLOUDSCOUTER_layout.sdfg')
    # # sdfg = dace.SDFG.from_file('CLOUDSCOUTER_fuse.sdfg')
    # permutation = change_data_layout(sdfg)
    # sdfg.save('CLOUDSCOUTER_layout.sdfg')
    # _to_gpu(sdfg)
    # csdfg = sdfg.compile()
    # print(benchmark(_func, (csdfg, inputs_dev, outputs_dev), n_repeat=10, n_warmup=10))


    # exit(0)



    # # fix_sdfg_symbols(sdfg)

    # for sd in sdfg.all_sdfgs_recursive():
    #     sd.openmp_sections = False

    # try:
    #     count = 1
    #     iteration = 0
    #     while count > 0:
    #         count = fission_sdfg(sdfg, sdfg.name, iteration)
    #         print(f"Fissioned {count} maps")
    #         iteration += 1
    #         # sdfg.compile()
    #         print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f)}")
    #     sdfg.simplify()
    # except:
    #     pass

    # last_iteration = iteration - 1
    # sdfg = dace.SDFG.from_file(f'CLOUDSCOUTER_fission_step_{last_iteration}.sdfg')

    # for sd, state, map_entry in find_toplevel_maps(sdfg):
    #     print(f"[{sd.label}, {state}, {map_entry}]: {count_map_transient_memory(sd, state, map_entry)}")

    # # sdfg = dace.SDFG.from_file('CLOUDSCOUTER_autoopt_loops_moved.sdfg')
    # sdfg = dace.SDFG.from_file('CLOUDSCOUTER_fission_step_2.sdfg')
    # # sdfg = dace.SDFG.from_file('CLOUDSCOUTER_autoopt_loops.sdfg')
    # # move_loops(sdfg)
    # count = 1
    # iteration = 3
    # while count > 0:
    #     count = fission_sdfg(sdfg, sdfg.name, iteration)
    #     print(f"Fissioned {count} maps")
    #     iteration += 1
    #     sdfg.compile()
    #     print(f"Validates? {validate_sdfg(sdfg, inputs, outputs_d, outputs_f)}")
    # sdfg.simplify()

    # for sd, state, map_entry in find_toplevel_maps(sdfg):
    #     print(f"[{sd.label}, {state}, {map_entry}]: {count_map_transient_memory(sd, state, map_entry)}")


    if device == dace.DeviceType.GPU:
        auto_optimize(sdfg, device)
    sdfg.simplify()
    # utils.make_dynamic_map_inputs_unique(sdfg)

    print("Running DaCe ...")
    sdfg(**inputs, **outputs_d)

    for k in outputs_f.keys():
        farr = outputs_f[k]
        darr = outputs_d[k]
        if np.allclose(farr, darr):
            print(f"{k}: OK!")
        else:
            print(f"{k}: relative error is {np.linalg.norm(farr - darr) / np.linalg.norm(farr)}")
        # assert np.allclose(farr, darr)
    
    # # print(outputs_f['DEBUG_EPSILON'])
    # # print(outputs_d['DEBUG_EPSILON'])
    # debug_auto_optimize(sdfg, inputs, outputs_d, outputs_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLOUDSC')
    parser.add_argument('-t',
                        '--target',
                        nargs="?",
                        choices=['cpu', 'gpu'],
                        default='cpu',
                        help='The target architecture.')
    parser.add_argument('-s',
                        '--sdfg_id',
                        type=int,
                        nargs="?",
                        default=7,
                        help='The SDFG to use.')
    args = parser.parse_args()
    
    device = dace.DeviceType.GPU if args.target == 'gpu' else dace.DeviceType.CPU
    sdfg_id = args.sdfg_id

    test_program('cloudscexp2_full_20230324', device, sdfg_id)
