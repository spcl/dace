import argparse
import copy
import dace
import os

from dace.frontend.fortran import fortran_parser
from dace.sdfg import utils
from dace.data import _prod
from dace.libraries.ttranspose import TensorTranspose
from dace.transformation import helpers
from dace.transformation.auto.auto_optimize import auto_optimize, greedy_fuse
from dace.transformation.dataflow import MapCollapse, TrivialMapElimination
from dace.transformation.interstate import LoopUnroll, MoveLoopIntoMap, TrivialLoopElimination
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes import ScalarFission, ScalarToSymbolPromotion
from typing import Dict, Set, Tuple, Union


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v: Union[str, bool]) -> bool:
    """
    Converts a boolean parameter potentially given as a string to boolean value.
    
    :param v: The boolean parameter.
    :return: The boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_source(filename: str, extension: str = 'f90') -> str:
    """
    Reads a Fortran source file from the current directory and returns its code as a string.

    :param filename: The name of the file to read.
    :param extension: The extension of the file to read.
    :return: The source code.
    """
    source = None
    with open(os.path.join(os.path.dirname(__file__), f'{filename}.{extension}'), 'r') as file:
        source = file.read()
    assert source
    return source


def get_unoptimized_sdfg(source: str, program_name: str) -> dace.SDFG:
    """
    Generates an unoptimized SDFG from a Fortran source code.
    
    :param source: The Fortran source code.
    :param program_name: The name of the program to generate.
    :return: The unoptimized SDFG.
    """
    source_fixed = source.replace("_JPRB", "")
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

    return sdfg


def simplify_sdfg(sdfg: dace.SDFG, specialize: bool) -> dace.SDFG:
    """
    Simplifies an SDFG. This operation applies in-place.
    
    :param sdfg: The SDFG to simplify.
    :param specialize: Whether to specialize the SDFG.
    :return: The simplified SDFG.
    """

    # Specialize
    if specialize:
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
                     'KFLDX': '1'}

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
    
    # Normalize offsets
    for sd in sdfg.all_sdfgs_recursive():
        promoted = ScalarToSymbolPromotion().apply_pass(sd, {})
        print(f"Promoted the following scalars: {promoted}")
    utils.normalize_offsets(sdfg)

    # Simplify (1)
    sdfg.simplify(verbose=True)

    # Specialize NPROMA
    if specialize:
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
    
    # Simplify (2)
    sdfg.simplify(verbose=True)

    # Eliminate trivial loops
    for sd in sdfg.all_sdfgs_recursive():
        helpers.split_interstate_edges(sd)
    sdfg.apply_transformations_repeated(TrivialLoopElimination, validate=False)

    # Rename (fission) scalars
    pipeline = Pipeline([ScalarFission()])
    for sd in sdfg.all_sdfgs_recursive():
        results = pipeline.apply_pass(sd, {})[ScalarFission.__name__]
        print(f"Fissioned the following scalars: {results}")
    
    # Simplify (3)
    sdfg.simplify(verbose=True)

    return sdfg


def change_strides(sdfg: dace.SDFG, klev_vals: Tuple[int], syms_to_add: Set[str] = None) -> Dict[str, int]:
    """
    Amends the arrays having a "KLEV" dimension so that this dimension has the largest stride.
    
    :param sdfg: The SDFG to change.
    :param klev_vals: The values of the "KLEV" dimension. Assists with identification of the "KLEV" dimension.
    :param syms_to_add: The symbols to add to the SDFG.
    :return: A dictionary mapping the names of the arrays to the new strides.
    """
    permutation = dict()
    syms_to_add = syms_to_add or set()

    print(f'TOP-LEVEL SDFG {sdfg.name}')

    # Gather inputs/outputs
    for state in sdfg.states():
        if state.name == 'stateCLOUDSC':
            main_state = state
        elif state.name == 'BeginFOR_l_1345_c_1345':
            init_state = state
        elif state.name == 'MergeFOR_l_1345_c_1345':
            exit_state = state


    inputs = set()
    outputs = set()

    for node in main_state.nodes():
        if isinstance(node, dace.sdfg.nodes.AccessNode):
            in_degree = main_state.in_degree(node)
            out_degree = main_state.out_degree(node)
            if in_degree > 0 and out_degree > 0:
                raise ValueError('Access node has both in and out edges')
            if in_degree == 0 and out_degree == 0:
                raise ValueError('Access node has no in or out edges')
            if in_degree > 0:
                outputs.add(node.data)
            else:
                inputs.add(node.data)
    
    pre_state = sdfg.add_state_before(init_state, 'pre')
    post_state = sdfg.add_state_after(exit_state, 'post')

    for name, desc in dict(sdfg.arrays).items():

        # We target arrays that have KLEV or KLEV + 1 in their shape
        shape_str = [str(s) for s in desc.shape]
        klev_idx = None
        divisor = None
        for v in klev_vals:
            if str(v) in shape_str:
                klev_idx = shape_str.index(str(v))
                divisor = v
                break
        if klev_idx is None:
            continue

        permutation[name] = klev_idx

        is_fortran = (desc.strides[0] == 1)

        # Add new array
        orig_desc = copy.deepcopy(desc)
        sdfg.arrays[name] = orig_desc
        # sdfg.add_datadesc(name + '_orig', orig_desc)
        sdfg.arrays[f"{name}_perm"] = desc
        desc.transient = True

        # Update the strides
        new_strides = list(desc.strides)
        if is_fortran:
            for idx in range(klev_idx + 1, len(desc.shape)):
                new_strides[idx] /= divisor
        else:
            for idx in range(klev_idx):
                new_strides[idx] /= divisor
        new_strides[klev_idx] = _prod(desc.shape) / divisor

        print(f"Changing strides of {name} with shape {desc.shape} from {desc.strides} to {new_strides}")

        desc.strides = tuple(new_strides)

        if name in inputs:
            orig_node = pre_state.add_access(name)
            perm_node = pre_state.add_access(name + '_perm')

            if is_fortran:
                corder = list(range(klev_idx)) + list(range(klev_idx + 1, len(desc.shape))) + [klev_idx]
                dorder = list(range(klev_idx)) + [len(desc.shape) - 1] + list(range(klev_idx, len(desc.shape) - 1))
            else:
                print(f"WARNING: {name} is not in Fortran order")
                corder = [klev_idx] + list(range(klev_idx)) + list(range(klev_idx + 1, len(desc.shape)))
                dorder = list(range(1, klev_idx + 1)) + [0] + list(range(klev_idx + 1, len(desc.shape)))

            vshape = [desc.shape[corder[i]] for i in range(len(desc.shape))]
            vstrides = [desc.strides[corder[i]] for i in range(len(desc.shape))]
            _, view = sdfg.add_view(name + '_in_view', vshape, desc.dtype, desc.storage, vstrides)

            view_node = pre_state.add_access(name + '_in_view')
            
            code_node = TensorTranspose(f"perm_{name}", corder)
            code_node.implementation = 'pure'

            pre_state.add_edge(orig_node, None, code_node, '_inp_tensor', dace.Memlet.from_array(name, orig_desc))
            pre_state.add_edge(code_node, '_out_tensor', view_node, None, dace.Memlet.from_array(name + '_in_view', view))
            pre_state.add_edge(view_node, 'views', perm_node, None, dace.Memlet(data=name + '_perm', subset=dace.subsets.Range.from_array(desc),
                                                                                other_subset=dace.subsets.Range.from_array(view)))
        
        if name in outputs:
            perm_node = post_state.add_access(name + '_perm')
            orig_node = post_state.add_access(name )

            if is_fortran:
                corder = list(range(klev_idx)) + [len(desc.shape) - 1] + list(range(klev_idx, len(desc.shape) - 1))
                dorder = list(range(klev_idx)) + list(range(klev_idx + 1, len(desc.shape))) + [klev_idx]
            else:
                print(f"WARNING: {name} is not in Fortran order")
                corder = list(range(1, klev_idx + 1)) + [0] + list(range(klev_idx + 1, len(desc.shape)))
                dorder = [klev_idx] + list(range(klev_idx)) + list(range(klev_idx + 1, len(desc.shape)))
            
            vshape = [desc.shape[dorder[i]] for i in range(len(desc.shape))]
            vstrides = [desc.strides[dorder[i]] for i in range(len(desc.shape))]
            _, view = sdfg.add_view(name + '_out_view', vshape, desc.dtype, desc.storage, vstrides)

            view_node = post_state.add_access(name + '_out_view')
            
            code_node = TensorTranspose(f"unperm_{name}", corder)
            code_node.implementation = 'pure'

            post_state.add_edge(perm_node, None, view_node, 'views', dace.Memlet(data=name + '_perm', subset=dace.subsets.Range.from_array(desc),
                                                                                 other_subset=dace.subsets.Range.from_array(view)))
            post_state.add_edge(view_node, None, code_node, '_inp_tensor', dace.Memlet.from_array(name + '_out_view', view))
            post_state.add_edge(code_node, '_out_tensor', orig_node, None, dace.Memlet.from_array(name, orig_desc))

    # Go to nested SDFGs
    # Assuming only 1 level of nested SDFG
    for sd in sdfg.all_sdfgs_recursive():

        if sd is sdfg:
            continue

        assert sd.parent_sdfg is sdfg

        print()
        print(f'NESTED SDFG {sd.name} (parent: {sd.parent_sdfg.name})')

        for s in syms_to_add:
            if s not in sd.parent_nsdfg_node.symbol_mapping:
                    sd.parent_nsdfg_node.symbol_mapping[s] = s
                    sd.add_symbol(s, dace.int32)

        for nname, ndesc in sd.arrays.items():

            if isinstance(ndesc, dace.data.Scalar):
                continue
            if ndesc.transient:
                continue
            
            nsdfg_node = sd.parent_nsdfg_node
            is_input = True
            edges = list(sd.parent.in_edges_by_connector(nsdfg_node, nname))
            if len(edges) == 0:
                is_input = False
                edges = list(sd.parent.out_edges_by_connector(nsdfg_node, nname))
                if len(edges) == 0:
                    raise ValueError
            edge = edges[0]
            if is_input:
                src = sd.parent.memlet_path(edge)[0].src
            else:
                src = sd.parent.memlet_path(edge)[-1].dst
            assert isinstance(src, dace.nodes.AccessNode)
            if src.data not in sdfg.arrays or src.data not in permutation:
                continue

            desc = sdfg.arrays[src.data + '_perm']
            new_strides = list(desc.strides)

            subset = edge.data.subset
            squeezed = copy.deepcopy(subset)
            rem_idx = squeezed.squeeze()
            if len(squeezed) != len(ndesc.shape):
                # print(f"WARNING: {nname}'s has subset {subset} is squeezed to {squeezed}, but the NestedSDFG connector has shape {ndesc.shape}")
                if len(ndesc.shape) - len(squeezed) == 1:
                    rem_idx.insert(0, 0)
                elif len(ndesc.shape) - len(squeezed) == 2:
                    rem_idx.insert(0, 0)
                    rem_idx.insert(2, 0)
                else:
                    raise NotImplementedError
            nnew_strides = [new_strides[i] for i in rem_idx]

            print(f"Changing strides of {nname} with shape {ndesc.shape} from {ndesc.strides} to {nnew_strides}")

            ndesc.strides = tuple(nnew_strides)

    # Fix names
    for node in main_state.nodes():
        if isinstance(node, dace.sdfg.nodes.AccessNode) and node.data in permutation:
            node.data += '_perm'
    for edge in main_state.edges():
        if edge.data.data in permutation:
            edge.data.data += '_perm'
    
    return permutation


def _to_gpu(sdfg: dace.SDFG):
    for _, desc in sdfg.arrays.items():
        if not desc.transient and isinstance(desc, dace.data.Array):
            desc.storage = dace.dtypes.StorageType.GPU_Global
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry):
                node.schedule = dace.ScheduleType.GPU_Device


def generate_all_sdfgs(specialize: bool, stride_transformation: bool, target: str, restart_index: int):
    """
    Generates all SDFGs for the CLOUDSCOUTER routine.
    
    :param specialize: If `True`, the SDFG will be specialized for know simulation parameters.
    :param stride_transformation: If `True`, the SDFG will be transformed to use the stride optimization.
    :param target: The target to generate the SDFG for.
    :param restart_index: The index to restart the generation from.
    """
    # Read Fortran code
    program = 'cloudscexp2_full_20230324'
    program_name = 'CLOUDPROGRAM'
    routine_name = 'CLOUDSCOUTER'
    fsource = read_source(program)

    # Generate unoptimized SDFG.
    if restart_index < 1:
        sdfg = get_unoptimized_sdfg(fsource, program_name)
        sdfg.save(os.path.join(os.path.dirname(__file__), f'{routine_name}_unoptimized.sdfg'))
    
    # Simplify the SDFG. This SDFG will also be specialized, if the corresponding optimization flag is set to `True`.
    if restart_index < 2:
        sdfg = dace.SDFG.from_file(os.path.join(os.path.dirname(__file__), f'{routine_name}_unoptimized.sdfg'))
        sdfg = simplify_sdfg(sdfg, specialize)
        sdfg.save(os.path.join(os.path.dirname(__file__), f'{routine_name}_simplified.sdfg'))

    # Apply stride transformation.
    if restart_index < 3 and stride_transformation:
        sdfg = dace.SDFG.from_file(os.path.join(os.path.dirname(__file__), f'{routine_name}_simplified.sdfg'))
        if specialize:
            klev_vals = (137, 138)
        else:
            klev_vals = ('KLEV', 'KLEV + 1')
        change_strides(sdfg, klev_vals, {'NBLOCKS'})
        sdfg.save(os.path.join(os.path.dirname(__file__), f'{routine_name}_strided.sdfg'))

    # Apply auto-optimization, including trivial map elimination and moving loops into maps.
    if restart_index < 4:
        if stride_transformation:
            sdfg = dace.SDFG.from_file(os.path.join(os.path.dirname(__file__), f'{routine_name}_strided.sdfg'))
        else:
            sdfg = dace.SDFG.from_file(os.path.join(os.path.dirname(__file__), f'{routine_name}_simplified.sdfg'))
        auto_optimize(sdfg, dace.DeviceType.Generic)
        sdfg.simplify(verbose=True)
        sdfg.apply_transformations_repeated(TrivialMapElimination)
        sdfg.simplify(verbose=True)
        for sd in sdfg.all_sdfgs_recursive():
            helpers.split_interstate_edges(sd)
        sdfg.apply_transformations_repeated(MoveLoopIntoMap)
        sdfg.simplify(verbose=True)
        sdfg.apply_transformations_repeated(MapCollapse)
        sdfg.simplify(verbose=True)
        sdfg.save(os.path.join(os.path.dirname(__file__), f'{routine_name}_optimized.sdfg'))
    
    # Unroll loops.
    if restart_index < 5:
        sdfg = dace.SDFG.from_file(os.path.join(os.path.dirname(__file__), f'{routine_name}_optimized.sdfg'))
        for sd in sdfg.all_sdfgs_recursive():
            helpers.split_interstate_edges(sd)
        sdfg.apply_transformations_repeated(LoopUnroll)
        sdfg.simplify(verbose=True)
        sdfg.save(os.path.join(os.path.dirname(__file__), f'{routine_name}_unrolled.sdfg'))

    # Fuse Maps.
    if restart_index < 6:
        sdfg = dace.SDFG.from_file(os.path.join(os.path.dirname(__file__), f'{routine_name}_unrolled.sdfg'))
        greedy_fuse(sdfg, False)
        sdfg.simplify(verbose=True)
        sdfg.save(os.path.join(os.path.dirname(__file__), f'{routine_name}_fused.sdfg'))
    
    sdfg = dace.SDFG.from_file(os.path.join(os.path.dirname(__file__), f'{routine_name}_fused.sdfg'))
    
    # Change storage and schedule types for GPU.
    if target == "gpu":
        sdfg.apply_gpu_transformations_cloudsc()
    
    # Try to compile
    sdfg.compile()

    # Save SDFG
    sdfg.save(os.path.join(os.path.dirname(__file__), f'{routine_name}_{target}_final.sdfg'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDFGs for CloudSC")
    parser.add_argument('--specialize', type=str2bool, nargs="?", default=True, help='Specialize SDFGs to use constants for known simulation parameters instead of symbols.')
    parser.add_argument('--stride-transformation', type=str2bool, nargs="?", default=True, help='Apply stride transformation to improve memory accesses (mainly for GPU execution).')
    parser.add_argument('--target', choices=['cpu', 'gpu'], nargs="?", default='cpu', help='Target device for final SDFG.')
    parser.add_argument('--restart-index', type=int, nargs="?", default=0, help='Index of the first SDFG to generate.')

    args = parser.parse_args()
    generate_all_sdfgs(args.specialize, args.stride_transformation, args.target, args.restart_index)
