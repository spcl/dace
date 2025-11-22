import collections
import functools
import logging
from typing import Dict, Optional, Set, Callable
import copy

import sympy

import dace
from dace import nodes as nd, data as dt
from dace.libraries import blas
from dace.sdfg import infer_types
from dace.sdfg.state import MultiConnectorEdge
from dace.transformation import dataflow
from dace import SDFG, SDFGState, dtypes
import dace.data as dt
from dace import dtypes, config
from dace.transformation.auto.auto_optimize import greedy_fuse, make_transients_persistent, move_small_arrays_to_stack, set_fast_implementations, tile_wcrs
from dace.transformation.dataflow import CopyToMap
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.trivial_map_elimination import TrivialMapElimination
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.sdfg_nesting import RefineNestedAccess

log = logging.getLogger(__name__)


def is_desc_contiguous(desc: dt.Data) -> bool:
    if type(desc) is dt.Scalar:
        return True
    elif type(desc) is dt.Array:
        contiguous_strides = [dt._prod(desc.shape[i + 1:]) for i in range(len(desc.shape))]
        return desc.strides == contiguous_strides
    else:
        raise ValueError("Unsupported data descriptor type {}".format(type(desc)))


def in_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG, name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to input connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the input connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[in_edge_with_name(node, state, name).data.data]


def out_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG, name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to output connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the output connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[out_edge_with_name(node, state, name).data.data]


def in_edge_with_name(node: nd.Node, state: SDFGState, name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to input connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the input connector name.
        :return: the edge that connects to connector `name`.
     """

    cands = list(state.in_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError("Expected to find exactly one edge with name '{}', found {}".format(name, len(cands)))
    return cands[0]


def out_edge_with_name(node: nd.Node, state: SDFGState, name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to output connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the output connector name.
        :return: the edge that connects to connector `name`.
     """
    cands = list(state.out_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError("Expected to find exactly one edge with name '{}', found {}".format(name, len(cands)))
    return cands[0]


def find_str_not_in_set(existing: Set[str], target_str: Optional[str]) -> str:
    """ Try to find a new str that is not in the set.

        :param existing: the existing strs.
        :param target_str: (optional) a target_str that should be used as a base for the new str.
        :return: a new str that is not in `existing`.
    """
    base_name = target_str or "temp"

    if base_name not in existing:
        return base_name

    i = 0
    while (base_name + "_" + str(i)) in existing:
        i += 1
    return base_name + "_" + str(i)


def vectorize_array_and_memlet(sdfg, array_name, type: dtypes.typeclass):
    '''
       Adjust the shape of a data container according to the vec width (only the last dimension).
       This will change its shape and strides
       together with the all the ingoin/outgoing memlets
    '''
    # find the array
    data = sdfg.arrays[array_name]
    if type == data.dtype:
        return
    #change the type
    data.dtype = type

    #adjust the shape
    vec_width = type.veclen
    if data.shape[-1] % vec_width != 0:
        raise ValueError("Shape of {} is not divisible by {}".format(data, vec_width))
    data.shape = data.shape[:-1] + (data.shape[-1] // vec_width, )

    # #adjust all the strides
    for stride in data.strides[:-1]:
        if stride % vec_width != 0:
            raise ValueError("Stride of {} is not divisible by {}".format(data.name, vec_width))

    data.strides = tuple(ti // vec_width for ti in data.strides[:-1]) + (data.strides[-1], )

    # Search for all the memlets
    for state in sdfg.nodes():
        for edge in state.edges():
            if edge.data.data == array_name:
                # get the range
                start, stop, skip = edge.data.subset.ranges[-1]

                # Let's be conservative for the moment
                if start != 0 or skip != 1 or (stop + 1) % vec_width != 0:
                    raise ValueError("Memlet {} not able to convert its range".format(edge.data))

                #update the range
                new_stop = (stop + 1) // vec_width - 1
                edge.data.subset.ranges[-1] = (start, new_stop, skip)


def expand_onnx_nodes(sdfg: dace.SDFG, predicate: Optional[Callable[[nd.Node], bool]] = None):
    """ Recursively expand all onnx library nodes in the SDFG, resulting in an SDFG that can be optimized by
        dace transformations. Will also specialize dace matmuls.

        :param sdfg: the sdfg to expand nodes on.
        :param predicate: a predicate that will be called to check if a node should be expanded.
    """

    try:
        from dace.libraries.onnx.nodes.onnx_op import ONNXOp  # avoid import loop
    except ImportError:
        raise ImportError("expand_onnx_nodes requires ONNX. Install with: pip install dace[ml]")

    if predicate is None:
        new_predicate = lambda n: isinstance(n, (ONNXOp, blas.MatMul))
    else:
        new_predicate = lambda n: predicate(n) and isinstance(n, (ONNXOp, blas.MatMul))

    expand_nodes(sdfg, new_predicate)


def expand_nodes(sdfg: dace.SDFG, predicate: Callable[[nd.Node], bool]):
    """ Recursively expand library nodes in the SDFG using a given predicate.

        :param sdfg: the sdfg to expand nodes on.
        :param predicate: a predicate that will be called to check if a node should be expanded.
    """
    if sdfg is None:
        return
    states = list(sdfg.states())
    while len(states) > 0:
        state = states.pop()
        expanded_something = False
        for node in list(state.nodes()):  # Make sure we have a copy
            if isinstance(node, nd.NestedSDFG):
                expand_nodes(node.sdfg, predicate=predicate)
            elif isinstance(node, nd.LibraryNode):
                if predicate(node):
                    impl_name = node.expand(sdfg, state)
                    if dace.Config.get_bool('debugprint'):
                        print("Automatically expanded library node \"{}\" with implementation \"{}\".".format(
                            str(node), impl_name))
                    # We made a copy of the original list of nodes, so we keep
                    # iterating even though this list has now changed
                    expanded_something = True

        if expanded_something:
            states.append(state)  # Nodes have changed. Check state again


def auto_optimize_onnx(sdfg: SDFG,
                       device: dtypes.DeviceType,
                       validate: bool = True,
                       validate_all: bool = False,
                       symbols: Dict[str, int] = None) -> SDFG:
    """ Automatically optimize an ``sdfg`` from the ONNX frontend.
    Runs a basic sequence of transformations to optimize a given SDFG to decent
    performance. In particular, performs the following:

        * Simplify
        * Auto-parallelization (loop-to-map)
        * Greedy application of SubgraphFusion
        * Tiled write-conflict resolution (MapTiling -> AccumulateTransient)
        * Tiled stream accumulation (MapTiling -> AccumulateTransient)
        * Collapse all maps to parallelize across all dimensions
        * Set all library nodes to expand to ``fast`` expansion, which calls
          the fastest library on the target device

    :param sdfg: The SDFG to optimize.
    :param device: the device to optimize for.
    :param validate: If True, validates the SDFG after all transformations
                     have been applied.
    :param validate_all: If True, validates the SDFG after every step.
    :param symbols: Optional dict that maps symbols (str/symbolic) to int/float
    :param use_gpu_storage: If True, changes the storage of non-transient data to GPU global memory.
    :return: The optimized SDFG.
    :note: Operates in-place on the given SDFG.
    :note: This function is still experimental and may harm correctness in
           certain cases. Please report an issue if it does.
    """
    debugprint = config.Config.get_bool('debugprint')
    try:
        from dace.transformation.onnx import ConstantFolding  # avoid import loop
    except ImportError:
        raise ImportError("auto_optimize_onnx requires ONNX. Install with: pip install dace[ml]")

    log.debug("ONNX-AutoOpt: Applying ONNX automatic optimizations")

    log.debug("ONNX-AutoOpt: Applying constant folding")
    sdfg.apply_transformations_repeated([ConstantFolding, dataflow.RedundantSecondArray], validate_all=True)

    log.debug("ONNX-AutoOpt: Expanding ONNX nodes")
    expand_onnx_nodes(sdfg)

    # Simplification and loop parallelization
    transformed = True
    sdfg.apply_transformations_repeated(TrivialMapElimination, validate=validate, validate_all=validate_all)

    while transformed:
        sdfg.simplify(validate=False, validate_all=validate_all)
        l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                   validate=False,
                                                   validate_all=validate_all)
        transformed = l2ms > 0

    # Collapse maps and eliminate trivial dimensions
    sdfg.simplify()
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)

    sdfg.simplify()
    sdfg.reset_cfg_list()

    log.debug("Setting fast implementations")
    set_fast_implementations(sdfg, device)

    # NOTE: We need to `infer_types` in case a LibraryNode expands to other LibraryNodes (e.g., np.linalg.solve)
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.expand_library_nodes()

    # fuse subgraphs greedily
    sdfg.simplify()
    sdfg.reset_cfg_list()

    greedy_fuse(sdfg, device=device, validate_all=validate_all)

    # fuse stencils greedily
    greedy_fuse(sdfg, device=device, validate_all=validate_all, recursive=False, stencil=True)

    # Move Loops inside Maps when possible
    from dace.transformation.interstate import MoveLoopIntoMap
    sdfg.apply_transformations_repeated([MoveLoopIntoMap])

    # Tiled WCR and streams
    for nsdfg in list(sdfg.all_sdfgs_recursive()):
        tile_wcrs(nsdfg, validate_all)

    # Collapse maps
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)
    for node, _ in sdfg.all_nodes_recursive():
        # Set OMP collapse property to map length
        if isinstance(node, dace.nodes.MapEntry):
            # FORNOW: Leave out
            # node.map.collapse = len(node.map.range)
            pass

    for nsdfg in sdfg.all_sdfgs_recursive():
        nsdfg.openmp_sections = False

    # Set all Default storage types that are constant sized to registers
    move_small_arrays_to_stack(sdfg)

    # Make all independent arrays persistent
    make_transients_persistent(sdfg, device)

    if symbols:
        # Specialize for all known symbols
        known_symbols = {}
        for (s, v) in symbols.items():
            if s in sdfg.free_symbols:
                if isinstance(v, (int, float)):
                    known_symbols[s] = v
                if isinstance(v, sympy.Integer):
                    try:
                        known_symbols[s] = int(v)
                    except TypeError:
                        pass

        if debugprint and len(known_symbols) > 0:
            print("Specializing the SDFG for symbols", known_symbols)
        sdfg.specialize(known_symbols)

    sdfg.reset_cfg_list()

    # Validate at the end
    if validate or validate_all:
        sdfg.validate()

    return sdfg


def remove_unnecessary_views(sdfg: dace.SDFG):
    """
    InlineSDFG generates some unnecessary views that make the SDFG more complex.
    Detect the pattern AN -> View -> reduction, check that the same data is being used and remove the view
    """
    for state in sdfg.nodes():
        for node in state.nodes():
            # If this is a view
            if isinstance(node, nd.AccessNode) and isinstance(sdfg.arrays[node.data], dt.ArrayView):
                # Check that the same data is passed to the view as the original AN
                if len(state.in_edges(node)) != 1 or len(state.out_edges(node)) != 1:
                    continue
                incoming_edge = state.in_edges(node)[0]
                outgoing_edge = state.out_edges(node)[0]
                view_desc = sdfg.arrays[node.data]
                if not isinstance(incoming_edge.src, nd.AccessNode) or isinstance(sdfg.arrays[incoming_edge.src.data],
                                                                                  dt.ArrayView):
                    continue
                original_node = incoming_edge.src
                original_desc = sdfg.arrays[original_node.data]

                # Additional restrictive condition: this should only cause a problem for library nodes
                if not isinstance(outgoing_edge.dst, nd.LibraryNode):
                    continue

                # They need to have the same shape
                if original_desc.shape != view_desc.shape:
                    continue

                # The memlet subsets need to be the same for the outgoing edge and incoming edge
                if not (incoming_edge.data.subset == outgoing_edge.data.subset
                        and incoming_edge.data.other_subset == outgoing_edge.data.other_subset):
                    continue

                # Remove the view node
                state.add_edge(incoming_edge.src, None, outgoing_edge.dst, outgoing_edge.dst_conn,
                               copy.deepcopy(incoming_edge.data))
                state.remove_node(node)


def iterables_equal(a, b) -> bool:
    """ Return whether the two iterables ``a`` and ``b`` are equal. """
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))


def prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)


def is_cuda(storage: dtypes.StorageType) -> bool:
    """ Check if a descriptor storage type is a GPU array """
    if dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, storage):
        return False
    elif dtypes.can_access(dtypes.ScheduleType.FPGA_Device, storage):
        return False
    elif dtypes.can_access(dtypes.ScheduleType.GPU_Default, storage):
        return True
    else:
        raise ValueError(f"Unsupported storage {storage}")


def platform_library_name(libname: str) -> str:
    """ Get the filename of a library.

        :param libname: the name of the library.
        :return: the filename of the library.
    """
    prefix = dace.Config.get('compiler', 'library_prefix')
    suffix = dace.Config.get('compiler', 'library_extension')
    return f"{prefix}{libname}.{suffix}"


def remove_output_connector(sdfg: dace.SDFG, state: dace.SDFGState, node: nd.Node, conn_name: str):
    """ Remove an output connector (only possible if the connector doesn't write to a non-transient).

        :param sdfg: the sdfg containing the node.
        :param state: the state containing the node.
        :param node: the node
        :param conn_name: the name of the connector to remove
    """
    queue = collections.deque(e.dst for e in state.out_edges_by_connector(node, conn_name))
    while len(queue) > 0:
        current_node = queue.popleft()

        edges = state.out_edges(current_node)
        state.remove_node(current_node)
        for e in edges:
            if not sdfg.arrays[e.data.data].transient:
                raise ValueError("Tried to remove a connector that wrote to a non-transient")

            queue.append(e.dst)


def get_library_node_by_name(sdfg, name):
    '''
    Searches for a library node with @param name
    in the SDFG @param sdfg and returns the library
    node and the associated state
    '''

    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.LibraryNode):
            if node.label == name:
                return node, state

    raise Exception(f"LibraryNode {name} not found")


def get_access_node_by_name(sdfg, name):
    '''
    Searches for an access node with @param name
    in the SDFG @param sdfg and returns the library
    node and the associated state
    '''

    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.AccessNode):
            if node.label == name:
                return node, state

    raise Exception("DataNode {} not found".format(name))


def all_equal(a, b) -> bool:
    """
    Check whether two iterables are equal
    """
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))
