# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import itertools
from dace import data, Memlet, subsets, symbolic, dtypes
from dace.sdfg import SDFGState, SDFG, nodes, utils as sdutil
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.graph import MultiConnectorEdge
from dace.frontend.python import memlet_parser
from typing import Callable, Dict, Optional, Set, Union, Tuple


class MemletReplacer(ast.NodeTransformer):
    """
    Iterates over all memlet expressions (name or subscript with matching array in SDFG) in a code block.
    The callable can also return another memlet to replace the current one.
    """

    def __init__(self,
                 arrays: Dict[str, data.Data],
                 process: Callable[[Memlet], Union[Memlet, None]],
                 array_filter: Optional[Set[str]] = None) -> None:
        """
        Create a new memlet replacer.

        :param arrays: A mapping from array names to data descriptors.
        :param process: A callable that takes a memlet and returns a memlet or None.
        :param array_filter: An optional subset of array names to process.
        """
        self.process = process
        self.arrays = arrays
        self.array_filter = array_filter or self.arrays.keys()

    def _parse_memlet(self, node: Union[ast.Name, ast.Subscript]) -> Memlet:
        """
        Parses a memlet from a subscript or name node.

        :param node: The node to parse.
        :return: The parsed memlet.
        """
        # Get array name
        if isinstance(node, ast.Name):
            data = node.id
        elif isinstance(node, ast.Subscript):
            data = node.value.id
        else:
            raise TypeError('Expected Name or Subscript')

        # Parse memlet subset
        array = self.arrays[data]
        subset, newaxes, _ = memlet_parser.parse_memlet_subset(array, node, self.arrays)
        if newaxes:
            raise NotImplementedError('Adding new axes to memlets is not supported')

        return Memlet(data=data, subset=subset)

    def _memlet_to_ast(self, memlet: Memlet) -> ast.Subscript:
        """
        Converts a memlet to a subscript node.

        :param memlet: The memlet to convert.
        :return: The converted node.
        """
        return ast.parse(f'{memlet.data}[{memlet.subset}]').body[0].value

    def _replace(self, node: Union[ast.Name, ast.Subscript]) -> ast.Subscript:
        cur_memlet = self._parse_memlet(node)
        new_memlet = self.process(cur_memlet)
        if new_memlet is None:
            return node

        new_node = self._memlet_to_ast(new_memlet)
        return ast.copy_location(new_node, node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.array_filter:
            return self._replace(node)
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in self.array_filter:
            return self._replace(node)
        return self.generic_visit(node)


def memlet_to_map(
    edge: MultiConnectorEdge,
    state: SDFGState,
    sdfg: SDFG,
    ignore_strides: bool = True,
) -> Tuple[nodes.MapEntry, nodes.MapExit]:
    """Change a Memlet into a Map.

    The function returns a pair consisting of the new Map entry and exit.
    Furthermore, the edge is removed from the state.

    In case the Memlet can not be turned into a Map, the function raises a
    `ValueError`. In order to test if a particular Memlet can be turned
    into a Map the function `can_memlet_be_turned_into_a_map()` be used.

    :param edge: The edge to translate.
    :param state: The state on which we operate.
    :param sdfg: The SDFG on which we operate.
    :param ignore_strides: If `True`, the default, also check for stride compatibility.
    """
    if not can_memlet_be_turned_into_a_map(edge=edge, state=state, sdfg=sdfg, ignore_strides=ignore_strides):
        raise ValueError(f'Tried to turn edge "{edge}" into a Map, but this is not possible.')

    avnode = edge.src
    av = avnode.data
    adesc = avnode.desc(sdfg)

    bvnode = edge.dst
    bv = bvnode.data
    bdesc = bvnode.desc(sdfg)

    src_subset = edge.data.get_src_subset(edge, state)
    if src_subset is None:
        src_subset = subsets.Range.from_array(adesc)
    src_subset_size = src_subset.size()
    red_src_subset_size = tuple(s for s in src_subset_size if s != 1)

    dst_subset = edge.data.get_dst_subset(edge, state)
    if dst_subset is None:
        dst_subset = subsets.Range.from_array(bdesc)
    dst_subset_size = dst_subset.size()
    red_dst_subset_size = tuple(s for s in dst_subset_size if s != 1)

    if len(adesc.shape) >= len(bdesc.shape):
        copy_shape = src_subset_size
        copy_a = True
    else:
        copy_shape = dst_subset_size
        copy_a = False

    if tuple(src_subset_size) == tuple(dst_subset_size):
        # The two subsets have exactly the same shape, so we can just copying with an offset.
        #  We use another index variables for the tests only.
        maprange = {f'__j{i}': (0, s - 1, 1) for i, s in enumerate(copy_shape)}
        a_index = [symbolic.pystr_to_symbolic(f'__j{i} + ({src_subset[i][0]})') for i in range(len(copy_shape))]
        b_index = [symbolic.pystr_to_symbolic(f'__j{i} + ({dst_subset[i][0]})') for i in range(len(copy_shape))]
    elif red_src_subset_size == red_dst_subset_size and (len(red_dst_subset_size) > 0):
        # If we remove all size 1 dimensions that the two subsets have the same size.
        #  This is essentially the memlet `a[0:10, 2, 0:10] -> 0:10, 10:20`
        #  We use another index variable only for the tests but we would have to
        #  recreate the index anyways.
        maprange = {f'__j{i}': (0, s - 1, 1) for i, s in enumerate(red_src_subset_size)}
        cnt = itertools.count(0)
        a_index = [
            symbolic.pystr_to_symbolic(f'{src_subset[i][0]}')
            if s == 1 else symbolic.pystr_to_symbolic(f'__j{next(cnt)} + ({src_subset[i][0]})')
            for i, s in enumerate(src_subset_size)
        ]
        cnt = itertools.count(0)
        b_index = [
            symbolic.pystr_to_symbolic(f'{dst_subset[i][0]}')
            if s == 1 else symbolic.pystr_to_symbolic(f'__j{next(cnt)} + ({dst_subset[i][0]})')
            for i, s in enumerate(dst_subset_size)
        ]
    else:
        # We have to delinearize and linearize
        #  We use another index variable for the tests.
        maprange = {f'__i{i}': (0, s - 1, 1) for i, s in enumerate(copy_shape)}
        if copy_a:
            a_index = [symbolic.pystr_to_symbolic(f'__i{i}') for i in range(len(copy_shape))]
            b_index = _memlet_to_copy_delinearize_linearize(bdesc, copy_shape, edge.data.get_dst_subset(edge, state))
        else:
            a_index = _memlet_to_copy_delinearize_linearize(adesc, copy_shape, edge.data.get_src_subset(edge, state))
            b_index = [symbolic.pystr_to_symbolic(f'__i{i}') for i in range(len(copy_shape))]

    a_subset = subsets.Range([(ind, ind, 1) for ind in a_index])
    b_subset = subsets.Range([(ind, ind, 1) for ind in b_index])

    # Set schedule based on GPU arrays
    schedule = dtypes.ScheduleType.Default
    if adesc.storage == dtypes.StorageType.GPU_Global or bdesc.storage == dtypes.StorageType.GPU_Global:
        # If already inside GPU kernel
        if is_devicelevel_gpu(sdfg, state, avnode):
            schedule = dtypes.ScheduleType.Sequential
        else:
            schedule = dtypes.ScheduleType.GPU_Device

    # Add copy map
    t, me, mx = state.add_mapped_tasklet(f'copy_{av}_{bv}',
                                         maprange,
                                         dict(__inp=Memlet(data=av, subset=a_subset)),
                                         '__out = __inp',
                                         dict(__out=Memlet(data=bv, subset=b_subset)),
                                         schedule,
                                         external_edges=True,
                                         input_nodes={av: avnode},
                                         output_nodes={bv: bvnode})

    # Set connector types (due to this transformation appearing in codegen, after connector
    # types have been resolved)
    t.in_connectors['__inp'] = adesc.dtype
    t.out_connectors['__out'] = bdesc.dtype

    # Remove old edge
    state.remove_edge(edge)

    return (me, mx)


def can_memlet_be_turned_into_a_map(
    edge: MultiConnectorEdge,
    state: SDFGState,
    sdfg: SDFG,
    ignore_strides: bool = True,
) -> bool:
    """Check if a Memlet can be turned into a Map.

    If this function returns `True` then `memlet_to_map()` will not
    raise an exception.

    :param edge: The edge to translate.
    :param state: The state on which we operate.
    :param sdfg: The SDFG on which we operate.
    :param ignore_strides: If `True`, the default, also check for stride compatibility.
    """
    a = edge.src
    b = edge.dst

    if not (isinstance(a, nodes.AccessNode) and isinstance(b, nodes.AccessNode)):
        return False
    if not isinstance(a.desc(sdfg), data.Array):
        return False
    if not isinstance(b.desc(sdfg), data.Array):
        return False
    if isinstance(a.desc(sdfg), data.View):
        if sdutil.get_view_node(state, a) == b:
            return False
    if isinstance(b.desc(sdfg), data.View):
        if sdutil.get_view_node(state, b) == a:
            return False
    if (not ignore_strides) and a.desc(sdfg).strides == b.desc(sdfg).strides:
        return False
    return True


def _memlet_to_copy_delinearize_linearize(desc: data.Array, copy_shape: Tuple[symbolic.SymbolicType],
                                          rng: subsets.Range) -> Tuple[symbolic.SymbolicType]:
    indices = [symbolic.pystr_to_symbolic(f'__i{i}') for i in range(len(copy_shape))]

    # Special case for when both dimensionalities are equal
    if tuple(desc.shape) == tuple(copy_shape):
        return subsets.Range([(ind, ind, 1) for ind in indices])

    if rng is not None:  # Deal with offsets and strides in range
        indices = rng.coord_at(indices)

    linear_index = sum(indices[i] * data._prod(copy_shape[i + 1:]) for i in range(len(indices)))

    cur_index = [0] * len(desc.shape)
    divide_by = 1
    for i in reversed(range(len(desc.shape))):
        cur_index[i] = (linear_index / divide_by) % desc.shape[i]
        divide_by = divide_by * desc.shape[i]

    return subsets.Range([(ind, ind, 1) for ind in cur_index])
