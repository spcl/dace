# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from dace import dtypes, symbolic, data, subsets, Memlet, properties
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation import transformation as xf
from dace.sdfg import SDFGState, SDFG, nodes, utils as sdutil
from typing import Tuple
import itertools

@properties.make_properties
class CopyToMap(xf.SingleStateTransformation):
    """
    Converts an access node -> access node copy into a map. Useful for generating manual code and
    controlling schedules for N-dimensional strided copies.
    """
    a = xf.PatternNode(nodes.AccessNode)
    b = xf.PatternNode(nodes.AccessNode)
    ignore_strides = properties.Property(
            default=False,
            desc='Ignore the stride of the data container; Defaults to `False`.',
    )

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.a, cls.b)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        # Only array->array or view<->array copies are supported
        if not isinstance(self.a.desc(sdfg), data.Array):
            return False
        if not isinstance(self.b.desc(sdfg), data.Array):
            return False
        if isinstance(self.a.desc(sdfg), data.View):
            if sdutil.get_view_node(graph, self.a) == self.b:
                return False
        if isinstance(self.b.desc(sdfg), data.View):
            if sdutil.get_view_node(graph, self.b) == self.a:
                return False
        if (not self.ignore_strides) and self.a.desc(sdfg).strides == self.b.desc(sdfg).strides:
            return False
        # Ensures that the edge goes from `a` -> `b`.
        if not any(edge.dst is self.b for edge in graph.out_edges(self.a)):
            return False

        return True

    def delinearize_linearize(self, desc: data.Array, copy_shape: Tuple[symbolic.SymbolicType],
                              rng: subsets.Range) -> Tuple[symbolic.SymbolicType]:
        """
        Converts one N-dimensional iteration space to another M-dimensional space via linearization
        followed by delinearization.
        """
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

    def apply(self, state: SDFGState, sdfg: SDFG):
        avnode = self.a
        av = avnode.data
        adesc = avnode.desc(sdfg)
        bvnode = self.b
        bv = bvnode.data
        bdesc = bvnode.desc(sdfg)

        edge = state.edges_between(avnode, bvnode)[0]
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
                if s == 1
                else symbolic.pystr_to_symbolic(f'__j{next(cnt)} + ({src_subset[i][0]})')
                for i, s in enumerate(src_subset_size)
            ]
            cnt = itertools.count(0)
            b_index = [
                symbolic.pystr_to_symbolic(f'{dst_subset[i][0]}')
                if s == 1
                else symbolic.pystr_to_symbolic(f'__j{next(cnt)} + ({dst_subset[i][0]})')
                for i, s in enumerate(dst_subset_size)
            ]
        else:
            # We have to delinearize and linearize
            #  We use another index variable for the tests.
            maprange = {f'__i{i}': (0, s - 1, 1) for i, s in enumerate(copy_shape)}
            if copy_a:
                a_index = [symbolic.pystr_to_symbolic(f'__i{i}') for i in range(len(copy_shape))]
                b_index = self.delinearize_linearize(bdesc, copy_shape, edge.data.get_dst_subset(edge, state))
            else:
                a_index = self.delinearize_linearize(adesc, copy_shape, edge.data.get_src_subset(edge, state))
                b_index = [symbolic.pystr_to_symbolic(f'__i{i}') for i in range(len(copy_shape))]

        a_subset = subsets.Range([(ind, ind, 1) for ind in a_index])
        b_subset = subsets.Range([(ind, ind, 1) for ind in b_index])

        # Set schedule based on GPU arrays
        schedule = dtypes.ScheduleType.Default
        if adesc.storage == dtypes.StorageType.GPU_Global or bdesc.storage == dtypes.StorageType.GPU_Global:
            # If already inside GPU kernel
            if is_devicelevel_gpu(sdfg, state, self.a):
                schedule = dtypes.ScheduleType.Sequential
            else:
                schedule = dtypes.ScheduleType.GPU_Device

        # Add copy map
        t, _, _ = state.add_mapped_tasklet(f'copy_{av}_{bv}',
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
