# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from dace import dtypes, symbolic, data, subsets, Memlet
from dace.transformation import transformation as xf
from dace.sdfg import SDFGState, SDFG, nodes, utils as sdutil
from typing import Tuple


class CopyToMap(xf.SingleStateTransformation):
    """
    Converts an access node -> access node copy into a map. Useful for generating manual code and
    controlling schedules for N-dimensional strided copies.
    """
    a = xf.PatternNode(nodes.AccessNode)
    b = xf.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.a, cls.b)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        # Only array->array copies are supported
        if type(self.a.desc(sdfg)) != data.Array:
            return False
        if type(self.b.desc(sdfg)) != data.Array:
            return False
        if self.a.desc(sdfg).strides == self.b.desc(sdfg).strides:
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

        linear_index = sum(indices[i] * data._prod(copy_shape[i + 1:]) for i in range(len(indices)))

        if rng is not None and any(mn != 0 for mn in rng.min_element()):
            raise NotImplementedError('Offsets not yet implemented')
        if rng is not None and any(rs != 1 for _, _, rs in rng):
            raise NotImplementedError('Strides not yet implemented')

        cur_index = [0] * len(desc.shape)
        divide_by = 1
        for i in reversed(range(len(desc.shape))):
            cur_index[i] = (linear_index / divide_by) % desc.shape[i]
            divide_by = divide_by * desc.shape[i]

        return subsets.Range([(ind, ind, 1) for ind in cur_index])

    def apply(self, state: SDFGState, sdfg: SDFG):
        adesc = self.a.desc(sdfg)
        bdesc = self.b.desc(sdfg)
        edge = state.edges_between(self.a, self.b)[0]

        if len(adesc.shape) >= len(bdesc.shape):
            copy_shape = edge.data.get_src_subset(edge, state).size()
            copy_a = True
        else:
            copy_shape = edge.data.get_dst_subset(edge, state).size()
            copy_a = False

        maprange = {f'__i{i}': (0, s - 1, 1) for i, s in enumerate(copy_shape)}

        av = self.a.data
        bv = self.b.data
        avnode = self.a
        bvnode = self.b

        # Linearize and delinearize to get index expression for other side
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
            schedule = dtypes.ScheduleType.GPU_Device

        # Add copy map
        state.add_mapped_tasklet('copy',
                                 maprange,
                                 dict(__inp=Memlet(data=av, subset=a_subset)),
                                 '__out = __inp',
                                 dict(__out=Memlet(data=bv, subset=b_subset)),
                                 schedule,
                                 external_edges=True,
                                 input_nodes={av: avnode},
                                 output_nodes={bv: bvnode})

        # Remove old edge
        state.remove_edge(edge)
