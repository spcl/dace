# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains classes that implement the double buffering pattern. """

import copy

from dace import data, sdfg as sd, subsets, symbolic, InterstateEdge, SDFGState, Memlet, dtypes
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties, Property, SymbolicProperty, CodeBlock
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion
import ast


@make_properties
class SystolicSplitStore(transformation.SingleStateTransformation):

    map_entry = transformation.PatternNode(nodes.MapEntry)
    accumulator = transformation.PatternNode(nodes.AccessNode)

    # Properties
    npe = Property(default=None, allow_none=True, desc="Number of processing elements")
    tM = Property(default=None, allow_none=True, desc="tM")
    tN = Property(default=None, allow_none=True, desc="tN")
    tK = Property(default=None, allow_none=True, desc="tK")
    M = SymbolicProperty(default=None, allow_none=True, desc="M")
    N = SymbolicProperty(default=None, allow_none=True, desc="N")
    K = SymbolicProperty(default=None, allow_none=True, desc="K")
    gi = SymbolicProperty(default=None, allow_none=True, desc="gi")
    gj = SymbolicProperty(default=None, allow_none=True, desc="gj")
    i = SymbolicProperty(default=None, allow_none=True, desc="i")
    j = SymbolicProperty(default=None, allow_none=True, desc="j")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.accumulator)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry
        accumulator = self.accumulator

        return True

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):
        map_entry = self.map_entry
        NPE = self.npe
        tM = self.tM
        tN = self.tN
        tK = self.tK
        M = self.M
        N = self.N
        K = self.K
        gi = self.gi
        gj = self.gj
        i = self.i
        j = self.j

        ##############################
        from dace.transformation.helpers import nest_state_subgraph
        from dace.sdfg import SDFG, SDFGState
        node = nest_state_subgraph(sdfg, graph, graph.scope_subgraph(map_entry, include_entry=False,
                                                                     include_exit=False))
        # node.schedule = map_entry.map.schedule
        nsdfg: SDFG = node.sdfg
        nstate: SDFGState = nsdfg.nodes()[0]

        for cfg in nsdfg.all_control_flow_blocks(recursive=True):
            if isinstance(cfg, LoopRegion) and (cfg == "loop" or "systolic_loop"):
                lr_param = cfg.loop_variable
                cond_code = cfg.loop_condition.code
                new_cond_code = CodeBlock(code=f"{lr_param} < (({gi}+{gj}+1) + ({K}/({tK})))",
                                          language=dtypes.Language.Python)
                cfg.loop_condition = new_cond_code

        last_systolic_sync: SDFGState = nsdfg.add_state("systolic_sync")
        nsdfg.add_edge(nstate, last_systolic_sync, InterstateEdge(None, None))
        last_systolic_sync.add_tasklet(name="SoftHier_sync",
                                       inputs=None,
                                       outputs=None,
                                       code=f'''
                            if (({i} >= {M} - {NPE}*{tM}) && ({j} >= {N} - {NPE}*{tN}))
                            {{
                                for (int sync_iter = 0; sync_iter < 2*{NPE} - 1 - {gi} - {gj} - 1; sync_iter++){{
                                    flex_global_barrier_xy();
                                }}
                                if (flex_is_dm_core()) {{
                                    flex_dma_async_wait_all();
                                }}
                                flex_intra_cluster_sync();
                                flex_global_barrier_xy();
                            }}
                            ''',
                                       language=dtypes.Language.CPP)

    @staticmethod
    def _modify_memlet(sdfg, subset, data_name):
        desc = sdfg.arrays[data_name]
        if len(subset) == len(desc.shape):
            # Already in the right shape, modify new dimension
            subset = list(subset)[1:]

        new_subset = subsets.Range([('__dace_db_param', '__dace_db_param', 1)] + list(subset))
        return new_subset

    @staticmethod
    def _replace_in_subset(subset, string_or_symbol, new_string_or_symbol):
        new_subset = copy.deepcopy(subset)

        repldict = {symbolic.pystr_to_symbolic(string_or_symbol): symbolic.pystr_to_symbolic(new_string_or_symbol)}

        for i, dim in enumerate(new_subset):
            try:
                new_subset[i] = tuple(d.subs(repldict) for d in dim)
            except TypeError:
                new_subset[i] = (dim.subs(repldict) if symbolic.issymbolic(dim) else dim)

        return new_subset
