# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains classes that implement the double buffering pattern. """

import copy
import ast
import random
from random import randint as rand
from dace import data, sdfg as sd, subsets, symbolic, InterstateEdge, SDFGState, Memlet, dtypes
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties, Property, SymbolicProperty, CodeBlock, CodeProperty
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion
from dace.codegen.targets.cpp import sym2cpp


@make_properties
class SplitKReduction(transformation.SingleStateTransformation):

    accumulator = transformation.PatternNode(nodes.AccessNode)
    global_hbm = transformation.PatternNode(nodes.AccessNode)

    # Properties
    npe_x = Property(default=None, allow_none=True, desc="Number of processing elements")
    npe_y = Property(default=None, allow_none=True, desc="Number of processing elements")
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
    kg_m = Property(default=None, allow_none=True, desc="kg_m")
    kg_n = Property(default=None, allow_none=True, desc="kg_n")

    reduce_cond = Property(default=None, allow_none=True, desc="decide which PE to reduce")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.accumulator, cls.global_hbm)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        for edge in graph.out_edges(self.accumulator):
            if edge.data.wcr is not None:
                print("WCR supported")
        return True

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):
        accumulator = self.accumulator
        acc_desc = sdfg.arrays[accumulator.data]
        accumulator_elem_num = acc_desc.total_size
        data_size = acc_desc.dtype.bytes
        accumulator_size = accumulator_elem_num * data_size

        global_hbm = self.global_hbm
        desc_hbm = sdfg.arrays[global_hbm.data]
        is_interleaved = desc_hbm.is_hbm_interleaved

        npe_x = self.npe_x
        npe_y = self.npe_y
        NPE = self.npe_x * self.npe_y
        tM = self.tM
        tN = self.tN
        tK = self.tK
        M = self.M
        N = self.N
        K = self.K
        gi = self.gi
        gj = self.gj
        kg_m = self.kg_m
        kg_n = self.kg_n
        reduce_cond = self.reduce_cond

        kg_i = gi // kg_m
        kg_j = gj // kg_n
        kg_oi = gi % kg_m
        kg_oj = gj % kg_n
        kg_num = kg_m * kg_n
        kg_off = kg_oi * kg_n + kg_oj
        from math import sqrt
        if reduce_cond is None:
            reduce_cond = f"({kg_off} == {((gi+gj*npe_x)//(int(sqrt(NPE))))%kg_num})"

        for edge in graph.out_edges(self.accumulator):
            if edge.data.wcr is not None:
                edge_to_replace = edge
                break

        print("Edge to replace: ", edge_to_replace)

        subset = edge_to_replace.data.subset
        beg, end, step = subset.ranges[0]
        length_0 = (end + 1) - beg
        beg, end, step = subset.ranges[1]
        length_1 = (end + 1) - beg
        src_strides = acc_desc.strides[-2:]
        dst_strides = desc_hbm.strides[-2:]

        edge_to_replace.data.wcr = None
        graph.remove_edge(edge_to_replace)
        reduction_tasklet_str = ""
        reduction_tasklet_str += f"if ({reduce_cond} && flex_is_dm_core()) {{\n"
        reduction_tasklet_str += (f"    flex_dma_async_1d_reduction(local(_in_accumulator), "
                                  f"dace_remote_xy(gi+{kg_m-1-kg_oi},gj+{kg_n-1-kg_oj},_in_accumulator,{npe_x}), "
                                  f"{accumulator_size}, COLLECTIVE_REDADD_FP_16);\n")

        reduction_tasklet_str += "    bare_dma_wait_all(); \n"
        reduction_tasklet_str += f"}}\n"
        reduction_tasklet_str += "flex_intra_cluster_sync();\n"
        reduction_tasklet = graph.add_tasklet(name="split_K_reduction",
                                              inputs={"_in_accumulator"},
                                              outputs={"_out_accumulator"},
                                              code=reduction_tasklet_str,
                                              language=dtypes.Language.CPP)

        tasklet_in_edge = graph.add_edge(accumulator, None, reduction_tasklet, "_in_accumulator",
                                         Memlet(f"{accumulator.data}"))

        # new access node of acc
        new_acc_an = graph.add_access(f"{accumulator.data}")
        tasklet_out_edge = graph.add_edge(reduction_tasklet, "_out_accumulator", new_acc_an, None,
                                          Memlet(f"{accumulator.data}"))

        # add a new nsdfg node
        nsdfg = sd.SDFG("conditional_store")

        n_acc_desc = copy.deepcopy(acc_desc)
        n_acc_desc.transient = False
        nsdfg.add_datadesc(accumulator.data, n_acc_desc)
        n_hbm_desc = copy.deepcopy(desc_hbm)
        n_hbm_desc.set_shape(new_shape=(n_acc_desc.shape[-2], n_acc_desc.shape[-1]), strides=n_hbm_desc.strides)
        n_hbm_desc.transient = False
        nsdfg.add_datadesc(global_hbm.data, n_hbm_desc)
        nsdfg.add_symbol("M", dtypes.typeclass(int))
        nsdfg.add_symbol("N", dtypes.typeclass(int))
        nsdfg.add_symbol("K", dtypes.typeclass(int))
        nsdfg.add_symbol("gi", dtypes.typeclass(int))
        nsdfg.add_symbol("gj", dtypes.typeclass(int))

        # add a state to the nsdfg
        nsdfg_start_state = nsdfg.add_state("conditional_store_start", is_start_block=True)

        cb_store = ConditionalBlock(label="conditional_store", sdfg=nsdfg)
        nsdfg.add_edge(nsdfg_start_state, cb_store, InterstateEdge())

        cb_store_cfg = ControlFlowRegion(label="cb_store")

        store_cfg_cond = CodeBlock(code=f"{reduce_cond}", language=dtypes.Language.Python)

        cb_store.add_branch(condition=store_cfg_cond, branch=cb_store_cfg)

        store_state = cb_store_cfg.add_state("store_state")
        cb_acc_an = store_state.add_access(accumulator.data)
        cb_hbm_an = store_state.add_access(global_hbm.data)
        store_state.add_edge(cb_acc_an, None, cb_hbm_an, None, Memlet(f"{global_hbm.data}"))

        nested_sdfg = graph.add_nested_sdfg(nsdfg, None, inputs={"accumulator"}, outputs={"C"})
        nsdfg_in_edge = graph.add_edge(new_acc_an, None, nested_sdfg, "accumulator", Memlet(f"{accumulator.data}"))

        nsdfg_out_edge = graph.add_edge(nested_sdfg, "C", edge_to_replace.dst, edge_to_replace.dst_conn,
                                        copy.deepcopy(edge_to_replace.data))
