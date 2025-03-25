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
    M  = SymbolicProperty(default=None, allow_none=True, desc="M")
    N  = SymbolicProperty(default=None, allow_none=True, desc="N")
    K  = SymbolicProperty(default=None, allow_none=True, desc="K")
    gi = SymbolicProperty(default=None, allow_none=True, desc="gi")
    gj = SymbolicProperty(default=None, allow_none=True, desc="gj")
    i = SymbolicProperty(default=None, allow_none=True, desc="i")
    j = SymbolicProperty(default=None, allow_none=True, desc="j")
    kg_m = Property(default=None, allow_none=True, desc="kg_m")
    kg_n = Property(default=None, allow_none=True, desc="kg_n")

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

        NPE = self.npe_x
        npe_x = self.npe_x
        npe_y = self.npe_y
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
        
        kg_i = gi // kg_m
        kg_j = gj // kg_n
        kg_oi = gi % kg_m
        kg_oj = gj % kg_n
        kg_num = kg_m * kg_n
        kg_off = kg_oi * kg_n + kg_oj
        
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
        reduction_tasklet_str += f"if ({kg_off} == 0 && flex_is_dm_core()) {{\n"
        reduction_tasklet_str += (
            f"    bare_dma_start_1d_reduction(local(_in_accumulator), "
            f"dace_remote_xy(gi+{kg_m-1},gj+{kg_n-1},_in_accumulator,{npe_x}), "
            f"{accumulator_size}, COLLECTIVE_REDADD_FP_16);\n"
        )
        

        reduction_tasklet_str += "    bare_dma_wait_all(); \n"
        reduction_tasklet_str += f"}}\n"
        reduction_tasklet_str += "flex_intra_cluster_sync();\n"
        reduction_tasklet = graph.add_tasklet(
            name="split_K_reduction",
            inputs={"_in_accumulator"},
            outputs={"_out_accumulator"},
            code=reduction_tasklet_str,
            language=dtypes.Language.CPP
        )
        
        tasklet_in_edge = graph.add_edge(
            accumulator, None,
            reduction_tasklet, "_in_accumulator",
            Memlet(f"{accumulator.data}")
        )
        

        # new access node of acc
        new_acc_an = graph.add_access(f"{accumulator.data}")
        tasklet_out_edge = graph.add_edge(
            reduction_tasklet, "_out_accumulator",
            new_acc_an, None,
            Memlet(f"{accumulator.data}")
        )

        # add a new nsdfg node
        nsdfg = sd.SDFG("conditional_store")
        
        n_acc_desc = copy.deepcopy(acc_desc)
        n_acc_desc.transient = False
        nsdfg.add_datadesc(accumulator.data, n_acc_desc)
        n_hbm_desc = copy.deepcopy(desc_hbm)
        n_hbm_desc.set_shape(new_shape=(n_acc_desc.shape[-2], n_acc_desc.shape[-1]),
                             strides=n_hbm_desc.strides)
        n_hbm_desc.transient = False
        nsdfg.add_datadesc(global_hbm.data, n_hbm_desc)
        nsdfg.add_symbol("M", dtypes.typeclass(int))
        nsdfg.add_symbol("N", dtypes.typeclass(int))
        nsdfg.add_symbol("K", dtypes.typeclass(int))
        nsdfg.add_symbol("gi", dtypes.typeclass(int))
        nsdfg.add_symbol("gj", dtypes.typeclass(int))

        # add a state to the nsdfg
        nsdfg_start_state = nsdfg.add_state("conditional_store_start", is_start_block=True)

        cb_store = ConditionalBlock(
            label="conditional_store",
            sdfg=nsdfg
        )
        nsdfg.add_edge(nsdfg_start_state, cb_store, InterstateEdge())

        cb_store_cfg = ControlFlowRegion(
            label="cb_store"
        )

        store_cfg_cond = CodeBlock(
            code=f"{kg_off} == 0",
            language=dtypes.Language.Python
        )

        cb_store.add_branch(
            condition=store_cfg_cond,
            branch=cb_store_cfg
        )

        store_state = cb_store_cfg.add_state("store_state")
        cb_acc_an = store_state.add_access(accumulator.data)
        cb_hbm_an = store_state.add_access(global_hbm.data)
        store_state.add_edge(
            cb_acc_an,
            None,
            cb_hbm_an,
            None,
            Memlet(f"{global_hbm.data}")
        )
        
        nested_sdfg = graph.add_nested_sdfg(nsdfg, None,
                                        inputs={"accumulator"},
                                        outputs={"C"})
        nsdfg_in_edge = graph.add_edge(
            new_acc_an, None,
            nested_sdfg, "accumulator",
            Memlet(f"{accumulator.data}")
        )

        nsdfg_out_edge = graph.add_edge(
            nested_sdfg, "C",
            edge_to_replace.dst, edge_to_replace.dst_conn,
            copy.deepcopy(edge_to_replace.data)
        )
          
    def _emit_hbm_interleaved_code(self, nodedesc, src_name, s, block_height, block_width, data_size, is_load):
        """
        Generates code for HBM interleaved memory access.

        Parameters:
            nodedesc: Descriptor of the HBM array (contains shape and split scheme).
            src_name: Name of the source array.
            s: Subset ranges (list of tuples), where s[0][0] is row start and s[1][0] is column start.
            block_height: Height of the block (number of elements in dimension 0).
            block_width: Width of the block (number of elements in dimension 1).
            data_size: Size of each data element in bytes.
            is_load: Boolean flag, True if loading from HBM, False if storing to HBM.
            
        Returns:
            A string containing the generated code.
        """
        lines = []
        hbm_width = nodedesc.shape[1]
        hbm_height = nodedesc.shape[0]
        row_start = sym2cpp(s[0][0])
        col_start = sym2cpp(s[1][0])
        height_split = nodedesc.hbm_split_scheme[0]
        width_split = nodedesc.hbm_split_scheme[1]
        lines.append(f"    const uint32_t tile_width = {src_name}_tile_width;")
        lines.append(f"    const uint32_t tile_height = {src_name}_tile_height;")
        lines.append(f"    const uint32_t row_start_offset = (({src_name} - {src_name}_base) / {data_size}) / {src_name}_width;")
        lines.append(f"    const uint32_t col_start_offset = (({src_name} - {src_name}_base) / {data_size}) % {src_name}_width;")
        lines.append(f"    const uint32_t col_start_temp = {col_start} + col_start_offset;")
        lines.append(f"    const uint32_t col_start = col_start_temp % {src_name}_width;")
        lines.append(f"    const uint32_t row_start = {row_start} + row_start_offset + col_start_temp / {src_name}_width;")
        lines.append(f"    const uint32_t tile_row_index = row_start / tile_height;")
        lines.append(f"    const uint32_t tile_col_index = col_start / tile_width;")
        lines.append(f"    const uint32_t tile_row_offset = row_start % tile_height;")
        lines.append(f"    const uint32_t tile_col_offset = col_start % tile_width;")
        lines.append(f"    const uint32_t tile_index = tile_row_index * {width_split} + tile_col_index;")
        lines.append(f"    const uint32_t channel_id = {src_name}_placement_info[tile_index].channel_id;")
        lines.append(f"    const uint32_t num_blocks_per_tile = (tile_height / {block_height}) * (tile_width / {block_width});")
        lines.append(f"    const uint32_t num_blocks_in_previous_tiles_in_channel = {src_name}_placement_info[tile_index].tile_offset * num_blocks_per_tile;")
        lines.append(f"    const uint32_t block_row_index = tile_row_offset / {block_height};")
        lines.append(f"    const uint32_t block_col_index = tile_col_offset / {block_width};")
        lines.append(f"    const uint32_t block_index = block_row_index * (tile_width / {block_width}) + block_col_index;")
        lines.append(f"    const uint32_t total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;")
        lines.append(f"    const uint64_t block_addr = {src_name}_base + channel_id * ARCH_HBM_NODE_ADDR_SPACE + total_block_index * {block_height} * {block_width} * {data_size};")
        funcname = "flex_dma_async_1d"
        if is_load:
            # For load operation: HBM -> local
            lines.append(f"    {funcname}(local(_in_accumulator), hbm_addr(block_addr), {block_height}*{block_width}*{data_size});")
        else:
            # For store operation: local -> HBM
            lines.append(f"    {funcname}(hbm_addr(block_addr), local(_in_accumulator), {block_height}*{block_width}*{data_size});")
        return "\n".join(lines)
            # if is_interleaved:
        #     # Use the adapted _emit_hbm_interleaved_code for interleaved HBM
        #     reduction_tasklet_str += "    bare_dma_wait_all(); \n"
        #     interleaved_code = self._emit_hbm_interleaved_code(
        #         desc_hbm,            # nodedesc
        #         global_hbm.data,     # src_name
        #         subset.ranges,       # s: list of ranges (s[0][0] and s[1][0] used for row and col start)
        #         length_0,            # block_height
        #         length_1,            # block_width
        #         data_size,           # data_size
        #         is_load=False        # store from local -> HBM in reduction
        #     )
        #     reduction_tasklet_str += interleaved_code + "\n"
        # else:
        #     reduction_tasklet_str += "    bare_dma_wait_all(); \n"
        #     funcname = "flex_dma_sync_2d"
        #     reduction_tasklet_str += (
        #         f"    {funcname}("
        #         f"hbm_addr(_out_C), "
        #         f"local(_in_accumulator), "
        #         f"{length_1}*{data_size}, "
        #         f"{dst_strides[0]}*{data_size}, "
        #         f"{src_strides[0]}*{data_size}, "
        #         f"{length_0});\n"
        #     )