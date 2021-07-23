from typing import Any, Dict, Iterable, List, Tuple, Union

import networkx
from dace import dtypes, properties, registry, subsets, symbolic
from dace.sdfg import utils, graph
from dace.codegen.targets import fpga
from dace.transformation import transformation, interstate
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet


@registry.autoregister
@properties.make_properties
class HbmTransform(transformation.Transformation):
    # type=List[Tuple[str, str, str]]
    update_array_banks = properties.Property(
        dtype=List,
        default=[],
        desc=
        "List of (arrayname, value for location['memorytype'], value for location['bank']])."
        "The shape of the array will be updated depending on it's previous placement."
        "DDR->HBM: Add a dimension, HBM->HBM: Scale first dimension, HBM->DDR: Remove first dimension."
    )

    # type=List[str]
    update_array_access = properties.Property(
        dtype=List,
        default=[],
        desc=
        "For all source and sink nodes accessing arrays defined here the distributed subset"
        " of the top-level map will be added. Additionally only the current bank will be passed to the"
        " nested SDFG for those arrays. At the moment such accessnode may only have exactly one"
        " ingoing/outgoing edge to have their accesses updated. ")

    # type=Tuple[str, str]
    outer_map_range = properties.Property(
        dtype=Tuple,
        default=("k", "0"),
        desc="Stores the range for the outer HBM map. Defaults to k = 0.")

    def _multiply_sdfg_executions(self, sdfg: SDFG):
        """
        Nests a whole SDFG and packs it into an unrolled map. 
        Depending on the values in update_array_access the first
        index of inputs/outputs is changed to the map param.
        """
        unrollparam = self.outer_map_range
        for state in sdfg.states():
            for outer_node in state.source_nodes() + state.sink_nodes():
                if (isinstance(outer_node, nd.AccessNode)
                        and outer_node.data in self.update_array_access
                        and len(state.all_edges(outer_node)) == 1):
                    self._update_memlet_hbm(state, outer_node, unrollparam[0])

        nesting = interstate.NestSDFG(sdfg.sdfg_id, -1, {}, self.expr_index)
        nesting.apply(sdfg)
        state = sdfg.states()[0]
        nsdfg_node = list(
            filter(lambda x: isinstance(x, nd.NestedSDFG), state.nodes()))[0]
        nsdfg_node.no_inline = True

        map_enter, map_exit = state.add_map("hbm_unrolled_map",
                                            {unrollparam[0]: unrollparam[1]},
                                            dtypes.ScheduleType.Unrolled)

        inputs = []
        outputs = []
        for attached in state.all_edges(nsdfg_node):
            if attached.data.data in self.update_array_access:
                para_sym = symbolic.pystr_to_symbolic(unrollparam[0])
                attached.data.subset[0] = (para_sym, para_sym, 1)
            if attached in state.in_edges(nsdfg_node):
                inputs.append(attached)
            else:
                outputs.append(attached)
            state.remove_edge(attached)
        for input in inputs:
            state.add_memlet_path(input.src,
                                  map_enter,
                                  nsdfg_node,
                                  memlet=input.data,
                                  src_conn=input.src_conn,
                                  dst_conn=input.dst_conn)
        for output in outputs:
            state.add_memlet_path(nsdfg_node,
                                  map_exit,
                                  output.dst,
                                  memlet=output.data,
                                  src_conn=output.src_conn,
                                  dst_conn=output.dst_conn)

    def _update_memlet_hbm(self, state: SDFGState,
                           convertible_node: nd.AccessNode,
                           inner_subset_index: symbolic.symbol):
        """
        Add the subset_index to the memlet path defined by convertible_node. If the end/start of
        the path is also an AccessNode, it will insert a tasklet before the access to 
        avoid validation failures due to dimensionality mismatch.
        :param convertible_node: An AccessNode with exactly one attached memlet path
        :param inner_subset_index: The distributed subset for the innermost edge on
            the memlet path defined by convertible_node
        """
        # get the inner edge:
        if len(state.out_edges(convertible_node)) == 1:
            inner_edge = state.memlet_path(
                state.out_edges(convertible_node)[0])[-1]
        else:
            inner_edge = state.memlet_path(
                state.in_edges(convertible_node)[0])[0]

        mem: memlet.Memlet = inner_edge.data
        new_subset = subsets.Range(
            [[inner_subset_index, inner_subset_index, 1]] +
            [x for x in mem.subset])

        path = state.memlet_path(inner_edge)
        edge_index = path.index(inner_edge)
        if edge_index == 0:
            is_write = True
            other_node = path[0].src
        elif edge_index == len(path) - 1:
            is_write = False
            other_node = path[-1].dst
        else:
            raise ValueError("The provided edge is not the innermost")

        if isinstance(other_node, nd.AccessNode):
            fwtasklet = state.add_tasklet("fwtasklet", set(["_in"]),
                                          set(["_out"]), "_out = _in")
            state.remove_edge_and_connectors(inner_edge)
            target_other_subset = mem.other_subset
            mem.other_subset = None
            if is_write:
                inner_edge = state.add_edge(fwtasklet, '_out', inner_edge.dst,
                                            inner_edge.dst_conn, mem)
                state.add_edge(
                    other_node, path[0].src_conn, fwtasklet, "_in",
                    memlet.Memlet(other_node.data, subset=target_other_subset))
            else:
                inner_edge = state.add_edge(inner_edge.src, inner_edge.src_conn,
                                            fwtasklet, '_in', mem)
                state.add_edge(
                    fwtasklet, "_out", other_node, path[-1].dst_conn,
                    memlet.Memlet(other_node.data, subset=target_other_subset))

        utils.update_path_subsets(state, inner_edge, new_subset)

    def _update_array_hbm(self, array_name: str, new_memory: str, new_bank: str,
                          sdfg: SDFG):
        """
        Updates bank assignments for the array
        """
        desc = sdfg.arrays[array_name]
        old_memory = None
        if 'memorytype' in desc.location and desc.location[
                "memorytype"] is not None:
            old_memory = desc.location["memorytype"]
        if new_memory == "HBM":
            low, high = fpga.get_multibank_ranges_from_subset(new_bank, sdfg)
        else:
            low, high = int(new_bank), int(new_bank) + 1
        if (old_memory is None or old_memory == "DDR") and new_memory == "HBM":
            desc = utils.update_array_shape(sdfg, array_name,
                                            (high - low, *desc.shape))
        elif old_memory == "HBM" and (new_memory == "DDR"
                                      or new_memory is None):
            desc = utils.update_array_shape(array_name, *(list(desc.shape)[1:]))
        elif old_memory == "HBM" and new_memory == "HBM":
            new_shape = list(desc.shape)
            new_shape[0] = high - low
            desc = utils.update_array_shape(sdfg, array_name, new_shape)
        desc.location["memorytype"] = new_memory
        desc.location['bank'] = new_bank
        desc.storage = dtypes.StorageType.FPGA_Global

    @staticmethod
    def can_be_applied(self, graph: Union[SDFG, SDFGState],
                       candidate: Dict['PatternNode', int], expr_index: int,
                       sdfg: SDFG, strict: bool) -> bool:
        return True

    @staticmethod
    def expressions():
        # Matches anything
        return [networkx.DiGraph()]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        # update array bank positions
        for array_name, memory_type, bank in self.update_array_banks:
            self._update_array_hbm(array_name, memory_type, bank, sdfg)

        # nest the sdfg and execute in parallel
        if (0, 0, 1) != subsets.Range.from_string(self.outer_map_range[1])[0]:
            self._multiply_sdfg_executions(sdfg)

        # set default on all outer arrays, such that FPGA_transformation can be used
        for desc in sdfg.arrays.items():
            desc[1].storage = dtypes.StorageType.Default
