# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
import copy
from typing import Any, Dict, Optional, Set
import dace
from dace import SDFG, InterstateEdge
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes import analysis as ap
from dace import properties
import dace.sdfg.construction_utils as cutil

@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveReduntantCopyTasklets(ppl.Pass):
    copy_tasklet_pattern = properties.Property(dtype=str, default='{_out} = {_in}', allow_none=True)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _arr_appears_in_loop(self, arr_name: str, loop: LoopRegion):
        pass

    def _arr_appears_in_conditional(self, arr_name: str, conditional: ConditionalBlock):
        pass

    def _scalar_or_len_one_array(self, arr_name: str, sdfg: dace.SDFG, only_if_transient: bool = False):
        arr = sdfg.arrays[arr_name]
        if only_if_transient and arr.transient is False:
            return False

        return (
            isinstance(arr, dace.data.Scalar) or
            (
                isinstance(arr, dace.data.Array) and 
                (
                    arr.shape == (1,) or arr.shape == [1,]
                )
            )
        )

    def _apply(self, sdfg: SDFG):
        # If AccessNode (A1) -> CopyTasklet -> AccessNode (A2)
        # If copy tasklet matches the pattern e.g. `{rhs} = {lhs}` or `vector_copy({rhs}, {lhs});`
        # If access node is not accessed anywhere at all (not appearing as a memlet data, interstate edge read, for/ifblocks)
        replacement_dict = dict()
        for state in sdfg.all_states():
            for tasklet in {n for n in state.nodes() if isinstance(n, dace.nodes.Tasklet)}:
                if len(tasklet.in_connectors) == 1 and len(tasklet.out_connectors) == 1:
                    in_conn = next(iter(tasklet.in_connectors))
                    out_conn = next(iter(tasklet.out_connectors))

                    copy_template = self.copy_tasklet_pattern.format(
                        _in=in_conn,
                        _out=out_conn
                    )

                    if tasklet.code.as_string == copy_template:
                        # We can remove it if source is transient
                        in_edges = state.in_edges(tasklet)
                        out_edges = state.out_edges(tasklet)

                        if (
                            len(in_edges) == 1 and len(out_edges) == 1 and
                            isinstance(in_edges[0].src, dace.nodes.AccessNode) and
                            isinstance(out_edges[0].dst, dace.nodes.AccessNode) and
                            self._scalar_or_len_one_array(out_edges[0].data.data, sdfg, True) and
                            state.out_degree(out_edges[0].dst) > 0
                           ):
                            dst_node = out_edges[0].dst
                            dst_out_edges = state.out_edges(dst_node)
                            state.remove_node(tasklet)
                            state.remove_node(dst_node)
                            in_edge = in_edges[0]

                            # Subset needs to be 0 so no need to change
                            for dst_out_edge in dst_out_edges:
                                state.add_edge(
                                    in_edge.src, None,
                                    dst_out_edge.dst, dst_out_edge.dst_conn,
                                    copy.deepcopy(in_edge.data)
                                )
                                replacement_dict[dst_out_edge.data.data] = in_edge.data.data

        sdfg.validate()

        for k in replacement_dict:
            if k in sdfg.arrays:
                sdfg.remove_data(k)

        sdfg.replace_dict(replacement_dict)

        for state in sdfg.all_states():
            for n in {n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG)}:
                self._apply(n.sdfg)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        self._apply(sdfg)

    def report(self, pass_retval: Any) -> Optional[str]:
        return f''
