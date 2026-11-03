# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loop tiling: strip-mine a sequential loop into a loop over tiles holding a loop over one tile's points."""

from dace import symbolic
from dace.properties import CodeBlock, Property, SymbolicProperty, make_properties
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.transformation import explicit_cf_compatible
from dace.utils import find_new_name


@make_properties
@explicit_cf_compatible
class LoopTiling(xf.MultiStateTransformation):
    """``for i in range(lo, hi)`` becomes ``for tile_i in range(lo, hi, K): for i in range(tile_i, min(hi, tile_i + K))``:
    the same iterations in the same order, so it is legal for every unit-stride loop. A tile size need not divide the
    range; the clamp keeps the last tile inside it. ``UntileLoops.untile_loop`` is the inverse."""

    loop = xf.PatternNode(LoopRegion)

    tile_size = SymbolicProperty(default=32, desc="Iterations per tile: a positive integer or a positive symbol")
    prefix = Property(dtype=str, default="tile", desc="Prefix of the tile loop's iteration variable")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        loop: LoopRegion = self.loop
        if not loop.loop_variable or loop.inverted:
            return False
        if loop_analysis.get_init_assignment(loop) is None or loop_analysis.get_loop_end(loop) is None:
            return False
        stride = loop_analysis.get_loop_stride(loop)
        if stride is None or symbolic.simplify(stride) != 1:
            return False
        size = symbolic.pystr_to_symbolic(str(self.tile_size))
        # a tile of one iteration is the loop itself
        return not (size.is_number and int(size) < 2)

    def apply(self, graph: ControlFlowRegion, sdfg):
        loop: LoopRegion = self.loop
        parent = loop.parent_graph
        var = loop.loop_variable
        start = loop_analysis.get_init_assignment(loop)
        stop = symbolic.simplify(loop_analysis.get_loop_end(loop) + 1)
        size = symbolic.pystr_to_symbolic(str(self.tile_size))
        taken = {
            *sdfg.symbols,
            *sdfg.arrays,
            *(
                r.loop_variable
                for r in sdfg.all_control_flow_regions(recursive=True)
                if isinstance(r, LoopRegion) and r.loop_variable
            ),
        }
        tile = find_new_name(f"{self.prefix}_{var}", taken)

        tiles = LoopRegion(
            f"{loop.label}_tiles",
            f"{tile} < {symbolic.symstr(stop)}",
            tile,
            f"{tile} = {symbolic.symstr(start)}",
            f"{tile} = {tile} + {symbolic.symstr(size)}",
        )
        parent.add_node(tiles, is_start_block=parent.start_block is loop)
        for edge in list(parent.in_edges(loop)):
            parent.add_edge(edge.src, tiles, edge.data)
            parent.remove_edge(edge)
        for edge in list(parent.out_edges(loop)):
            parent.add_edge(tiles, edge.dst, edge.data)
            parent.remove_edge(edge)
        parent.remove_node(loop)
        tiles.add_node(loop, is_start_block=True)
        loop.init_statement = CodeBlock(f"{var} = {tile}")
        loop.loop_condition = CodeBlock(f"{var} < min({symbolic.symstr(stop)}, {tile} + {symbolic.symstr(size)})")
        sdutil.set_nested_sdfg_parent_references(sdfg)
        sdfg.reset_cfg_list()
        return tiles
