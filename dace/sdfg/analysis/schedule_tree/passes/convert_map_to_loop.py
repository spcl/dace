from dace.sdfg import state
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def convert_map_to_loop(stree: tn.ScheduleTreeRoot) -> int:
    """Convert all maps to loop.
    The hypothesis is that there is no dynamic maps here"""
    class ConvertMapToLoop(tn.ScheduleNodeTransformer):
        def __init__(self) -> None:
            super().__init__()
            self.map_converted = 0

        def __str__(self) -> str:
            return "ConvertMapToLoop"

        def visit_MapScope(self, the_map: tn.MapScope) -> tn.ForScope:
            new_childs = []
            for child in the_map.children:
                new_childs.append(self.visit(child))

            # The below is taken directly from `MapToForLoop` SDFG transform
            loop_idx = the_map.node.map.params[0]
            loop_from, loop_to, loop_step = the_map.node.map.range[0]
            loop_region = state.LoopRegion(
                "loop_" + the_map.node.map.label, f"{loop_idx} < {loop_to + 1}",
                loop_idx, f"{loop_idx} = {loop_from}",
                f"{loop_idx} = {loop_idx} + {loop_step}"
            )
            for_scope = tn.ForScope(
                loop=loop_region,
                children=new_childs,
                parent=the_map.parent,
            )        

            for child in new_childs:
                child.parent = for_scope

            self.map_converted += 1
            return for_scope


    converter = ConvertMapToLoop()
    converter.visit(stree)
    return converter.map_converted

