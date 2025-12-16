# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Any, Dict
from dace import SDFG, InterstateEdge, properties
from dace.sdfg import nodes
from dace.sdfg.state import BreakBlock, ControlFlowRegion
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.eliminate_branches import EliminateBranches
from dace.transformation.passes.vectorization.vectorization_utils import *
import dace
import dace.sdfg.construction_utils as cutil
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

safe_mask_template_vv = """
    // Phase 1: compute condition array
    bool had_break_cond = false;
    for (int i = 0; i < {VLEN}; ++i) {{
        bool break_cond = ({a}[i] {op} {b}[i]);
        // Grace not grace update before else update after
        if (!{grace}){{
            had_break_cond = had_break_cond || break_cond;
        }}
        if (had_break_cond){{
            {out}[i] = 0.0;
        }} else {{
            {out}[i] = 1.0;
        }}
        if ({grace}){{
            had_break_cond = had_break_cond || break_cond;
        }}
    }}
"""

safe_mask_template_vs = """
    // Phase 1: compute condition array
    bool had_break_cond = false;
    for (int i = 0; i < {VLEN}; ++i) {{
        bool break_cond = ({a}[i] {op} {sc});
        // Grace not grace update before else update after
        if (!{grace}){{
            had_break_cond = had_break_cond || break_cond;
        }}
        if (had_break_cond){{
            {out}[i] = 0.0;
        }} else {{
            {out}[i] = 1.0;
        }}
        if ({grace}){{
            had_break_cond = had_break_cond || break_cond;
        }}
    }}
"""

safe_mask_template_sv = """
    // Phase 1: compute condition array
    bool had_break_cond = false;
    for (int i = 0; i < {VLEN}; ++i) {{
        bool break_cond = ({sc} {op} {a}[i]);
        // Grace not grace update before else update after
        if (!{grace}){{
            had_break_cond = had_break_cond || break_cond;
        }}
        if (had_break_cond){{
            {out}[i] = 0.0;
        }} else {{
            {out}[i] = 1.0;
        }}
        if ({grace}){{
            had_break_cond = had_break_cond || break_cond;
        }}
    }}
"""


def move_state_inside_if(condition_str: str, cfg: ControlFlowRegion, state: SDFGState):
    ies = cfg.in_edges(state)
    oes = cfg.out_edges(state)

    cfg.remove_node(state)

    if_block = ConditionalBlock(label=f"wr_{state.label}", sdfg=state.sdfg, parent=cfg)
    if_cfg = ControlFlowRegion(label=f"body_{state.label}", sdfg=state.sdfg, parent=cfg)
    if_block.add_branch(condition=CodeBlock(condition_str), branch=if_cfg)
    if_body_state = if_cfg.add_state(f"st_{state.label}", is_start_block=True)

    cfg.add_node(if_block, is_start_block=state == cfg.start_block)

    cutil.copy_state_contents(state, if_body_state)

    for ie in ies:
        cfg.add_edge(ie.src, if_block, copy.deepcopy(ie.data))
    for oe in oes:
        cfg.add_edge(if_block, oe.dst, copy.deepcopy(oe.data))


def non_transient_sink_data(state: SDFGState):
    return {
        node.data
        for node in state.data_nodes() if (state.out_degree(node) == 0 and node.data in state.sdfg.arrays
                                           and state.sdfg.arrays[node.data].transient is False)
    }


def move_map_to_loop(state: SDFGState, map: dace.nodes.MapEntry):
    copystate = copy.deepcopy(state)
    map_entries = {n for n in copystate.nodes() if isinstance(n, dace.nodes.MapEntry)}
    assert len(map_entries) == 1
    new_map_entry = map_entries.pop()
    new_map_exit = copystate.exit_node(new_map_entry)
    all_nodes = copystate.all_nodes_between(new_map_entry, copystate.exit_node(new_map_entry))
    for n in all_nodes:
        copystate.remove_node(n)
    for ie in copystate.in_edges(new_map_entry):
        copystate.remove_node(ie.src)
    for oe in copystate.out_edges(new_map_exit):
        copystate.remove_node(oe.dst)
    copystate.remove_node(new_map_entry)
    copystate.remove_node(new_map_exit)
    assert len(copystate.nodes()) == 0

    map_params = map.map.params
    map_ranges = map.map.range
    assert len(map_params) == 1
    map_param = map_params[0]
    b, e, s = map_ranges[0]

    loop = LoopRegion(label="l1",
                      condition_expr=f"{map_param} < {e} + 1",
                      loop_var=f"{map_param}",
                      initialize_expr=f"{map_param} = {b}",
                      update_expr=f"{map_param} = {map_param} + {s}")

    copy_in = loop.add_state(label="l1_copy_in", is_start_block=True)

    assert map in state.nodes(), f"{map} not in {state.nodes()}"
    for oe in state.out_edges(map):
        if isinstance(oe.dst, dace.nodes.AccessNode):
            srca = copy_in.add_access(oe.data.data)
            dsta = copy_in.add_access(oe.dst.data)
            copy_in.add_edge(srca, None, dsta, None, dace.memlet.Memlet(data=oe.data.data, subset=oe.data.subset))

    nsdfgs = {n for n in state.all_nodes_between(map, state.exit_node(map)) if isinstance(n, dace.nodes.NestedSDFG)}
    assert len(nsdfgs) == 1, f"{nsdfgs}"
    nsdfg = nsdfgs.pop()

    nmap = dict()
    for node in nsdfg.sdfg.nodes():
        cnode = copy.deepcopy(node)
        nmap[node] = cnode
        loop.add_node(cnode)
        if nsdfg.sdfg.in_degree(node) == 0:
            loop.add_edge(copy_in, cnode, InterstateEdge())

    for edge in nsdfg.sdfg.edges():
        src = nmap[edge.src]
        dst = nmap[edge.dst]
        loop.add_edge(src, dst, copy.deepcopy(edge.data))

    sink_nodes = {n for n in loop.nodes() if loop.out_degree(n) == 0}
    assert len(sink_nodes) == 1, f"{sink_nodes}"
    sink_node = sink_nodes.pop()

    # Find safe write for copy out
    nns = {n for n in loop.nodes() if n.label == "safe_write"}
    assert len(nns) == 1
    nn = nns.pop()

    copy_out = loop.add_state_after(nn, label="l1_copy_out")
    #loop.add_edge(nn, copy_out, InterstateEdge())

    for ie in state.in_edges(state.exit_node(map)):
        if isinstance(ie.src, dace.nodes.AccessNode):
            srca = copy_out.add_access(ie.src.data)
            dsta = copy_out.add_access(ie.data.data)
            copy_out.add_edge(srca, None, dsta, None, dace.memlet.Memlet(data=ie.data.data, subset=ie.data.subset))

    g = state.parent_graph
    ies = g.in_edges(state)
    oes = g.out_edges(state)
    g.remove_node(state)
    for ie in ies:
        g.add_edge(ie.src, loop, copy.deepcopy(ie.data))
    for oe in oes:
        g.add_edge(loop, oe.dst, copy.deepcopy(oe.data))

    was_start = g.start_block == state
    g.remove_node(state)
    g.add_node(loop, is_start_block=was_start)

    for arr_name, arr in nsdfg.sdfg.arrays.items():
        if arr.transient and arr_name not in g.sdfg.arrays:
            copydesc = copy.deepcopy(arr)
            g.sdfg.add_datadesc(arr_name, copydesc)
        if arr.transient is False and arr_name not in g.sdfg.arrays:
            raise Exception("UWU")

    return loop


def extract_identifiers(expr: str):
    """Extract variable names and the single operator from a Python expression."""

    class Visitor(ast.NodeVisitor):

        def __init__(self):
            self.names = set()
            self.op = None  # operator string

        def visit_Name(self, node):
            self.names.add(node.id)

        def visit_Compare(self, node):
            # comparison like a > b or a == b
            if len(node.ops) != 1:
                raise ValueError("Expression must contain exactly one operator.")
            op = node.ops[0]

            op_map = {
                ast.Gt: ">",
                ast.Lt: "<",
                ast.GtE: ">=",
                ast.LtE: "<=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }
            self.op = op_map[type(op)]
            self.generic_visit(node)

        def visit_BinOp(self, node):
            # arithmetic operator like +, -, *, /, **
            if self.op is not None:
                raise ValueError("Expression must contain exactly one operator.")

            op_map = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.Pow: "**",
            }
            self.op = op_map[type(node.op)]
            self.generic_visit(node)

    tree = ast.parse(expr)
    vis = Visitor()
    vis.visit(tree)

    if vis.op is None:
        raise ValueError("No operator found in expression.")

    return vis.names, vis.op


def expand_identifier(name: str, lane: int, vector_arrays, scalar_symbols):
    """Return the lane-specific identifier."""
    if name in vector_arrays:
        # vector array -> element access
        return f"{name}[{lane}]"
    elif name in scalar_symbols:
        # scalar symbol -> lane-specific symbolic identifier
        return f"{name}_laneid_{lane}"
    else:
        return name


def generate_lane_expression(expr: str, lane: int, identifiers, vector_arrays, scalar_symbols):
    """Produce a lane-specific expression by replacing identifiers."""
    lane_expr = expr
    for name in identifiers:
        lane_expr = lane_expr.replace(name, expand_identifier(name, lane, vector_arrays, scalar_symbols))
    return lane_expr


c = 0


@properties.make_properties
@transformation.explicit_cf_compatible
class VectorizeBreak(ppl.Pass):
    # This pass is testes as part of the vectorization pipeline
    CATEGORY: str = 'Optimization Preparation'
    vector_width = properties.SymbolicProperty(default=8)

    def __init__(self, vector_width):
        self.vector_width = vector_width

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.InterstateEdges | ppl.Modifies.Tasklets | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def _has_break(self, cfg: LoopRegion):
        for node in cfg.all_control_flow_blocks():
            if isinstance(node, BreakBlock):
                return True
        return False

    def _has_no_nsdfg(self, cfg: LoopRegion):
        for state in cfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    return False
        return True

    def _has_no_inner_loops(self, cfg: LoopRegion):
        for node in cfg.all_control_flow_regions():
            if node == cfg:
                continue
            if isinstance(node, LoopRegion):
                return False
        return True

    def _apply(self, sdfg: SDFG):
        vlen = str(self.vector_width)
        GRACE_VAL = True
        global c
        mask_counter = 0
        #sdfg.append_global_code('#include "dace/vector_intrinsics/break_safe_mask.h"')
        sdfg.append_global_code('#include <iostream>')
        loops: Set[LoopRegion] = set()
        for cfg in sdfg.all_control_flow_regions():
            if isinstance(cfg, LoopRegion):
                print(cfg, self._has_break(cfg), self._has_no_nsdfg(cfg))
                if self._has_break(cfg) and self._has_no_nsdfg(cfg) and self._has_no_inner_loops(cfg):
                    print(cfg, " has a break but no nsdfg and innermsot loop")
                    loops.add(cfg)

        old_existing_maps = {n for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
        for loop in loops:
            LoopToMap().apply_to(sdfg=loop.sdfg, loop=loop, permissive=True)
        now_existing_maps = {(n, g) for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)}
        c += 1
        new_maps = now_existing_maps - old_existing_maps
        condition_strs = dict()
        print("New Maps", new_maps)
        for m, state in new_maps:
            m.map.schedule = dace.dtypes.ScheduleType.Sequential

            # Find break block condition
            for n in state.nodes():
                if isinstance(n, dace.nodes.NestedSDFG):
                    condition_str = None

                    for inner_node in n.sdfg.nodes():
                        # Find the break block inside if statement
                        if (isinstance(inner_node, ConditionalBlock) and len(inner_node.branches) == 1
                                and len(inner_node.branches[0][1].nodes()) == 1
                                and isinstance(inner_node.branches[0][1].nodes()[0], BreakBlock)):
                            assert condition_str is None
                            condition_str = inner_node.branches[0][0].as_string

                    assert condition_str is not None
                    condition_strs[m] = condition_str

                    #for nested_node in n.sdfg.nodes():
                    #    if isinstance(nested_node, SDFGState) and len(nested_node.nodes()) > 0:
                    #        move_state_inside_if(f"1", n.sdfg, nested_node)
        c += 1
        EliminateBranches().apply_pass(sdfg, {})
        for new_map, state in new_maps:
            print(new_map)
            VectorizeCPU(vector_width=self.vector_width,
                         try_to_demote_symbols_in_nsdfgs=True,
                         fuse_overlapping_loads=False,
                         apply_on_maps=[new_map],
                         insert_copies=True,
                         eliminate_trivial_vector_map=True,
                         fail_on_unvectorizable=True).apply_pass(sdfg, {})

        for new_map, state in new_maps:
            backup = state
            condition_str = condition_strs[new_map]

            for n in state.nodes():
                if isinstance(n, dace.nodes.NestedSDFG):
                    # Need to know if we are before or after break
                    node_ids = {_n: i for i, _n in enumerate(n.sdfg.bfs_nodes())}
                    _break_node = None
                    for inner_node in n.sdfg.nodes():
                        # Find the break block inside if statement
                        if (isinstance(inner_node, ConditionalBlock) and len(inner_node.branches) == 1
                                and len(inner_node.branches[0][1].nodes()) == 1
                                and isinstance(inner_node.branches[0][1].nodes()[0], BreakBlock)):
                            _break_node = inner_node
                    assert _break_node is not None
                    if n.sdfg.out_degree(_break_node) == 0:
                        break_at_end = "true"
                    else:
                        break_at_end = "false"
                        ies = n.sdfg.in_edges(_break_node)
                        oes = n.sdfg.out_edges(_break_node)
                        n.sdfg.remove_node(_break_node)
                        assert len(ies) == 1
                        assert len(oes) <= 1
                        s = n.sdfg.add_state("emtpy")
                        for ie in ies:
                            n.sdfg.add_edge(ie.src, s, copy.deepcopy(ie.data))
                        for oe in oes:
                            n.sdfg.add_edge(s, oe.dst, copy.deepcopy(oe.data))
                        sink = {_n for _n in n.sdfg.nodes() if n.sdfg.out_degree(_n) == 0}.pop()
                        n.sdfg.add_node(_break_node)
                        n.sdfg.add_edge(sink, _break_node, InterstateEdge())

                    print(condition_str)
                    ccondition_str = condition_str[1:-1] if condition_str.startswith("(") and condition_str.endswith(
                        ")") else condition_str
                    tokens = ccondition_str.split(" ")
                    assert len(tokens) == 3
                    lhs, op, rhs = tokens
                    identifiers, op2 = extract_identifiers(condition_str)
                    assert op == op2
                    print(identifiers)
                    for lane in range(self.vector_width):
                        lane_expr = generate_lane_expression(
                            condition_str, lane, identifiers,
                            {s
                             for s, arr in n.sdfg.arrays.items() if isinstance(arr, dace.data.Array)},
                            {str(s)
                             for s in n.sdfg.symbols})
                        print(lane_expr)
                    sink_nodes_dicts = dict()
                    for inner_node in n.sdfg.nodes():
                        if isinstance(inner_node, SDFGState):
                            sink_nodes_dicts[inner_node] = {
                                s: f"{s}_breaksafe"
                                for s in non_transient_sink_data(inner_node)
                            }

                            for k, v in sink_nodes_dicts[inner_node].items():
                                for state_node in inner_node.nodes():
                                    if isinstance(state_node, dace.nodes.AccessNode) and inner_node.out_degree(
                                            state_node) == 0 and state_node.data == k:
                                        if v not in inner_node.sdfg.arrays:
                                            pdesc = inner_node.sdfg.arrays[k]
                                            inner_node.sdfg.add_array(name=v,
                                                                      shape=(self.vector_width, ),
                                                                      dtype=pdesc.dtype,
                                                                      storage=dace.dtypes.StorageType.Register,
                                                                      transient=True)
                                            assert pdesc.shape == (self.vector_width, )
                                        state_node.data = v
                                        for ie in inner_node.in_edges(state_node):
                                            if ie.data.data == k:
                                                ie.data.data = v

                    break_if_block = None
                    for inner_node in n.sdfg.nodes():
                        # Find the break block inside if statement
                        if (isinstance(inner_node, ConditionalBlock) and len(inner_node.branches) == 1
                                and len(inner_node.branches[0][1].nodes()) == 1
                                and isinstance(inner_node.branches[0][1].nodes()[0], BreakBlock)):
                            assert break_if_block is None
                            break_if_block = inner_node
                    assert break_if_block is not None

                    # Need to add 2 states after this state
                    # 1. Generate break safe mask
                    # This should be after writing to the break safe array

                    safe_mask_state = n.sdfg.add_state_before(break_if_block,
                                                              is_start_block=n.sdfg.start_block == break_if_block)
                    in_accesses = dict()
                    for identifier in identifiers:
                        in_accesses[identifier] = safe_mask_state.add_access(identifier)
                    out_access = safe_mask_state.add_access(f"safe_mask_c{mask_counter}")
                    mask_counter += 1
                    first_id = next(iter(identifiers))
                    inner_node.sdfg.add_array(
                        name=f"safe_mask_c{mask_counter-1}",
                        shape=(self.vector_width, ),
                        dtype=n.sdfg.arrays[first_id].dtype if first_id in n.sdfg.arrays else n.sdfg.symbols[first_id],
                        storage=dace.dtypes.StorageType.Register,
                        transient=True)
                    joined_inputs = ", ".join([f"_in_{identifier}" for identifier in identifiers])
                    assert len(identifiers) == 2 or len(identifiers) == 1
                    if len(identifiers) == 2:
                        id1, id2 = lhs, rhs
                        array1 = id1 in safe_mask_state.sdfg.arrays and isinstance(safe_mask_state.sdfg.arrays[id1],
                                                                                   dace.data.Array)
                        array2 = id2 in safe_mask_state.sdfg.arrays and isinstance(safe_mask_state.sdfg.arrays[id2],
                                                                                   dace.data.Array)
                        ss = ""
                        if array1 and array2:
                            ss = safe_mask_template_vv.format(T=safe_mask_state.sdfg.arrays[id1].dtype.ctype,
                                                              a=f"_in_{id1}",
                                                              b=f"_in_{id2}",
                                                              out=f"_out_{out_access}",
                                                              op=f"{op}",
                                                              VLEN=vlen,
                                                              grace=break_at_end)
                        elif array1 and not array2:
                            ss = safe_mask_template_vs.format(T=safe_mask_state.sdfg.arrays[id1].dtype.ctype,
                                                              a=f"_in_{id1}",
                                                              sc=f"_in_{id2}",
                                                              out=f"_out_{out_access}",
                                                              op=f"{op}",
                                                              VLEN=vlen,
                                                              grace=break_at_end)
                        elif not array1 and array2:
                            ss = safe_mask_template_sv.format(T=safe_mask_state.sdfg.arrays[id1].dtype.ctype,
                                                              sc=f"_in_{id1}",
                                                              b=f"_in_{id2}",
                                                              out=f"_out_{out_access}",
                                                              op=f"{op}",
                                                              VLEN=vlen,
                                                              grace=break_at_end)
                        else:
                            pass
                        assert ss != ""
                    elif len(identifiers) == 1:
                        ccondition_str = condition_str[1:-1] if condition_str.startswith(
                            "(") and condition_str.endswith(")") else condition_str
                        tokens = ccondition_str.split(" ")
                        assert len(tokens) == 3
                        lhs, op, rhs = tokens
                        print(tokens, lhs in safe_mask_state.sdfg.arrays, rhs in safe_mask_state.sdfg.arrays)
                        ss = ""
                        if lhs in safe_mask_state.sdfg.arrays:
                            assert isinstance(safe_mask_state.sdfg.arrays[lhs], dace.data.Array)
                            ss = safe_mask_template_vs.format(T=safe_mask_state.sdfg.arrays[lhs].dtype.ctype,
                                                              a=f"_in_{lhs}",
                                                              sc=f"{rhs}",
                                                              out=f"_out_{out_access}",
                                                              op=f"{op}",
                                                              VLEN=vlen,
                                                              grace=break_at_end)
                        elif rhs in safe_mask_state.sdfg.arrays:
                            assert isinstance(safe_mask_state.sdfg.arrays[rhs], dace.data.Array)
                            ss = safe_mask_template_vs.format(T=safe_mask_state.sdfg.arrays[rhs].dtype.ctype,
                                                              a=f"_in_{rhs}",
                                                              sc=f"{lhs}",
                                                              out=f"_out_{out_access}",
                                                              op=f"{op}",
                                                              VLEN=vlen,
                                                              grace=break_at_end)
                        else:
                            pass
                        assert ss != ""

                    t = safe_mask_state.add_tasklet("safe_mask", {f"_in_{identifier}"
                                                                  for identifier in identifiers},
                                                    {f"_out_{out_access}"},
                                                    code=ss,
                                                    language=dace.dtypes.Language.CPP)
                    for name, in_access in in_accesses.items():
                        safe_mask_state.add_edge(
                            in_access, None, t, f"_in_{name}",
                            dace.memlet.Memlet.from_array(in_access.data, safe_mask_state.sdfg.arrays[in_access.data]))
                    safe_mask_state.add_edge(
                        t, f"_out_{out_access}", out_access, None,
                        dace.memlet.Memlet.from_array(out_access.data, safe_mask_state.sdfg.arrays[out_access.data]))

                    # 2. Add safe write
                    name_map_rev = dict()
                    for inner_node in n.sdfg.nodes():
                        if isinstance(inner_node, SDFGState):
                            name_map = sink_nodes_dicts.get(inner_node, dict())
                            name_map_rev.update({v: k for k, v in name_map.items()})

                    for k, v in name_map_rev.items():
                        print(k, v)
                        # Find last write to k
                        last_node_with_write_to_k = None
                        for inner_node in n.sdfg.bfs_nodes():
                            if isinstance(inner_node, SDFGState):
                                has_write_to_k = {
                                    node.data
                                    for node in state.data_nodes() if
                                    (state.out_degree(node) == 0 and node.data in state.sdfg.arrays and node.data == k)
                                }
                                last_node_with_write_to_k = inner_node
                        assert last_node_with_write_to_k is not None
                        self.add_safe_write(n.sdfg, last_node_with_write_to_k, f"safe_mask_c{mask_counter-1}",
                                            self.vector_width, {k: v})

                    # Update break condition
                    lane_exprs = [
                        generate_lane_expression(
                            condition_str, lane, identifiers,
                            {s
                             for s, arr in n.sdfg.arrays.items() if isinstance(arr, dace.data.Array)},
                            {str(s)
                             for s in n.sdfg.symbols}) for lane in range(self.vector_width)
                    ]
                    new_cond = " or ".join(lane_exprs)
                    print("NEW COND", new_cond)

                    break_if_block.branches[0][0] = CodeBlock(new_cond)

                    for arr_name, arr in n.sdfg.arrays.items():
                        if arr.transient is True:
                            arr.storage = dace.dtypes.StorageType.Register

                    # Replace with connector names
                    connector_name_map = dict()
                    for in_connector in n.in_connectors:
                        connector_name_map[in_connector] = next(iter(state.in_edges_by_connector(
                            n, in_connector))).data.data
                    for out_connector in n.out_connectors:
                        connector_name_map[out_connector] = next(iter(state.out_edges_by_connector(
                            n, out_connector))).data.data
                    state.sdfg.validate()
                    n.sdfg.replace_dict(connector_name_map)
                    for k, v in connector_name_map.items():
                        if k in n.in_connectors:
                            connval = n.in_connectors[k]
                            del n.in_connectors[k]
                            n.in_connectors[str(v)] = copy.deepcopy(connval)
                    for k, v in connector_name_map.items():
                        if k in n.out_connectors:
                            connval = n.out_connectors[k]
                            del n.out_connectors[k]
                            n.out_connectors[str(v)] = copy.deepcopy(connval)
                    for ie in state.in_edges(n):
                        if ie.dst_conn in connector_name_map:
                            ie.dst_conn = connector_name_map[ie.dst_conn]
                    for oe in state.out_edges(n):
                        if oe.src_conn in connector_name_map:
                            oe.src_conn = connector_name_map[oe.src_conn]

            # new map doesnt exit anymore because after map tiling map gets changed
            # and after that only option is to get the map through parent of nsdfg
            parent_map = state.scope_dict()[n]
            assert isinstance(parent_map, dace.nodes.MapEntry)
            for_cfg = move_map_to_loop(state, parent_map)
            self.flip_first_zero_in_conditions_before_break(for_cfg)

    def flip_first_zero_in_conditions_before_break(self, for_cfg: LoopRegion):
        return
        for node in for_cfg.bfs_nodes():
            if (isinstance(node, ConditionalBlock) and len(node.branches) == 1 and len(node.branches[0][1].nodes()) == 1
                    and isinstance(node.branches[0][1].nodes(), BreakBlock)):
                # Found break block return
                return
            else:
                if isinstance(node, SDFGState):
                    # Find first tasklet that starts with
                    for snode in node.nodes():
                        if isinstance(snode,
                                      dace.nodes.Tasklet) and snode.label.startswith("condition_symbol_to_scalar"):
                            ttemplate = """
                                // Flip the FIRST 0.0 to 1.0 in the input array
                                {{
                                bool flipped = false;

                                for (int i = 0; i < {VLEN}; ++i) {{
                                    if (!flipped && {arr}[i] == 0.0) {{
                                        {arr}[i] = 1.0;
                                        flipped = true;
                                    }}
                                }}
                                }}
                            """
                            assert snode.code.language == dace.dtypes.Language.CPP
                            snode.code = CodeBlock(snode.code.as_string + "\n" + ttemplate.format(
                                VLEN=self.vector_width,
                                arr=next(iter(snode.out_connectors)),
                            ),
                                                   language=dace.dtypes.Language.CPP)
                            # Only do it once
                            break

    def add_safe_write(self, sdfg: SDFG, state_to_gen_after: SDFGState, mask_name: str, VLEN, names: Dict[str, str]):
        loopreg = LoopRegion(label="safe_write",
                             condition_expr=f"swi < {VLEN}",
                             loop_var="swi",
                             initialize_expr="swi = 0",
                             update_expr="swi = swi + 1")
        ifb = ConditionalBlock(label="safe_write_cond", sdfg=sdfg, parent=loopreg)
        loopreg.add_node(ifb, is_start_block=True)
        ifcfg = ControlFlowRegion(label="safe_write_body_cfg", sdfg=sdfg, parent=ifb)
        ifb.add_branch(CodeBlock(f"{mask_name}[swi] == 1.0"), ifcfg)
        state = ifcfg.add_state(label="safe_write_body", is_start_block=True)
        sdfg.add_node(loopreg)
        for oe in sdfg.out_edges(state_to_gen_after):
            sdfg.remove_edge(oe)
            sdfg.add_edge(oe.src, loopreg, copy.deepcopy(oe.data))
            sdfg.add_edge(loopreg, oe.dst, InterstateEdge())
        if len(sdfg.out_edges(state_to_gen_after)) == 0:
            sdfg.add_edge(state_to_gen_after, loopreg, InterstateEdge())

        for k, v in names.items():
            ka = state.add_access(k)
            va = state.add_access(v)
            t = state.add_tasklet(f"safe_write_{v}", {"_in"}, {"_out"}, "_out = _in")
            state.add_edge(ka, None, t, "_in", dace.memlet.Memlet(f"{k}[swi]"))
            state.add_edge(t, "_out", va, None, dace.memlet.Memlet(f"{v}[swi]"))
            assert sdfg.arrays[v].shape == (self.vector_width, )

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> None:
        # If we have the pattern:
        # for (int i = 0; i < N; i ++){
        #    if (A[i] < 0.0){
        #       break;
        #    }
        #    A[i] = expr
        # }
        # Then we van vectorize it by:
        # for (int i = 0; i < N; i += VL) {
        #
        #     auto vA  = load8(&A[i]);          // load 8 elements
        #
        #     auto mbr = (vA < 0.0);            // break mask
        #     auto tmp = expr(vA);              // compute expression in temp
        #     store8_masked(&A[i], tmp, !mbr);  // masked assignment
        #
        #     if (any(mbr))                     // early exit if any lane < 0
        #         break;
        # }
        self._apply(sdfg)
        sdfg.validate()
