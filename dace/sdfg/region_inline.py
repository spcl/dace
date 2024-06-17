# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Tuple, Set
from dace.frontend.python import astutils
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalRegion, ControlFlowBlock, ControlFlowRegion, LoopRegion, ReturnState, SDFGState


def inline(block: ControlFlowBlock) \
    -> Tuple[set[LoopRegion.BreakState], set[LoopRegion.ContinueState], set[ReturnState]]:
    """
    Inline all ControlFlowRegions inside this region recursively.
    Returns three sets containing the Break, Continue and Return states which have to be handled by
    the caller.
    """

    break_states: set[LoopRegion.BreakState] = set()
    continue_states: set[LoopRegion.ContinueState] = set()
    return_states: set[ReturnState] = set()

    for node in block.nodes():
        bs, cs, rs = set(), set(), set()
        if isinstance(node, ConditionalRegion):
            bs, cs, rs = inline_conditional_region(node, block)
        elif isinstance(node, LoopRegion):
            bs, cs, rs = inline_loop_region(node, block)
        elif isinstance(node, LoopRegion.BreakState):
            break_states.add(node)
        elif isinstance(node, LoopRegion.ContinueState):
            continue_states.add(node)
        elif isinstance(node, ReturnState):
            return_states.add(node)        
        elif isinstance(node, ControlFlowRegion):
            bs, cs, rs = inline_control_flow_region(node, block)
        break_states.update(bs)
        continue_states.update(cs)
        return_states.update(rs)
    
    if isinstance(block, ControlFlowRegion):
        block.reset_cfg_list()
    
    return break_states, continue_states, return_states

def inline_control_flow_region(region: ControlFlowRegion, parent: ControlFlowRegion):
    from dace.sdfg.sdfg import InterstateEdge

    break_states, continue_states, return_states = inline(region)

    # Add all region states and make sure to keep track of all the ones that need to be connected in the end.
    to_connect: Set[ControlFlowBlock] = set()
    for node in region.nodes():
        parent.add_node(node)
        if region.out_degree(node) == 0 and not isinstance(node, (LoopRegion.BreakState, LoopRegion.ContinueState, ReturnState)):
            to_connect.add(node)

    end_state = parent.add_state(region.label + '_end')
    if len(region.nodes()) > 0:
        internal_start = region.start_block
    else:
        internal_start = end_state

    # Add all region edges.
    for edge in region.edges():
        parent.add_edge(edge.src, edge.dst, edge.data)

    # Redirect all edges to the region to the internal start state.
    for b_edge in parent.in_edges(region):
        parent.add_edge(b_edge.src, internal_start, b_edge.data)
        parent.remove_edge(b_edge)
    # Redirect all edges exiting the region to instead exit the end state.
    for a_edge in parent.out_edges(region):
        parent.add_edge(end_state, a_edge.dst, a_edge.data)
        parent.remove_edge(a_edge)

    for node in to_connect:
        parent.add_edge(node, end_state, InterstateEdge())

    # Remove the original loop.
    parent.remove_node(region)

    if parent.in_degree(end_state) == 0:
        parent.remove_node(end_state)
    return break_states, continue_states, return_states


def inline_loop_region(loop: LoopRegion, parent: ControlFlowRegion):
    from dace.sdfg.sdfg import InterstateEdge

    break_states, continue_states, return_states = inline(loop)

    internal_start = loop.start_block

    # Add all boilerplate loop states necessary for the structure.
    init_state = parent.add_state(loop.label + '_init')
    guard_state = parent.add_state(loop.label + '_guard')
    end_state = parent.add_state(loop.label + '_end')
    loop_tail_state = parent.add_state(loop.label + '_tail')

    # Add all loop states and make sure to keep track of all the ones that need to be connected in the end.
    connect_to_tail: Set[SDFGState] = set()
    for node in loop.nodes():
        node.label = loop.label + '_' + node.label
        parent.add_node(node)
        if loop.out_degree(node) == 0 and not isinstance(node, (LoopRegion.BreakState, LoopRegion.ContinueState, ReturnState)):
            connect_to_tail.add(node)

    # Add all internal loop edges.
    for edge in loop.edges():
        parent.add_edge(edge.src, edge.dst, edge.data)

    # Redirect all edges to the loop to the init state.
    for b_edge in parent.in_edges(loop):
        parent.add_edge(b_edge.src, init_state, b_edge.data)
        parent.remove_edge(b_edge)
    # Redirect all edges exiting the loop to instead exit the end state.
    for a_edge in parent.out_edges(loop):
        parent.add_edge(end_state, a_edge.dst, a_edge.data)
        parent.remove_edge(a_edge)

    # Add an initialization edge that initializes the loop variable if applicable.
    init_edge = InterstateEdge()
    if loop.init_statement is not None:
        init_edge.assignments = {}
        for stmt in loop.init_statement.code:
            assign: astutils.ast.Assign = stmt
            init_edge.assignments[assign.targets[0].id] = astutils.unparse(assign.value)
    if loop.inverted:
        parent.add_edge(init_state, internal_start, init_edge)
    else:
        parent.add_edge(init_state, guard_state, init_edge)

    # Connect the loop tail.
    update_edge = InterstateEdge()
    if loop.update_statement is not None:
        update_edge.assignments = {}
        for stmt in loop.update_statement.code:
            assign: astutils.ast.Assign = stmt
            update_edge.assignments[assign.targets[0].id] = astutils.unparse(assign.value)
    parent.add_edge(loop_tail_state, guard_state, update_edge)

    # Add condition checking edges and connect the guard state.
    cond_expr = loop.loop_condition.code
    parent.add_edge(guard_state, end_state,
                    InterstateEdge(CodeBlock(astutils.negate_expr(cond_expr)).code))
    parent.add_edge(guard_state, internal_start, InterstateEdge(CodeBlock(cond_expr).code))

    # Connect any end states from the loop's internal state machine to the tail state so they end a
    # loop iteration. Do the same for any continue states, and connect any break states to the end of the loop.
    for node in continue_states + connect_to_tail:
        parent.add_edge(node, loop_tail_state, InterstateEdge())
    for node in break_states:
        parent.add_edge(node, end_state, InterstateEdge())

    # Remove the original loop.
    parent.remove_node(loop)
    if parent.in_degree(end_state) == 0:
        parent.remove_node(end_state)
    return set(), set(), return_states

def inline_conditional_region(conditional: ConditionalRegion, parent: ControlFlowRegion):
    from dace.sdfg.sdfg import InterstateEdge

    break_states, continue_states, return_states = inline(conditional)

    # Add all boilerplate states necessary for the structure.
    guard_state = parent.add_state(conditional.label + '_guard')
    endif_state = parent.add_state(conditional.label + '_endinf')

    connect_to_end : Set[ControlFlowBlock] = set()
    # Add all loop states and make sure to keep track of all the ones that need to be connected in the end.
    for node in conditional.nodes():
        node.label = conditional.label + '_' + node.label
        parent.add_node(node)
        if conditional.out_degree(node) == 0 and not isinstance(node, (LoopRegion.BreakState, LoopRegion.ContinueState, ReturnState)):
            connect_to_end.add(node)

    # Add all internal region edges.
    for edge in conditional.edges():
        parent.add_edge(edge.src, edge.dst, edge.data)

    # Redirect all edges entering the region to the init state.
    for b_edge in parent.in_edges(conditional):
        parent.add_edge(b_edge.src, guard_state, b_edge.data)
        parent.remove_edge(b_edge)
    # Redirect all edges exiting the region to instead exit the end state.
    for a_edge in parent.out_edges(conditional):
        parent.add_edge(endif_state, a_edge.dst, a_edge.data)
        parent.remove_edge(a_edge)

    # Add condition checking edges and connect the guard state.
    parent.add_edge(guard_state, conditional.start_block, InterstateEdge(conditional.condition_expr))
    parent.add_edge(guard_state, conditional.else_branch, InterstateEdge(conditional.condition_else_expr))

    for node in connect_to_end:
        parent.add_edge(node, endif_state, InterstateEdge())
    parent.add_edge(conditional.else_branch, endif_state, InterstateEdge())
    bs, cs, rs = inline_control_flow_region(conditional.else_branch, parent)
    break_states.update(bs)
    continue_states.update(cs)
    return_states.update(rs)

    parent.remove_node(conditional)
    if parent.in_degree(endif_state) == 0:
        parent.remove_node(endif_state)
    return break_states, continue_states, return_states
    