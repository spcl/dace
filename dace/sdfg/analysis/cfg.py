# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Various analyses related to control flow in SDFGs. """
from collections import defaultdict
from dace.sdfg import SDFGState, InterstateEdge, graph as gr, utils as sdutil
from dace import graphlib as nx
from dace import symbolic
import sympy as sp
from typing import Dict, Iterator, List, Optional, Tuple

from dace.sdfg.state import BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowBlock, ControlFlowRegion, ReturnBlock
from dace.ordered import OrderedSet


def collect_enclosing_conditions(block: ControlFlowBlock, stop: Optional[ControlFlowRegion] = None) -> sp.Basic:
    """The conjunction of branch conditions that must hold for ``block`` to execute.

    Walks out through every enclosing :class:`ConditionalBlock`, accumulating the guard of the
    branch that contains ``block``; an ``else`` branch (``cond is None``) contributes the
    negation of all preceding conditions. Returns ``sp.true`` when nothing guards ``block``.

    ``stop`` bounds the walk at a region: guards at or above it are not collected. Pass the
    enclosing loop when the caller reasons in that loop's iteration space, so a condition
    written in terms of symbols an interstate edge reassigns on the way in never enters the
    answer. Left ``None`` the walk runs to the root.

    A condition that does not parse is dropped rather than guessed at. Every consumer uses the
    result to SHRINK the set of states it must consider, so a missing conjunct can only make an
    answer more conservative, never wrong.
    """
    conditions: List[sp.Basic] = []
    current: Optional[ControlFlowBlock] = block
    while current is not None and current is not stop:
        parent = current.parent_graph
        if parent is None or parent is stop:
            break
        # ``block`` lives inside a branch region; the ConditionalBlock is that region's parent.
        cond_block = parent.parent_graph
        if isinstance(cond_block, ConditionalBlock):
            our_cond: Optional[str] = None
            seen_else = False
            for cond_codeblock, branch in cond_block.branches:
                if branch is parent:
                    if cond_codeblock is not None and cond_codeblock.as_string:
                        our_cond = cond_codeblock.as_string
                    else:
                        seen_else = True
                    break
            if our_cond is not None:
                parsed = parse_condition(our_cond)
                if parsed is not None:
                    conditions.append(parsed)
            elif seen_else:
                negs: List[sp.Basic] = []
                for cond_codeblock, _branch in cond_block.branches:
                    if cond_codeblock is None:
                        break
                    if cond_codeblock.as_string:
                        parsed = parse_condition(cond_codeblock.as_string)
                        negated = negate_condition(parsed) if parsed is not None else None
                        if negated is not None:
                            negs.append(negated)
                if negs:
                    conditions.append(sp.And(*negs) if len(negs) > 1 else negs[0])
        current = parent
    if not conditions:
        return sp.true
    if len(conditions) == 1:
        return conditions[0]
    try:
        return sp.And(*conditions)
    except TypeError:
        # ``sp.And`` rejects DaCe's own ``AND`` / ``OR`` nodes -- ``pystr_to_symbolic`` builds
        # them with ``evaluate=False`` and sympy does not count a Function as Boolean. Fold with
        # DaCe's connective instead of dropping the guard; both consumers understand it.
        out = conditions[0]
        for cond in conditions[1:]:
            out = symbolic.AND(out, cond)
        return out


def parse_condition(cond_str: str) -> Optional[sp.Basic]:
    """``cond_str`` as a sympy expression, or ``None`` if it does not parse."""
    try:
        return symbolic.pystr_to_symbolic(cond_str)
    except Exception:
        return None


def negate_condition(expr: sp.Basic) -> Optional[sp.Basic]:
    """``not expr``, or ``None`` when it has no form the callers can use.

    ``sp.Not`` RAISES on DaCe's own ``AND`` / ``OR`` nodes: ``pystr_to_symbolic`` builds them with
    ``evaluate=False`` to keep parse trees verbatim, and sympy does not count a Function as
    Boolean. So the connectives are negated by De Morgan here instead of being handed to sympy,
    and only a bare relational reaches ``sp.Not``.
    """
    func = str(expr.func) if isinstance(expr, sp.Basic) else ''
    if func == 'NOT':
        return expr.args[0]
    if func in ('AND', 'OR'):
        parts = [negate_condition(a) for a in expr.args]
        if any(part is None for part in parts):
            return None
        joiner = symbolic.OR if func == 'AND' else symbolic.AND
        out = parts[0]
        for part in parts[1:]:
            out = joiner(out, part)
        return out
    try:
        return sp.Not(expr)
    except TypeError:
        return None  # not a form sympy can negate; dropping the guard stays conservative


def acyclic_dominance_frontier(cfg: ControlFlowRegion, idom=None) -> Dict[ControlFlowBlock, OrderedSet]:
    """
    Finds the dominance frontier for a CFG while ignoring any back edges.

    This is a modified version of the dominance frontiers algorithm as implemented by networkx.

    :param cfg: The CFG for which to compute the acyclic dominance frontier.
    :param idom: Optional precomputed immediate dominators.
    :return: A dictionary keyed by control flow blocks, containing the dominance frontier for each control flow block.
    """
    idom = idom or nx.immediate_dominators(cfg.nx, cfg.start_block)

    dom_frontiers = {block: OrderedSet() for block in cfg.nodes()}
    for u in idom:
        if len(cfg.nx.pred[u]) >= 2:
            for v in cfg.nx.pred[u]:
                if v in idom:
                    df_candidates = OrderedSet()
                    while v != idom[u]:
                        if v == u:
                            df_candidates = None
                            break
                        df_candidates.add(v)
                        v = idom[v]
                    if df_candidates is not None:
                        for candidate in df_candidates:
                            dom_frontiers[candidate].add(u)

    return dom_frontiers


def block_immediate_dominators(cfg: ControlFlowRegion) -> Dict[ControlFlowBlock, ControlFlowBlock]:
    """ Returns the immediate dominator of every block, including those unreachable from the start block.

    ``nx.immediate_dominators`` only covers what the start block reaches, which leaves a legitimately
    unreachable block -- dead code the frontend emitted, a branch never taken -- absent from the map
    rather than mapped. A block nothing can reach is dominated by nothing, so it is its own immediate
    dominator: a root, exactly like the start block. That keeps every dominator-based analysis total
    over the whole CFG instead of raising ``KeyError`` on part of it.

    Note that this deliberately does not distinguish dead code from a region some transformation
    severed by mistake -- the two are the same graph shape, and nothing local to the CFG can tell them
    apart. Codegen is what catches the severed case, by refusing to emit a program that is missing a
    block (see ``DaCeCodeGenerator.generate_code``).

    :param cfg: The control flow graph to compute immediate dominators for.
    :return: A dictionary mapping each block to its immediate dominator.
    """
    idom = nx.immediate_dominators(cfg.nx, cfg.start_block)
    for block in cfg.nodes():
        idom.setdefault(block, block)
    return idom


def all_dominators(cfg: ControlFlowRegion,
                   idom: Dict[ControlFlowBlock, ControlFlowBlock] = None) -> Dict[ControlFlowBlock, OrderedSet]:
    """ Returns a mapping between each control flow block and all its dominators. """
    idom = idom or block_immediate_dominators(cfg)
    # Create a dictionary of all dominators of each node by using the transitive closure of the DAG induced by the idoms
    g = nx.DiGraph()
    for node, dom in idom.items():
        if node is dom:  # Skip root
            continue
        g.add_edge(node, dom)
    tc = nx.transitive_closure_dag(g)
    # Seeded with every block, not just the start one: a root is dominated by nothing, and an
    # unreachable block is a root, so both need an entry the transitive closure will never supply.
    alldoms: Dict[ControlFlowBlock, OrderedSet] = {block: OrderedSet() for block in cfg.nodes()}
    for node in tc:
        alldoms[node] = OrderedSet(dst for _, dst in tc.out_edges(node))

    return alldoms


def all_postdominators(cfg: ControlFlowRegion,
                       ipostdom: Dict[ControlFlowBlock, ControlFlowBlock] = None,
                       sink: Optional[ControlFlowBlock] = None) -> Dict[ControlFlowBlock, OrderedSet]:
    """ Returns a mapping between each control flow block and all its postdominators. """
    remove_sink = False
    if sink is None:
        remove_sink = True
        sinks = OrderedSet()
        for block in cfg.nodes():
            if cfg.out_degree(block) == 0 or isinstance(block, (ContinueBlock, BreakBlock, ReturnBlock)):
                sinks.add(block)
        sink = ControlFlowBlock('__DACE_dummy_sink')
        cfg.add_node(sink)
        for s in sinks:
            cfg.add_edge(s, sink, InterstateEdge())

    ipostdom = ipostdom or nx.immediate_dominators(cfg.nx.reverse(), sink)

    # Create a dictionary of all postdominators of each node by using the transitive closure of the DAG induced by the
    # ipostdoms
    g = nx.DiGraph()
    for node, pdom in ipostdom.items():
        if node is pdom:
            continue
        g.add_edge(node, pdom)
    tc = nx.transitive_closure_dag(g)
    all_postdoms: Dict[ControlFlowBlock, OrderedSet] = defaultdict(OrderedSet)
    for node in tc:
        all_postdoms[node] = OrderedSet(dst for _, dst in tc.out_edges(node))

    if remove_sink:
        cfg.remove_node(sink)

    return all_postdoms


def find_sese_region(
        graph: ControlFlowRegion,
        target_nodes: OrderedSet) -> Tuple[OrderedSet, Optional[ControlFlowBlock], Optional[ControlFlowBlock]]:
    """
    Find the smallest SESE region containing the target nodes.

    :param graph: The control flow graph to analyze.
    :param target_nodes: The set of target nodes to include in the SESE region.
    :param start_node: The starting node of the SESE region. If None, the start node of the graph is used.
    :param end_nodes: The end node of the SESE region. If None, a virtual sink node is created temporarily.
    :return: A tuple containing:
        - A set of nodes in the SESE region.
        - The entry node of the SESE region.
        - The exit node of the SESE region.
    :raises ValueError: If no start node or end nodes are found and none are provided.
    """
    if not target_nodes:
        return OrderedSet(), None, None

    sinks = OrderedSet()
    for block in graph.nodes():
        if graph.out_degree(block) == 0 or isinstance(block, (ContinueBlock, BreakBlock, ReturnBlock)):
            sinks.add(block)
    sink = ControlFlowBlock('__DACE_dummy_sink')
    graph.add_node(sink)
    for s in sinks:
        graph.add_edge(s, sink, InterstateEdge())

    # Compute dominators and post-dominators
    dominators = all_dominators(graph)
    post_dominators = all_postdominators(graph, sink=sink)

    # Find the entry node: the lowest common dominator of all target nodes
    common_dominators = None
    for node in target_nodes:
        if node not in dominators:
            continue
        if common_dominators is None:
            common_dominators = dominators[node].copy()
        else:
            common_dominators &= dominators[node]

    if not common_dominators:
        return OrderedSet(), None, None

    # The entry is the dominator closest to the target nodes
    entry_node = None
    min_distance = float('inf')
    for dom in common_dominators:
        # Find maximum distance to any target node
        max_dist_to_targets = 0
        for target in target_nodes:
            if target in dominators and dom in dominators[target]:
                # Count nodes between dom and target
                try:
                    dist = nx.shortest_path_length(graph.nx, dom, target)
                    max_dist_to_targets = max(max_dist_to_targets, dist)
                except nx.NetworkXNoPath:
                    max_dist_to_targets = float('inf')

        if max_dist_to_targets < min_distance:
            min_distance = max_dist_to_targets
            entry_node = dom

    # Find the exit node: the lowest common post-dominator of all target nodes
    common_post_dominators = None
    for node in target_nodes:
        if node not in post_dominators:
            continue
        if common_post_dominators is None:
            common_post_dominators = post_dominators[node].copy()
        else:
            common_post_dominators &= post_dominators[node]

    if not common_post_dominators:
        return OrderedSet(), entry_node, None

    # The exit is the post-dominator closest to the target nodes, from which none of the target nodes can be reached
    # anymore.
    exit_node = None
    min_distance = float('inf')
    for post_dom in common_post_dominators:
        max_dist_from_targets = 0
        if any(nx.has_path(graph.nx, post_dom, t) for t in target_nodes):
            continue
        for target in target_nodes:
            if target in post_dominators and post_dom in post_dominators[target]:
                path_exists = nx.has_path(graph.nx, target, post_dom)
                if path_exists:
                    try:
                        dist = nx.shortest_path_length(graph.nx, target, post_dom)
                        max_dist_from_targets = max(max_dist_from_targets, dist)
                    except nx.NetworkXNoPath:
                        max_dist_from_targets = float('inf')

        if max_dist_from_targets < min_distance:
            min_distance = max_dist_from_targets
            exit_node = post_dom

    # Find all nodes in the SESE region
    if entry_node is None or exit_node is None:
        return target_nodes.copy(), entry_node, exit_node

    # The region includes all nodes on paths from entry to exit that are reachable from entry and
    # can reach exit. ``nx.descendants`` answers with a plain set, whose iteration order follows
    # allocation addresses; the caller REMOVES these nodes from the graph and codegen structures
    # what is left, so that order reaches the emitted program. Rank both by the graph's own block
    # order, which is insertion order, before intersecting.
    reachable = OrderedSet()
    if entry_node in graph:
        descendants = nx.descendants(graph.nx, entry_node)
        reachable = OrderedSet(b for b in graph.nodes() if b is entry_node or b in descendants)

    backwards = nx.descendants(graph.nx.reverse(), exit_node)
    can_reach_exit = OrderedSet(b for b in graph.nodes() if b is exit_node or b in backwards)

    region_nodes = reachable & can_reach_exit

    # Remove the dummy sink
    graph.remove_node(sink)
    if sink in region_nodes:
        region_nodes.remove(sink)
    if exit_node == sink:
        exit_node = None

    return region_nodes, entry_node, exit_node


def back_edges(cfg: ControlFlowRegion,
               idom: Dict[ControlFlowBlock, ControlFlowBlock] = None,
               alldoms: Optional[Dict[ControlFlowBlock, OrderedSet]] = None) -> List[gr.Edge[InterstateEdge]]:
    """ Returns a list of back-edges in a control flow graph. """
    alldoms = alldoms or all_dominators(cfg, idom)
    return [e for e in cfg.edges() if e.dst in alldoms[e.src]]


def branch_merges(
        cfg: ControlFlowRegion,
        idom: Dict[ControlFlowBlock, ControlFlowBlock] = None,
        alldoms: Optional[Dict[ControlFlowBlock, OrderedSet]] = None) -> Dict[ControlFlowBlock, ControlFlowBlock]:
    alldoms = alldoms or all_dominators(cfg, idom)

    # Annotate branches
    result: Dict[SDFGState, SDFGState] = {}
    adf = acyclic_dominance_frontier(cfg)
    # ipostdom = sdutil.postdominators(cfg)
    for block in cfg.nodes():
        oedges = cfg.out_edges(block)
        # Skip if not branch
        if len(oedges) <= 1:
            continue

        # If branch without else (adf of one successor is equal to the other)
        if len(oedges) == 2:
            if {oedges[0].dst} & adf[oedges[1].dst]:
                merge = oedges[0].dst
                if block in alldoms[merge]:
                    result[block] = oedges[0].dst
                continue
            elif {oedges[1].dst} & adf[oedges[0].dst]:
                merge = oedges[1].dst
                if block in alldoms[merge]:
                    result[block] = oedges[1].dst
                continue

        # Try to obtain common DF to find merge state
        common_frontier = OrderedSet()
        descendants_blacklist = OrderedSet()
        disjoint_edges = OrderedSet()
        for oedge in oedges:
            branch_descendants = OrderedSet(cfg.dfs_edges(oedge.dst))
            branch_descendants.add(oedge.dst)
            frontier = adf[oedge.dst]
            if not frontier:
                # If no dominance frontier is found for this edge, there are two possible scenarios under which this
                # may still lead to a valid merge state:
                # 1: The edge destination is itself the branch merge state. To cover this, the frontier consisits of
                #    the destination block itself, and if there is a concrete merge state, that will result in a single
                #    common frontier block.
                # 2: The edge leads to a completely separate control flow path that does not reconnect to the branch
                #    merge state and can not reach any of the other branch descendants.
                if not (branch_descendants & descendants_blacklist):
                    disjoint_edges.add(oedge)
                    continue
                else:
                    frontier = OrderedSet((oedge.dst, ))
            common_frontier |= frontier
            descendants_blacklist.update(branch_descendants)
        if len(common_frontier) == 1:
            merge = next(iter(common_frontier))
            if block in alldoms[merge]:
                result[block] = merge
        elif len(common_frontier) == 0 and len(disjoint_edges) == len(oedges):
            result[block] = None  # No merge state found, but the branches are disjoint.

    return result


def block_parent_tree(cfg: ControlFlowRegion,
                      loopexits: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                      idom: Dict[ControlFlowBlock, ControlFlowBlock] = None,
                      with_loops: bool = True) -> Dict[ControlFlowBlock, ControlFlowBlock]:
    """
    Computes an upward-pointing tree of each control flow block, pointing to the "parent block" it belongs to (in terms
    of structured control flow). More formally, each block is either mapped to its immediate dominator with out
    degree >= 2, one block upwards if the block occurs after a loop and `with_loops` is True, or the start block if
    no such block exist.

    :param sdfg: The SDFG to analyze.
    :param idom: An optional, pre-computed immediate dominator dictionary.
    :param with_loops: Respect loops in the parent computation, mapping blocks to a parent one block upwards of a loop
                       if the block occurs after a loop. Defaults to true.
    :return: A dictionary that maps each block to a parent block, or None if the root (start) block.
    """
    idom = idom or block_immediate_dominators(cfg)
    merges = branch_merges(cfg, idom)
    if with_loops:
        alldoms = all_dominators(cfg, idom)
        loopexits = loopexits if loopexits is not None else defaultdict(lambda: None)

        # First, annotate loops
        for be in back_edges(cfg, idom, alldoms):
            guard = be.dst
            laststate = be.src
            if loopexits[guard] is not None:
                continue
            if guard in merges:
                continue

            # Natural loops = one edge leads back to loop, another leads out
            in_edges = cfg.in_edges(guard)
            out_edges = cfg.out_edges(guard)

            # A loop guard has at least one incoming edges (the backedge, performing the increment), and exactly two
            # outgoing edges (loop and exit loop).
            if len(in_edges) < 1 or len(out_edges) != 2:
                continue

            # The outgoing edges must be negations of one another.
            if out_edges[0].data.condition_sympy() != (sp.Not(out_edges[1].data.condition_sympy())):
                continue

            # Find all nodes that are between each branch and the guard.
            # Condition makes sure the entire cycle is dominated by this node.
            # If not, we're looking at a guard for a nested cycle, which we ignore for
            # this cycle.
            oa, ob = out_edges[0].dst, out_edges[1].dst

            reachable_a = False
            a_reached_guard = False

            def cond_a(parent, child):
                nonlocal reachable_a
                nonlocal a_reached_guard
                if reachable_a:  # If last state has been reached, stop traversal
                    return False
                if parent is laststate or child is laststate:  # Reached back edge
                    reachable_a = True
                    a_reached_guard = True
                    return False
                if oa not in alldoms[child]:  # Traversed outside of the loop
                    return False
                if child is guard:  # Traversed back to guard
                    a_reached_guard = True
                    return False
                return True  # Keep traversing

            reachable_b = False
            b_reached_guard = False

            def cond_b(parent, child):
                nonlocal reachable_b
                nonlocal b_reached_guard
                if reachable_b:  # If last state has been reached, stop traversal
                    return False
                if parent is laststate or child is laststate:  # Reached back edge
                    reachable_b = True
                    b_reached_guard = True
                    return False
                if ob not in alldoms[child]:  # Traversed outside of the loop
                    return False
                if child is guard:  # Traversed back to guard
                    b_reached_guard = True
                    return False
                return True  # Keep traversing

            list(sdutil.dfs_conditional(cfg, (oa, ), cond_a))
            list(sdutil.dfs_conditional(cfg, (ob, ), cond_b))

            # Check which candidate states led back to guard
            is_a_begin = a_reached_guard and reachable_a
            is_b_begin = b_reached_guard and reachable_b

            loop_state = None
            exit_state = None
            if is_a_begin and not is_b_begin:
                loop_state = oa
                exit_state = ob
            elif is_b_begin and not is_a_begin:
                loop_state = ob
                exit_state = oa
            if loop_state is None or exit_state is None:
                continue
            loopexits[guard] = exit_state

    # Get dominators
    parents: Dict[ControlFlowBlock, ControlFlowBlock] = {}
    step_up: OrderedSet = OrderedSet()
    for block in cfg.nodes():
        curdom = idom[block]
        if curdom == block:
            parents[block] = None
            continue

        while curdom != idom[curdom]:
            if cfg.out_degree(curdom) > 1:
                break
            curdom = idom[curdom]

        if with_loops and cfg.out_degree(curdom) == 2 and loopexits[curdom] is not None:
            p = block
            while p != curdom and p != loopexits[curdom]:
                p = idom[p]
            if p == loopexits[curdom]:
                # Dominated by loop exit: do one more step up
                step_up.add(block)

        parents[block] = curdom

    if with_loops:
        # Step up for post-loop blocks.
        for block in step_up:
            if parents[block] is not None and parents[parents[block]] is not None:
                parents[block] = parents[parents[block]]

    return parents


def _blockorder_topological_sort(
        cfg: ControlFlowRegion,
        start: ControlFlowBlock,
        ptree: Dict[ControlFlowBlock, ControlFlowBlock],
        branch_merges: Dict[ControlFlowBlock, ControlFlowBlock],
        stop: ControlFlowBlock = None,
        visited: Optional[OrderedSet] = None,
        loopexits: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None) -> Iterator[ControlFlowBlock]:
    """
    Helper function for ``blockorder_topological_sort``.

    :param cfg: CFG.
    :param start: Starting block for traversal.
    :param ptree: Block parent tree (computed from ``block_parent_tree``).
    :param branch_merges: Dictionary mapping from branch blocks to its merge block.
    :param stop: Stopping blocks to not traverse through (e.g., merge blocks of a branch or guard block of a loop).
    :param visited: Optionally, a set of already visited blocks.
    :param loopexits: An optional dictionary of already identified loop guard to exit block mappings.
    :return: Generator that yields control flow blocks in execution order from ``start`` to ``stop``.
    """
    loopexits = loopexits if loopexits is not None else defaultdict(lambda: None)

    # Traverse blocks in custom order
    visited = visited if visited is not None else OrderedSet()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited or node is stop:
            continue
        yield node
        visited.add(node)

        oe = cfg.out_edges(node)
        if len(oe) == 0:  # End block
            continue
        elif len(oe) == 1:  # No traversal change
            stack.append(oe[0].dst)
            continue
        elif len(oe) == 2:  # Loop or branch
            # If loop, traverse body, then exit
            if node in loopexits:
                if oe[0].dst == loopexits[node]:
                    for s in _blockorder_topological_sort(cfg,
                                                          oe[1].dst,
                                                          ptree,
                                                          branch_merges,
                                                          stop=node,
                                                          visited=visited,
                                                          loopexits=loopexits):
                        yield s
                        visited.add(s)
                    stack.append(oe[0].dst)
                    continue
                elif oe[1].dst == loopexits[node]:
                    for s in _blockorder_topological_sort(cfg,
                                                          oe[0].dst,
                                                          ptree,
                                                          branch_merges,
                                                          stop=node,
                                                          visited=visited,
                                                          loopexits=loopexits):
                        yield s
                        visited.add(s)
                    stack.append(oe[1].dst)
                    continue
            # Otherwise, passthrough to branch
        # Branch
        if node in branch_merges and branch_merges[node] is not None:
            # Try to find merge block and traverse until reaching that
            mergeblock = branch_merges[node]
        else:
            try:
                # Otherwise (e.g., with return/break statements), traverse through each branch,
                # stopping at the end of the current tree level.
                mergeblock = next(e.dst for e in cfg.out_edges(stop) if ptree[e.dst] != stop)
            except (StopIteration, KeyError):
                # If that fails, simply traverse branches in arbitrary order
                mergeblock = stop

        for branch in oe:
            if branch.dst is mergeblock:
                # If we hit the merge block (if without else), defer to end of branch traversal
                continue
            for s in _blockorder_topological_sort(cfg,
                                                  branch.dst,
                                                  ptree,
                                                  branch_merges,
                                                  stop=mergeblock,
                                                  visited=visited,
                                                  loopexits=loopexits):
                yield s
                visited.add(s)
        stack.append(mergeblock)


def blockorder_topological_sort(cfg: ControlFlowRegion,
                                recursive: bool = True,
                                ignore_nonstate_blocks: bool = False) -> Iterator[ControlFlowBlock]:
    """
    Returns a generator that produces control flow blocks in the order that they will be executed, disregarding multiple
    loop iterations and employing topological sort for branches.

    :param cfg: The CFG to iterate over.
    :param recursive: Whether or not to recurse down hierarchies of control flow regions (not across Nested SDFGs).
    :param ignore_nonstate_blocks: If true, only produce basic blocks / SDFGStates. Defaults to False.
    :return: Generator that yields control flow blocks in execution-order.
    """
    # Get parent states
    loopexits: Dict[ControlFlowBlock, ControlFlowBlock] = defaultdict(lambda: None)
    idom = block_immediate_dominators(cfg)
    ptree = block_parent_tree(cfg, loopexits, idom=idom)

    # Annotate branches
    merges = branch_merges(cfg, idom)

    for block in _blockorder_topological_sort(cfg, cfg.start_block, ptree, merges, loopexits=loopexits):
        if isinstance(block, ControlFlowRegion):
            if not ignore_nonstate_blocks:
                yield block
            if recursive:
                yield from blockorder_topological_sort(block, recursive, ignore_nonstate_blocks)
        elif isinstance(block, ConditionalBlock):
            if not ignore_nonstate_blocks:
                yield block
            if recursive:
                for _, branch in block.branches:
                    if not ignore_nonstate_blocks:
                        yield branch
                    yield from blockorder_topological_sort(branch, recursive, ignore_nonstate_blocks)
        elif isinstance(block, SDFGState):
            yield block
        else:
            # Other control flow block.
            if not ignore_nonstate_blocks:
                yield block
