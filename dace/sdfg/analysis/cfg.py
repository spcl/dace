# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Various analyses related to control flow in SDFGs. """
from collections import defaultdict
from dace.sdfg import SDFGState, InterstateEdge, graph as gr, utils as sdutil
import networkx as nx
import sympy as sp
from typing import Dict, Iterator, List, Optional, Set

from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion


def acyclic_dominance_frontier(cfg: ControlFlowRegion, idom=None) -> Dict[ControlFlowBlock, Set[ControlFlowBlock]]:
    """
    Finds the dominance frontier for a CFG while ignoring any back edges.

    This is a modified version of the dominance frontiers algorithm as implemented by networkx.

    :param cfg: The CFG for which to compute the acyclic dominance frontier.
    :param idom: Optional precomputed immediate dominators.
    :return: A dictionary keyed by control flow blocks, containing the dominance frontier for each control flow block.
    """
    idom = idom or nx.immediate_dominators(cfg.nx, cfg.start_block)

    dom_frontiers = {block: set() for block in cfg.nodes()}
    for u in idom:
        if len(cfg.nx.pred[u]) >= 2:
            for v in cfg.nx.pred[u]:
                if v in idom:
                    df_candidates = set()
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


def all_dominators(
        cfg: ControlFlowRegion,
        idom: Dict[ControlFlowBlock, ControlFlowBlock] = None) -> Dict[ControlFlowBlock, Set[ControlFlowBlock]]:
    """ Returns a mapping between each control flow block and all its dominators. """
    idom = idom or nx.immediate_dominators(cfg.nx, cfg.start_block)
    # Create a dictionary of all dominators of each node by using the transitive closure of the DAG induced by the idoms
    g = nx.DiGraph()
    for node, dom in idom.items():
        if node is dom:  # Skip root
            continue
        g.add_edge(node, dom)
    tc = nx.transitive_closure_dag(g)
    alldoms: Dict[ControlFlowBlock, Set[ControlFlowBlock]] = {cfg.start_block: set()}
    for node in tc:
        alldoms[node] = set(dst for _, dst in tc.out_edges(node))

    return alldoms


def back_edges(cfg: ControlFlowRegion,
               idom: Dict[ControlFlowBlock, ControlFlowBlock] = None,
               alldoms: Dict[ControlFlowBlock, Set[ControlFlowBlock]] = None) -> List[gr.Edge[InterstateEdge]]:
    """ Returns a list of back-edges in a control flow graph. """
    alldoms = alldoms or all_dominators(cfg, idom)
    return [e for e in cfg.edges() if e.dst in alldoms[e.src]]


def branch_merges(
        cfg: ControlFlowRegion,
        idom: Dict[ControlFlowBlock, ControlFlowBlock] = None,
        alldoms: Dict[ControlFlowBlock, Set[ControlFlowBlock]] = None) -> Dict[ControlFlowBlock, ControlFlowBlock]:
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
        common_frontier = set()
        for oedge in oedges:
            frontier = adf[oedge.dst]
            if not frontier:
                frontier = {oedge.dst}
            common_frontier |= frontier
        if len(common_frontier) == 1:
            merge = next(iter(common_frontier))
            if block in alldoms[merge]:
                result[block] = merge
        # elif len(common_frontier) > 1 and ipostdom and ipostdom[block] in common_frontier:
        #     result[block] = ipostdom[block]

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
    idom = idom or nx.immediate_dominators(cfg.nx, cfg.start_block)
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
    step_up: Set[ControlFlowBlock] = set()
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
        visited: Set[ControlFlowBlock] = None,
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
    visited = visited or set()
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
        if node in branch_merges:
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
    idom = nx.immediate_dominators(cfg.nx, cfg.start_block)
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
