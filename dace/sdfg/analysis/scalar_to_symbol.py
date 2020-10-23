# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Scalar to symbol promotion functionality. """

import ast
from dace.sdfg.sdfg import InterstateEdge
from dace.frontend.python.astutils import ASTFindReplace
from dace import (dtypes, nodes, sdfg as sd, data as dt, properties as props,
                  memlet as mm)
from dace.sdfg import graph as gr
from dace.frontend.python import astutils
from dace.transformation import helpers as xfh
import re
from typing import Any, Dict, Set


def find_promotable_scalars(sdfg: sd.SDFG) -> Set[str]:
    """
    Finds scalars that can be promoted to symbols in the given SDFG.
    Conditions for matching a scalar for symbol-promotion are as follows:
        * Size of data must be 1, it must not be a stream and must be transient.
        * Only inputs to candidate scalars must be either arrays or tasklets.
        * All tasklets that lead to it must have one statement, one output, 
          and may have zero or more **array** inputs and not be in a scope.
        * Scalar must not be accessed with a write-conflict resolution.
        * Scalar must not be written to more than once in a state.

    These conditions must apply on all occurences of the scalar in order for
    it to be promotable.

    :param sdfg: The SDFG to query.
    :return: A set of promotable scalar names.
    """
    # Keep set of active candidates
    candidates: Set[str] = set()

    # General array checks
    for aname, desc in sdfg.arrays.items():
        if not desc.transient or isinstance(desc, dt.Stream):
            continue
        if desc.total_size != 1:
            continue
        candidates.add(aname)

    # Check all occurrences of candidates in SDFG and filter out
    candidates_seen: Set[str] = set()
    for state in sdfg.nodes():
        candidates_in_state: Set[str] = set()

        for node in state.nodes():
            if not isinstance(node, nodes.AccessNode):
                continue
            candidate = node.data
            if candidate not in candidates:
                continue

            # If candidate is read-only, continue normally
            if state.in_degree(node) == 0:
                continue

            # Candidate may only be accessed in a top-level scope
            if state.entry_node(node) is not None:
                candidates.remove(candidate)
                continue

            # Candidate may only be written to once within a state
            if candidate in candidates_in_state:
                if state.in_degree(node) == 1:
                    candidates.remove(candidate)
                    continue
            candidates_in_state.add(candidate)

            # If input is not a single array nor tasklet, skip
            if state.in_degree(node) > 1:
                candidates.remove(candidate)
                continue
            edge = state.in_edges(node)[0]

            # Edge must not be WCR
            if edge.data.wcr is not None:
                candidates.remove(candidate)
                continue

            # Check inputs
            if isinstance(edge.src, nodes.AccessNode):
                # If input is array, ensure it is not a stream
                if isinstance(sdfg.arrays[edge.src.data], dt.Stream):
                    candidates.remove(candidate)
                    continue
                # Ensure no inputs exist to the array
                if state.in_degree(edge.src) > 0:
                    candidates.remove(candidate)
                    continue
            elif isinstance(edge.src, nodes.Tasklet):
                # If input tasklet has more than one output, skip
                if state.out_degree(edge.src) > 1:
                    candidates.remove(candidate)
                    continue
                # If inputs to tasklets are not arrays, skip
                for tinput in state.in_edges(edge.src):
                    if not isinstance(tinput.src, nodes.AccessNode):
                        candidates.remove(candidate)
                        break
                    if isinstance(sdfg.arrays[tinput.src.data], dt.Stream):
                        candidates.remove(candidate)
                        break
                else:
                    # Check that tasklets have only one statement
                    cb: props.CodeBlock = edge.src.code
                    if cb.language is dtypes.Language.Python:
                        if (len(cb.code) > 1
                                or not isinstance(cb.code[0], ast.Assign)):
                            candidates.remove(candidate)
                            continue
                        # Ensure the candidate is assigned to
                        if (len(cb.code[0].targets) != 1 or astutils.rname(
                                cb.code[0].targets[0]) != edge.src_conn):
                            candidates.remove(candidate)
                            continue
                    elif cb.language is dtypes.Language.CPP:
                        # Try to match a single C assignment
                        cstr = cb.as_string.strip()
                        # Since we cannot remove subscripts from C++ tasklets,
                        # if the type of the data is an array we will also skip
                        if re.match(r'^[a-zA-Z_][a-zA-Z_0-9]*\s*=.*;$',
                                    cstr) is None:
                            candidates.remove(candidate)
                            continue
                    else:  # Other languages are currently unsupported
                        candidates.remove(candidate)
                        continue
            else:  # If input is not an acceptable node type, skip
                candidates.remove(candidate)
        candidates_seen |= candidates_in_state

    # Only keep candidates that were found in SDFG
    candidates &= candidates_seen

    return candidates


class TaskletPromoter(ast.NodeTransformer):
    """
    Promotes scalars to symbols in Tasklets.
    If connector name is used in tasklet as subscript, modifies to symbol name.
    If connector is used as a standard name, modify tasklet code to use symbol.
    """
    def __init__(self, connector: str, symbol: str) -> None:
        """
        Initializes AST transformer.
        :param connector: Connector name (replacement source).
        :param symbol: Symbol name (replacement target).
        """
        self.conn = connector
        self.symbol = symbol

    def visit_Name(self, node: ast.Name) -> Any:
        # Convert connector to symbol
        if node.id == self.conn:
            node.id = self.symbol
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # Convert subscript to symbol name
        node_name = astutils.rname(node)
        if node_name == self.conn:
            return ast.copy_location(ast.Name(id=self.symbol, ctx=ast.Load()),
                                     node)
        return self.generic_visit(node)


def promote_scalars_to_symbols(sdfg: sd.SDFG):
    """
    Promotes all matching transient scalars to SDFG symbols, changing all
    tasklets to inter-state assignments. This enables the transformed symbols
    to be used within states as part of memlets, and allows further
    transformations (such as loop detection) to use the information for
    optimization.

    :param sdfg: The SDFG to run the pass on.
    :note: Operates in-place.
    """
    # Process:
    # 1. Find scalars to promote
    # 2. For every assignment tasklet/access:
    #    2.1. Fission state to isolate assignment
    #    2.2. Replace assignment with inter-state edge assignment
    # 3. For every read of the scalar:
    #    3.1. If destination is tasklet, remove node, edges, and connectors
    #    3.2. If used in tasklet as subscript or connector, modify tasklet code
    #    3.3. If destination is array, change to tasklet that copies symbol data
    # 4. Remove newly-isolated access nodes
    # 5. Remove data descriptors and add symbols to SDFG
    # 6. Replace subscripts in all interstate conditions and assignments
    # 7. Make indirections with symbols a single memlet

    to_promote = find_promotable_scalars(sdfg)

    for state in sdfg.nodes():
        scalar_nodes = [
            n for n in state.nodes()
            if isinstance(n, nodes.AccessNode) and n.data in to_promote
        ]
        # Step 2: Assignment tasklets
        for node in scalar_nodes:
            # There is only zero or one incoming edges by definition
            if state.in_degree(node) == 0:
                continue
            in_edge = state.in_edges(node)[0]
            input = in_edge.src
            tasklet_inputs = [e.src for e in state.in_edges(input)]
            # Step 2.1
            new_state = xfh.state_fission(
                sdfg, gr.SubgraphView(state, [input, node] + tasklet_inputs))
            new_isedge: sd.InterstateEdge = sdfg.out_edges(new_state)[0]
            # Step 2.2
            node: nodes.AccessNode = new_state.sink_nodes()[0]
            input = new_state.in_edges(node)[0].src
            if isinstance(input, nodes.Tasklet):
                # Convert tasklet to interstate edge
                newcode: str = ''
                if input.language is dtypes.Language.Python:
                    newcode = astutils.unparse(input.code.code[0].value)
                elif input.language is dtypes.Language.CPP:
                    newcode = re.findall(r'.*=\s*(.*);',
                                         input.code.as_string.strip())[0]
                # Replace tasklet inputs with incoming edges
                for e in new_state.in_edges(input):
                    memlet_str: str = e.data.data
                    if e.data.subset is not None:
                        memlet_str += '[%s]' % e.data.subset
                    newcode = re.sub(r'\b%s\b' % re.escape(e.dst_conn),
                                     memlet_str, newcode)
                # Add interstate edge assignment
                new_isedge.data.assignments[node.data] = newcode
            elif isinstance(input, nodes.AccessNode):
                memlet: mm.Memlet = in_edge.data
                if memlet.src_subset:
                    new_isedge.data.assignments[
                        node.data] = '%s[%s]' % (input.data, memlet.src_subset)
                else:
                    new_isedge.data.assignments[node.data] = input.data

            # Clean up all nodes after assignment was transferred
            new_state.remove_nodes_from(new_state.nodes())

        scalar_nodes = [
            n for n in state.nodes()
            if isinstance(n, nodes.AccessNode) and n.data in to_promote
        ]

        # Step 3: Scalar reads
        for node in scalar_nodes:
            for out_edge in state.out_edges(node):
                for e in state.memlet_tree(out_edge):
                    # Step 3.1
                    dst = e.dst
                    state.remove_edge_and_connectors(e)
                    if isinstance(dst, nodes.Tasklet):
                        # Step 3.2
                        if dst.language is dtypes.Language.Python:
                            promo = TaskletPromoter(e.dst_conn, node.data)
                            for stmt in dst.code.code:
                                promo.visit(stmt)
                        elif dst.language is dtypes.Language.CPP:
                            # Replace whole-word matches (identifiers) in code
                            dst.code = re.sub(r'\b%s\b' % re.escape(e.dst_conn),
                                              node.data, dst.code.as_string)
                    elif isinstance(dst, nodes.AccessNode):
                        # Step 3.3
                        t = state.add_tasklet('symassign', {}, {'__out'},
                                              '__out = %s' % node.data)
                        state.add_edge(
                            t, '__out', dst, e.dst_conn,
                            mm.Memlet(data=dst.data,
                                      subset=e.data.dst_subset,
                                      volume=1))
                        # Reassign destination for check below
                        dst = t

                    # If nodes were disconnected, reconnect with empty memlet
                    if (isinstance(e.src, nodes.EntryNode)
                            and len(state.edges_between(e.src, dst)) == 0):
                        state.add_nedge(e.src, dst, mm.Memlet())

    # Step 4: Isolated nodes
    for state in sdfg.nodes():
        scalar_nodes = [
            n for n in state.nodes()
            if isinstance(n, nodes.AccessNode) and n.data in to_promote
        ]
        state.remove_nodes_from(
            [n for n in scalar_nodes if len(state.all_edges(n)) == 0])

    # Step 5: Data descriptor management
    for scalar in to_promote:
        desc = sdfg.arrays[scalar]
        sdfg.remove_data(scalar, validate=False)
        sdfg.add_symbol(scalar, desc.dtype)

    # Step 6: Inter-state edge cleanup
    for edge in sdfg.edges():
        ise: InterstateEdge = edge.data
        for scalar in to_promote:
            # Condition
            if ise.condition.language is dtypes.Language.Python:
                promo = TaskletPromoter(scalar, scalar)
                for stmt in ise.condition.code:
                    promo.visit(stmt)
            elif ise.condition.language is dtypes.Language.CPP:
                ise.condition = re.sub(r'\b%s\[.*\]' % re.escape(scalar),
                                       scalar, ise.condition.as_string)
            # Assignments
            for aname, assignment in ise.assignments.items():
                ise.assignments[aname] = re.sub(
                    r'\b%s\[.*\]' % re.escape(scalar), scalar,
                    assignment.strip())

    # Step 7: Indirection
    # TODO