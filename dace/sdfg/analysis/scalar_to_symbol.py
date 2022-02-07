# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Scalar to symbol promotion functionality. """

import ast
import collections
from platform import node
from dace import symbolic
from dace.sdfg.sdfg import InterstateEdge
from dace import (dtypes, nodes, sdfg as sd, data as dt, properties as props, memlet as mm, subsets)
from dace.sdfg import graph as gr
from dace.frontend.python import astutils
from dace.sdfg import utils as sdutils
from dace.transformation import helpers as xfh
import re
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union


class AttributedCallDetector(ast.NodeVisitor):
    """ Detects attributed calls in Tasklets.
    """
    def __init__(self):
        self.detected = False

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Attribute):
            self.detected = True
            return
        return self.generic_visit(node)


def find_promotable_scalars(sdfg: sd.SDFG, transients_only: bool = True, integers_only: bool = True) -> Set[str]:
    """
    Finds scalars that can be promoted to symbols in the given SDFG.
    Conditions for matching a scalar for symbol-promotion are as follows:
        * Size of data must be 1, it must not be a stream and must be transient.
        * Only inputs to candidate scalars must be either arrays or tasklets.
        * All tasklets that lead to it must have one statement, one output, 
          and may have zero or more **array** inputs and not be in a scope.
        * Scalar must not be accessed with a write-conflict resolution.
        * Scalar must not be written to more than once in a state.
        * If scalar is not integral (i.e., int type), it must also appear in
          an inter-state condition to be promotable.

    These conditions must apply on all occurences of the scalar in order for
    it to be promotable.

    :param sdfg: The SDFG to query.
    :param transients_only: If False, also considers global data descriptors (e.g., arguments).
    :param integers_only: If False, also considers non-integral descriptors for promotion.
    :return: A set of promotable scalar names.
    """
    # Keep set of active candidates
    candidates: Set[str] = set()

    # General array checks
    for aname, desc in sdfg.arrays.items():
        if (transients_only and not desc.transient) or isinstance(desc, dt.Stream):
            continue
        if desc.total_size != 1:
            continue
        if desc.lifetime is dtypes.AllocationLifetime.Persistent:
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

            # If candidate is read by a library node, skip
            removed = False
            for oe in state.out_edges(node):
                for e in state.memlet_tree(oe):
                    if isinstance(e.dst, nodes.LibraryNode):
                        candidates.remove(candidate)
                        removed = True
                        break
                if removed:
                    break
            if removed:
                continue
            # End of read check

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
                    # If input is not a single-element memlet, skip
                    if (tinput.data.dynamic or tinput.data.subset.num_elements() != 1):
                        candidates.remove(candidate)
                        break
                    # If input array has inputs of its own (cannot promote within same state), skip
                    if state.in_degree(tinput.src) > 0:
                        candidates.remove(candidate)
                        break
                else:
                    # Check that tasklets have only one statement
                    cb: props.CodeBlock = edge.src.code
                    if cb.language is dtypes.Language.Python:
                        if (len(cb.code) > 1 or not isinstance(cb.code[0], ast.Assign)):
                            candidates.remove(candidate)
                            continue
                        # Ensure the candidate is assigned to
                        if (len(cb.code[0].targets) != 1 or astutils.rname(cb.code[0].targets[0]) != edge.src_conn):
                            candidates.remove(candidate)
                            continue
                        # Ensure that the candidate is not assigned through
                        # an "attribute" call, e.g., "dace.int64". These calls
                        # are not supported currently by the SymPy-based
                        # symbolic module.
                        detector = AttributedCallDetector()
                        detector.visit(cb.code[0].value)
                        if detector.detected:
                            candidates.remove(candidate)
                            continue
                    elif cb.language is dtypes.Language.CPP:
                        # Try to match a single C assignment
                        cstr = cb.as_string.strip()
                        # Since we cannot remove subscripts from C++ tasklets,
                        # if the type of the data is an array we will also skip
                        if re.match(r'^[a-zA-Z_][a-zA-Z_0-9]*\s*=.*;$', cstr) is None:
                            candidates.remove(candidate)
                            continue
                        newcode = translate_cpp_tasklet_to_python(cstr)
                        try:
                            parsed_ast = ast.parse(str(newcode))
                        except SyntaxError:
                            #if we cannot parse the expression to pythonize it, we cannot promote the candidate
                            candidates.remove(candidate)
                            continue
                    else:  # Other languages are currently unsupported
                        candidates.remove(candidate)
                        continue
            else:  # If input is not an acceptable node type, skip
                candidates.remove(candidate)
        candidates_seen |= candidates_in_state

    # Filter out non-integral symbols that do not appear in inter-state edges
    interstate_symbols = set()
    for edge in sdfg.edges():
        interstate_symbols |= edge.data.free_symbols
    for candidate in (candidates - interstate_symbols):
        if integers_only and sdfg.arrays[candidate].dtype not in dtypes.INTEGER_TYPES:
            candidates.remove(candidate)

    # Only keep candidates that were found in SDFG
    candidates &= (candidates_seen | interstate_symbols)

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
            return ast.copy_location(ast.Name(id=self.symbol, ctx=ast.Load()), node)
        return self.generic_visit(node)


class TaskletPromoterDict(ast.NodeTransformer):
    """
    Promotes scalars to symbols in Tasklets.
    If connector name is used in tasklet as subscript, modifies to symbol name.
    If connector is used as a standard name, modify tasklet code to use symbol.
    """
    def __init__(self, conn_to_sym: Dict[str, str]) -> None:
        """
        Initializes AST transformer.
        :param conn_to_sym: Connector name (replacement source) to symbol name (replacement target)
                            replacement dictionary.
        """
        self.conn_to_sym = conn_to_sym

    def visit_Name(self, node: ast.Name) -> Any:
        # Convert connector to symbol
        if node.id in self.conn_to_sym:
            node.id = self.conn_to_sym[node.id]
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # Convert subscript to symbol name
        node_name = astutils.rname(node)
        if node_name in self.conn_to_sym:
            return ast.copy_location(ast.Name(id=self.conn_to_sym[node_name], ctx=ast.Load()), node)
        return self.generic_visit(node)


class TaskletIndirectionPromoter(ast.NodeTransformer):
    """
    Promotes indirect memory access in Tasklets to symbolic memlets.
    After visiting an AST, self.{in,out}_mapping will be filled with mappings
    from unique new connector names to sets of individual memlets.
    """
    def __init__(self, in_connectors: Set[str], out_connectors: Set[str], sdfg: sd.SDFG,
                 defined_syms: Set[str]) -> None:
        """
        Initializes AST transformer.
        
        """
        self.iconns = in_connectors
        self.oconns = out_connectors
        self.sdfg = sdfg
        self.defined = defined_syms
        self.in_mapping: Dict[str, Tuple[str, subsets.Range]] = {}
        self.out_mapping: Dict[str, Tuple[str, subsets.Range]] = {}
        self.do_not_remove: Set[str] = set()
        self.latest: DefaultDict[str, int] = collections.defaultdict(int)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # Convert subscript to symbol name
        node_name = astutils.rname(node)
        if node_name in self.iconns:
            self.latest[node_name] += 1
            new_name = f'{node_name}_{self.latest[node_name]}'
            subset = subsets.Range(astutils.subscript_to_slice(node, self.sdfg.arrays)[1])
            # Check if range can be collapsed
            if _range_is_promotable(subset, self.defined):
                self.in_mapping[new_name] = (node_name, subset)
                return ast.copy_location(ast.Name(id=new_name, ctx=ast.Load()), node)
            else:
                self.do_not_remove.add(node_name)
        elif node_name in self.oconns:
            self.latest[node_name] += 1
            new_name = f'{node_name}_{self.latest[node_name]}'
            subset = subsets.Range(astutils.subscript_to_slice(node, self.sdfg.arrays)[1])
            # Check if range can be collapsed
            if _range_is_promotable(subset, self.defined):
                self.out_mapping[new_name] = (node_name, subset)
                return ast.copy_location(ast.Name(id=new_name, ctx=ast.Store()), node)
            else:
                self.do_not_remove.add(node_name)
        return self.generic_visit(node)


def _range_is_promotable(subset: subsets.Range, defined: Set[str]) -> bool:
    """ Helper function that determines whether a range is promotable. """
    # Some free symbols remain, we cannot promote
    if len(subset.free_symbols - defined) > 0:
        return False
    return True


def _handle_connectors(state: sd.SDFGState, node: nodes.Tasklet, mapping: Dict[str, Tuple[str, subsets.Range]],
                       ignore: Set[str], in_edges: bool) -> bool:
    """ 
    Adds new connectors and removes unused connectors after indirection
    promotion. 
    """
    if in_edges:
        orig_edges = {e.dst_conn: e for e in state.in_edges(node)}
    else:
        orig_edges = {e.src_conn: e for e in state.out_edges(node)}
    for cname, (orig, subset) in mapping.items():
        if in_edges:
            node.add_in_connector(cname)
        else:
            node.add_out_connector(cname)
        # Add new edge
        orig_edge = orig_edges[orig]
        if in_edges:
            state.add_edge(orig_edge.src, orig_edge.src_conn, orig_edge.dst, cname,
                           mm.Memlet(data=orig_edge.data.data, subset=subset))
        else:
            state.add_edge(orig_edge.src, cname, orig_edge.dst, orig_edge.dst_conn,
                           mm.Memlet(data=orig_edge.data.data, subset=subset))
    # Remove connectors and edges
    conns_to_remove = set(v[0] for v in mapping.values()) - ignore
    for conn in conns_to_remove:
        state.remove_edge(orig_edges[conn])
        if in_edges:
            node.remove_in_connector(conn)
        else:
            node.remove_out_connector(conn)


def _cpp_indirection_promoter(
    code: str, in_edges: Dict[str, mm.Memlet], out_edges: Dict[str, mm.Memlet], sdfg: sd.SDFG, defined_syms: Set[str]
) -> Tuple[str, Dict[str, Tuple[str, subsets.Range]], Dict[str, Tuple[str, subsets.Range]], Set[str]]:
    """
    Promotes indirect memory access in C++ Tasklets to symbolic memlets.
    """
    in_mapping: Dict[str, Tuple[str, subsets.Range]] = {}
    out_mapping: Dict[str, Tuple[str, subsets.Range]] = {}
    do_not_remove: Set[str] = set()
    latest: DefaultDict[str, int] = collections.defaultdict(int)

    # String replacement
    repl: Dict[Tuple[int, int], str] = {}

    # Find all occurrences of "aname[subexpr]"
    for m in re.finditer(r'([a-zA-Z_][a-zA-Z_0-9]*?)\[(.*?)\]', code):
        node_name = m.group(1)
        subexpr = m.group(2)
        if node_name in (set(in_edges.keys()) | set(out_edges.keys())):
            try:
                # NOTE: This is not necessarily a Python string. If fails,
                #       we skip this indirection.
                symexpr = symbolic.pystr_to_symbolic(subexpr)
            except TypeError:
                do_not_remove.add(node_name)
                continue

            latest[node_name] += 1
            new_name = f'{node_name}_{latest[node_name]}'

            # subexpr is always a one-dimensional index
            # Find non-scalar dimension to replace in memlet
            if node_name in in_edges:
                orig_subset = in_edges[node_name].subset
            else:
                orig_subset = out_edges[node_name].subset

            try:
                first_nonscalar_dim = next(i for i, s in enumerate(orig_subset.size()) if s != 1)
            except StopIteration:
                first_nonscalar_dim = 0

            # Make subset out of range and new sub-expression
            subset = subsets.Range(orig_subset.ndrange()[:first_nonscalar_dim] + [(subexpr, subexpr, 1)] +
                                   orig_subset.ndrange()[first_nonscalar_dim + 1:])

            # Check if range can be collapsed
            if _range_is_promotable(subset, defined_syms):
                if node_name in in_edges:
                    in_mapping[new_name] = (node_name, subset)
                else:
                    out_mapping[new_name] = (node_name, subset)
                repl[m.span()] = new_name
            else:
                do_not_remove.add(node_name)

    # Make all string replacements
    for (begin, end), replacement in reversed(sorted(repl.items())):
        code = code[:begin] + replacement + code[end:]

    return code, in_mapping, out_mapping, do_not_remove


def remove_symbol_indirection(sdfg: sd.SDFG):
    """
    Converts indirect memory accesses that involve only symbols into explicit
    memlets.

    :param sdfg: The SDFG to run the pass on.
    :note: Operates in-place.
    """
    for state, node, defined_syms in sdutils.traverse_sdfg_with_defined_symbols(sdfg):
        if not isinstance(node, nodes.Tasklet):
            continue
        # Strip subscripts one by one
        while True:
            in_mapping = {}
            out_mapping = {}
            do_not_remove = {}
            if node.code.language is dtypes.Language.Python:
                promo = TaskletIndirectionPromoter(set(node.in_connectors.keys()), set(node.out_connectors.keys()),
                                                   sdfg, defined_syms.keys())
                for stmt in node.code.code:
                    promo.visit(stmt)
                in_mapping = promo.in_mapping
                out_mapping = promo.out_mapping
                do_not_remove = promo.do_not_remove
            elif node.code.language is dtypes.Language.CPP:
                (node.code.code, in_mapping, out_mapping,
                 do_not_remove) = _cpp_indirection_promoter(node.code.as_string,
                                                            {e.dst_conn: e.data
                                                             for e in state.in_edges(node)},
                                                            {e.src_conn: e.data
                                                             for e in state.out_edges(node)}, sdfg, defined_syms.keys())

            # Nothing more to do
            if len(in_mapping) + len(out_mapping) == 0:
                break

            # Handle input/output connectors
            _handle_connectors(state, node, in_mapping, do_not_remove, True)
            _handle_connectors(state, node, out_mapping, do_not_remove, False)


def remove_scalar_reads(sdfg: sd.SDFG, array_names: Dict[str, str]):
    """
    Removes all instances of a promoted symbol's read accesses in an SDFG.
    This removes each read-only access node as well as all of its descendant
    edges (in memlet trees) and connectors. Descends recursively to nested
    SDFGs and modifies tasklets (Python and C++).
    :param sdfg: The SDFG to operate on.
    :param array_names: Mapping between scalar names to replace and their
                        replacement symbol name.
    :note: Operates in-place on the SDFG.
    """
    for state in sdfg.nodes():
        scalar_nodes = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data in array_names]
        for node in scalar_nodes:
            symname = array_names[node.data]
            for out_edge in state.out_edges(node):
                for e in state.memlet_tree(out_edge):
                    # Step 3.1
                    dst = e.dst
                    state.remove_edge_and_connectors(e)
                    if isinstance(dst, nodes.Tasklet):
                        # Step 3.2
                        if dst.language is dtypes.Language.Python:
                            promo = TaskletPromoter(e.dst_conn, symname)
                            for stmt in dst.code.code:
                                promo.visit(stmt)
                        elif dst.language is dtypes.Language.CPP:
                            # Replace whole-word matches (identifiers) in code
                            dst.code.code = re.sub(r'\b%s\b' % re.escape(e.dst_conn), symname, dst.code.as_string)
                    elif isinstance(dst, nodes.AccessNode):
                        # Step 3.3
                        t = state.add_tasklet('symassign', {}, {'__out'}, '__out = %s' % symname)
                        state.add_edge(t, '__out', dst, e.dst_conn,
                                       mm.Memlet(data=dst.data, subset=e.data.dst_subset, volume=1))
                        # Reassign destination for check below
                        dst = t
                    elif isinstance(dst, nodes.NestedSDFG):
                        tmp_symname = symname
                        val = 1
                        while (tmp_symname in dst.sdfg.symbols or tmp_symname in dst.sdfg.arrays):
                            # Find new symbol name
                            tmp_symname = f'{symname}_{val}'
                            val += 1

                        # Descend recursively to remove scalar
                        remove_scalar_reads(dst.sdfg, {e.dst_conn: tmp_symname})
                        for ise in dst.sdfg.edges():
                            ise.data.replace(e.dst_conn, tmp_symname)
                            # Remove subscript occurrences as well
                            for aname, aval in ise.data.assignments.items():
                                vast = ast.parse(aval)
                                vast = astutils.RemoveSubscripts({tmp_symname}).visit(vast)
                                ise.data.assignments[aname] = astutils.unparse(vast)
                            ise.data.replace(tmp_symname + '[0]', tmp_symname)

                        # Set symbol mapping
                        dst.sdfg.remove_data(e.dst_conn, validate=False)
                        dst.remove_in_connector(e.dst_conn)
                        dst.sdfg.symbols[tmp_symname] = sdfg.arrays[node.data].dtype
                        dst.symbol_mapping[tmp_symname] = symname
                    elif isinstance(dst, (nodes.EntryNode, nodes.ExitNode)):
                        # Skip
                        continue
                    else:
                        raise ValueError('Node type "%s" not supported for promotion' % type(dst).__name__)

                    # If nodes were disconnected, reconnect with empty memlet
                    if (isinstance(e.src, nodes.EntryNode) and len(state.edges_between(e.src, dst)) == 0):
                        state.add_nedge(e.src, dst, mm.Memlet())

        # Remove newly-isolated nodes
        state.remove_nodes_from([n for n in scalar_nodes if len(state.all_edges(n)) == 0])


def translate_cpp_tasklet_to_python(code: str):
    newcode: str = ''
    newcode = re.findall(r'.*?=\s*(.*);', code)[0]
    # We need to also translate the tasklet itself from CPP to Python
    newcode = re.sub(r'\|\|', ' or ', newcode)
    newcode = re.sub(r'\&\&', ' and ', newcode)
    return newcode


def promote_scalars_to_symbols(sdfg: sd.SDFG,
                               ignore: Optional[Set[str]] = None,
                               transients_only: bool = True,
                               integers_only: bool = True) -> Set[str]:
    """
    Promotes all matching transient scalars to SDFG symbols, changing all
    tasklets to inter-state assignments. This enables the transformed symbols
    to be used within states as part of memlets, and allows further
    transformations (such as loop detection) to use the information for
    optimization.

    :param sdfg: The SDFG to run the pass on.
    :param ignore: An optional set of strings of scalars to ignore.
    :param transients_only: If False, also considers global data descriptors (e.g., arguments).
    :param integers_only: If False, also considers non-integral descriptors for promotion.
    :return: Set of promoted scalars.
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
    to_promote = find_promotable_scalars(sdfg, transients_only=transients_only, integers_only=integers_only)
    if ignore:
        to_promote -= ignore
    if len(to_promote) == 0:
        return to_promote

    for state in sdfg.nodes():
        scalar_nodes = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data in to_promote]
        # Step 2: Assignment tasklets
        for node in scalar_nodes:
            if state.in_degree(node) == 0:
                continue
            in_edge = state.in_edges(node)[0]
            input = in_edge.src

            # There is only zero or one incoming edges by definition
            tasklet_inputs = [e.src for e in state.in_edges(input)]
            # Step 2.1
            new_state = xfh.state_fission(sdfg, gr.SubgraphView(state, set([input, node] + tasklet_inputs)))
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
                    newcode = translate_cpp_tasklet_to_python(input.code.as_string.strip())

                # Replace tasklet inputs with incoming edges
                for e in new_state.in_edges(input):
                    memlet_str: str = e.data.data
                    if (e.data.subset is not None and not isinstance(sdfg.arrays[memlet_str], dt.Scalar)):
                        memlet_str += '[%s]' % e.data.subset
                    newcode = re.sub(r'\b%s\b' % re.escape(e.dst_conn), memlet_str, newcode)
                # Add interstate edge assignment
                new_isedge.data.assignments[node.data] = newcode
            elif isinstance(input, nodes.AccessNode):
                memlet: mm.Memlet = in_edge.data
                if (memlet.src_subset and not isinstance(sdfg.arrays[memlet.data], dt.Scalar)):
                    new_isedge.data.assignments[node.data] = '%s[%s]' % (input.data, memlet.src_subset)
                else:
                    new_isedge.data.assignments[node.data] = input.data

            # Clean up all nodes after assignment was transferred
            new_state.remove_nodes_from(new_state.nodes())

    # Step 3: Scalar reads
    remove_scalar_reads(sdfg, {k: k for k in to_promote})

    # Step 4: Isolated nodes
    for state in sdfg.nodes():
        scalar_nodes = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data in to_promote]
        state.remove_nodes_from([n for n in scalar_nodes if len(state.all_edges(n)) == 0])

    # Step 5: Data descriptor management
    for scalar in to_promote:
        desc = sdfg.arrays[scalar]
        sdfg.remove_data(scalar, validate=False)
        # If the scalar is already a symbol (e.g., as part of an array size),
        # do not re-add the symbol
        if scalar not in sdfg.symbols:
            sdfg.add_symbol(scalar, desc.dtype)

    # Step 6: Inter-state edge cleanup
    cleanup_re = {s: re.compile(fr'\b{re.escape(s)}\[.*?\]') for s in to_promote}
    promo = TaskletPromoterDict({k: k for k in to_promote})
    for edge in sdfg.edges():
        ise: InterstateEdge = edge.data
        # Condition
        if not edge.data.is_unconditional():
            if ise.condition.language is dtypes.Language.Python:
                for stmt in ise.condition.code:
                    promo.visit(stmt)
            elif ise.condition.language is dtypes.Language.CPP:
                for scalar in to_promote:
                    ise.condition = cleanup_re[scalar].sub(scalar, ise.condition.as_string)

        # Assignments
        for aname, assignment in ise.assignments.items():
            for scalar in to_promote:
                if scalar in assignment:
                    ise.assignments[aname] = cleanup_re[scalar].sub(scalar, assignment.strip())

    # Step 7: Indirection
    remove_symbol_indirection(sdfg)

    return to_promote
