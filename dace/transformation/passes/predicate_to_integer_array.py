# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that stores transient boolean arrays used as predicates in loops as integers.

A 1-byte ``bool`` array that selects between floating-point values in a loop (``mask[i] ? a : b``) keeps compilers
from vectorizing the loop: gcc 13 vectorizes a loop by the widest vector of its narrowest element type, and cannot
combine a vector of 32 byte-sized masks with vectors of 8 floats (``relevant stmt not supported: mask != 0 ? float :
float``). Storing the mask as a 32-bit integer removes the mismatch. On the pyFV3 D_SW dynamical core this made gcc
3.3-3.8x faster on the PPM advection loops (and clang up to 3x), with identical results.

The array keeps holding only ``0`` and ``1``: every value written to it is converted with ``bool(...)`` first, and
C++ treats an integer ``0``/``1`` exactly like the ``bool`` it came from in every expression that reads it.
"""
import ast
from typing import Dict, List, Optional, Set, Tuple

from dace import SDFG, data, dtypes, properties
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl, transformation


def _output_assignments(tasklet: nodes.Tasklet, connector: str) -> Optional[List[ast.Assign]]:
    """The statements of a Python tasklet that assign ``connector``, if every one of them is a plain
    ``connector = <expression>``; ``None`` otherwise (other languages, augmented or tuple assignments)."""
    if tasklet.language != dtypes.Language.Python:
        return None
    assignments = []
    for node in ast.walk(ast.Module(body=list(tasklet.code.code), type_ignores=[])):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
        names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        if connector not in names:
            continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            return None
        assignments.append(node)
    return assignments


def _names(node: ast.AST) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _predicate_flow(tasklet: nodes.Tasklet, connector: str) -> Tuple[bool, Set[str]]:
    """How a Python tasklet uses input ``connector`` as a truth value: whether it chooses between alternatives with it
    (in the test of a conditional expression or ``if``), and which names it passes it on to by boolean operations
    (``t = m and a > 0``; among them, output connectors). Other languages use nothing."""
    if tasklet.language != dtypes.Language.Python:
        return False, set()
    module = ast.Module(body=list(tasklet.code.code), type_ignores=[])
    derived = {connector}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(module):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, (ast.Name, ast.BoolOp, ast.UnaryOp, ast.Compare))
                    and _names(node.value) & derived and node.targets[0].id not in derived):
                derived.add(node.targets[0].id)
                changed = True
    tested = any(
        isinstance(node, (ast.IfExp, ast.If, ast.While)) and _names(node.test) & derived for node in ast.walk(module))
    return tested, derived


def _in_loop(region: ControlFlowRegion) -> bool:
    """Whether code in ``region`` (a state or control-flow region) can run repeatedly: it is within a loop region, or
    within a nested SDFG inside a map or loop."""
    while region is not None:
        if isinstance(region, LoopRegion):
            return True
        parent = region.parent_graph
        if parent is None and isinstance(region, SDFG) and region.parent_nsdfg_node is not None:
            state = region.parent
            if state.entry_node(region.parent_nsdfg_node) is not None:
                return True
            parent = state
        region = parent
    return False


def _predicate_containers(sdfg: SDFG) -> Set[str]:
    """Containers used as predicates inside loops: read in branch, loop and interstate-edge conditions that run in a
    loop, or read by a tasklet in a loop (or map) that tests them, or combined by boolean operations -- in a tasklet
    or an interstate-edge assignment in a loop -- into another predicate (e.g. ``flag = above[i - 1] or above[i]``
    feeding ``if flag``)."""
    names: Set[str] = set()
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, ConditionalBlock) and _in_loop(block):
            for condition, _ in block.branches:
                if condition is not None:
                    names |= set(map(str, condition.get_free_symbols()))
        elif isinstance(block, LoopRegion) and block.loop_condition is not None:
            names |= set(map(str, block.loop_condition.get_free_symbols()))
    for region in sdfg.all_control_flow_regions():
        if _in_loop(region):
            for edge in region.edges():
                if edge.data is not None and not edge.data.is_unconditional():
                    names |= set(map(str, edge.data.condition.get_free_symbols()))

    # Through tasklets: (read container, written container) pairs where the value flows by boolean operations
    flows: Set[Tuple[str, str]] = set()
    for state in sdfg.states():
        for tasklet in state.nodes():
            if not isinstance(tasklet, nodes.Tasklet) or not (state.entry_node(tasklet) is not None or _in_loop(state)):
                continue
            outputs = {e.src_conn: e.data.data for e in state.out_edges(tasklet) if not e.data.is_empty()}
            for edge in state.in_edges(tasklet):
                if edge.data.is_empty() or edge.dst_conn is None:
                    continue
                source = state.memlet_path(edge)[0].data.data
                tested, derived = _predicate_flow(tasklet, edge.dst_conn)
                if tested:
                    names.add(source)
                flows |= {(source, outputs[o]) for o in derived & outputs.keys()}
    # Through interstate-edge assignments in loops (``flag = above[i - 1] or above[i]`` feeding ``if flag``)
    for region in sdfg.all_control_flow_regions():
        if not _in_loop(region):
            continue
        for edge in region.edges():
            for target, value in (edge.data.assignments.items() if edge.data is not None else ()):
                try:
                    expr = ast.parse(str(value), mode='eval').body
                except SyntaxError:
                    continue
                if (isinstance(expr, (ast.Name, ast.Subscript, ast.BoolOp, ast.Compare))
                        or (isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not))):
                    flows |= {(name, target) for name in _names(expr)}
    changed = True
    while changed:
        changed = False
        for source, target in flows:
            if target in names and source not in names:
                names.add(source)
                changed = True
    return names


@properties.make_properties
@transformation.explicit_cf_compatible
class PredicateToIntegerArray(ppl.Pass):
    """
    Store transient boolean arrays that act as predicates in loops as integers (``dtype``), so that vectorized loops
    can select with them.

    An array is retyped if it is a transient ``bool`` array (not a view, reference or scalar) that
    (a) is used as a predicate in a loop: some read chooses between alternatives inside a map or loop, i.e. in the
        test of a conditional expression or ``if`` in a tasklet (possibly through local boolean values), or in a
        branch, loop or interstate-edge condition, or is combined by boolean operations into another container that
        is used so (``flag = above[i - 1] or above[i]`` feeding ``if flag``); and
    (b) is only accessed by single-element memlets without write-conflict resolution at tasklet connectors that are
        untyped or typed ``bool``.
    Arrays used only as data, or only outside loops, stay ``bool``: widening them would cost memory traffic without
    enabling vectorization.

    The accessing connectors become ``dtype``, so elements are loaded and stored as values. Every Python assignment
    ``c = <expression>`` to such an output connector becomes ``c = bool(<expression>)``, so the array only ever holds
    ``0`` or ``1`` (outputs assigned otherwise, or by tasklets in other languages, keep the array ``bool``). Reading
    an integer ``0``/``1`` gives the same result as reading the ``bool`` in every C++ expression, in tasklets as well
    as in conditions, which therefore need no change. Arrays passed into nested SDFGs are left alone (the nested
    SDFG's own descriptor would have to change with them); run after inlining.
    """

    CATEGORY: str = 'Optimization Preparation'

    # The integer width should match the floating-point values the predicate selects between, so that masks and
    # values fill vectors with the same number of lanes: int32 for float (measured: gcc 3.3-3.8x on pyFV3 D_SW);
    # for double, int64 is the matching width (4 lanes on AVX2), which has not been measured yet. A future version may
    # derive the width from the predicate's uses instead.
    dtype = properties.TypeClassProperty(default=dtypes.int32,
                                         desc='Integer type to store predicate arrays as (match the width of the '
                                         'floating-point values they select, e.g. int64 for double)')

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, _: dict) -> Optional[Dict[str, Set[str]]]:
        """
        :return: The retyped arrays, by SDFG label, or ``None`` if nothing was retyped.
        """
        result: Dict[str, Set[str]] = {}
        for nested in sdfg.all_sdfgs_recursive():
            retyped = self._retype(nested)
            if retyped:
                result[nested.label] = retyped
        return result or None

    def _retype(self, sdfg: SDFG) -> Set[str]:
        candidates = {
            name
            for name, desc in sdfg.arrays.items()
            if desc.transient and desc.dtype == dtypes.bool_ and type(desc) is data.Array
        }
        # Per array: (tasklet, connector, is output) of every access
        accesses: Dict[str, List[Tuple[nodes.Tasklet, str, bool]]] = {name: [] for name in candidates}
        for state in sdfg.states():
            for node in state.data_nodes():
                if node.data not in candidates:
                    continue
                for edge in state.all_edges(node):
                    if edge.data.is_empty():
                        continue
                    path = state.memlet_path(edge)
                    output = path[0].src is not node
                    end = path[0] if output else path[-1]
                    owner, connector = (end.src, end.src_conn) if output else (end.dst, end.dst_conn)
                    if (not isinstance(owner, nodes.Tasklet) or connector is None
                            or any(e.data.wcr is not None or e.data.data != node.data for e in path)
                            or end.data.other_subset is not None or end.data.subset.num_elements() != 1):
                        candidates.discard(node.data)
                        break
                    ctype = (owner.out_connectors if output else owner.in_connectors).get(connector, None)
                    if ctype is not None and ctype.type is not None and ctype != dtypes.bool_:
                        candidates.discard(node.data)  # Would be loaded or stored through a pointer cast
                        break
                    if output and _output_assignments(owner, connector) is None:
                        candidates.discard(node.data)
                        break
                    accesses[node.data].append((owner, connector, output))
        candidates &= _predicate_containers(sdfg)
        for name in candidates:
            sdfg.arrays[name].dtype = self.dtype
            for tasklet, connector, output in accesses[name]:
                (tasklet.out_connectors if output else tasklet.in_connectors)[connector] = self.dtype
                if not output:
                    continue
                for assignment in _output_assignments(tasklet, connector):
                    value = assignment.value
                    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                            and value.func.id == 'bool'):
                        assignment.value = ast.copy_location(
                            ast.Call(func=ast.Name(id='bool', ctx=ast.Load()), args=[value], keywords=[]), value)
                tasklet.code = CodeBlock([ast.fix_missing_locations(s) for s in tasklet.code.code],
                                         tasklet.code.language)
        return candidates
