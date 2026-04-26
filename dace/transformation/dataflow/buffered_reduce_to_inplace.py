# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite the ``Map -> buffer -> Reduce`` anti-pattern into a sequential
``Map`` that accumulates straight into the destination — no WCR edges.

::

    BEFORE
    ------
                   ┌──────────┐
       a ─▶ MapEntry ─▶ Tasklet ─▶ MapExit ─▶ buf ─▶ Reduce ─▶ out
                       _out=expr     IN_buf  OUT_buf  wcr,id

    AFTER
    -----
       (init state placed before)         (sequential schedule)
                                            ┌────────────┐
                          ┌─▶ Tasklet ─▶│ acc tasklet│
       a ─▶ MapEntry ─────┤  _out=expr  │  __out =    ├─▶ MapExit ─▶ out
       out ─▶ MapEntry ───┘   (unchanged) │ wcr(acc,t) │   IN_out
                                            └────────────┘   OUT_out
       init: out[*] = identity     buf + Reduce removed

What changed:
  * Map schedule → ``Sequential`` (read-modify-write of ``out`` is race-free).
  * Buffer ``buf`` and the ``Reduce`` libnode are deleted.
  * A new accumulator tasklet sits between the original tasklet and the
    map exit; it reads the running ``out`` value through the map entry
    and writes the updated value through the map exit (regular memlet,
    *no WCR*).
  * An init state seeds ``out[*] = identity`` once before the loop.
  * Downstream tiling/partitioning passes can split this sequential map
    into a parallel chunk loop with a small final reduce step.

Not auto-applied; users invoke explicitly when they want the buffer-
free shape (e.g. before ``MapReduceFusion`` — once the Reduce libnode
is gone, that fusion has nothing to match).
"""
import ast

from dace import dtypes
from dace.memlet import Memlet
from dace.properties import make_properties
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.subsets import Range
from dace.transformation import transformation as pm

import dace.libraries.standard as stdlib  # for Reduce libnode


def _wcr_body_substitute(wcr: str, lhs_name: str, rhs_name: str) -> str:
    """Render the WCR lambda's body with the formal parameters replaced
    by ``lhs_name`` / ``rhs_name``.

    ``wcr`` is a string like ``"lambda a, b: a + b"`` (or ``min`` / ``max``
    / ``or`` / ``and``).  We parse it as a Python expression, find the
    two formal arg names, and substitute them throughout the body.
    """
    tree = ast.parse(wcr, mode='eval').body
    if not isinstance(tree, ast.Lambda) or len(tree.args.args) != 2:
        raise ValueError(f"WCR must be a 2-arg lambda; got {wcr!r}")
    a_name = tree.args.args[0].arg
    b_name = tree.args.args[1].arg

    class Sub(ast.NodeTransformer):

        def visit_Name(self, n):
            if n.id == a_name:
                return ast.copy_location(ast.Name(id=lhs_name, ctx=n.ctx), n)
            if n.id == b_name:
                return ast.copy_location(ast.Name(id=rhs_name, ctx=n.ctx), n)
            return n

    body = Sub().visit(tree.body)
    ast.fix_missing_locations(body)
    return ast.unparse(body)


@make_properties
class BufferedReduceToInplace(pm.SingleStateTransformation):
    """Rewrite ``Tasklet -> MapExit -> buf -> Reduce -> out`` as a
    sequential Map that accumulates directly into ``out``.

    The intermediate ``buf`` and the ``Reduce`` library node are removed.
    No WCR edges are introduced — instead the Map's schedule is set to
    ``Sequential`` so the read-modify-write through the destination
    AccessNode is race-free at codegen time.  An init state is inserted
    before the current state to seed ``out`` with the reduction's
    identity.
    """

    tasklet = pm.PatternNode(nodes.Tasklet)
    map_exit = pm.PatternNode(nodes.MapExit)
    in_array = pm.PatternNode(nodes.AccessNode)
    reduce_node = pm.PatternNode(stdlib.Reduce)
    out_array = pm.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.tasklet, cls.map_exit, cls.in_array, cls.reduce_node, cls.out_array)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        tmap_exit = self.map_exit
        in_array = self.in_array
        reduce_node = self.reduce_node
        tasklet = self.tasklet

        # Buffer must be a transient written only by the map exit and
        # read only by the reduce node, with no other accessors.
        if any(src is not tmap_exit for src, _, _, _, _ in graph.in_edges(in_array)):
            return False
        if any(dst is not reduce_node for _, _, dst, _, _ in graph.out_edges(in_array)):
            return False
        if not sdfg.arrays[in_array.data].transient:
            return False
        if (len([n for n in graph.nodes() if isinstance(n, nodes.AccessNode) and n.data == in_array.data]) > 1
                or in_array.data in sdfg.shared_transients()):
            return False

        # Need an identity to seed the destination.
        if reduce_node.identity is None:
            return False

        # MapExit→buffer subset must match Reduce input subset (post-
        # aggregation full range on both sides).  Mirrors MapReduceFusion.
        tout_memlet = graph.in_edges(in_array)[0].data
        rin_memlet = graph.out_edges(in_array)[0].data
        if tout_memlet.subset != rin_memlet.subset:
            return False
        # The tasklet must have exactly one output edge into the map exit
        # — keeps the rewrite trivial (one output value to accumulate).
        out_edges = [e for e in graph.out_edges(tasklet) if e.dst is tmap_exit]
        if len(out_edges) != 1:
            return False
        if len(tasklet.out_connectors) != 1:
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        tasklet = self.tasklet
        tmap_exit = self.map_exit
        in_array = self.in_array
        reduce_node = self.reduce_node
        out_array = self.out_array

        tmap = tmap_exit.map
        # Find the matching MapEntry by scanning nodes for one with the same
        # ``map`` attribute — ``graph.entry_node`` walks the scope dict which
        # is unhappy with the dangling buffer node we're about to remove.
        map_entry = next(n for n in graph.nodes() if isinstance(n, nodes.MapEntry) and n.map is tmap)
        # Sequential schedule keeps the in-place accumulation race-free.
        tmap.schedule = dtypes.ScheduleType.Sequential

        # Compute the destination subset (output array indices that
        # remain after dropping the reduction axes from the buffer's
        # memlet subset).
        rin_memlet = graph.out_edges(in_array)[0].data
        rout_memlet = graph.out_edges(reduce_node)[0].data
        axes = reduce_node.axes
        if axes is None:
            axes = list(range(len(rin_memlet.subset)))
        dst_subset_list = [d for i, d in enumerate(rin_memlet.subset) if i not in axes]
        if not dst_subset_list:
            dst_subset_list = [(0, 0, 1)]
        dst_subset = Range(dst_subset_list)

        # Output dtype + identity init.
        identity_str = str(reduce_node.identity)
        init_state = graph.parent_graph.add_state_before(graph, label=f'reduce_init_{out_array.data}')
        init_iters = [(f'_oi{i}', f'{r[0]}:{r[1] + 1}:{r[2]}') for i, r in enumerate(dst_subset_list)]
        if init_iters:
            init_state.add_mapped_tasklet(
                f'reduce_init_{out_array.data}',
                init_iters,
                {},
                f'__out = {identity_str}',
                {'__out': Memlet.simple(out_array.data, ','.join(it[0] for it in init_iters))},
                external_edges=True,
            )
        else:
            t = init_state.add_tasklet('reduce_init', {}, {'__out'}, f'__out = {identity_str}')
            wnode = init_state.add_write(out_array.data)
            init_state.add_edge(t, '__out', wnode, None, Memlet.simple(out_array.data, '0'))

        # The original tasklet's single output connector + the WCR lambda's
        # body give us the new in-place accumulator code.
        old_out_conn = next(iter(tasklet.out_connectors))
        acc_in_conn = '__acc_in'
        # Pick a fresh expr-input name that does not collide with existing connectors.
        expr_in_conn = '__expr_in'
        i = 0
        while expr_in_conn in tasklet.in_connectors or expr_in_conn in tasklet.out_connectors:
            expr_in_conn = f'__expr_in_{i}'
            i += 1

        # New accumulator tasklet placed between the original tasklet and the map exit.
        body = _wcr_body_substitute(reduce_node.wcr, acc_in_conn, expr_in_conn)
        acc_tasklet = graph.add_tasklet(
            f'reduce_acc_{out_array.data}',
            {acc_in_conn, expr_in_conn},
            {'__out'},
            f'__out = {body}',
        )

        # Wire: original_tasklet[old_out] -> acc_tasklet[expr_in].
        # The inter-tasklet memlet carries the scalar value in flight and
        # has no associated data array — leaving ``data`` blank lets
        # codegen mint a unique register name and avoids colliding with
        # the destination array's name.
        old_te_edge = next(e for e in graph.out_edges(tasklet) if e.dst is tmap_exit)
        graph.remove_edge(old_te_edge)
        graph.add_edge(tasklet, old_out_conn, acc_tasklet, expr_in_conn, Memlet())

        # Read current accumulator value: out_array(read) -> map_entry -> acc_tasklet[acc_in].
        out_read = graph.add_read(out_array.data)
        # Add a passthrough connector pair on map_entry.
        new_in_conn = f'IN_{out_array.data}'
        new_out_conn = f'OUT_{out_array.data}'
        suffix = 0
        while new_in_conn in map_entry.in_connectors or new_out_conn in map_entry.out_connectors:
            suffix += 1
            new_in_conn = f'IN_{out_array.data}_{suffix}'
            new_out_conn = f'OUT_{out_array.data}_{suffix}'
        map_entry.add_in_connector(new_in_conn)
        map_entry.add_out_connector(new_out_conn)
        graph.add_edge(out_read, None, map_entry, new_in_conn,
                       Memlet(data=out_array.data, subset=str(dst_subset), volume=1))
        graph.add_edge(map_entry, new_out_conn, acc_tasklet, acc_in_conn,
                       Memlet(data=out_array.data, subset=str(dst_subset), volume=1))

        # acc_tasklet[__out] -> map_exit -> out_array (write back, no WCR).
        # Replace the MapExit's old IN_<buf>/OUT_<buf> connector pair with
        # a new IN_<out>/OUT_<out> pair so codegen-side variable names line
        # up with the data they carry — leaving the old name in place
        # makes codegen think it has to redeclare ``<out>`` as a scalar
        # local inside the scope.
        old_in_conn = old_te_edge.dst_conn  # 'IN_<bufname>'
        old_out_map_conn = 'OUT_' + old_in_conn[3:]
        # Drop the now-stale buf-named edge from MapExit before retiring its connectors.
        old_buf_edge = next(e for e in graph.out_edges(tmap_exit)
                            if e.src_conn == old_out_map_conn and e.dst is in_array)
        graph.remove_edge(old_buf_edge)
        tmap_exit.remove_in_connector(old_in_conn)
        tmap_exit.remove_out_connector(old_out_map_conn)

        new_exit_in = f'IN_{out_array.data}'
        new_exit_out = f'OUT_{out_array.data}'
        # Disambiguate against any pre-existing same-named connector on
        # the exit (rare but cheap to handle).
        suffix = 0
        while new_exit_in in tmap_exit.in_connectors or new_exit_out in tmap_exit.out_connectors:
            suffix += 1
            new_exit_in = f'IN_{out_array.data}_{suffix}'
            new_exit_out = f'OUT_{out_array.data}_{suffix}'
        tmap_exit.add_in_connector(new_exit_in)
        tmap_exit.add_out_connector(new_exit_out)

        graph.add_edge(acc_tasklet, '__out', tmap_exit, new_exit_in,
                       Memlet(data=out_array.data, subset=str(dst_subset), volume=1))
        graph.add_edge(tmap_exit, new_exit_out, out_array, None,
                       Memlet(data=out_array.data, subset=str(dst_subset), volume=1))

        # Remove buffer + reduce nodes.
        graph.remove_node(in_array)
        graph.remove_node(reduce_node)

        # Remove the (now-orphaned) reduce -> out edge if it survived.
        for e in list(graph.in_edges(out_array)):
            if e.src is None:
                graph.remove_edge(e)

        # Drop buffer descriptor if no longer referenced.
        try:
            sdfg.remove_data(in_array.data)
        except ValueError:
            pass
