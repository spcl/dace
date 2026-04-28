"""Library-node + terminator emissions.

These are the shortest per-kind emitters: each one stamps a single DaCe
library node (CopyLibraryNode / MemsetLibraryNode / MatMul / Transpose /
Dot / Reduce) or control-flow terminator (BreakBlock / ReturnBlock) and
wires its memlets.

All share the same shape: flush pending scalars, ensure a state, add the
node, attach edges.  Kept together because they're structurally cousins
and none is big enough to earn its own file.
"""
from __future__ import annotations

import importlib
import math

from dace import InterstateEdge, Memlet

from dace.frontend.hlfir.builder.access import acc

# Per-library-node connector conventions.  Kept here rather than on
# ``LibNodeIntrinsic`` because the names are a property of the DaCe
# library node, not of the Fortran intrinsic.  Each entry maps a
# bridge-side ``LibNodeIntrinsic`` callee tag to ``(input_conns, output_conn)``;
# library nodes with their own dedicated emitters (CopyLibraryNode,
# MemsetLibraryNode, MergeLibraryNode, CountLibraryNode) bypass this
# generic dispatch table and live in the per-emitter functions below.
_LIBCALL_CONNECTORS = {
    "MatMul": (("_a", "_b"), "_c"),
    "Dot": (("_x", "_y"), "_result"),
    "Transpose": (("_inp", ), "_out"),
    "MergeLibraryNode": (("_t", "_f", "_mask"), "_out"),
    "CountLibraryNode": (("_mask", ), "_out"),
}


def emit_copy(builder, ctx, n, region):
    """Whole-array ``b = a`` → ``CopyLibraryNode`` with ``_in`` / ``_out``
    memlets covering the full source / destination arrays."""
    from dace.libraries.standard.nodes import CopyLibraryNode
    ctx.flush(builder)
    ctx.ensure(region)
    state = ctx.cur

    src_name = n.reduce_src  # buildCopyNode stored the source here
    tgt_name = n.target
    src_desc = ctx.sdfg.arrays[src_name]
    tgt_desc = ctx.sdfg.arrays[tgt_name]

    cp = CopyLibraryNode(f"copy_{tgt_name}_{builder.nid()}")
    state.add_node(cp)

    src_access = acc(builder, state, src_name)
    tgt_access = acc(builder, state, tgt_name)
    state.add_edge(src_access, None, cp, "_in", Memlet.from_array(src_name, src_desc))
    state.add_edge(cp, "_out", tgt_access, None, Memlet.from_array(tgt_name, tgt_desc))


def emit_memset(builder, ctx, n, region):
    """Scalar-zero → array fill → ``MemsetLibraryNode`` with a single
    ``_out`` memlet covering the destination.  The memset transitions
    to a fresh successor state so any later element write to the same
    array lands in a new state (and on a new access node) instead of
    racing with the array-wide write inside one state's DAG."""
    from dace.libraries.standard.nodes import MemsetLibraryNode
    ctx.flush(builder)
    ctx.ensure(region)
    state = ctx.cur

    tgt_name = n.target
    tgt_desc = ctx.sdfg.arrays[tgt_name]

    ms = MemsetLibraryNode(f"memset_{tgt_name}_{builder.nid()}")
    ms.add_out_connector("_out")
    state.add_node(ms)

    tgt_access = acc(builder, state, tgt_name)
    state.add_edge(ms, "_out", tgt_access, None, Memlet.from_array(tgt_name, tgt_desc))

    # Force a state break so a subsequent element write doesn't share
    # the memset's access node.  Two incoming memlets on one access
    # node race in DaCe's dataflow DAG.
    ctx.new_state(builder, region)


def emit_libcall(builder, ctx, n, region):
    """``target = matmul(a, b)`` / ``transpose(a)`` / ``dot_product(x, y)``
    lowered to the matching DaCe library node.  ``MatMul`` specializes
    internally (GEMM / GEMV / Dot) based on operand ranks.
    """
    from dace.frontend.hlfir.intrinsics import libnode_spec

    ctx.flush(builder)
    ctx.ensure(region)
    state = ctx.cur

    spec = libnode_spec(n.callee)
    if spec is None:
        raise RuntimeError(f"unregistered libnode intrinsic {n.callee!r}")
    mod = importlib.import_module(f"dace.libraries.{spec.module}.nodes")
    cls = getattr(mod, spec.node_cls)
    in_conns, out_conn = _LIBCALL_CONNECTORS[spec.node_cls]

    # ``Transpose`` needs an explicit ``dtype`` so its expansion can
    # produce the right element type; ``CountLibraryNode`` consumes its
    # Fortran-1-based ``dim`` from ``reduce_axes`` (set by the bridge's
    # ``buildLibCallNode`` when the source ``hlfir.count`` carries a dim
    # operand); every other library node picks types up from the
    # attached memlets.
    tgt_desc = ctx.sdfg.arrays[n.target]
    if spec.node_cls == "Transpose":
        node = cls(f"{spec.name}_{n.target}_{builder.nid()}", dtype=tgt_desc.dtype)
    elif spec.node_cls == "CountLibraryNode":
        # Bridge stores the (0-based) reduce axis the same way it does
        # for whole-array vs per-dim Reduce nodes.  CountLibraryNode's
        # constructor wants Fortran 1-based, so convert back.
        dim = (n.reduce_axes[0] + 1) if n.reduce_axes else -1
        node = cls(f"{spec.name}_{n.target}_{builder.nid()}", dim=dim)
    else:
        node = cls(f"{spec.name}_{n.target}_{builder.nid()}")
    state.add_node(node)

    # ``call_arg_subsets`` is parallel to ``call_args``; an empty entry =
    # whole-array source, a non-empty entry = a DaCe-0-based subset like
    # ``"0:3"`` for ``dot_product(arg1(1:3), arg2(1:3))``.  Older bridge
    # builds may not populate the field; default to empty for each arg.
    arg_subsets = list(getattr(n, 'call_arg_subsets', None) or [])
    arg_subsets += [''] * (len(n.call_args) - len(arg_subsets))
    for conn, src, sub in zip(in_conns, n.call_args, arg_subsets):
        src_desc = ctx.sdfg.arrays[src]
        if sub:
            in_memlet = Memlet(f"{src}[{sub}]")
        else:
            in_memlet = Memlet.from_array(src, src_desc)
        state.add_edge(acc(builder, state, src), None, node, conn, in_memlet)

    # Element-designate destination (``res1(1) = dot_product(...)``):
    # the bridge populates ``n.accesses[0]`` with the per-dim write
    # index so the output memlet covers a single element instead of
    # the whole array (which would fail validation for scalar-output
    # libcalls like dot_product, count, …).
    write_acc = next((ac for ac in n.accesses if ac.is_write), None)
    if write_acc is not None:
        from dace.frontend.hlfir.builder.access import build_memlet_index
        ix = build_memlet_index(builder, n.target, write_acc, ctx.iter_map)
        out_memlet = Memlet(f"{n.target}[{ix}]")
    else:
        out_memlet = Memlet.from_array(n.target, tgt_desc)
    state.add_edge(node, out_conn, acc(builder, state, n.target), None, out_memlet)


def emit_reduce(builder, ctx, n, region):
    """``target = sum(src)`` (and product / minval / maxval) lowered as a
    DaCe ``standard.Reduce`` library node via
    ``state.add_reduce(wcr, axes, identity)``.

    ``axes=None`` reduces all dimensions (whole-array scalar result); a
    non-empty ``reduce_axes`` list reduces along those dims only.

    When ``n.target_is_array`` is true and ``n.accesses[0]`` carries a
    write AccessInfo (LHS was ``res(i) = MINVAL(...)``), the output
    memlet covers only that element — otherwise multiple reductions
    in the same routine all write through the whole destination and
    the last one wins.
    """
    from dace.frontend.hlfir.builder.access import build_memlet_index

    ctx.flush(builder)
    ctx.ensure(region)
    state = ctx.cur

    src_name = n.reduce_src
    src_desc = ctx.sdfg.arrays.get(src_name)
    if src_desc is None:
        raise RuntimeError(f"reduction source {src_name!r} not registered as SDFG data")
    axes = list(n.reduce_axes) if n.reduce_axes else None

    # DaCe's Reduce expects a value (or None) for ``identity``.  The
    # bridge emits the float-extreme identities as bare ``inf`` /
    # ``-inf`` (so the section-reduce init tasklet renders to a valid
    # ``INFINITY`` C++ literal); patch the eval namespace so this
    # whole-array path resolves them too.
    #
    # Fortran spec: ``MINVAL`` / ``MAXVAL`` on an empty array returns
    # ``HUGE(x)`` / ``-HUGE(x)`` (the dtype's representable extreme),
    # not ``±inf``.  Substitute the identity per destination dtype so
    # the empty-array case matches gfortran exactly and the integer
    # path doesn't truncate ``inf`` to a garbage int.
    import numpy as _np
    tgt_desc = ctx.sdfg.arrays[n.target]
    identity_val = None
    if n.reduce_identity:
        identity_val = eval(n.reduce_identity, {'math': math, 'inf': math.inf})
        if identity_val in (math.inf, -math.inf):
            np_dt = tgt_desc.dtype.as_numpy_dtype()
            if _np.issubdtype(np_dt, _np.integer):
                info = _np.iinfo(np_dt)
                identity_val = info.max if identity_val == math.inf else info.min
            elif _np.issubdtype(np_dt, _np.floating):
                info = _np.finfo(np_dt)
                identity_val = float(info.max if identity_val == math.inf else info.min)

    red = state.add_reduce(n.reduce_wcr, axes, identity_val)

    src_access = acc(builder, state, src_name)
    tgt_access = acc(builder, state, n.target)
    state.add_edge(src_access, None, red, None, Memlet.from_array(src_name, src_desc))

    write_acc = next((ac for ac in n.accesses if ac.is_write), None) if n.accesses else None
    if n.target_is_array and write_acc is not None and write_acc.index_exprs:
        subset = build_memlet_index(builder, n.target, write_acc, iter_map={})
        out_memlet = Memlet(f"{n.target}[{subset}]")
    else:
        out_memlet = Memlet.from_array(n.target, tgt_desc)
    state.add_edge(red, None, tgt_access, None, out_memlet)


def emit_break(builder, ctx, n, region):
    """Fortran ``EXIT`` → ``BreakBlock`` added to the current region.
    The block is a leaf and implicitly transfers control to the nearest
    enclosing loop's exit edge at codegen time.  When the break is the
    region's first block (a branch body whose only statement is
    ``exit``), it becomes the region's start block.
    """
    from dace.sdfg.state import BreakBlock
    ctx.flush(builder, region)
    is_start = ctx.cur is None
    blk = BreakBlock(f"break_{builder.nid()}")
    region.add_node(blk, is_start_block=is_start)
    if ctx.cur is not None:
        region.add_edge(ctx.cur, blk, InterstateEdge())
    ctx.cur = blk


def emit_return(builder, ctx, n, region):
    """Fortran ``RETURN`` → ``ReturnBlock``.  Added to the current region
    so RETURNs nested inside a loop or conditional get placed correctly;
    codegen still emits a plain ``return`` that bails out of the whole
    subroutine.
    """
    from dace.sdfg.state import ReturnBlock
    ctx.flush(builder, region)
    is_start = ctx.cur is None
    blk = ReturnBlock(f"return_{builder.nid()}")
    region.add_node(blk, is_start_block=is_start)
    if ctx.cur is not None:
        region.add_edge(ctx.cur, blk, InterstateEdge())
    ctx.cur = blk
