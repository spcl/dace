# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Post-descent invariants for the vectorization pipeline.

The K-dim ``PromoteNSDFGBodyToTiles`` descent and the legacy
``VectorizeCPU`` ``_vectorize_map`` path both transform an SDFG region
in place. Two structural invariants must hold once each finishes; both
are silent-correctness frontiers (codegen would otherwise compile but
miscompute):

1. No memlet may carry a non-``None`` ``other_subset``. The K-dim
   descent's ``_break_an_to_an_with_other_subset`` runs upfront to
   convert every cross-side AN -> AN copy into an explicit
   ``_out = _in`` tasklet whose two flanking memlets each carry their
   own ``data + subset``. A surviving ``other_subset`` would let the
   ``CopyND`` codegen fall back to default strides and silently emit a
   wrong-stride copy.

2. The body holds at most one WCR-tagged memlet, and that memlet must
   be the LAST edge of its chain (its destination is the surrounding
   ``MapExit``). DaCe's accumulator codegen handles exactly one WCR
   write per destination per iteration -- a mid-chain WCR breaks the
   accumulator's atomicity contract, and two WCR chains targeting the
   same destination race.

Both checks are loud refusals (``NotImplementedError``) so the
miscompile never reaches the runtime.
"""
from typing import Iterable

import dace
from dace import subsets
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState


def _node_handle(n: nodes.Node) -> str:
    """Return a short display label for ``n`` for diagnostic strings.

    AccessNodes carry ``.data``; other node kinds (Tasklet, MapEntry,
    LibraryNode, ...) carry ``.label``. Both fields are guaranteed by
    DaCe's node hierarchy, so the previous defensive
    ``getattr(n, 'data', getattr(n, 'label', '?'))`` chain is replaced by
    an explicit isinstance dispatch.

    :param n: The SDFG node to describe.
    :returns: ``n.data`` for AccessNodes; ``n.label`` otherwise.
    """
    return n.data if isinstance(n, nodes.AccessNode) else n.label


def cleanup_an_to_an_edges(sdfg: dace.SDFG) -> None:
    """Convert every direct ``AccessNode -> AccessNode`` edge in ``sdfg`` into a
    canonical ``AN -> [_out = _in] -> AN`` form (or drop empty-data ones).

    Per user directive: "we will never vectorize an AN -> AN edge". Runs
    BEFORE any boundary widening / fan-out / collapse, so both vectorization
    paths (legacy ``_vectorize_map`` and the K-dim
    :class:`PromoteNSDFGBodyToTiles` descent) see a body NSDFG free of direct
    copies. Each surviving copy gets a fresh assignment tasklet; the original
    in-side subset goes on the ``AN -> tasklet`` edge as
    ``data=src.data, subset=src_subset`` and the original out-side subset on
    the ``tasklet -> AN`` edge as ``data=dst.data, subset=dst_subset``.
    Neither flanking memlet carries ``other_subset``, so the descent's
    downstream passes never see one.

    Empty-data edges (``memlet=None`` or ``memlet.data is None``) are
    structural ordering links between two unrelated AccessNodes and carry no
    data; they are simply removed.

    :param sdfg: The body SDFG to clean.
    :raises NotImplementedError: When the source and destination volumes of
        a non-empty AN -> AN edge cannot be reconciled by a scalar
        ``_out = _in`` tasklet.
    """
    for istate in sdfg.states():
        for ed in list(istate.edges()):
            if not (isinstance(ed.src, dace.nodes.AccessNode) and isinstance(ed.dst, dace.nodes.AccessNode)):
                continue
            if ed.data is None or ed.data.data is None:
                # Empty-data ordering edge: no data flow, drop the link so
                # ``assert_no_an_to_an_edges`` is satisfied.
                istate.remove_edge(ed)
                continue
            in_data = ed.data.data
            # Decide source / destination sides from the memlet's data field.
            # When ``data == dst.data``, the memlet's ``subset`` is
            # destination-side and ``other_subset`` (if any) is source-side;
            # vice versa for ``data == src.data``.
            if in_data == ed.dst.data:
                dst_subset = ed.data.subset
                src_subset = ed.data.other_subset if ed.data.other_subset is not None else ed.data.subset
            else:
                src_subset = ed.data.subset
                dst_subset = ed.data.other_subset if ed.data.other_subset is not None else ed.data.subset
            try:
                in_ne = src_subset.num_elements_exact()
                out_ne = dst_subset.num_elements_exact()
            except Exception:
                in_ne = out_ne = None
            if in_ne is None or out_ne is None or not bool(dace.symbolic.simplify(in_ne - out_ne) == 0):
                raise NotImplementedError(
                    f"vectorization: AccessNode {ed.src.data!r} -> AccessNode {ed.dst.data!r} edge "
                    f"in state {istate.label!r} has mismatched src ({src_subset}, ne={in_ne}) vs "
                    f"dst ({dst_subset}, ne={out_ne}) volume; cannot insert a ``_out = _in`` "
                    f"assignment tasklet -- refusing vectorization.")
            t = istate.add_tasklet(name=f"_break_an_an_{ed.src.data}_to_{ed.dst.data}",
                                   inputs={"_in"},
                                   outputs={"_out"},
                                   code="_out = _in",
                                   language=dace.dtypes.Language.Python)
            istate.add_edge(ed.src, ed.src_conn, t, "_in",
                            dace.Memlet(data=ed.src.data, subset=subsets.Range(list(src_subset.ranges))))
            istate.add_edge(t, "_out", ed.dst, ed.dst_conn,
                            dace.Memlet(data=ed.dst.data, subset=subsets.Range(list(dst_subset.ranges))))
            istate.remove_edge(ed)


def assert_no_residual_other_subsets(sdfg: dace.SDFG) -> None:
    """Refuse vectorization if any memlet in ``sdfg`` carries an ``other_subset``.

    Per user directive: "we will never vectorize an AN -> AN edge". The
    cleanup phase converts every AN -> AN edge into the canonical
    ``AN -> [_out=_in] -> AN`` form so each flanking memlet carries only
    ``data + subset``. A surviving ``other_subset`` means a downstream pass
    re-introduced a direct copy edge; refuse loudly so codegen never falls
    back to a wrong-stride ``CopyND``.

    :param sdfg: The SDFG (top-level or body NSDFG) whose memlets to scan.
    :raises NotImplementedError: When at least one edge memlet has a
        non-``None`` ``other_subset``.
    """
    for istate in sdfg.states():
        for ed in istate.edges():
            if ed.data is None or ed.data.other_subset is None:
                continue
            raise NotImplementedError(
                f"vectorization: residual ``other_subset`` on edge "
                f"{type(ed.src).__name__}({_node_handle(ed.src)!r}) "
                f"-> {type(ed.dst).__name__}({_node_handle(ed.dst)!r}) "
                f"in state {istate.label!r}: memlet={ed.data.data}[{ed.data.subset}] "
                f"other_subset={ed.data.other_subset}. Direct AN -> AN copies must be converted to "
                f"``_out = _in`` tasklets at vectorization cleanup (before any widening); a "
                f"reappearance here means a downstream pass re-introduced one -- refusing.")


def assert_no_an_to_an_edges(sdfg: dace.SDFG) -> None:
    """Refuse vectorization if any direct ``AN -> AN`` edge remains in ``sdfg``.

    The cleanup phase converts every AN -> AN edge into the canonical
    ``AN -> [_out=_in] -> AN`` form. A direct AN -> AN edge that survives
    means a downstream pass re-introduced one; refuse loudly because
    codegen would emit a stride-blind ``CopyND``.

    :param sdfg: The SDFG (top-level or body NSDFG) whose edges to scan.
    :raises NotImplementedError: When at least one direct ``AN -> AN`` edge
        remains.
    """
    for istate in sdfg.states():
        for ed in istate.edges():
            if not (isinstance(ed.src, dace.nodes.AccessNode) and isinstance(ed.dst, dace.nodes.AccessNode)):
                continue
            raise NotImplementedError(
                f"vectorization: direct AN -> AN edge {ed.src.data!r} -> {ed.dst.data!r} survived "
                f"the descent in state {istate.label!r} (memlet={ed.data.data}[{ed.data.subset}]). "
                f"All AN -> AN edges must be converted to ``_out = _in`` tasklets at vectorization "
                f"cleanup (before any widening). A survivor here means a downstream pass "
                f"re-introduced one -- refusing.")


def assert_wcr_is_last_out_of_map(state: SDFGState, map_entry: nodes.MapEntry) -> None:
    """Refuse vectorization if the body of ``map_entry`` carries more than one
    WCR memlet, or the WCR memlet is not the last edge of its chain.

    :param state: The state holding ``map_entry``.
    :param map_entry: The map entry whose body to inspect.
    :raises NotImplementedError: When the body holds more than one
        WCR-tagged memlet, or the WCR memlet's destination is not the
        surrounding ``MapExit``.
    """
    map_exit = state.exit_node(map_entry)
    wcr_edges = [
        ed for ed in state.scope_subgraph(map_entry).edges() if ed.data is not None and ed.data.wcr is not None
    ]
    if len(wcr_edges) > 1:
        raise NotImplementedError(f"vectorization: body of map {map_entry.label!r} carries {len(wcr_edges)} WCR-tagged "
                                  f"memlets; at most one WCR chain per map is supported (multiple chains race on the "
                                  f"destination tile) -- refusing.")
    for ed in wcr_edges:
        if ed.dst is not map_exit:
            raise NotImplementedError(f"vectorization: WCR memlet {ed.data.data}[{ed.data.subset}] in map "
                                      f"{map_entry.label!r} is not the last edge of its chain (dst is "
                                      f"{type(ed.dst).__name__}({_node_handle(ed.dst)!r}), "
                                      f"expected MapExit). A mid-chain WCR breaks DaCe's accumulator atomicity "
                                      f"contract -- refusing.")


def assert_post_descent_invariants(state: SDFGState, map_entry: nodes.MapEntry,
                                   body_sdfgs: Iterable[dace.SDFG]) -> None:
    """Convenience: run both checks for a vectorized map and its body NSDFGs.

    :param state: State containing ``map_entry``.
    :param map_entry: The vectorized map entry.
    :param body_sdfgs: Body NSDFGs to scan for ``other_subset``. Typically
        ``[nsdfg_node.sdfg for nsdfg_node in body_nsdfgs]``; the empty
        iterable is valid when the body holds no NSDFG.
    """
    assert_wcr_is_last_out_of_map(state, map_entry)
    for inner in body_sdfgs:
        assert_no_residual_other_subsets(inner)
        assert_no_an_to_an_edges(inner)
