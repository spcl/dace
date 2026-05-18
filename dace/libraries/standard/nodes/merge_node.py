"""``MergeLibraryNode`` — Fortran ``MERGE(tsource, fsource, mask)`` intrinsic.

Mirrors the modularity pattern of ``CopyLibraryNode`` /
``MemsetLibraryNode``: the bridge / frontend can drop a ``MergeLibraryNode``
into the SDFG instead of inlining a per-element conditional tasklet, so
later passes (vectorisation, GPU offload, alternative backends) can pick
their own expansion without touching the surrounding graph.

Per-element semantics: ``_out[i] = _t[i] if _mask[i] else _f[i]``.

Supported variants (Fortran standard):

- **All-array** — ``_t``, ``_f``, ``_mask`` all have the result's shape.
- **Scalar broadcast** — any of ``_t`` / ``_f`` (or both) is a single-
  element input; the expansion broadcasts that value across every
  iteration.  Mask is typically array-shaped (the broadcast variants
  in Fortran).  Detected by inspecting each input edge's memlet
  subset volume; the bridge wires the input memlet appropriately
  (full-shape for arrays, single-element for scalars).
- ``_mask`` itself can also be scalar (degenerate; Flang usually folds).

Today only the ``pure`` expansion is provided (a mapped tasklet); GPU /
CPU-vectorised expansions slot in the same way ``CopyLibraryNode``'s
storage-aware variants do.
"""
import dace
from dace import library, nodes
from dace.transformation.transformation import ExpandTransformation

# Outer connector names this libnode publishes. Republished as
# ``MergeLibraryNode.{TRUE,FALSE,MASK,OUTPUT}_CONNECTOR_NAME`` so
# external consumers reference class constants instead of string
# literals (mirrors ``copy_node`` / ``memset_node``).
_TRUE_CONNECTOR_NAME = "_mrg_t"
_FALSE_CONNECTOR_NAME = "_mrg_f"
_MASK_CONNECTOR_NAME = "_mrg_mask"
_OUTPUT_CONNECTOR_NAME = "_mrg_out"


def _subset_volume(subset):
    """Number of elements covered by ``subset``.  ``1`` means the input
    is a single value (the broadcast case); anything else is the full
    iteration shape."""
    vol = 1
    for (b, e, s) in subset:
        try:
            vol *= int((e + 1 - b) // s)
        except (TypeError, ValueError):
            return None  # symbolic — assume non-scalar
    return vol


@library.expansion
class ExpandPure(ExpandTransformation):
    """Pure SDFG expansion — one mapped tasklet doing the per-element
    select.  Each input is independently broadcast or per-iteration:

    - **Array input** (subset volume > 1 or symbolic): the inner SDFG
      mirrors the full shape; the tasklet reads ``_t[i, j, …]`` etc.
    - **Scalar input** (subset volume == 1): the inner SDFG declares a
      length-1 array; the tasklet reads element ``0`` uniformly across
      iterations.  Same shape for the operand whether it came in as a
      Fortran scalar dummy or a sliced single element.
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        t_oe, f_oe, m_oe, out_oe = node.validate(parent_sdfg, parent_state)
        # Iteration shape = the output's subset (the result shape).
        out_subset = out_oe.data.subset
        out_arr = parent_sdfg.arrays[out_oe.data.data]
        iter_shape = [(e + 1 - b) // s for (b, e, s) in out_subset]
        params = [f"__i{i}" for i in range(len(iter_shape))]
        rng = {p: f"0:{s}" for p, s in zip(params, iter_shape)}
        full_idx = ", ".join(params)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.schedule = dace.dtypes.ScheduleType.Sequential

        # Per-input descriptor + access expression.  Single-element
        # inputs become a length-1 array on the inner SDFG and are
        # indexed by ``0``; multi-element inputs match the iteration
        # shape and are indexed per-iteration.
        def add_input(conn: str, edge):
            arr = parent_sdfg.arrays[edge.data.data]
            vol = _subset_volume(edge.data.subset)
            if vol == 1:
                inner_shape = [1]
                idx_expr = "0"
            else:
                inner_shape = iter_shape
                idx_expr = full_idx
            sdfg.add_array(conn, inner_shape, arr.dtype, arr.storage)
            return idx_expr

        t_idx = add_input(_TRUE_CONNECTOR_NAME, t_oe)
        f_idx = add_input(_FALSE_CONNECTOR_NAME, f_oe)
        m_idx = add_input(_MASK_CONNECTOR_NAME, m_oe)
        sdfg.add_array(_OUTPUT_CONNECTOR_NAME, iter_shape, out_arr.dtype, out_arr.storage, strides=out_arr.strides)

        state = sdfg.add_state(f"{node.label}_state")
        state.add_mapped_tasklet(
            f"{node.label}_tasklet",
            rng,
            inputs={
                "_in_t": dace.memlet.Memlet(f"{_TRUE_CONNECTOR_NAME}[{t_idx}]"),
                "_in_f": dace.memlet.Memlet(f"{_FALSE_CONNECTOR_NAME}[{f_idx}]"),
                "_in_mask": dace.memlet.Memlet(f"{_MASK_CONNECTOR_NAME}[{m_idx}]"),
            },
            code="_out_v = (_in_t if _in_mask else _in_f)",
            outputs={"_out_v": dace.memlet.Memlet(f"{_OUTPUT_CONNECTOR_NAME}[{full_idx}]")},
            external_edges=True,
        )
        return sdfg


@library.node
class MergeLibraryNode(nodes.LibraryNode):
    """Library node for the Fortran ``MERGE(tsource, fsource, mask)``
    intrinsic.

    Inputs ``_t``, ``_f``, ``_mask``; output ``_out``.  Each input may
    cover the result shape (per-element) or a single element (broadcast
    scalar).  The expansion reads the per-input subset volume to pick
    its broadcast strategy.
    """

    implementations = {"pure": ExpandPure}
    default_implementation = "pure"

    # Connector names this libnode publishes. External consumers (tests,
    # the Fortran frontend's emitter) must reference these constants
    # instead of string literals so a future rename is a single-line
    # change (mirrors ``CopyLibraryNode`` / ``MemsetLibraryNode``).
    TRUE_CONNECTOR_NAME = _TRUE_CONNECTOR_NAME
    FALSE_CONNECTOR_NAME = _FALSE_CONNECTOR_NAME
    MASK_CONNECTOR_NAME = _MASK_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = _OUTPUT_CONNECTOR_NAME

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={_TRUE_CONNECTOR_NAME, _FALSE_CONNECTOR_NAME, _MASK_CONNECTOR_NAME},
                         outputs={_OUTPUT_CONNECTOR_NAME},
                         **kwargs)

    def validate(self, sdfg, state):
        """Return the four edge descriptors after asserting the expected
        connector layout.  Each connector takes exactly one edge."""
        t_es = [ie for ie in state.in_edges(self) if ie.dst_conn == _TRUE_CONNECTOR_NAME]
        f_es = [ie for ie in state.in_edges(self) if ie.dst_conn == _FALSE_CONNECTOR_NAME]
        m_es = [ie for ie in state.in_edges(self) if ie.dst_conn == _MASK_CONNECTOR_NAME]
        o_es = [oe for oe in state.out_edges(self) if oe.src_conn == _OUTPUT_CONNECTOR_NAME]
        if len(t_es) != 1 or len(f_es) != 1 or len(m_es) != 1 or len(o_es) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one edge per connector "
                             f"({_TRUE_CONNECTOR_NAME}, {_FALSE_CONNECTOR_NAME}, "
                             f"{_MASK_CONNECTOR_NAME}, {_OUTPUT_CONNECTOR_NAME})")
        return t_es[0], f_es[0], m_es[0], o_es[0]
