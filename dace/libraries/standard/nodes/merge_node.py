"""``MergeLibraryNode`` — Fortran ``MERGE(tsource, fsource, mask)`` intrinsic.

Per-element semantics: ``_out[i] = _t[i] if _mask[i] else _f[i]``. Each of
``_t``/``_f``/``_mask`` may be a full-shape array or a scalar broadcast;
picked per input from its memlet subset volume (see ``_subset_volume``).
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
            vol *= int(dace.symbolic.int_floor(e + 1 - b, s))
        except (TypeError, ValueError):
            return None  # symbolic — assume non-scalar
    return vol


@library.expansion
class ExpandPure(ExpandTransformation):
    """Pure SDFG expansion: one mapped tasklet performing the per-element select."""
    environments = []

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        t_oe, f_oe, m_oe, out_oe = node.validate(parent_sdfg, parent_state)
        # Iteration shape = the output's subset (the result shape).
        out_subset = out_oe.data.subset
        out_arr = parent_sdfg.arrays[out_oe.data.data]
        iter_shape = [dace.symbolic.int_floor(e + 1 - b, s) for (b, e, s) in out_subset]
        params = [f"__i{i}" for i in range(len(iter_shape))]
        rng = {p: f"0:{s}" for p, s in zip(params, iter_shape)}
        full_idx = ", ".join(params)

        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.schedule = dace.dtypes.ScheduleType.Sequential

        def add_input(conn: str, edge):
            # Array input keeps the operand's own strides, exactly as the
            # output below does: the connector is a view onto the caller's
            # buffer, so assuming a packed C layout silently reads a
            # Fortran-layout or otherwise non-packed operand at the wrong
            # addresses.
            arr = parent_sdfg.arrays[edge.data.data]
            vol = _subset_volume(edge.data.subset)
            if vol == 1:
                sdfg.add_array(conn, [1], arr.dtype, arr.storage)
                return "0"
            sdfg.add_array(conn, iter_shape, arr.dtype, arr.storage, strides=arr.strides)
            return full_idx

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
    intrinsic. Inputs ``_t``, ``_f``, ``_mask``; output ``_out``."""

    implementations = {"pure": ExpandPure}
    default_implementation = "pure"

    TRUE_CONNECTOR_NAME = _TRUE_CONNECTOR_NAME
    FALSE_CONNECTOR_NAME = _FALSE_CONNECTOR_NAME
    MASK_CONNECTOR_NAME = _MASK_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = _OUTPUT_CONNECTOR_NAME

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={
                             _TRUE_CONNECTOR_NAME: None,
                             _FALSE_CONNECTOR_NAME: None,
                             _MASK_CONNECTOR_NAME: None
                         },
                         outputs={_OUTPUT_CONNECTOR_NAME: None},
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
