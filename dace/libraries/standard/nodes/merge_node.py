"""``MergeLibraryNode`` — Fortran ``MERGE(tsource, fsource, mask)`` intrinsic.

Mirrors the modularity pattern of ``CopyLibraryNode`` /
``FillLibraryNode``: the bridge / frontend can drop a ``MergeLibraryNode``
into the SDFG instead of inlining a per-element conditional tasklet, so
later passes (vectorisation, GPU offload, alternative backends) can pick
their own expansion without touching the surrounding graph.

Per-element semantics: ``_out[i] = _t[i] if _mask[i] else _f[i]`` -- the
same operation as a per-element if-then-else (ITE) and as NumPy's
``np.where(mask, t, f)``, only with the operands named after the Fortran
intrinsic.  A frontend that has one of those spellings wants this node.

Every operand -- ``_t``, ``_f``, ``_mask`` alike -- is broadcast against the result
by the NumPy rule: right-align the axes, and read an operand axis of extent 1 at
index ``0`` for every iteration of the result axis it lines up with.  That covers
Fortran's all-array and scalar-broadcast ``MERGE`` variants and NumPy's partial
broadcasts (a ``(N, 1)`` operand against an ``(N, M)`` result) in one rule.  Each
operand's shape comes from ITS OWN memlet subset, so the caller says which variant it
means by the memlet it wires, not by a flag.

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
# literals (mirrors ``copy`` / ``fill``).
_TRUE_CONNECTOR_NAME = "_mrg_t"
_FALSE_CONNECTOR_NAME = "_mrg_f"
_MASK_CONNECTOR_NAME = "_mrg_mask"
_OUTPUT_CONNECTOR_NAME = "_mrg_out"


@library.expansion
class ExpandPure(ExpandTransformation):
    """Pure SDFG expansion — one mapped tasklet doing the per-element select.

    Each input keeps its OWN shape on the inner SDFG and is indexed by NumPy
    broadcasting against the iteration space: axes right-align, an operand axis of
    extent 1 is read at index ``0`` for every iteration of the matching output axis,
    and any other axis is read at that axis' iterator.  A Fortran scalar dummy is the
    degenerate rank-1 case of the same rule, so ``MERGE`` broadcast operands and
    ``np.where`` partial broadcasts go through one code path.
    """
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

        # Per-input descriptor + access expression.  The connector mirrors the operand's
        # own subset shape and strides -- it is a view onto the caller's buffer, so
        # assuming the iteration shape or a packed C layout would read a Fortran-layout,
        # sliced or broadcast operand at the wrong addresses.  The index expression is
        # what applies the broadcast, one axis at a time.
        def add_input(conn: str, edge):
            arr = parent_sdfg.arrays[edge.data.data]
            shape = list(edge.data.subset.size())
            if len(shape) > len(iter_shape):
                raise ValueError(f"{node.label}: operand on {conn} has rank {len(shape)}, above the "
                                 f"rank-{len(iter_shape)} result; it cannot broadcast")
            sdfg.add_array(conn, shape, arr.dtype, arr.storage, strides=arr.strides)
            # Right-align the operand's axes against the result's, NumPy style.
            offset = len(iter_shape) - len(shape)
            idx = ["0" if extent == 1 else params[offset + k] for k, extent in enumerate(shape)]
            return ", ".join(idx)

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
            # Inherit the node's schedule instead of taking the Default. On a GPU graph the
            # operands are already GPU_Global by the time this expands, so a Default map inlines
            # as host code reading device memory and validation rejects the whole SDFG with
            # "stored as StorageType.GPU_Global but accessed on host".
            schedule=node.schedule,
        )
        return sdfg


@library.node
class MergeLibraryNode(nodes.LibraryNode):
    """Library node for the per-element select: Fortran
    ``MERGE(tsource, fsource, mask)``, NumPy ``where(mask, t, f)``, and the
    element-wise if-then-else generally.

    Inputs ``_t``, ``_f``, ``_mask``; output ``_out``.  Each input is broadcast
    against the result shape by the NumPy rule; see :class:`ExpandPure`.
    """

    implementations = {"pure": ExpandPure}
    default_implementation = "pure"

    # Connector names this libnode publishes. External consumers (tests,
    # the Fortran frontend's emitter) must reference these constants
    # instead of string literals so a future rename is a single-line
    # change (mirrors ``CopyLibraryNode`` / ``FillLibraryNode``).
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
