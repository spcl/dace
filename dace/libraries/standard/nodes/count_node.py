# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``CountLibraryNode`` -- Fortran ``COUNT(mask [, dim])`` intrinsic as a
single library node.

The node owns the Fortran-COUNT semantics (input is a logical / integer
mask, output is always integer regardless of mask kind, ``dim`` is in
Fortran 1-based numbering) and delegates the actual reduction to the
generic ``Reduce`` library node so all of Reduce's existing
specialisations (CPU sequential, OpenMP, CUDA device / block, GPUAuto)
are inherited for free.

Expansions:

``pure``      Default expansion -- builds a small inner SDFG containing
              a single ``Reduce`` node with WCR ``lambda a,b: a + b``
              and identity ``0``.  Avoids any WCR-on-tasklet pattern
              (no per-element atomic add); the inner Reduce picks its
              own schedule based on its own implementation.
"""

import dace
from dace import library, nodes, properties
from dace.transformation.transformation import ExpandTransformation
from .reduce import Reduce

# Outer connector names this libnode publishes. Republished as
# ``CountLibraryNode.INPUT_CONNECTOR_NAME`` / ``.OUTPUT_CONNECTOR_NAME``
# so external consumers reference a class constant instead of a string
# literal (mirrors ``copy_node`` / ``memset_node``).
_INPUT_CONNECTOR_NAME = "_cnt_in"
_OUTPUT_CONNECTOR_NAME = "_cnt_out"


def _validate_count_edges(node, sdfg, state):
    """Validate the COUNT node's edges -> ``(mask_name, mask_desc, mask_subset, out_name,
    out_desc, out_subset)``. Mirrors the ``_validate_*_edges`` pattern from ``copy_node`` /
    ``memset_node`` -- one edge per connector, dtype checked at the boundary."""
    in_edges = [ie for ie in state.in_edges(node) if ie.dst_conn == _INPUT_CONNECTOR_NAME]
    if len(in_edges) != 1:
        raise ValueError(f"{type(node).__name__} expects exactly one "
                         f"``{_INPUT_CONNECTOR_NAME}`` input edge.")
    ie = in_edges[0]
    mask = sdfg.arrays[ie.data.data]
    mask_subset = ie.data.subset
    mask_name = ie.dst_conn

    out_edges = [oe for oe in state.out_edges(node) if oe.src_conn == _OUTPUT_CONNECTOR_NAME]
    if len(out_edges) != 1:
        raise ValueError(f"{type(node).__name__} expects exactly one "
                         f"``{_OUTPUT_CONNECTOR_NAME}`` output edge.")
    oe = out_edges[0]
    out = sdfg.arrays[oe.data.data]
    out_subset = oe.data.subset
    out_name = oe.src_conn

    return mask_name, mask, mask_subset, out_name, out, out_subset


def _fortran_dim_to_axes(dim, mask_rank):
    """Translate Fortran 1-based ``dim`` to DaCe ``axes`` for ``Reduce``.

    ``dim == -1`` (the default) means whole-array collapse → ``axes=None``
    (Reduce reduces over every axis).  Any other value is the 1-based
    axis index Fortran's ``COUNT(mask, dim=k)`` selects; convert to
    0-based and validate.
    """
    if dim is None or dim == -1:
        return None
    if not (1 <= dim <= mask_rank):
        raise ValueError(f"CountLibraryNode `dim={dim}` is out of range "
                         f"for a rank-{mask_rank} mask.")
    return [dim - 1]


@library.expansion
class ExpandPure(ExpandTransformation):
    """
    Default expansion: build a one-state SDFG containing a single
    ``Reduce`` library node with WCR ``add`` and identity ``0``.  The
    Reduce node is responsible for its own implementation choice
    (sequential / OpenMP / CUDA …) -- this expansion stays target-
    neutral.

    No WCR edges land in the parent graph: the per-element accumulation
    happens entirely inside the Reduce node's own expansion.  An
    explicit cast tasklet narrows non-integer mask dtypes to ``int32``;
    DaCe's simplification folds it for already-integer masks.
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        mask_name, mask, mask_subset, out_name, out, out_subset = node.validate(parent_sdfg, parent_state)

        mask_shape = [(e + 1 - b) // s for (b, e, s) in mask_subset]
        out_shape = [(e + 1 - b) // s for (b, e, s) in out_subset] if out_subset.dims() else []
        axes = _fortran_dim_to_axes(node.dim, len(mask_shape))

        # Inner SDFG: cast → reduce.  Cast turns a non-integer mask
        # (LOGICAL(1) i1 / LOGICAL(8) i64) into the int kind the
        # Fortran COUNT spec returns.  For already-int masks the cast
        # is a copy; simplification will fuse it with the reduction
        # source when no cross-storage boundary intervenes.
        sdfg = dace.SDFG(f"{node.label}_sdfg")
        sdfg.schedule = dace.dtypes.ScheduleType.Sequential

        sdfg.add_array(mask_name, mask_shape, mask.dtype, mask.storage, strides=mask.strides)
        sdfg.add_transient("_mask_int", mask_shape, dace.int32, storage=mask.storage)
        # ``out`` is a scalar when ``axes is None``; otherwise an array
        # of the rank-reduced shape.  The parent-side memlet already
        # carries the right shape, so we mirror it.
        if out_shape:
            sdfg.add_array(out_name, out_shape, out.dtype, out.storage, strides=out.strides)
        else:
            sdfg.add_scalar(out_name, out.dtype, storage=out.storage)

        # State 1: int cast.  Mapped tasklet writing the int32 mask.
        cast_state = sdfg.add_state(f"{node.label}_cast", is_start_block=True)
        params = [f"__i{i}" for i in range(len(mask_shape))]
        rng = {p: f"0:{s}" for p, s in zip(params, mask_shape)}
        idx = ", ".join(params)
        cast_state.add_mapped_tasklet(
            f"{node.label}_cast_tasklet",
            rng,
            inputs={"_in_v": dace.memlet.Memlet(f"{mask_name}[{idx}]")},
            code="_out_v = _in_v",
            outputs={"_out_v": dace.memlet.Memlet(f"_mask_int[{idx}]")},
            external_edges=True,
        )

        # State 2: Reduce node with sum / identity 0.  Reduce's own
        # implementation choice (which itself can be ``pure``,
        # ``OpenMP``, ``CUDA (device)``, …) decides whether this becomes
        # a mapped sum, a parallel reduction, or a CUB call.
        reduce_state = sdfg.add_state_after(cast_state, f"{node.label}_reduce")
        red = Reduce(name=f"{node.label}_reduce_node", wcr="lambda a, b: a + b", axes=axes, identity=0)
        reduce_state.add_node(red)
        mask_in = reduce_state.add_access("_mask_int")
        out_w = reduce_state.add_access(out_name)
        reduce_state.add_edge(mask_in, None, red, "_in_data",
                              dace.memlet.Memlet.from_array("_mask_int", sdfg.arrays["_mask_int"]))
        reduce_state.add_edge(red, "_out", out_w, None, dace.memlet.Memlet.from_array(out_name, sdfg.arrays[out_name]))

        return sdfg


@library.node
class CountLibraryNode(nodes.LibraryNode):
    """Library node for the Fortran ``COUNT(mask [, dim])`` intrinsic.

    Inputs / output:

    ``_mask``  Logical / integer mask array.  Any dtype is accepted; the
               expansion casts to ``int32`` before reducing.
    ``_out``   Integer scalar (whole-array reduce) or array of rank
               ``mask_rank - 1`` (per-dim reduce).

    Properties:

    ``dim``    Fortran 1-based reduction axis.  Default ``-1`` collapses
               every axis to a scalar (the ``COUNT(mask)`` form); any
               value in ``1..mask_rank`` selects ``COUNT(mask, dim=k)``.

    Implementations:

    ``pure``   Inner ``cast → Reduce`` SDFG.  Default and only entry;
               Reduce's own per-target expansions inherit transparently.
    """

    implementations = {"pure": ExpandPure}
    default_implementation = "pure"

    # Connector names this libnode publishes. External consumers (tests,
    # the Fortran frontend's emitter) must reference these constants
    # instead of string literals so a future rename is a single-line
    # change (mirrors ``CopyLibraryNode`` / ``MemsetLibraryNode``).
    INPUT_CONNECTOR_NAME = _INPUT_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = _OUTPUT_CONNECTOR_NAME

    dim = properties.Property(
        dtype=int,
        default=-1,
        desc="Fortran 1-based reduction axis (``-1`` collapses every axis to a scalar).",
    )

    def __init__(self, name, dim=-1, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={CountLibraryNode.INPUT_CONNECTOR_NAME},
                         outputs={CountLibraryNode.OUTPUT_CONNECTOR_NAME},
                         **kwargs)
        self.dim = dim

    def validate(self, sdfg, state):
        return _validate_count_edges(self, sdfg, state)
