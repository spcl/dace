# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``AllNode`` / ``AnyNode`` library nodes -- Fortran ``ALL(mask [, dim])``
/ ``ANY(mask [, dim])`` logical reductions.

``ALL`` is the logical-AND reduction (true iff every element is true,
identity ``True``); ``ANY`` is the logical-OR reduction (true iff any
element is true, identity ``False``).  They are DISTINCT library nodes
(``AllNode`` and ``AnyNode``) sharing this file and their expansion
builders.

Both expansions are built with the DaCe **Python frontend**
(``@dace.program``) rather than hand-assembled state by state.  This is
the rule for these Fortran intrinsic nodes: a pure expansion either
LOWERS to another library node (as ``CountLibraryNode`` delegates to
``Reduce``) or is written as a ``@dace.program`` -- it must NOT
hand-build an SDFG with ``add_state`` / ``add_mapped_tasklet``.  The
frontend lowers ``dace.reduce`` to a ``Reduce`` node (so the CPU OpenMP
``&&``/``||`` and GPU CUB specialisations are inherited for free) and
natively supports ``break``, which is what makes the short-circuit
variant a four-line loop.

Expansions (shared by both nodes):

``reduction``   Default.  ``dace.reduce(and/or, _mask, identity=...)`` ->
                a ``Reduce`` node.  Scans every element with no early
                exit, but RETARGETS: GPU CUB tree-reduction / CPU OpenMP
                ``reduction(&&)`` / plain loop.  The parallel strategy.

``sequential``  Whole-array short-circuit: a ``for`` loop that ``break``s
                at the first ``.false.`` (ALL) / ``.true.`` (ANY) -- does
                <= N iterations, but the break is a loop-carried
                dependency so it is strictly serial (no GPU/OpenMP).
                The CPU early-exit strategy.  Only the rank-1 whole-array
                form is handled; anything else raises so the caller falls
                back to ``reduction``.

Same result either way; the choice is a cost-model trade-off (early exit
vs parallelisability).
"""
import dace
from dace import library, nodes, properties
from dace.transformation.transformation import ExpandTransformation

_INPUT_CONNECTOR_NAME = "_mask"
_OUTPUT_CONNECTOR_NAME = "_out"


def _validate_edges(node, sdfg, state):
    """``(mask_desc, mask_subset, out_desc, out_subset)`` from the wired
    edges -- one edge per connector, dtype taken at the boundary."""
    in_edges = [ie for ie in state.in_edges(node) if ie.dst_conn == _INPUT_CONNECTOR_NAME]
    if len(in_edges) != 1:
        raise ValueError(f"{type(node).__name__} expects exactly one "
                         f"``{_INPUT_CONNECTOR_NAME}`` input edge.")
    out_edges = [oe for oe in state.out_edges(node) if oe.src_conn == _OUTPUT_CONNECTOR_NAME]
    if len(out_edges) != 1:
        raise ValueError(f"{type(node).__name__} expects exactly one "
                         f"``{_OUTPUT_CONNECTOR_NAME}`` output edge.")
    ie, oe = in_edges[0], out_edges[0]
    return sdfg.arrays[ie.data.data], ie.data.subset, sdfg.arrays[oe.data.data], oe.data.subset


def _fortran_dim_to_axis(dim, mask_rank):
    """Fortran 1-based ``dim`` -> numpy 0-based axis; ``None`` for a
    whole-array (``dim == -1``) collapse."""
    if dim is None or dim == -1:
        return None
    if not (1 <= dim <= mask_rank):
        raise ValueError(f"All/Any `dim={dim}` is out of range for a rank-{mask_rank} mask.")
    return dim - 1


def _descriptors(node, parent_sdfg, parent_state):
    """``(mask_desc, out_desc, mask_shape, axis)`` rebuilt from the wired
    edges -- the inner ``@dace.program`` is specialised against these so
    it matches the parent connectors' shapes / dtypes / storage exactly
    (sections, ``LOGICAL(1)`` masks, GPU storage all flow through).

    The OUTPUT is always a BOOLEAN: ``ALL`` / ``ANY`` return a Fortran
    LOGICAL, which the HLFIR bridge maps to ``dace.bool_``.  The mask
    keeps its own dtype (any logical kind / int); the reduction lowering
    casts the accumulator to bool (``!= 0``).  The caller (bridge /
    test) must therefore wire ``_out`` to a ``bool`` array."""
    mask, mask_subset, out, out_subset = _validate_edges(node, parent_sdfg, parent_state)
    mask_shape = [(e + 1 - b) // s for (b, e, s) in mask_subset]
    out_shape = [(e + 1 - b) // s for (b, e, s) in out_subset] if out_subset.dims() else [1]
    axis = _fortran_dim_to_axis(node.dim, len(mask_shape))
    mask_desc = dace.data.Array(mask.dtype, mask_shape, storage=mask.storage)
    out_desc = dace.data.Array(dace.bool_, out_shape, storage=out.storage)
    return mask_desc, out_desc, mask_shape, axis


def _reduction_sdfg(node, parent_state, parent_sdfg, is_all):
    """Parallel-reduction expansion via the Python frontend.

    ``dace.reduce`` lowers to a ``Reduce`` node, so the per-target
    schedule (pure map / OpenMP ``&&``/``||`` / CUB) is inherited.  The
    logical identity MUST be explicit -- ``np.all`` / ``np.any`` lower
    with identity 0, which is wrong for ALL (``0 and x ... == 0`` even
    when every element is true).  ALL = AND-reduce with identity 1,
    ANY = OR-reduce with identity 0.

    The reduction accumulates in the MASK's element type (``a and b``
    keeps an integer logical), so the result is cast to a normalised
    ``0``/``1`` via ``!= 0`` before it lands in ``_out`` -- the output is
    a boolean (``ALL`` / ``ANY`` return a LOGICAL scalar, and the
    condition path wires a ``bool`` ``_out``).
    """
    mask_desc, out_desc, mask_shape, axis = _descriptors(node, parent_sdfg, parent_state)
    ident = 1 if is_all else 0

    # The reduce accumulates in the MASK's element type; assigning the
    # result into the boolean ``_out`` casts it.  Do NOT wrap in an
    # explicit ``!= 0`` -- that introduces an untyped intermediate the
    # frontend defaults to ``double``, which then trips the integer/bool
    # ``Logical_And`` (``operator&`` is undefined on ``double``).  The
    # mask is normalised to a clean ``0``/``1`` first so the accumulator
    # type is the mask's int/bool (never a float logical kind).
    if axis is None:
        if is_all:
            @dace.program
            def kernel(_mask, _out):
                _out[0] = dace.reduce(lambda a, b: a and b, _mask != 0, identity=ident)
        else:
            @dace.program
            def kernel(_mask, _out):
                _out[0] = dace.reduce(lambda a, b: a or b, _mask != 0, identity=ident)
    else:
        if is_all:
            @dace.program
            def kernel(_mask, _out):
                _out[:] = dace.reduce(lambda a, b: a and b, _mask != 0, axis=axis, identity=ident)
        else:
            @dace.program
            def kernel(_mask, _out):
                _out[:] = dace.reduce(lambda a, b: a or b, _mask != 0, axis=axis, identity=ident)

    return kernel.to_sdfg(_mask=mask_desc, _out=out_desc, simplify=True)


def _sequential_sdfg(node, parent_state, parent_sdfg, is_all):
    """Whole-array short-circuit via the Python frontend's ``break``."""
    mask_desc, out_desc, mask_shape, axis = _descriptors(node, parent_sdfg, parent_state)
    if axis is not None or len(mask_shape) != 1:
        raise NotImplementedError(f"{type(node).__name__} 'sequential' (short-circuit) handles "
                                  "only a rank-1 whole-array reduce; use 'reduction' for "
                                  "dim-wise / multi-rank.")
    n = mask_shape[0]
    if is_all:
        @dace.program
        def kernel(_mask, _out):
            _out[0] = 1
            for i in range(n):
                if _mask[i] == 0:
                    _out[0] = 0
                    break
    else:
        @dace.program
        def kernel(_mask, _out):
            _out[0] = 0
            for i in range(n):
                if _mask[i] != 0:
                    _out[0] = 1
                    break

    return kernel.to_sdfg(_mask=mask_desc, _out=out_desc, simplify=True)


@library.expansion
class ExpandAllReduction(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _reduction_sdfg(node, parent_state, parent_sdfg, is_all=True)


@library.expansion
class ExpandAllSequential(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _sequential_sdfg(node, parent_state, parent_sdfg, is_all=True)


@library.expansion
class ExpandAnyReduction(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _reduction_sdfg(node, parent_state, parent_sdfg, is_all=False)


@library.expansion
class ExpandAnySequential(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _sequential_sdfg(node, parent_state, parent_sdfg, is_all=False)


@library.node
class AllNode(nodes.LibraryNode):
    """Fortran ``ALL(mask [, dim])`` -- logical-AND reduction (identity
    true).  ``_mask`` logical array in, ``_out`` logical scalar (whole
    array) or rank ``mask_rank-1`` array (``dim``-wise)."""

    implementations = {"reduction": ExpandAllReduction, "sequential": ExpandAllSequential}
    default_implementation = "reduction"

    INPUT_CONNECTOR_NAME = _INPUT_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = _OUTPUT_CONNECTOR_NAME

    dim = properties.Property(dtype=int, default=-1,
                              desc="Fortran 1-based reduction axis (-1 = collapse to scalar).")

    def __init__(self, name, dim=-1, *args, **kwargs):
        super().__init__(name, *args, inputs={_INPUT_CONNECTOR_NAME},
                         outputs={_OUTPUT_CONNECTOR_NAME}, **kwargs)
        self.dim = dim

    def validate(self, sdfg, state):
        return _validate_edges(self, sdfg, state)


@library.node
class AnyNode(nodes.LibraryNode):
    """Fortran ``ANY(mask [, dim])`` -- logical-OR reduction (identity
    false).  ``_mask`` logical array in, ``_out`` logical scalar (whole
    array) or rank ``mask_rank-1`` array (``dim``-wise)."""

    implementations = {"reduction": ExpandAnyReduction, "sequential": ExpandAnySequential}
    default_implementation = "reduction"

    INPUT_CONNECTOR_NAME = _INPUT_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = _OUTPUT_CONNECTOR_NAME

    dim = properties.Property(dtype=int, default=-1,
                              desc="Fortran 1-based reduction axis (-1 = collapse to scalar).")

    def __init__(self, name, dim=-1, *args, **kwargs):
        super().__init__(name, *args, inputs={_INPUT_CONNECTOR_NAME},
                         outputs={_OUTPUT_CONNECTOR_NAME}, **kwargs)
        self.dim = dim

    def validate(self, sdfg, state):
        return _validate_edges(self, sdfg, state)
