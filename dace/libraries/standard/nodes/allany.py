# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``AllNode`` / ``AnyNode`` -- Fortran ``ALL(mask [, dim])`` / ``ANY(mask [, dim])`` logical reductions.

``reduction`` (default) lowers ``dace.reduce`` to a ``Reduce`` node via a ``@dace.program``
(GPU CUB / OpenMP / plain loop).  ``sequential`` is a rank-1 whole-array short-circuit ``break``
loop.  Same result either way; the choice is a cost-model trade-off (early exit vs parallelism).
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
    """``(mask_desc, out_desc, mask_shape, axis)`` rebuilt from the wired edges, for the inner ``@dace.program``."""
    mask, mask_subset, out, out_subset = _validate_edges(node, parent_sdfg, parent_state)
    mask_shape = [(e + 1 - b) // s for (b, e, s) in mask_subset]
    out_shape = [(e + 1 - b) // s for (b, e, s) in out_subset] if out_subset.dims() else [1]
    axis = _fortran_dim_to_axis(node.dim, len(mask_shape))
    mask_desc = dace.data.Array(mask.dtype, mask_shape, storage=mask.storage)
    out_desc = dace.data.Array(dace.bool_, out_shape, storage=out.storage)
    return mask_desc, out_desc, mask_shape, axis


def _reduction_sdfg(node, parent_state, parent_sdfg, is_all):
    """Parallel-reduction expansion: lowers ``dace.reduce`` to a ``Reduce`` node (AND id=1 for ALL, OR id=0 for ANY)."""
    mask_desc, out_desc, mask_shape, axis = _descriptors(node, parent_sdfg, parent_state)
    ident = 1 if is_all else 0

    # Mask is normalised via ``_mask != 0`` going IN, not the result coming
    # out: wrapping the reduce result in ``!= 0`` makes an untyped
    # intermediate that the frontend defaults to ``double``, breaking
    # ``Logical_And``.
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

    dim = properties.Property(dtype=int, default=-1, desc="Fortran 1-based reduction axis (-1 = collapse to scalar).")

    def __init__(self, name, dim=-1, *args, **kwargs):
        super().__init__(name, *args, inputs={_INPUT_CONNECTOR_NAME}, outputs={_OUTPUT_CONNECTOR_NAME}, **kwargs)
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

    dim = properties.Property(dtype=int, default=-1, desc="Fortran 1-based reduction axis (-1 = collapse to scalar).")

    def __init__(self, name, dim=-1, *args, **kwargs):
        super().__init__(name, *args, inputs={_INPUT_CONNECTOR_NAME}, outputs={_OUTPUT_CONNECTOR_NAME}, **kwargs)
        self.dim = dim

    def validate(self, sdfg, state):
        return _validate_edges(self, sdfg, state)
