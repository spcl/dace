# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ArgMin`` / ``ArgMax`` library nodes -- Fortran ``MINLOC`` / ``MAXLOC``.

Architecture: a multi-state pipeline that keeps every WCR a pure
``min`` / ``max`` over a scalar.  No ``(val, idx)`` struct WCR with
nested if/else for index disambiguation -- the index selection is a
separate filtered-min/-max reduction.

Pipeline (whole-array ``dim is None``):

1. ``init_best_val``: write the identity (``+inf`` / ``-inf``) into a
   scalar transient ``__best_val``.
2. ``reduce_best_val``: ``Map`` over the input with ``__out = __in``
   feeding a ``WCR=min`` / ``WCR=max`` into ``__best_val[0]``.
3. ``init_best_idx``: write the index-identity sentinel into
   ``__best_idx`` (``num_elements`` for ``back=False``, ``-1`` for
   ``back=True``).
4. ``reduce_best_idx``: ``Map`` over the input emitting the matching
   flat index when the element equals ``__best_val``, else the
   sentinel; the WCR ``min`` (``back=False``) / ``max`` (``back=True``)
   picks the first / last occurrence.
5. ``extract``: decode the flat index to per-dim subscripts and apply
   the Fortran 1-based offset.

For ``dim is not None`` the same five states run with a Map whose
outer iterators are the dim-reduced output coordinates.

Future backends (not yet implemented; ``pure`` is the only registered
implementation):

* ``CUB`` one-pass: ``gpucub::DeviceReduce::ArgMax`` / ``ArgMin`` return a
  ``gpucub::KeyValuePair<int, T>``, which would collapse the two reduction
  passes below into one.  Both passes already go through ``Reduce``
  nodes, so each of them reaches CUB's ``DeviceReduce`` on its own; a
  dedicated expansion would only save the second pass.
* ``OpenMP``: a user-defined reduction
  (``#pragma omp declare reduction``) over a ``pair<val, idx>`` struct
  with a custom combiner.  Lets the OpenMP 5.0 runtime own the parallel
  scan.
* ``stdpar``: ``std::min_element`` / ``std::max_element`` with
  ``std::execution::par`` (or sequential, when the SDFG schedule is
  sequential).  Index = iterator distance.
"""
import dace
import dace.dtypes as dtypes
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, SDFGState, memlet as mm, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.libraries.standard.nodes.reduce import Reduce
from dace.transformation.transformation import ExpandTransformation


def _flat_size(shape) -> int:
    """Return the product of the static-int extents in ``shape``.

    :raises ValueError: if any extent isn't a literal int (the flat-decode
        path needs a static total size).
    """
    total = 1
    for s in shape:
        total *= int(s)
    return total


def _flat_index_expr(rank: int, shape) -> str:
    """Build the row-major flat-index expression for ``__i0, __i1, ...``.

    ``flat = i0 * (s1 * s2 * .. * s_{R-1}) + i1 * (s2 * .. * s_{R-1}) + .. + i_{R-1}``.
    """
    if rank == 1:
        return "__i0"
    terms = []
    for d in range(rank):
        if d == rank - 1:
            terms.append(f"__i{d}")
        else:
            mult = 1
            for k in range(d + 1, rank):
                mult *= int(shape[k])
            terms.append(f"(__i{d} * {mult})")
    return " + ".join(terms)


def _emit_pure(node, parent_state: SDFGState, parent_sdfg: SDFG, func: str):
    """Shared pure expansion for :class:`ArgMin` / :class:`ArgMax`.

    Multi-state pipeline (see module docstring); every WCR is a pure
    ``min`` / ``max`` over a scalar.  Tasklet bodies are
    single-statement: state init = ``__out = <literal>``, value reduce
    = ``__out = __in``, idx reduce = ``__out = flat if (__in == bv) else sentinel``,
    extract = one ``__o<d> = (__flat // stride) % extent + offset``
    per dim.

    :param node: ``ArgMin`` or ``ArgMax`` instance being expanded.
    :param parent_state: enclosing state.
    :param parent_sdfg: enclosing SDFG.
    :param func: ``"min"`` or ``"max"`` selecting the comparison direction.
    :returns: a self-contained nested SDFG implementing the scan.
    :raises NotImplementedError: when a ``_mask`` connector is wired
        (mask handling is not yet implemented).
    """
    desc_x, desc_idx, mask_desc, dim_zero = node.validate(parent_sdfg, parent_state)
    if mask_desc is not None:
        raise NotImplementedError(f"{type(node).__name__}: mask= argument is not yet supported in the pure expansion. "
                                  "Materialise the masked input first (e.g. set masked-out entries to +inf / -inf).")

    dtype = desc_x.dtype.base_type
    idx_dtype = desc_idx.dtype.base_type
    shape = list(desc_x.shape)
    rank = len(shape)
    flat = dim_zero is None

    sdfg = dace.SDFG(node.label + "_sdfg")
    # Carry the operand strides, not just the shape: a compact declaration reads a strided view (``a[:, 0:2*h:2]``) at the wrong offsets and returns a plausible array of the right shape and dtype.
    sdfg.add_array("_x", shape, dtype, strides=desc_x.strides, storage=desc_x.storage)
    if flat:
        idx_shape, per_slice_shape = [rank], [1]
    else:
        per_slice_shape = [s for d, s in enumerate(shape) if d != dim_zero]
        if not per_slice_shape:
            per_slice_shape = [1]
        idx_shape = per_slice_shape
    # The result connector takes the writer's strides only when the expansion's own result shape
    # IS the outer one; the reduced/flat spellings below build a shape of their own, and a
    # stride list from a differently-ranked descriptor would be nonsense.
    idx_matches = symbolic.shapes_equal(idx_shape, desc_idx.shape)
    sdfg.add_array("_idx",
                   idx_shape,
                   idx_dtype,
                   strides=desc_idx.strides if idx_matches else None,
                   storage=desc_idx.storage)
    # Every intermediate takes the INPUT's storage rather than the default. A transient left at
    # ``Default`` under a GPU-scheduled expansion resolves to ``GPU_Shared``, and
    # ``ExpandReduceGPUAuto`` refuses anything but ``GPU_Global``: it falls back to the pure
    # expansion, which is a WCR map -- the per-element atomic the Reduce nodes exist to avoid.
    sdfg.add_transient("__best_val", per_slice_shape, dtype, storage=desc_x.storage)
    sdfg.add_transient("__best_idx", per_slice_shape, idx_dtype, storage=desc_x.storage)

    val_identity_value = dtypes.max_value(dtype) if func == "min" else dtypes.min_value(dtype)
    # Sentinel for the idx reduction: ``num_elements`` is larger than any
    # valid flat idx, so a final WCR=min gives "no match" -> num_elements
    # which the extract step ignores.  For ``back=True`` we want the LAST
    # occurrence so the sentinel is ``-1`` and the WCR is ``max``.
    if flat:
        total = _flat_size(shape)
    else:
        total = int(shape[dim_zero])
    idx_sentinel = -1 if node.back else total
    idx_wcr_op = "max" if node.back else "min"

    one_offset = 1 if node.one_based else 0

    # ---- state 1: reduce __best_val ----
    #
    # A ``Reduce`` node, not a WCR on a map over ``_x``. A WCR into one accumulator emits a global
    # atomic per element on a GPU, every one of them contending for the same address; the node owns
    # its per-target lowering and reaches CUB's ``DeviceReduce`` on CUDA. It also carries the
    # identity, so the separate seeding state this used to need is gone.
    axes = None if flat else [dim_zero]
    map_rng = {f"__i{d}": f"0:{shape[d]}" for d in range(rank)}
    x_subs = ", ".join([f"__i{d}" for d in range(rank)])
    bv_subs = "0" if flat else ", ".join([f"__i{d}" for d in range(rank) if d != dim_zero])
    reduce_val = sdfg.add_state(node.label + "_reduce_val", is_start_block=True)
    val_red = Reduce(name=node.label + "_val_reduce",
                     wcr=f"lambda a, b: {func}(a, b)",
                     axes=axes,
                     identity=val_identity_value)
    reduce_val.add_node(val_red)
    reduce_val.add_edge(reduce_val.add_read("_x"), None, val_red, "_in",
                        dace.Memlet.from_array("_x", sdfg.arrays["_x"]))
    reduce_val.add_edge(val_red, "_out", reduce_val.add_access("__best_val"), None,
                        dace.Memlet.from_array("__best_val", sdfg.arrays["__best_val"]))

    # ---- state 2: candidate index per element ----
    #
    # Elementwise, so it parallelizes anywhere: each element emits its own flat index if it matches
    # the winning value and the sentinel otherwise. Materializing the candidates is what lets the
    # pick below be a ``Reduce`` node rather than a second WCR accumulator.
    candidates = sdfg.add_state_after(reduce_val, node.label + "_candidates")
    sdfg.add_transient("__idx_candidates", shape, idx_dtype, storage=desc_x.storage)
    flat_idx = _flat_index_expr(rank, shape) if flat else f"__i{dim_zero}"
    # Single-statement Python tasklet: emit the flat index for matches,
    # the sentinel otherwise.  Conditional expression in a tasklet body
    # transpiles cleanly via astutils.
    code = f"__out = ({flat_idx}) if (__in == __bv) else {idx_sentinel}"
    candidates.add_mapped_tasklet(
        name="candidate_idx",
        map_ranges=map_rng,
        inputs={
            "__in": dace.Memlet(f"_x[{x_subs}]"),
            "__bv": dace.Memlet(f"__best_val[{bv_subs}]"),
        },
        code=code,
        outputs={"__out": dace.Memlet(f"__idx_candidates[{x_subs}]")},
        external_edges=True,
    )

    # ---- state 3: pick the first (or last) matching index ----
    reduce_idx = sdfg.add_state_after(candidates, node.label + "_reduce_idx")
    idx_red = Reduce(name=node.label + "_idx_reduce",
                     wcr=f"lambda a, b: {idx_wcr_op}(a, b)",
                     axes=axes,
                     identity=idx_sentinel)
    reduce_idx.add_node(idx_red)
    reduce_idx.add_edge(reduce_idx.add_read("__idx_candidates"), None, idx_red, "_in",
                        dace.Memlet.from_array("__idx_candidates", sdfg.arrays["__idx_candidates"]))
    reduce_idx.add_edge(idx_red, "_out", reduce_idx.add_access("__best_idx"), None,
                        dace.Memlet.from_array("__best_idx", sdfg.arrays["__best_idx"]))

    # ---- state 5: extract / decode ----
    extract = sdfg.add_state_after(reduce_idx, node.label + "_extract")
    if flat:
        # Decode flat -> per-dim subscripts.  One tasklet per dim,
        # single statement each: ``__o<d> = (__in // stride) % extent + offset``.
        strides = [1]
        for d in range(rank - 1, 0, -1):
            strides.insert(0, strides[0] * int(shape[d]))
        for d in range(rank):
            stride_d = strides[d]
            extent_d = int(shape[d])
            extract.add_mapped_tasklet(
                name=f"decode_dim_{d}",
                map_ranges={"__t": "0:1"},
                inputs={"__in": dace.Memlet("__best_idx[0]")},
                code=f"__out = (__in // {stride_d}) % {extent_d} + {one_offset}",
                outputs={"__out": dace.Memlet(f"_idx[{d}]")},
                external_edges=True,
            )
    else:
        rng = {f"__o{d}": f"0:{shape[d]}" for d in range(rank) if d != dim_zero}
        out_subs = ", ".join([f"__o{d}" for d in range(rank) if d != dim_zero])
        extract.add_mapped_tasklet(
            name="extract_idx",
            map_ranges=rng,
            inputs={"__in": dace.Memlet(f"__best_idx[{out_subs}]")},
            code=f"__out = __in + {one_offset}",
            outputs={"__out": dace.Memlet(f"_idx[{out_subs}]")},
            external_edges=True,
        )

    return sdfg


@dace.library.expansion
class ExpandArgMinPure(ExpandTransformation):
    """Pure expansion of :class:`ArgMin` -- multi-state min/min pipeline."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _emit_pure(node, parent_state, parent_sdfg, "min")


@dace.library.expansion
class ExpandArgMaxPure(ExpandTransformation):
    """Pure expansion of :class:`ArgMax` -- multi-state max/min pipeline."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _emit_pure(node, parent_state, parent_sdfg, "max")


def _validate_argminmax(node, sdfg, state):
    """Shared validation for ``ArgMin`` / ``ArgMax``.

    :returns: ``(desc_x, desc_idx, mask_desc_or_None, dim_zero_or_None)``.
    :raises ValueError: if the connector wiring or shapes don't match the
        configured ``dim`` reduction axis.
    """
    in_x = None
    in_mask = None
    for e in state.in_edges(node):
        if e.dst_conn == "_x":
            in_x = e
        elif e.dst_conn == "_mask":
            in_mask = e
    if in_x is None:
        raise ValueError(f"{type(node).__name__} requires an `_x` input")
    out_edges = state.out_edges(node)
    if len(out_edges) != 1 or out_edges[0].src_conn != "_idx":
        raise ValueError(f"{type(node).__name__} requires exactly one `_idx` output")
    desc_x = sdfg.arrays[in_x.data.data]
    desc_idx = sdfg.arrays[out_edges[0].data.data]
    mask_desc = sdfg.arrays[in_mask.data.data] if in_mask is not None else None
    rank = len(desc_x.shape)
    dim_zero = None
    if node.dim is not None:
        if not (1 <= node.dim <= rank):
            raise ValueError(f"{type(node).__name__}: dim={node.dim} out of range for rank-{rank} input")
        dim_zero = node.dim - 1
        expected_out_rank = rank - 1 or 1
        if len(desc_idx.shape) != expected_out_rank:
            raise ValueError(f"{type(node).__name__}: dim={node.dim} reduction expects rank-{expected_out_rank} "
                             f"output, got rank-{len(desc_idx.shape)}")
    return desc_x, desc_idx, mask_desc, dim_zero


@dace.library.node
class ArgMin(dace.sdfg.nodes.LibraryNode):
    """Index of the smallest element -- Fortran ``MINLOC`` / numpy ``argmin``."""

    implementations = {"pure": ExpandArgMinPure}
    default_implementation = "pure"

    one_based = dace.properties.Property(
        dtype=bool, default=True, desc="Return a Fortran 1-based index (``True``) or a 0-based index (``False``).")
    back = dace.properties.Property(
        dtype=bool,
        default=False,
        desc="Tie-break direction.  ``False`` keeps the first occurrence; ``True`` keeps the last.")
    dim = dace.properties.Property(
        dtype=int,
        default=None,
        allow_none=True,
        desc="Fortran 1-based reduction axis.  ``None`` reduces the whole array to a scalar.")

    def __init__(self, name, *, one_based=True, back=False, dim=None, mask=False, **kwargs):
        """:param mask: if ``True``, expose an optional ``_mask`` input connector."""
        inputs = {"_x"}
        if mask:
            inputs.add("_mask")
        super().__init__(name, inputs=inputs, outputs={"_idx"}, **kwargs)
        self.one_based = one_based
        self.back = back
        self.dim = dim

    def validate(self, sdfg, state):
        return _validate_argminmax(self, sdfg, state)


@dace.library.node
class ArgMax(dace.sdfg.nodes.LibraryNode):
    """Index of the largest element -- Fortran ``MAXLOC`` / numpy ``argmax``."""

    implementations = {"pure": ExpandArgMaxPure}
    default_implementation = "pure"

    one_based = dace.properties.Property(
        dtype=bool, default=True, desc="Return a Fortran 1-based index (``True``) or a 0-based index (``False``).")
    back = dace.properties.Property(
        dtype=bool,
        default=False,
        desc="Tie-break direction.  ``False`` keeps the first occurrence; ``True`` keeps the last.")
    dim = dace.properties.Property(
        dtype=int,
        default=None,
        allow_none=True,
        desc="Fortran 1-based reduction axis.  ``None`` reduces the whole array to a scalar.")

    def __init__(self, name, *, one_based=True, back=False, dim=None, mask=False, **kwargs):
        """:param mask: if ``True``, expose an optional ``_mask`` input connector."""
        inputs = {"_x"}
        if mask:
            inputs.add("_mask")
        super().__init__(name, inputs=inputs, outputs={"_idx"}, **kwargs)
        self.one_based = one_based
        self.back = back
        self.dim = dim

    def validate(self, sdfg, state):
        return _validate_argminmax(self, sdfg, state)


@oprepo.replaces('dace.libraries.standard.argmin')
@oprepo.replaces('dace.libraries.standard.ArgMin')
def argmin_libnode(pv: 'ProgramVisitor',
                   sdfg: SDFG,
                   state: SDFGState,
                   x,
                   idx,
                   *,
                   one_based=True,
                   back=False,
                   dim=None,
                   mask=None):
    x_in = state.add_read(x)
    out = state.add_write(idx)
    node = ArgMin('argmin', one_based=one_based, back=back, dim=dim, mask=mask is not None)
    state.add_node(node)
    state.add_edge(x_in, None, node, '_x', mm.Memlet(x))
    if mask is not None:
        mask_in = state.add_read(mask)
        state.add_edge(mask_in, None, node, '_mask', mm.Memlet(mask))
    state.add_edge(node, '_idx', out, None, mm.Memlet(idx))
    return []


@oprepo.replaces('dace.libraries.standard.argmax')
@oprepo.replaces('dace.libraries.standard.ArgMax')
def argmax_libnode(pv: 'ProgramVisitor',
                   sdfg: SDFG,
                   state: SDFGState,
                   x,
                   idx,
                   *,
                   one_based=True,
                   back=False,
                   dim=None,
                   mask=None):
    x_in = state.add_read(x)
    out = state.add_write(idx)
    node = ArgMax('argmax', one_based=one_based, back=back, dim=dim, mask=mask is not None)
    state.add_node(node)
    state.add_edge(x_in, None, node, '_x', mm.Memlet(x))
    if mask is not None:
        mask_in = state.add_read(mask)
        state.add_edge(mask_in, None, node, '_mask', mm.Memlet(mask))
    state.add_edge(node, '_idx', out, None, mm.Memlet(idx))
    return []
