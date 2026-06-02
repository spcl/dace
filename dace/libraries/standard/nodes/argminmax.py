# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ArgMin`` / ``ArgMax`` library nodes -- Fortran ``MINLOC`` / ``MAXLOC``.

Mirrors the ``numpy.argmin`` / ``numpy.argmax`` replacement pattern in
``dace/frontend/python/replacements/reduction.py`` (the
``_argminmax`` helper): a single Map writes a ``(val, idx)`` struct
into a per-slice transient via WCR; a second Map extracts the index
field into the user-facing result.

Configurable:

* ``one_based`` (default ``True``) -- offset added to the raw scan
  index before writing.  ``True`` matches Fortran's ``MINLOC`` /
  ``MAXLOC`` convention; set ``False`` for a 0-based result.
* ``back`` (default ``False``) -- tie-break direction.  ``False``
  keeps the first occurrence; ``True`` keeps the last.
* ``dim`` (default ``None``) -- 1-based reduction axis (Fortran
  convention).  ``None`` flattens to a scalar.
* ``mask`` constructor flag adds a ``_mask`` input connector.  Mask
  semantics are intentionally NOT yet wired in the pure expansion --
  the validate step accepts the connector for downstream bridge
  patterns, and an explicit ``NotImplementedError`` is raised at
  expansion time.
"""

import dace
import dace.dtypes as dtypes
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, SDFGState, memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.transformation.transformation import ExpandTransformation


def _emit_pure(node, parent_state: SDFGState, parent_sdfg: SDFG, func: str):
    """Shared pure expansion for :class:`ArgMin` / :class:`ArgMax`.

    Builds a nested SDFG with two states: ``init`` seeds the
    per-output-slice ``(val, idx)`` struct transient with a sentinel,
    and ``reduce`` runs a Map with a min-/max-of-pair WCR over the
    input slice.  A trailing Map extracts the ``.idx`` field into the
    user-facing index array.

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
    # Whole-array form (``dim is None``) reduces over every element and
    # returns a rank-1 array of length ``rank(input)`` holding the
    # multi-dim subscript -- Fortran ``MINLOC`` / ``MAXLOC`` semantics
    # (F2018 13.7.110 / 13.7.111).  ``flat`` selects between this
    # "decode flat index back to subscripts" path and the per-axis
    # reduction path that returns rank-(R-1).
    flat = dim_zero is None

    sdfg = dace.SDFG(node.label + "_sdfg")
    sdfg.add_array("_x", shape, dtype)
    if flat:
        sdfg.add_array("_idx", [rank], idx_dtype)
        # Internal pair transient is a singleton -- holds the running
        # (best val, flat idx) for the whole-array scan; the trailing
        # ``extract`` state decodes the flat idx into per-dim
        # subscripts and fans out to the rank-1 ``_idx`` output.
        out_shape = [1]
    else:
        out_shape = [s for d, s in enumerate(shape) if d != dim_zero]
        if not out_shape:
            out_shape = [1]
        sdfg.add_array("_idx", out_shape, idx_dtype)

    pair = dtypes.struct(f"_val_and_idx_{func}", idx=idx_dtype, val=dtype)
    _, pair_arr = sdfg.add_transient("_pair", out_shape, pair)

    init_code_val = (str(dtypes.min_value(dtype)) if func == "max" else str(dtypes.max_value(dtype)))
    init_state = sdfg.add_state(node.label + "_init")
    if flat:
        init_state.add_mapped_tasklet(
            name=f"_arg{func}_init",
            map_ranges={"__i0": "0:1"},
            inputs={},
            code=f"__init = _val_and_idx_{func}(val={init_code_val}, idx=-1)",
            outputs={"__init": dace.Memlet("_pair[0]")},
            external_edges=True,
        )
    else:
        init_rng = {f"__o{d}": f"0:{shape[d]}" for d in range(rank) if d != dim_zero}
        out_subs = ", ".join([f"__o{d}" for d in range(rank) if d != dim_zero])
        init_state.add_mapped_tasklet(
            name=f"_arg{func}_init",
            map_ranges=init_rng,
            inputs={},
            code=f"__init = _val_and_idx_{func}(val={init_code_val}, idx=-1)",
            outputs={"__init": dace.Memlet(f"_pair[{out_subs}]")},
            external_edges=True,
        )

    reduce_state = sdfg.add_state_after(init_state, node.label + "_reduce")
    map_rng = {f"__i{d}": f"0:{shape[d]}" for d in range(rank)}
    x_subs = ", ".join([f"__i{d}" for d in range(rank)])

    # WCR x/y convention in DaCe codegen: ``x`` is the incoming value
    # and ``y`` is the accumulator.  For ``back=False`` (first
    # occurrence) the accumulator wins on ties; for ``back=True`` the
    # incoming wins.  When the WCR is scheduled sequentially, the
    # strict ``<`` (for min) keeps the accumulator on ties (only
    # taking incoming when strictly smaller) -- first occurrence.
    # ``back=True`` needs the parallel pair to break ties toward the
    # higher SCAN INDEX, not the accumulator-vs-incoming role:
    # compare the indices to disambiguate.
    cmp_strict = ">" if func == "max" else "<"  # "x is strictly better than y"
    cmp_op = cmp_strict

    if flat:
        scan_idx_expr = " + ".join([(f"__i{d}" if d == rank - 1 else f"(__i{d} * " +
                                     " * ".join([str(int(s)) for s in shape[d + 1:]]) + ")")
                                    for d in range(rank)]) if rank > 1 else "__i0"
        out_pair_subs = "0"
    else:
        scan_idx_expr = f"__i{dim_zero}"
        out_pair_subs = ", ".join([f"__i{d}" for d in range(rank) if d != dim_zero])

    one_offset = 1 if node.one_based else 0
    # On tie (``x.val == y.val``), pick the larger or smaller of the
    # two scan indices depending on ``back``: ``back=True`` keeps the
    # larger index (last occurrence), ``back=False`` keeps the smaller
    # (first occurrence).  Three-way pick: x strictly better -> x;
    # y strictly better -> y; tied -> index comparison decides.
    tie_idx = "max(x.idx, y.idx)" if node.back else "min(x.idx, y.idx)"
    wcr = ("lambda x, y: "
           f"_val_and_idx_{func}("
           f"val={func}(x.val, y.val), "
           f"idx=(x.idx if x.val {cmp_op} y.val else (y.idx if y.val {cmp_op} x.val else {tie_idx})))")
    reduce_code = f"__out = _val_and_idx_{func}(val=__in, idx=({scan_idx_expr}) + {one_offset})"
    reduce_state.add_mapped_tasklet(
        name=f"_arg{func}_reduce",
        map_ranges=map_rng,
        inputs={"__in": dace.Memlet(f"_x[{x_subs}]")},
        code=reduce_code,
        outputs={"__out": dace.Memlet(f"_pair[{out_pair_subs}]", wcr=wcr)},
        external_edges=True,
    )

    extract_state = sdfg.add_state_after(reduce_state, node.label + "_extract")
    if flat:
        # Decode flat idx -> per-dim subscripts.  Each subscript lives
        # in its own ``_idx[d]`` slot; the tasklet recovers them with
        # the Horner-style divide/mod ladder that matches the encoding
        # in ``scan_idx_expr`` above.  The +1 (Fortran 1-based) is
        # already baked into the stored flat idx -- subtract it once
        # before decoding, then re-add per-dim if ``one_based``.
        flat_zero = f"(__in.idx - {one_offset})"
        decode_lines = ["__flat = " + flat_zero]
        # Strides per dim from rightmost (== 1) toward leftmost.
        strides = [1]
        for d in range(rank - 1, 0, -1):
            strides.insert(0, strides[0] * int(shape[d]))
        for d in range(rank):
            stride_d = strides[d]
            if d < rank - 1:
                decode_lines.append(f"__i{d} = (__flat // {stride_d}) % {int(shape[d])}")
            else:
                decode_lines.append(f"__i{d} = __flat % {int(shape[d])}")
        body = "\n".join(decode_lines)
        outputs = {}
        out_lines = []
        for d in range(rank):
            out_lines.append(f"__o{d} = __i{d} + {one_offset}")
            outputs[f"__o{d}"] = dace.Memlet(f"_idx[{d}]")
        body += "\n" + "\n".join(out_lines)
        extract_state.add_mapped_tasklet(
            name=f"_arg{func}_extract",
            map_ranges={"__t": "0:1"},
            inputs={"__in": dace.Memlet("_pair[0]")},
            code=body,
            outputs=outputs,
            external_edges=True,
        )
    else:
        out_rng = {f"__o{d}": f"0:{shape[d]}" for d in range(rank) if d != dim_zero}
        out_subs = ", ".join([f"__o{d}" for d in range(rank) if d != dim_zero])
        extract_state.add_mapped_tasklet(
            name=f"_arg{func}_extract",
            map_ranges=out_rng,
            inputs={"__in": dace.Memlet(f"_pair[{out_subs}]")},
            code="__out = __in.idx",
            outputs={"__out": dace.Memlet(f"_idx[{out_subs}]")},
            external_edges=True,
        )

    return sdfg


@dace.library.expansion
class ExpandArgMinPure(ExpandTransformation):
    """Pure expansion of :class:`ArgMin` -- WCR over a ``(val, idx)`` struct."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _emit_pure(node, parent_state, parent_sdfg, "min")


@dace.library.expansion
class ExpandArgMaxPure(ExpandTransformation):
    """Pure expansion of :class:`ArgMax` -- WCR over a ``(val, idx)`` struct."""
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
        desc="Tie-break direction. ``False`` keeps the first occurrence; ``True`` keeps the last.")
    dim = dace.properties.Property(dtype=int,
                                   default=None,
                                   allow_none=True,
                                   desc="Fortran 1-based reduction axis. ``None`` reduces the whole array to a scalar.")

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
        desc="Tie-break direction. ``False`` keeps the first occurrence; ``True`` keeps the last.")
    dim = dace.properties.Property(dtype=int,
                                   default=None,
                                   allow_none=True,
                                   desc="Fortran 1-based reduction axis. ``None`` reduces the whole array to a scalar.")

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
