# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Scan`` library node: in-place / out-of-place prefix scan over a 1-D array.

A *scan* (also "prefix-reduction") computes per-position partial reductions of an
input array: for an inclusive scan with op ``+``::

    out[k] = in[0] + in[1] + ... + in[k]    for k in 0..N-1

For an exclusive scan with identity ``e``::

    out[k] = e + in[0] + ... + in[k-1]      for k in 0..N-1  (out[0] == e)

This is the canonical *parallel-prefix* primitive: a true scan keeps every
partial sum visible, so a downstream consumer can read any ``out[k]``. It is
*not* equivalent to a single ``Reduce`` -- that collapses to one scalar and
loses the per-position values.

Why a library node, not a transformation: GPU implementations (Blelloch upsweep
+ downsweep) have O(N log N) work / O(log N) depth and are non-trivial to
emit from generic SDFG primitives; the CPU implementation is a tight
memory-bandwidth-bound sequential loop. Both already exist as battle-tested
library functions (``gpucub::DeviceScan`` on GPU, ``std::partial_sum`` /
``std::inclusive_scan`` / ``std::exclusive_scan`` on CPU). The libnode is the
right level of abstraction.

Implementations:

- ``CPU`` -- ``dace::scan::*`` from :file:`dace/runtime/include/dace/scan.hpp`: a
  blocked three-phase scan over one OpenMP region, each phase vectorised by an
  OpenMP 5.0 ``simd`` ``inscan``. Below the header's element threshold it is that
  same vectorised pass on one thread, which is what the cloudsc-style
  vertical-flux pattern (N ~ 137) gets.
- ``CUDA`` -- ``gpucub::DeviceScan::InclusiveScan`` / ``ExclusiveScan`` (Blelloch
  upsweep + downsweep; memory-bandwidth-bound on modern NVIDIA GPUs at
  GKeys/s rates).
- ``pure`` -- portable single-loop fallback (used when neither CPU nor CUDA
  expansion applies, e.g. for FPGA backends in v1).

For supported binary ops, see :data:`_OP_TO_STD_CPP` and :data:`_OP_TO_CUB`. The
op must be associative -- ``+``, ``*``, ``min``, ``max`` -- so the order of the
partial reductions does not change the result.
"""

import numpy

import dace
from dace import dtypes, library, nodes, symbolic
from dace.codegen.common import sym2cpp
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.properties import Property, EnumProperty
from dace.transformation.transformation import ExpandTransformation
import enum

# CUB env is imported lazily inside ``ExpandCUDA.expansion`` to break the
# ``dace.libraries.standard.nodes.scan`` <-> ``dace.libraries.sort.environments.cub``
# circular import (cub.py pulls in standard.environments, which loads this module).
from dace.libraries.standard.environments.cpu import CPU as CPUEnv

# Connector names exposed for library-node builders.
INPUT_CONNECTOR_NAME = "_scan_in"
OUTPUT_CONNECTOR_NAME = "_scan_out"
#: Optional scalar input. When wired, the expansion emits an inclusive scan
#: with an initial accumulator value (``out[k] = init OP in[0] OP ... OP in[k]``),
#: which lets the LoopToScan rewrite skip its separate seed-add Map.
INIT_CONNECTOR_NAME = "_scan_init"

#: Second input array, wired only when ``op is ScanOp.AFFINE``: the per-element
#: coefficient ``c`` of ``out[k] = c[k] * out[k-1] + d[k]``. ``_scan_in`` carries ``d``.
COEF_CONNECTOR_NAME = "_scan_coef"


def in_connector(chain: int = 0) -> str:
    """Input connector of scan chain ``chain`` (chain 0 keeps the bare name)."""
    return INPUT_CONNECTOR_NAME if chain == 0 else f'{INPUT_CONNECTOR_NAME}_{chain}'


def out_connector(chain: int = 0) -> str:
    """Output connector of scan chain ``chain``."""
    return OUTPUT_CONNECTOR_NAME if chain == 0 else f'{OUTPUT_CONNECTOR_NAME}_{chain}'


def coef_connector(chain: int = 0) -> str:
    """Coefficient-input connector name for ``chain`` (affine scans only)."""
    return COEF_CONNECTOR_NAME if chain == 0 else f'{COEF_CONNECTOR_NAME}_{chain}'


def init_connector(chain: int = 0) -> str:
    """Optional init connector of scan chain ``chain``."""
    return INIT_CONNECTOR_NAME if chain == 0 else f'{INIT_CONNECTOR_NAME}_{chain}'


class ScanOp(enum.Enum):
    """Associative binary operations supported by the :class:`Scan` libnode."""
    SUM = 'sum'
    PRODUCT = 'product'
    MIN = 'min'
    MAX = 'max'
    #: ``out[k] = c[k] * out[k-1] + d[k]`` -- a first-order LINEAR recurrence. The four ops above
    #: carry a value; this one carries the affine map ``x -> a*x + b``, and its monoid is map
    #: composition. It is a scan in every structural sense (associative, fixed-width carry, same
    #: blocked lowering) but it reads a SECOND array through ``_scan_coef`` and cannot borrow any
    #: of the scalar-op plumbing: no ``std`` functor, no OpenMP built-in reduction identifier, no
    #: CUB functor. Every shape that would need one refuses it explicitly rather than KeyError.
    #:
    #: Only linearity in the carry makes the map close under composition. ``out[k] = f(out[k-1])``
    #: for a nonlinear ``f`` is still associative under composition, but the carry is then the
    #: whole function and there is nothing of bounded width to scan over; ``LoopToScan``'s matcher
    #: proves linearity symbolically before it ever builds these buffers.
    AFFINE = 'affine'


#: Map op enum to the C++ binary-op functor for ``std::inclusive_scan`` / ``std::exclusive_scan``
#: (used by the ``pure`` expansion). These are functor *values* (constructed via ``Op{}``).
_OP_TO_STD_CPP = {
    ScanOp.SUM: 'std::plus<>{}',
    ScanOp.PRODUCT: 'std::multiplies<>{}',
    ScanOp.MIN: '[](auto a, auto b){ return std::min(a, b); }',
    ScanOp.MAX: '[](auto a, auto b){ return std::max(a, b); }',
}

#: Map op enum to the suffix of the OpenMP-scan function in ``dace/scan.hpp``.
#: The ``CPU`` expansion emits ``dace::scan::inclusive_<suffix>`` / ``exclusive_<suffix>``.
_OP_TO_OMP_SUFFIX = {
    ScanOp.SUM: 'sum',
    ScanOp.PRODUCT: 'product',
    ScanOp.MIN: 'min',
    ScanOp.MAX: 'max',
    ScanOp.AFFINE: 'affine',
}

#: Map op enum to the OpenMP reduction identifier used by the ``reduction(inscan, <id>: ...)``
#: clause the multi-chain expansion emits. All four are OpenMP built-ins, so no
#: ``declare reduction`` is needed.
_OP_TO_OMP_REDUCTION = {
    ScanOp.SUM: '+',
    ScanOp.PRODUCT: '*',
    ScanOp.MIN: 'min',
    ScanOp.MAX: 'max',
}

#: Map op enum to the CUB-side binary functor for ``gpucub::DeviceScan::InclusiveScan``.
#: Routed through the ``DACE_CUB_*_OP`` macros from ``dace/cub_compat.cuh`` so the
#: same SDFG source builds against CUDA Toolkit 12 (CUB <= 2.x has ``gpucub::Sum`` /
#: ``gpucub::Min`` / ``gpucub::Max``) and 13 (CCCL 3.x dropped those in favour of
#: ``cuda::std::plus`` + device lambdas).
_OP_TO_CUB = {
    ScanOp.SUM: 'DACE_CUB_SUM_OP',
    ScanOp.PRODUCT: 'DACE_CUB_MUL_OP',
    ScanOp.MIN: 'DACE_CUB_MIN_OP',
    ScanOp.MAX: 'DACE_CUB_MAX_OP',
}

#: Fold identity per op as a C++ expression at element type ``{ct}``. The blocked
#: parallel shape needs one for its per-block reduce pass; ``min``/``max`` take the
#: runtime's neutral elements, which are what OpenMP seeds a reduction private copy
#: to, so a block that the fold skips cannot move the result.
_OP_TO_FOLD_IDENTITY = {
    ScanOp.SUM: 'static_cast<{ct}>(0)',
    ScanOp.PRODUCT: 'static_cast<{ct}>(1)',
    ScanOp.MIN: '::dace::scan::detail::min_identity<{ct}>()',
    ScanOp.MAX: '::dace::scan::detail::max_identity<{ct}>()',
}

#: Default identity literal for ``exclusive`` scans, per op.
_OP_TO_IDENTITY_CPP = {
    ScanOp.SUM:
    '0',
    ScanOp.PRODUCT:
    '1',
    # ``min``/``max`` have no universal identity in C++ literal form -- callers must
    # supply ``identity`` explicitly for exclusive ``min``/``max`` scans.
    ScanOp.MIN:
    None,
    ScanOp.MAX:
    None,
}


def _validate_chain(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG, chain: int):
    """Resolve and validate one chain's in/out/init edges; raise on any wiring/shape/dtype
    mismatch. ``_scan_init`` is optional; when present it must be a single scalar /
    length-1 edge whose dtype matches the input array's element type.
    """
    in_conn, out_conn, init_conn = in_connector(chain), out_connector(chain), init_connector(chain)
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == in_conn]
    out_edges = [e for e in state.out_edges(node) if e.src_conn == out_conn]
    init_edges = [e for e in state.in_edges(node) if e.dst_conn == init_conn]
    if len(in_edges) != 1 or len(out_edges) != 1:
        raise ValueError(f"Scan node {node.label} expects exactly one ``{in_conn}`` "
                         f"in-edge and one ``{out_conn}`` out-edge.")
    if len(init_edges) > 1:
        raise ValueError(f"Scan node {node.label}: ``{init_conn}`` is optional but at "
                         f"most one in-edge is allowed; got {len(init_edges)}.")
    in_desc = sdfg.arrays[in_edges[0].data.data]
    out_desc = sdfg.arrays[out_edges[0].data.data]
    if not isinstance(in_desc, dace.data.Array) or not isinstance(out_desc, dace.data.Array):
        raise ValueError(f"Scan requires Array inputs/outputs; got {type(in_desc).__name__} -> "
                         f"{type(out_desc).__name__}.")
    if in_desc.dtype != out_desc.dtype and not widening_is_value_preserving(in_desc.dtype, out_desc.dtype):
        raise ValueError(f"Scan input/output dtype mismatch: {in_desc.dtype} vs {out_desc.dtype}. Only a "
                         f"value-preserving integer WIDENING is allowed, and only on the unit-stride "
                         f"single-chain host expansions.")
    coef_edges = [e for e in state.in_edges(node) if e.dst_conn == coef_connector(chain)]
    if node.op is ScanOp.AFFINE:
        if len(coef_edges) != 1:
            raise ValueError(f"Scan node {node.label}: ``op=AFFINE`` requires exactly one "
                             f"``{coef_connector(chain)}`` in-edge; got {len(coef_edges)}.")
        coef_desc = sdfg.arrays[coef_edges[0].data.data]
        if not isinstance(coef_desc, dace.data.Array):
            raise ValueError(f"Scan node {node.label}: ``{coef_connector(chain)}`` must be an Array; "
                             f"got {type(coef_desc).__name__}.")
        # The ACCUMULATOR's type: the coefficient multiplies the carry, so a coefficient in a
        # different type would silently pick the promotion C++ happens to give it.
        if coef_desc.dtype != out_desc.dtype:
            raise ValueError(f"Scan node {node.label}: ``{coef_connector(chain)}`` dtype "
                             f"{coef_desc.dtype} must match output dtype {out_desc.dtype}.")
        if symbolic.equal(coef_edges[0].data.subset.num_elements(), in_edges[0].data.subset.num_elements()) is False:
            raise ValueError(f"Scan node {node.label}: ``{coef_connector(chain)}`` spans "
                             f"{coef_edges[0].data.subset.num_elements()} elements against "
                             f"``{in_conn}``'s {in_edges[0].data.subset.num_elements()}.")
    elif coef_edges:
        raise ValueError(f"Scan node {node.label}: ``{coef_connector(chain)}`` is wired but "
                         f"``op`` is {node.op.value!r}, not AFFINE.")
    if init_edges:
        init_desc = sdfg.arrays[init_edges[0].data.data]
        # The OUTPUT dtype, not the input's: ``_scan_init`` is the accumulator's entry value, and
        # the accumulator is the output element type (identical to the input's unless widening).
        if init_desc.dtype != out_desc.dtype:
            raise ValueError(f"Scan node {node.label}: ``{init_conn}`` dtype "
                             f"{init_desc.dtype} must match output dtype {out_desc.dtype}.")
    return in_desc, out_desc, in_edges[0], out_edges[0], (init_edges[0] if init_edges else None)


def _validate_inputs_and_outputs(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG):
    """Validate every chain and return chain 0's ``(in_desc, out_desc, in_edge, out_edge)``.

    The chains are INDEPENDENT scans sharing one libnode (and, on the parallel CPU
    schedule, one OpenMP region), so they must agree on element count and dtype --
    they are lowered as list items of a single ``reduction(inscan, ...)`` clause.
    """
    first = None
    for chain in range(node.chains):
        in_desc, out_desc, in_edge, out_edge, _ = _validate_chain(node, state, sdfg, chain)
        if first is None:
            first = (in_desc, out_desc, in_edge, out_edge)
            continue
        if in_desc.dtype != first[0].dtype or out_desc.dtype != first[1].dtype:
            raise ValueError(f"Scan node {node.label}: chain {chain} dtypes {in_desc.dtype} -> "
                             f"{out_desc.dtype} differ from chain 0's {first[0].dtype} -> "
                             f"{first[1].dtype}; chains share one scan loop.")
        if symbolic.equal(in_edge.data.subset.num_elements(), first[2].data.subset.num_elements()) is False:
            raise ValueError(f"Scan node {node.label}: chain {chain} spans "
                             f"{in_edge.data.subset.num_elements()} elements against chain 0's "
                             f"{first[2].data.subset.num_elements()}; chains share one scan loop.")
    return first


def widening_is_value_preserving(in_dtype, out_dtype) -> bool:
    """Whether a scan may read ``in_dtype`` and accumulate into a wider ``out_dtype``.

    The accumulator is the OUTPUT element type, so a narrow input is read and widened per element.
    That is what lets stream compaction carry its predicate mask as ``int8`` -- one byte per
    element instead of eight -- while the ranks it prefix-sums into stay ``int64``, which they must
    (a rank is an index). Summing in the input type would wrap at 127.

    Only integers, only strictly wider, and never signed -> unsigned: every other pair either loses
    range (``float64 -> float32``) or reinterprets negatives, and a scan that silently changes a
    value is worse than one that refuses.

    :param in_dtype: the input array's element type.
    :param out_dtype: the output array's element type, which is also the accumulator's.
    :returns: True if the widening is value-preserving.
    """
    if not (numpy.issubdtype(in_dtype.type, numpy.integer) and numpy.issubdtype(out_dtype.type, numpy.integer)):
        return False
    if out_dtype.bytes <= in_dtype.bytes:
        return False
    return not (numpy.issubdtype(in_dtype.type, numpy.signedinteger)
                and numpy.issubdtype(out_dtype.type, numpy.unsignedinteger))


def refuse_widening(node: "Scan", in_desc, out_desc, shape: str) -> None:
    """Raise when a widening scan reaches a shape that has no widening implementation.

    The widening accumulator lives in the unit-stride single-chain host routines only. The strided
    form carries one accumulator per residue class seeded from the input, the multi-chain form
    spells K accumulators into one ``inscan`` clause at a single ctype, and ``gpucub::DeviceScan``
    deduces its accumulator from the input iterator -- each would need its own widening design.
    Refuse loudly; a silent narrow accumulator is a wrong answer, not a slow one.
    """
    if in_desc.dtype != out_desc.dtype:
        raise NotImplementedError(f"Scan {node.label}: a widening scan ({in_desc.dtype} -> {out_desc.dtype}) "
                                  f"is not supported with {shape}.")


def refuse_affine_shape(node: "Scan", shape: str) -> None:
    """Raise when an affine scan reaches a shape whose lowering only exists for the scalar ops.

    The affine carry is a PAIR, so every place the scalar plumbing names a single accumulator has
    to be redesigned rather than reused: an ``inscan`` reduction needs a built-in or declared
    identifier for the pair, and a residue-class split needs one pair per class. The unit-stride
    CUDA path has its pair and its functor (``dace/cuda/scan_affine.cuh``); the shapes still listed
    here do not, and a silent fallback to a scalar op would compute a different function. Refuse.
    """
    if node.op is ScanOp.AFFINE:
        raise NotImplementedError(f"Scan {node.label}: ``op=AFFINE`` is not supported with {shape}.")


def has_coef(node: "Scan", chain: int = 0) -> bool:
    """``True`` iff chain ``chain`` has the affine ``_scan_coef`` connector wired."""
    return coef_connector(chain) in node.in_connectors


def _has_init(node: "Scan", chain: int = 0) -> bool:
    """``True`` iff chain ``chain`` has the optional ``_scan_init`` connector wired."""
    return init_connector(chain) in node.in_connectors


def seed_desc(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG, chain: int):
    """Descriptor behind chain ``chain``'s wired ``_scan_init``, or None when it carries no seed."""
    conn = init_connector(chain)
    edge = next((e for e in state.in_edges(node) if e.dst_conn == conn), None)
    return None if edge is None else sdfg.arrays[edge.data.data]


def seed_arg(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG, chain: int) -> str:
    """The ``init_value`` argument of the cub call for chain ``chain``."""
    conn = init_connector(chain)
    desc = seed_desc(node, state, sdfg, chain)
    if desc is None or desc.storage not in GPU_RESIDENT_STORAGES:
        return conn
    return f"::gpucub::FutureValue<{desc.dtype.base_type.ctype}>({conn})"


def coef_desc(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG, chain: int = 0):
    """Descriptor behind chain ``chain``'s ``_scan_coef``; affine scans always wire one."""
    conn = coef_connector(chain)
    edge = next(e for e in state.in_edges(node) if e.dst_conn == conn)
    return sdfg.arrays[edge.data.data]


def _resolve_length(node: "Scan", state: dace.SDFGState, _sdfg: dace.SDFG) -> str:
    """C++ expression for the number of elements ``N`` in the input edge."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    return sym2cpp(in_edges[0].data.subset.num_elements())


def _is_length_one(node: "Scan", state: dace.SDFGState) -> bool:
    """``True`` if the input subset is statically a single element. Single-element scans
    degenerate to a trivial copy (inclusive) or identity write (exclusive) -- no array
    iteration, no iterator-based template instantiation that would conflict with the
    codegen's scalar-typing of single-element subsets."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    n = symbolic.simplify(in_edges[0].data.subset.num_elements())
    return getattr(n, 'is_Integer', False) and int(n) == 1


def _degenerate_single_element_tasklet(node: "Scan", in_desc) -> nodes.Tasklet:
    """Return the single-element degenerate scan tasklet.

    For an inclusive single-element scan the result is just the input itself; for an
    exclusive single-element scan the result is the user-supplied identity. Both are
    expressed as Python tasklets so the codegen handles scalar connector typing
    naturally (no array indexing, no iterator templates).
    """
    if node.exclusive:
        # Treat the identity as a Python literal; the codegen casts via the connector
        # type (a scalar of ``in_desc.dtype``).
        seed_py = node.identity if node.identity is not None else (0 if node.op is ScanOp.SUM else
                                                                   (1 if node.op is ScanOp.PRODUCT else None))
        if seed_py is None:
            raise ValueError(f"Scan op {node.op.value!r} has no universal identity; set ``identity`` explicitly.")
        code = f"{OUTPUT_CONNECTOR_NAME} = {seed_py}"
    else:
        code = f"{OUTPUT_CONNECTOR_NAME} = {INPUT_CONNECTOR_NAME}"
    return nodes.Tasklet(node.name,
                         inputs={INPUT_CONNECTOR_NAME: None},
                         outputs={OUTPUT_CONNECTOR_NAME: None},
                         code=code,
                         language=dace.Language.Python)


def _identity_expr(node: "Scan", acc_desc) -> str:
    """C++ expression for the exclusive-scan identity element, at the ACCUMULATOR's type.

    ``acc_desc`` is the OUTPUT descriptor: the identity is the accumulator's entry value, and on a
    widening scan the accumulator is wider than the input.

    The user-supplied ``identity`` property wins. Otherwise the per-op default
    from :data:`_OP_TO_IDENTITY_CPP` is used; if the op has no universal
    identity (``min``/``max``) the user *must* set ``identity``.
    """
    literal = node.identity
    if literal is None:
        literal = _OP_TO_IDENTITY_CPP[node.op]
        if literal is None:
            raise ValueError(f"Scan op {node.op.value!r} has no universal identity in C++ literal form; "
                             f"set ``identity`` explicitly when using ``exclusive=True``.")
    # ALWAYS cast, including a user-supplied identity. Beyond avoiding signed/unsigned warnings, the
    # cast is what pins the accumulator's width: ``gpucub::DeviceScan::ExclusiveScan`` deduces ``AccumT``
    # from the init value, so a bare ``0`` would make an int8 -> int64 scan accumulate in ``int``.
    return f"static_cast<{acc_desc.dtype.ctype}>({literal})"


def _combine_expr(op: ScanOp, ctype: str, a: str, b: str) -> str:
    """C++ expression for ``a OP b`` at element type ``ctype``."""
    if op is ScanOp.SUM:
        return f'{a} + {b}'
    if op is ScanOp.PRODUCT:
        return f'{a} * {b}'
    return f'std::{"min" if op is ScanOp.MIN else "max"}<{ctype}>({a}, {b})'


def _chain_seed_expr(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG, chain: int, ctype: str) -> str:
    """C++ expression the chain's accumulator starts at.

    A wired ``_scan_init`` wins (inclusive-with-init: ``out[k] = init OP in[0] OP
    ... OP in[k]``). Otherwise an exclusive scan takes the op identity and an
    inclusive one takes ``+``/``*``'s identity, or -- for ``min``/``max``, which
    have none in C++ literal form -- the chain's own first element, exactly as the
    single-chain routines in ``dace/scan.hpp`` do.
    """
    _, _, _, _, init_edge = _validate_chain(node, state, sdfg, chain)
    if init_edge is not None:
        return init_connector(chain)
    if node.identity is not None or node.op in (ScanOp.SUM, ScanOp.PRODUCT):
        return f'static_cast<{ctype}>({node.identity if node.identity is not None else _OP_TO_IDENTITY_CPP[node.op]})'
    if node.exclusive:
        raise ValueError(f"Scan op {node.op.value!r} has no universal identity; set ``identity`` explicitly.")
    return f'{in_connector(chain)}[0]'


#: Compound assignment per op, for the ``declare reduction`` combiner below.
_OP_TO_COMPOUND = {ScanOp.SUM: '+=', ScanOp.PRODUCT: '*='}


def _multi_chain_udr(op: ScanOp, dtype, ctype: str) -> str:
    """``declare reduction`` for an element type OpenMP has no built-in one for, else ``''``.

    An OpenMP UDR is found by ORDINARY UNQUALIFIED LOOKUP from the point of use, and this
    expansion splices its pragmas into the generated program -- so ``dace/scan.hpp``'s copies,
    which sit inside ``dace::scan::detail``, are unreachable from here and every complex-typed
    chain past the first failed to compile ("user defined reduction not found"). OpenMP allows
    the directive at block scope, so the tasklet declares its own next to the accumulators.

    ``min`` / ``max`` need none: complex has no ordering, and the fold identity
    (``detail::min_identity`` / ``max_identity``) already refuses it with a ``static_assert``.
    """
    if dtype not in (dace.complex64, dace.complex128):
        return ''
    compound = _OP_TO_COMPOUND.get(op)
    if compound is None:
        return ''
    ident = _OP_TO_IDENTITY_CPP[op]
    return (f'#pragma omp declare reduction({_OP_TO_OMP_REDUCTION[op]} : {ctype} : omp_out {compound} omp_in) '
            f'initializer(omp_priv = {ctype}({ident}))')


def _multi_chain_parallel_code(node: "Scan", ctype: str, n_expr: str, accs, acc_list: str, seeds, first: str,
                               second: str, scan_kind: str, udr: str) -> str:
    """Blocked three-phase body for ``chains > 1``, K chains wide.

    Same algorithm as ``dace::scan::detail::blocked_scan``; it cannot BE that
    routine, because the ``inscan`` clause needs the K accumulator names spelled
    out and an array section is rejected there. So the scan pass is a lambda and
    the driver around it mirrors the header: one region, tiles sized to keep the
    reduce pass and the scan pass on the same L2-resident block, one padded block
    total per thread carrying all K chains. The lambda is what keeps the emitted
    code to a single ``omp scan`` directive despite the two call sites.

    ``udr`` is the block-scope reduction declaration the element type needs, or ``''``
    -- see :func:`_multi_chain_udr`. It heads the block so both the ``inscan`` pass and
    the fold pass below it are inside its scope.
    """
    k = node.chains
    d = '::dace::scan::detail'
    op = _OP_TO_OMP_REDUCTION[node.op]
    ident = _OP_TO_FOLD_IDENTITY[node.op].format(ct=ctype)
    sums = [f'__s{c}' for c in range(k)]
    alls = [f'__all{c}' for c in range(k)]
    carries = [f'__carry{c}' for c in range(k)]
    offs = [f'__off{c}' for c in range(k)]
    call_offs = ', '.join(offs)
    call_carries = ', '.join(carries)
    call_seeds = ', '.join(seeds)
    return '\n'.join([
        '{',
    ] + ([udr] if udr else []) + [
        f'const long __n = static_cast<long>({n_expr});',
        'if (__n > 0) {',
        'auto __scan_block = [&](long __lo, long __hi, ' + ', '.join(f'{ctype} {a}' for a in accs) + ') {',
        f'    #pragma omp simd reduction(inscan, {op}:{acc_list})',
        '    for (long __i = __lo; __i < __hi; ++__i) {',
        first,
        f'        #pragma omp scan {scan_kind}({acc_list})',
        second,
        '    }',
        '};',
        f'const int __want = {d}::team_size();',
        # No size test: the multi-chain shape follows the single-chain one, where whether a
        # scan earns a team is decided ONCE at compile time against the host's calibrated
        # break-even, not re-tested on every call. ``__want > 1`` is not a threshold.
        'if (__want > 1) {',
        f'    {d}::TeamSlot<{ctype}, {k}> __tot[{d}::MAX_TEAM];',
        '    #pragma omp parallel num_threads(__want)',
        '    {',
        f'        const long __team = {d}::team_count();',
        f'        const long __me = {d}::team_rank();',
        f'        const long __per = {d}::block_span(__n, __team, static_cast<long>({k} * sizeof({ctype})));',
        '        const long __tile = __per * __team;',
    ] + [f'        {ctype} {carries[c]} = {seeds[c]};' for c in range(k)] + [
        '        for (long __base = 0; __base < __n; __base += __tile) {',
        '            const long __end = (__base + __tile < __n) ? __base + __tile : __n;',
        '            const long __lo = (__base + __me * __per < __end) ? __base + __me * __per : __end;',
        '            const long __hi = (__lo + __per < __end) ? __lo + __per : __end;',
        '            if (__team > 1) {',
    ] + [f'                {ctype} {sums[c]} = {ident};' for c in range(k)] + [
        f'                #pragma omp simd reduction({op}:{", ".join(sums)})',
        '                for (long __i = __lo; __i < __hi; ++__i) {',
    ] + [
        f'                    {sums[c]} = {_combine_expr(node.op, ctype, sums[c], f"{in_connector(c)}[__i]")};'
        for c in range(k)
    ] + ['                }'] + [f'                __tot[__me].v[{c}] = {sums[c]};' for c in range(k)] + [
        '                #pragma omp barrier',
    ] + [f'                {ctype} {offs[c]} = {carries[c]}, {alls[c]} = {carries[c]};' for c in range(k)] + [
        '                for (long __q = 0; __q < __team; ++__q) {',
        '                    if (__q == __me) {',
    ] + [f'                        {offs[c]} = {alls[c]};' for c in range(k)] + ['                    }'] + [
        f'                    {alls[c]} = {_combine_expr(node.op, ctype, alls[c], f"__tot[__q].v[{c}]")};'
        for c in range(k)
    ] + ['                }'] + [
        f'                __scan_block(__lo, __hi, {call_offs});',
    ] + [f'                {carries[c]} = {alls[c]};' for c in range(k)] + [
        '                #pragma omp barrier',
        '            } else {',
        f'                __scan_block(__lo, __hi, {call_carries});',
        '            }',
        '        }',
        '    }',
        '} else {',
        f'    __scan_block(0, __n, {call_seeds});',
        '}',
        '}',
        '}',
    ])


def _multi_chain_tasklet(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG, parallel: bool) -> nodes.Tasklet:
    """``node.chains`` INDEPENDENT scans over the same index range, lowered as ONE loop.

    OpenMP 5.0 allows at most one ``scan`` directive per loop, but places no limit
    on how many list items its ``inclusive``/``exclusive`` clause -- and the
    enclosing ``reduction(inscan, op: ...)`` clause -- may carry, so K independent
    chains fit in a single ``omp parallel for simd`` loop: one fork/join, one pass
    over the index space, K accumulators live in registers. (List items must be
    scalars: an array section is explicitly rejected in an ``inscan`` reduction, so
    the accumulator names are spelled out here at expansion time rather than living
    behind a K-templated routine in ``dace/scan.hpp``.)

    ``parallel=False`` emits the same loop without the pragmas -- the shape used
    when the scan is already inside a parallel scope.

    FP: reassociation happens only WITHIN each chain's own combining op, and only
    on the parallel shape, where the ``inscan`` lowering folds chunk-wise (see the
    header's note); the chains never mix.
    """
    dtype = _validate_inputs_and_outputs(node, state, sdfg)[0].dtype
    ctype = dtype.ctype
    if symbolic.pystr_to_symbolic(sym2cpp(node.stride)) != 1:
        raise NotImplementedError("Scan: ``chains > 1`` with ``stride > 1`` is not supported; emit one "
                                  "Scan libnode per strided chain.")
    k = node.chains
    n_expr = _resolve_length(node, state, sdfg)
    accs = [f'__acc{c}' for c in range(k)]
    acc_list = ', '.join(accs)
    seeds = [_chain_seed_expr(node, state, sdfg, c, ctype) for c in range(k)]

    updates = '\n'.join(f'        {accs[c]} = {_combine_expr(node.op, ctype, accs[c], f"{in_connector(c)}[__i]")};'
                        for c in range(k))
    stores = '\n'.join(f'        {out_connector(c)}[__i] = {accs[c]};' for c in range(k))
    if node.exclusive:
        first, second = stores, updates
        scan_kind = 'exclusive'
    else:
        first, second = updates, stores
        scan_kind = 'inclusive'
    if not parallel:
        # Sequential shape: the plain loop, no pragmas, no blocking. ``min``/``max``
        # seed from the chain's first element, so an empty range must not run at all.
        decls = '\n'.join(f'    {ctype} {accs[c]} = {seeds[c]};' for c in range(k))
        code = (f'{{\n'
                f'    const long __n = static_cast<long>({n_expr});\n'
                f'    if (__n > 0) {{\n'
                f'{decls}\n'
                f'    for (long __i = 0; __i < __n; ++__i) {{\n'
                f'{first}\n'
                f'{second}\n'
                f'    }}\n'
                f'    }}\n'
                f'}}')
    else:
        code = _multi_chain_parallel_code(node, ctype, n_expr, accs, acc_list, seeds, first, second, scan_kind,
                                          _multi_chain_udr(node.op, dtype, ctype))
    inputs = {in_connector(c): None for c in range(node.chains)}
    inputs.update({init_connector(c): None for c in range(node.chains) if init_connector(c) in node.in_connectors})
    return nodes.Tasklet(node.name,
                         inputs=inputs,
                         outputs={out_connector(c): None
                                  for c in range(node.chains)},
                         code=code,
                         language=dace.Language.CPP)


#: Map op enum to the identity the header's single-block scan is seeded with when the caller
#: supplies none. ``min``/``max`` have no C++ literal identity, so the header's own neutral
#: elements are named instead of a number.
_OP_TO_SEED_CPP = {
    ScanOp.SUM: '{ct}(0)',
    ScanOp.PRODUCT: '{ct}(1)',
    ScanOp.MIN: '::dace::scan::detail::min_identity<{ct}>()',
    ScanOp.MAX: '::dace::scan::detail::max_identity<{ct}>()',
}


def affine_scan_body(node: "Scan", ctype: str, n_expr: str, parallel: bool) -> str:
    """Body for ``out[k] = c[k] * out[k-1] + d[k]``, entered at ``_scan_init`` (or 0).

    ``parallel`` picks the blocked runtime entry point over the naked loop. The two agree
    exactly within a block -- the blocked form's seeded pass IS this loop -- so they differ
    only in the association at block boundaries, same contract as every other op here.

    :param node: the Scan node, read for ``_scan_init``.
    :param ctype: the accumulator's C type, which is the output element type.
    :param n_expr: C++ expression for the element count.
    :param parallel: emit the blocked runtime call rather than the sequential loop.
    """
    seed = INIT_CONNECTOR_NAME if _has_init(node) else f'static_cast<{ctype}>(0)'
    if parallel:
        return (f'::dace::scan::inclusive_affine({COEF_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME}, '
                f'{OUTPUT_CONNECTOR_NAME}, static_cast<long>({n_expr}), {seed});')
    return (f'{{ const long _n = static_cast<long>({n_expr});\n'
            f'  {ctype} _acc = {seed};\n'
            f'  for (long _k = 0; _k < _n; ++_k) {{\n'
            f'      _acc = {COEF_CONNECTOR_NAME}[_k] * _acc + {INPUT_CONNECTOR_NAME}[_k];\n'
            f'      {OUTPUT_CONNECTOR_NAME}[_k] = _acc;\n'
            f'  }}\n'
            f'}}')


def degenerate_affine_tasklet(node: "Scan") -> nodes.Tasklet:
    """The one-element affine scan: ``out[0] = c[0]*seed + d[0]``, as a PYTHON tasklet.

    A statically length-1 subset arrives at the codegen scalar-typed, not as a pointer, so the
    C++ loop shape would index a scalar. The four scalar ops hit the same wall and answer it the
    same way (:func:`_degenerate_single_element_tasklet`); affine needs its own because the
    answer is an expression over two inputs rather than a copy.
    """
    seed = INIT_CONNECTOR_NAME if _has_init(node) else '0'
    inputs = {INPUT_CONNECTOR_NAME: None, COEF_CONNECTOR_NAME: None}
    if _has_init(node):
        inputs[INIT_CONNECTOR_NAME] = None
    return nodes.Tasklet(node.name,
                         inputs=inputs,
                         outputs={OUTPUT_CONNECTOR_NAME: None},
                         code=f'{OUTPUT_CONNECTOR_NAME} = {COEF_CONNECTOR_NAME} * {seed} + {INPUT_CONNECTOR_NAME}',
                         language=dace.Language.Python)


def affine_tasklet(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG, out_desc, n_expr: str,
                   parallel: bool) -> nodes.Tasklet:
    """Assemble the affine-scan tasklet with its coefficient (and optional init) connectors."""
    inputs = {INPUT_CONNECTOR_NAME: None, COEF_CONNECTOR_NAME: None}
    if _has_init(node):
        inputs[INIT_CONNECTOR_NAME] = None
    return nodes.Tasklet(node.name,
                         inputs=inputs,
                         outputs={OUTPUT_CONNECTOR_NAME: None},
                         code=affine_scan_body(node, out_desc.dtype.ctype, n_expr, parallel),
                         language=dace.Language.CPP)


def refuse_unsupported_affine_flags(node: "Scan") -> None:
    """Refuse the affine shapes whose lowering does not exist, before anything is emitted."""
    if node.op is not ScanOp.AFFINE:
        return
    if node.exclusive:
        refuse_affine_shape(node, '``exclusive=True``')
    if node.chains > 1:
        refuse_affine_shape(node, '``chains > 1``')
    if symbolic.pystr_to_symbolic(sym2cpp(node.stride)) != 1:
        refuse_affine_shape(node, '``stride > 1``')


def single_block_scan_call(op: ScanOp, exclusive: bool, n_expr: str, seed: str) -> str:
    """A unit-stride scan as ONE call into the runtime header's single-block routine.

    That routine is a ``#pragma omp simd reduction(inscan, op:acc)`` loop and nothing else: no
    parallel region, no allocation, no barrier -- which is what a scan that already sits inside an
    OpenMP region or a loop should be. It is also the SAME function the blocked parallel shape runs
    per block, so there is one implementation of the vector scan rather than one per call site.
    (The four op variants exist because an OpenMP reduction identifier cannot be a template
    parameter; ``complex`` works through them because the header declares its ``+``/``*`` UDRs in
    the same namespace, where unqualified lookup finds them.)

    Measured against the scalar dependent loop, fp64, one thread, GCC 15.2: 2.5x at n=1024, 2.5x at
    n=65536, 1.3x at n=8.4M. Clang 21 declines to vectorize the pragma and stays at 1.0x -- correct
    either way. FP association becomes the vector network's, not left-to-right, so a float result
    moves by ~3e-10 relative; ``min``/``max`` and every integer type stay exact.

    :param op: the scan's binary operator.
    :param exclusive: call the exclusive variant.
    :param n_expr: C++ expression for the element count.
    :param seed: C++ expression the accumulator starts at; it takes part in the prefix.
    :returns: the tasklet body.
    """
    kind = 'excl' if exclusive else 'incl'
    fn = f'::dace::scan::detail::scan_{kind}_{_OP_TO_OMP_SUFFIX[op]}'
    return (f'{fn}({INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, 0L, '
            f'static_cast<long>({n_expr}), {seed});')


@library.expansion
class ExpandPure(ExpandTransformation):
    """Portable fallback: a hand-written single-loop scan."""

    environments = [CPUEnv]

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        refuse_unsupported_affine_flags(node)
        in_desc, out_desc, in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        if node.op is ScanOp.AFFINE:
            refuse_widening(node, in_desc, out_desc, 'op=AFFINE')
            if _is_length_one(node, state):
                return degenerate_affine_tasklet(node)
            return affine_tasklet(node, state, sdfg, out_desc, _resolve_length(node, state, sdfg), parallel=False)
        if node.chains > 1:
            refuse_widening(node, in_desc, out_desc, 'chains > 1')
            return _multi_chain_tasklet(node, state, sdfg, parallel=False)
        if _is_length_one(node, state):
            return _degenerate_single_element_tasklet(node, in_desc)
        n_expr = _resolve_length(node, state, sdfg)
        op_cpp = _OP_TO_STD_CPP[node.op]
        # The ACCUMULATOR's type, which is the output's: a widening scan reads a narrower input.
        ctype = out_desc.dtype.ctype
        stride_expr = sym2cpp(node.stride)
        is_stride_one = (symbolic.pystr_to_symbolic(stride_expr) == 1)

        if not is_stride_one:
            refuse_widening(node, in_desc, out_desc, 'stride > 1')
            # No ``inscan`` here: the reduction spans one canonical loop, and a strided scan is one
            # independent chain PER RESIDUE CLASS, so the vectorizable axis is across classes -- not
            # the axis the dependence runs along. The classes stay scalar.
            if node.exclusive:
                raise NotImplementedError("Scan(pure): exclusive with stride > 1 is not supported.")
            # Outer loop over residue classes ``_k in [0, s)``; inner sequential scan.
            # Initialise the accumulator from the first valid input in each class so we
            # don't need a per-op identity literal (matches the residue-class oracle).
            body = (f"{{ long _s = (long)({stride_expr}); long _n = (long)({n_expr});\n"
                    f"  if (_s <= 0) std::abort();\n"
                    f"  for (long _k = 0; _k < _s; ++_k) {{\n"
                    f"      if (_k >= _n) continue;\n"
                    f"      auto _acc = {INPUT_CONNECTOR_NAME}[_k];\n"
                    f"      {OUTPUT_CONNECTOR_NAME}[_k] = _acc;\n"
                    f"      for (long _j = _k + _s; _j < _n; _j += _s) {{\n"
                    f"          _acc = ({op_cpp})(_acc, {INPUT_CONNECTOR_NAME}[_j]);\n"
                    f"          {OUTPUT_CONNECTOR_NAME}[_j] = _acc;\n"
                    f"      }}\n"
                    f"  }}\n"
                    f"}}")
        elif node.exclusive:
            body = single_block_scan_call(node.op, True, n_expr, _identity_expr(node, out_desc))
        elif _has_init(node):
            # Inclusive with an explicit init: ``out[k] = init OP in[0] OP ... OP in[k]``. The seed
            # is the accumulator's entry value, which the inscan prefix carries.
            body = single_block_scan_call(node.op, False, n_expr, INIT_CONNECTOR_NAME)
        else:
            body = single_block_scan_call(node.op, False, n_expr, _OP_TO_SEED_CPP[node.op].format(ct=ctype))
        inputs = {INPUT_CONNECTOR_NAME: None}
        if _has_init(node):
            inputs[INIT_CONNECTOR_NAME] = None
        return nodes.Tasklet(
            node.name,
            inputs=inputs,
            outputs={OUTPUT_CONNECTOR_NAME: None},
            code=body,
            language=dace.Language.CPP,
        )


@library.expansion
class ExpandCPU(ExpandTransformation):
    """PARALLEL-schedule scan: the blocked ``dace::scan`` runtime header.

    Emits a single call into ``dace::scan::{inclusive,exclusive}_{sum,product,min,max}``
    or ``dace::scan::strided_inclusive_*``. The unit-stride entry points run a
    blocked three-phase scan -- per-thread block folds, a prefix over the block
    totals, then a seeded ``simd`` ``inscan`` pass writing ``out`` -- over one
    parallel region, tiled so the fold pass and the scan pass share an L2-resident
    block. Work O(2N), depth O(N/P + log P), DRAM traffic 1R + 1W.

    NOT ``#pragma omp parallel for simd reduction(inscan, ...)``: GCC lowers that
    composite to a per-thread ``malloc`` of the whole chunk plus a scratch
    round-trip, which measured 0.62x against a plain sequential scan at n=1e6 and
    0.03x at n=137. See the header for the full measurement.

    The blocking makes the floating-point association depend on the team size, so
    results MOVE WITH ``OMP_NUM_THREADS`` above the header's parallel threshold.
    Callers that need a reproducible scan want the SEQUENTIAL shape instead --
    :class:`ExpandPure`, a plain loop with no pragma and no call into the header.
    """

    environments = [CPUEnv]

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        refuse_unsupported_affine_flags(node)
        in_desc, out_desc, _in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        # SCOPE decides the shape, not ``node.schedule``: that is storage-derived, so a Scan
        # nested in a parallel map (directly, or one level down through a NestedSDFG) arrives
        # carrying ``CPU_Multicore``. A re-entered node opens no region of its own.
        from dace.transformation.auto.auto_optimize import libnode_is_sequential
        if libnode_is_sequential(node, state, sdfg):
            # Already inside an OpenMP region or a loop: take the sequential naked-loop shape.
            return ExpandPure.expansion(node, state, sdfg)
        if node.op is ScanOp.AFFINE:
            refuse_widening(node, in_desc, out_desc, 'op=AFFINE')
            if _is_length_one(node, state):
                return degenerate_affine_tasklet(node)
            return affine_tasklet(node, state, sdfg, out_desc, _resolve_length(node, state, sdfg), parallel=True)
        if node.chains > 1:
            # K independent chains, ONE ``inscan`` loop == one fork/join. See
            # :func:`_multi_chain_tasklet` for the OpenMP-spec argument.
            refuse_widening(node, in_desc, out_desc, 'chains > 1')
            return _multi_chain_tasklet(node, state, sdfg, parallel=True)
        if _is_length_one(node, state):
            return _degenerate_single_element_tasklet(node, in_desc)
        n_expr = _resolve_length(node, state, sdfg)
        suffix = _OP_TO_OMP_SUFFIX[node.op]
        stride_expr = sym2cpp(node.stride)
        is_stride_one = (symbolic.pystr_to_symbolic(stride_expr) == 1)

        if not is_stride_one:
            refuse_widening(node, in_desc, out_desc, 'stride > 1')
            if node.exclusive:
                raise NotImplementedError("Scan: ``exclusive=True`` with ``stride > 1`` is not yet supported.")
            if _has_init(node):
                raise NotImplementedError("Scan: ``_scan_init`` with ``stride > 1`` is not yet supported.")
            call = (f"::dace::scan::strided_inclusive_{suffix}("
                    f"{INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, ({n_expr}), ({stride_expr}));")
        elif node.exclusive:
            seed = _identity_expr(node, out_desc)
            call = (f"::dace::scan::exclusive_{suffix}("
                    f"{INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n_expr}), "
                    f"{OUTPUT_CONNECTOR_NAME}, {seed});")
        else:
            # A wired ``_scan_init`` is just the accumulator's starting value, which the
            # runtime's seeded overload takes directly -- it is the SAME blocked parallel
            # scan, not the sequential ``std::inclusive_scan`` this used to fall back to.
            init = f", {INIT_CONNECTOR_NAME}" if _has_init(node) else ""
            call = (f"::dace::scan::inclusive_{suffix}("
                    f"{INPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME} + ({n_expr}), "
                    f"{OUTPUT_CONNECTOR_NAME}{init});")
        inputs = {INPUT_CONNECTOR_NAME: None}
        if _has_init(node):
            inputs[INIT_CONNECTOR_NAME] = None
        return nodes.Tasklet(
            node.name,
            inputs=inputs,
            outputs={OUTPUT_CONNECTOR_NAME: None},
            code=call,
            language=dace.Language.CPP,
        )


def affine_cuda_tasklet(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG, out_desc) -> nodes.Tasklet:
    """The first-order linear recurrence on the device, as a call into ``dace::cuda_scan``.

    The header's monoid is affine-map composition, so the recurrence is a plain cub prefix scan
    over the maps -- see :file:`dace/runtime/include/dace/cuda/scan_affine.cuh` for why the seed is
    folded into element 0 rather than handed to cub as an init value. Only the shape
    :func:`refuse_unsupported_affine_flags` admits arrives here: one chain, unit stride, inclusive.

    The seed reaches the wrapper twice over, as a pointer and as a value, and exactly one of the
    two is live: a device-resident seed must not be dereferenced by the host code issuing the
    launch, and a host-readable one has no device address to hand over.
    """
    coef = coef_desc(node, state, sdfg)
    in_desc = sdfg.arrays[next(e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME).data.data]
    e_ctype = out_desc.dtype.base_type.ctype
    c_ctype = coef.dtype.base_type.ctype
    d_ctype = in_desc.dtype.base_type.ctype

    seed = seed_desc(node, state, sdfg, 0) if _has_init(node) else None
    on_device = seed is not None and seed.storage in GPU_RESIDENT_STORAGES
    s_ctype = seed.dtype.base_type.ctype if seed is not None else e_ctype
    seed_ptr = init_connector(0) if on_device else f'static_cast<const {s_ctype}*>(nullptr)'
    live_seed_value = seed is not None and not on_device
    seed_val = f'static_cast<{e_ctype}>({init_connector(0)})' if live_seed_value else f'static_cast<{e_ctype}>(0)'

    state_id = state.parent_graph.node_id(state)
    wrapper = f'__dace_scan_affine_{sdfg.name}_{state_id}_{state.node_id(node)}'
    params = (f'const {c_ctype}* __sc_c, const {d_ctype}* __sc_d, const {s_ctype}* __sc_seed_ptr, '
              f'{e_ctype} __sc_seed_val, {e_ctype}* __sc_out, long long __sc_n, gpuStream_t __sc_stream')
    prototype = f'DACE_EXPORTED gpuError_t {wrapper}({params});'
    sdfg.append_global_code(prototype + '\n')
    sdfg.append_global_code(
        f'{prototype}\n'
        f'gpuError_t {wrapper}({params}) {{\n'
        f'    return ::dace::cuda_scan::inclusive_affine<{e_ctype}, {c_ctype}, {d_ctype}, {s_ctype}>(\n'
        f'        __sc_c, __sc_d, __sc_seed_ptr, __sc_seed_val, __sc_out, __sc_n, __sc_stream);\n'
        f'}}\n', 'cuda')

    inputs = {INPUT_CONNECTOR_NAME: None, COEF_CONNECTOR_NAME: None}
    if _has_init(node):
        inputs[init_connector(0)] = dtypes.pointer(seed.dtype.base_type) if on_device else None
    code = (f'DACE_GPU_CHECK({wrapper}({COEF_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME}, {seed_ptr}, {seed_val}, '
            f'{OUTPUT_CONNECTOR_NAME}, ({_resolve_length(node, state, sdfg)}), __dace_current_stream));')
    return nodes.Tasklet(node.name,
                         inputs=inputs,
                         outputs={OUTPUT_CONNECTOR_NAME: None},
                         code=code,
                         language=dace.Language.CPP)


def strided_cuda_tasklet(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG, out_desc) -> nodes.Tasklet:
    """``s`` independent residue-class scans, one device thread per class.

    ``gpucub::DeviceScan`` scans one contiguous sequence, so it cannot serve a stride: CUB has a
    segmented reduce and a segmented sort but no segmented scan, and driving the residue classes
    through ``s`` separate strided-iterator ``DeviceScan`` calls would be ``s`` kernel launches for
    the shape that produces most of them (``LoopToScan``'s composite body, where ``s`` is the inner
    size and runs to 1e5). One launch over ``dace::cuda_scan::strided_inclusive_<op>``
    (:file:`dace/runtime/include/dace/cuda/scan.cuh`) walks every class in parallel instead.

    Emitted as a wrapper in the CUDA translation unit and CALLED from the host tasklet, the same
    shape :func:`affine_cuda_tasklet` and the cub path use: the kernel launch is nvcc/hipcc-only
    syntax, and the Scan libnode sits at host schedule. Templated at the point of emission, so
    every dtype the header accepts works.
    """
    if node.chains != 1:
        raise NotImplementedError("Scan(CUDA, stride > 1): multi-chain scans are not yet supported.")
    if node.exclusive:
        raise NotImplementedError("Scan(CUDA, stride > 1): ``exclusive=True`` is not yet supported.")
    if _has_init(node):
        raise NotImplementedError("Scan(CUDA, stride > 1): ``_scan_init`` is not yet supported.")
    ctype = out_desc.dtype.base_type.ctype
    suffix = _OP_TO_OMP_SUFFIX[node.op]
    state_id = state.parent_graph.node_id(state)
    wrapper = f'__dace_scan_strided_{sdfg.name}_{state_id}_{state.node_id(node)}'
    params = f'const {ctype}* __sc_in, {ctype}* __sc_out, long __sc_n, long __sc_s, gpuStream_t __sc_stream'
    prototype = f'DACE_EXPORTED gpuError_t {wrapper}({params});'
    sdfg.append_global_code(prototype + '\n')
    sdfg.append_global_code(
        f'{prototype}\n'
        f'gpuError_t {wrapper}({params}) {{\n'
        f'    ::dace::cuda_scan::strided_inclusive_{suffix}<{ctype}>('
        f'__sc_in, __sc_out, __sc_n, __sc_s, __sc_stream);\n'
        f'    return gpuGetLastError();\n'
        f'}}\n', 'cuda')
    code = (f'DACE_GPU_CHECK({wrapper}({INPUT_CONNECTOR_NAME}, {OUTPUT_CONNECTOR_NAME}, '
            f'(long)({_resolve_length(node, state, sdfg)}), (long)({sym2cpp(node.stride)}), '
            f'__dace_current_stream));')
    return nodes.Tasklet(node.name,
                         inputs={INPUT_CONNECTOR_NAME: None},
                         outputs={OUTPUT_CONNECTOR_NAME: None},
                         code=code,
                         language=dace.Language.CPP)


#: Threads per block for the in-kernel collectives. Four wavefronts on CDNA (64 wide), eight warps
#: on NVIDIA (32 wide): wide enough that ``gpucub::BlockScan``'s cross-warp step has work to do,
#: narrow enough that a short row does not leave most of the block idle.
BLOCK_COLLECTIVE_THREADS = 256


@library.expansion
class ExpandCUDABlock(ExpandTransformation):
    """One thread BLOCK scans the whole range: ``gpucub::BlockScan`` fed by a block-strided loop.

    :class:`ExpandCUDA` issues ``gpucub::DeviceScan``, which is a HOST call. A ``Scan`` that lands
    INSIDE a kernel therefore had only ``pure`` to fall back on -- one thread walking every element
    -- and that fallback is self-perpetuating: :func:`~dace.transformation.passes.offloading.
    taskloop.encloses_device_wide_libnode` keeps a map on the host precisely BECAUSE it holds a
    device-wide library node, so the per-row scan
    (``for i: for j: b[i, j] = b[i, j - 1] + a[i, j]``) becomes one host-issued ``DeviceScan``
    launch per row rather than one kernel. This expansion is what lets that row map be a kernel.

    The emitted subgraph is a ``GPU_ThreadBlock`` map around a single call to
    ``dace::cuda_scan::detail::block_inclusive_scan_strided`` (:file:`dace/runtime/include/dace/
    cuda/scan.cuh`), the same collective the residue-class kernel is built from. The map's
    parameter is deliberately unused: the collective indexes threads through ``threadIdx`` the way
    CUB itself does, and the map is here to tell the code generator two things it can learn no
    other way -- that the enclosing device map runs one iteration per BLOCK rather than per thread,
    and that the block is :data:`BLOCK_COLLECTIVE_THREADS` wide (``get_kernel_dimensions`` reads
    the block size off contained thread-block maps).

    Refuses rather than approximates: the exclusive, multi-chain, affine and explicit-init shapes
    all fall through to another expansion instead of being silently lowered as something else.
    """

    environments = []

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> dace.SDFG:
        if not ExpandCUDABlock.environments:
            from dace.libraries.sort.environments.cub import BlockCollectives
            ExpandCUDABlock.environments = [BlockCollectives]
        in_desc, out_desc, in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        if node.op is ScanOp.AFFINE:
            raise NotImplementedError("Scan(CUDA (block)): op=AFFINE is not supported.")
        if node.chains != 1:
            raise NotImplementedError("Scan(CUDA (block)): multi-chain scans are not supported.")
        if node.exclusive:
            raise NotImplementedError("Scan(CUDA (block)): exclusive scans are not supported.")
        if _has_init(node):
            raise NotImplementedError("Scan(CUDA (block)): an explicit ``_scan_init`` is not supported.")
        # The collective accumulates in ONE type, the array's; a widening scan would accumulate at
        # the input's width and lose the range the wider output was asked for.
        refuse_widening(node, in_desc, out_desc, 'the CUDA (block) expansion')

        n_expr = _resolve_length(node, state, sdfg)
        stride_expr = sym2cpp(node.stride)
        ctype = out_desc.dtype.base_type.ctype
        op_functor = f'::dace::cuda_scan::detail::Scan{node.op.value.capitalize()}<{ctype}>'
        identity = _OP_TO_IDENTITY_CPP[node.op]
        if identity is None:
            # min/max have no universal identity, so the out-of-range lanes have nothing safe to
            # read. Refusing sends the node to ``pure``, which is slow rather than wrong.
            raise NotImplementedError(f"Scan(CUDA (block)): op {node.op.value!r} has no identity to pad "
                                      "the final partial chunk with.")

        n_sym = in_edge.data.subset.num_elements()
        nsdfg = dace.SDFG(node.label + '_block')
        nsdfg.add_array(INPUT_CONNECTOR_NAME, [n_sym], in_desc.dtype, storage=in_desc.storage)
        nsdfg.add_array(OUTPUT_CONNECTOR_NAME, [n_sym], out_desc.dtype, storage=out_desc.storage)
        nstate = nsdfg.add_state(node.label + '_block_state')
        read = nstate.add_read(INPUT_CONNECTOR_NAME)
        write = nstate.add_write(OUTPUT_CONNECTOR_NAME)
        code = (f'::dace::cuda_scan::detail::block_inclusive_scan_strided'
                f'<{ctype}, {op_functor}, {BLOCK_COLLECTIVE_THREADS}>('
                f'__bsin, __bsout, '
                f'(long)({n_expr}), (long)({stride_expr}), {op_functor}(), '
                f'static_cast<{ctype}>({identity}));')
        tasklet = nstate.add_tasklet(node.label + '_block_scan',
                                     inputs={'__bsin': dtypes.pointer(in_desc.dtype.base_type)},
                                     outputs={'__bsout': dtypes.pointer(out_desc.dtype.base_type)},
                                     code=code,
                                     language=dace.Language.CPP)
        entry, exit_node = nstate.add_map(node.label + '_block_lanes', {'__lane': f'0:{BLOCK_COLLECTIVE_THREADS}'},
                                          schedule=dtypes.ScheduleType.GPU_ThreadBlock)
        # EVERY lane sees the WHOLE range -- the map does not partition the data, it only supplies
        # the threads. Slicing by ``__lane`` here would hand each thread its own scan.
        whole = dace.Memlet.simple(INPUT_CONNECTOR_NAME, f'0:{sym2cpp(n_sym)}')
        nstate.add_memlet_path(read, entry, tasklet, dst_conn='__bsin', memlet=whole)
        nstate.add_memlet_path(tasklet,
                               exit_node,
                               write,
                               src_conn='__bsout',
                               memlet=dace.Memlet.simple(OUTPUT_CONNECTOR_NAME, f'0:{sym2cpp(n_sym)}'))
        return nsdfg


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """``gpucub::DeviceScan::InclusiveScan`` / ``ExclusiveScan`` over device-global memory.

    Temporary storage is obtained from the per-libnode-class, per-stream CUB scratch pool
    tagged ``ScanTag`` (see :file:`dace/runtime/include/dace/cub_scratch.cuh` and the
    :class:`ScanScratch` environment): the default-stream entry is pre-allocated to 128 MB
    at SDFG init; additional streams allocate lazily on first use. Each per-stream entry is
    reused across every ``Scan`` call on that stream, grown in place if a request exceeds
    the current allocation, and released at SDFG exit. The libnode threads
    ``__dace_current_stream`` to both the scratch lookup and the underlying ``gpucub::DeviceScan``
    call, so concurrent launches on different streams cannot race on the pool.
    """

    # Populated lazily in :meth:`expansion` (and below) to dodge the sort<->standard cycle.
    environments = []

    @staticmethod
    def expansion(node: "Scan", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        if not ExpandCUDA.environments:
            from dace.libraries.sort.environments.cub import ScanScratch
            ExpandCUDA.environments = [ScanScratch]
        in_desc, out_desc, _in_edge, _out_edge = _validate_inputs_and_outputs(node, state, sdfg)
        if node.op is ScanOp.AFFINE:
            refuse_unsupported_affine_flags(node)
            if _is_length_one(node, state):
                return degenerate_affine_tasklet(node)
            return affine_cuda_tasklet(node, state, sdfg, out_desc)
        # A widening scan is safe here exactly where the accumulator's type is pinned by an argument
        # this expansion controls. ``ExclusiveScan`` deduces ``AccumT`` from the init value, so
        # seeding at the OUTPUT type accumulates there. An inclusive scan has no such argument -- it
        # deduces from the input iterator, and a device-resident seed arrives as a ``FutureValue`` of
        # the seed's own type -- so that shape still refuses rather than accumulate narrow.
        if not node.exclusive:
            refuse_widening(node, in_desc, out_desc, 'the CUDA expansion without an exclusive seed')
        if _is_length_one(node, state):
            return _degenerate_single_element_tasklet(node, in_desc)
        n_expr = _resolve_length(node, state, sdfg)
        op_cub = _OP_TO_CUB[node.op]
        stride_expr = sym2cpp(node.stride)
        is_stride_one = (symbolic.pystr_to_symbolic(stride_expr) == 1)

        if not is_stride_one:
            # ``gpucub::DeviceScan`` walks one contiguous sequence and would run past each
            # residue's boundary, so stride > 1 takes the residue-class kernel instead. Same
            # implementation key: a caller that asks for the GPU should not also have to know
            # the stride, and the fast-library priority lists only ever name ``CUDA``.
            return strided_cuda_tasklet(node, state, sdfg, out_desc)

        # The chains are independent, so on the device they stay independent cub
        # launches -- the CPU-side fork/join fusion the multi-chain shape exists for
        # has no GPU analogue (a kernel launch is not a parallel region).
        # ``cub/cub.cuh`` is a CUDA header: the host translation unit is compiled by the host
        # compiler and cannot parse it, which is why the CUB call cannot be emitted here directly.
        # Emit a wrapper into the CUDA unit and CALL it from the host tasklet -- the same shape
        # ``ExpandFindFirstCUDA`` uses for ``dace::find_first_index_device``.
        state_id = state.parent_graph.node_id(state)
        idstr = f'{sdfg.name}_{state_id}_{state.node_id(node)}'
        in_ctype, out_ctype = in_desc.dtype.base_type.ctype, out_desc.dtype.base_type.ctype
        blocks = []
        for chain in range(node.chains):
            in_conn, out_conn = in_connector(chain), out_connector(chain)
            seed_param, seed_expr, seed_actual = '', '', ''
            if node.exclusive:
                # The OUTPUT descriptor, not the input: this argument is what fixes cub's accumulator
                # width, and on a widening scan the accumulator is the output's type.
                seed_expr = _identity_expr(node, out_desc)
                call = 'ExclusiveScan'
                extra = f', {seed_expr}'
            elif _has_init(node, chain):
                # Inclusive scan with init. ``gpucub::DeviceScan::InclusiveScanInit`` is the
                # direct API (CUB >= 2.0 / CUDA 12+); on older CUB it'd need an
                # ``ExclusiveScan`` + tail-add fallback, which can be added when
                # supporting CUDA 11 becomes a requirement.
                # A seed the host cannot read is passed as a ``gpucub::FutureValue``, which cub
                # dereferences on the device; a host-resident one goes by value as before.
                desc = seed_desc(node, state, sdfg, chain)
                seed_ctype = desc.dtype.base_type.ctype
                if desc is not None and desc.storage in GPU_RESIDENT_STORAGES:
                    seed_param = f', const {seed_ctype}* __sc_init'
                    extra = f', ::gpucub::FutureValue<{seed_ctype}>(__sc_init)'
                else:
                    seed_param = f', {seed_ctype} __sc_init'
                    extra = ', __sc_init'
                seed_actual = f', {init_connector(chain)}'
                call = 'InclusiveScanInit'
            else:
                call = 'InclusiveScan'
                extra = ''

            wrapper = f'__dace_scan_{idstr}_c{chain}'
            params = (f'const {in_ctype}* __sc_in, {out_ctype}* __sc_out{seed_param}, '
                      f'long long __sc_n, gpuStream_t __sc_stream')
            prototype = f'DACE_EXPORTED gpuError_t {wrapper}({params});'
            args = f'__sc_in, __sc_out, {op_cub}{extra}, __sc_n, __sc_stream'
            sdfg.append_global_code(prototype + '\n')
            sdfg.append_global_code(
                f'{prototype}\n'
                f'gpuError_t {wrapper}({params}) {{\n'
                f'    size_t _sc_needed = 0;\n'
                f'    ::gpucub::DeviceScan::{call}(nullptr, _sc_needed, {args});\n'
                f'    void* _sc_scratch = ::dace::cub::get_scratch<::dace::cub::ScanTag>('
                f'_sc_needed, __sc_stream);\n'
                f'    return ::gpucub::DeviceScan::{call}(_sc_scratch, _sc_needed, {args});\n'
                f'}}\n', 'cuda')
            blocks.append(f'DACE_GPU_CHECK({wrapper}({in_conn}, {out_conn}{seed_actual}, '
                          f'({n_expr}), __dace_current_stream));')
        inputs = {in_connector(c): None for c in range(node.chains)}
        # A device-resident seed reaches ``FutureValue`` as a POINTER; a scalar connector would be
        # dereferenced by the host code issuing the launch, which validation rejects.
        for chain in range(node.chains):
            if not _has_init(node, chain):
                continue
            desc = seed_desc(node, state, sdfg, chain)
            device = desc is not None and desc.storage in GPU_RESIDENT_STORAGES
            inputs[init_connector(chain)] = dtypes.pointer(desc.dtype.base_type) if device else None
        return nodes.Tasklet(
            node.name,
            inputs=inputs,
            outputs={out_connector(c): None
                     for c in range(node.chains)},
            code='\n'.join(blocks),
            language=dace.Language.CPP,
        )


@library.node
class Scan(nodes.LibraryNode):
    """Per-position prefix reduction over a 1-D array.

    Inputs / outputs:

    - ``_scan_in``:  input 1-D contiguous array of length ``N``. For ``op=AFFINE`` this is the
      recurrence's per-element DELTA ``d``.
    - ``_scan_coef``: (``op=AFFINE`` only) the per-element coefficient ``c``, same length/dtype.
    - ``_scan_out``: output 1-D contiguous array, same dtype, same shape.
    - chain ``c > 0`` (only when ``chains > 1``) adds ``_scan_in_c`` / ``_scan_out_c``
      and the optional ``_scan_init_c``: an INDEPENDENT scan over the same index range,
      lowered into the SAME OpenMP region as chain 0.

    Properties:

    - ``op``: one of :class:`ScanOp` (``SUM`` / ``PRODUCT`` / ``MIN`` / ``MAX`` / ``AFFINE``).
      ``AFFINE`` is the first-order linear recurrence ``out[k] = c[k]*out[k-1] + d[k]``; it
      carries the affine map ``x -> a*x + b`` instead of a value, reads the extra ``_scan_coef``
      array, and is supported on the host expansions at unit stride, single chain, inclusive
      only. Every other shape refuses rather than falling back to a scalar op.
    - ``exclusive``: ``False`` (inclusive: ``out[k] = in[0] OP ... OP in[k]``);
      ``True`` (exclusive: ``out[0] = identity``, ``out[k] = identity OP in[0] OP ... OP in[k-1]``).
    - ``identity``: the exclusive-scan seed. Defaults to ``0`` for ``SUM`` and ``1`` for
      ``PRODUCT``; ``MIN``/``MAX`` exclusive scans require this to be set explicitly.

    Implementations:

    - ``'CPU'`` (default) -- ``std::inclusive_scan`` / ``std::exclusive_scan`` (C++17 ``<numeric>``),
      or ``dace::scan::inclusive_affine`` for ``op=AFFINE``.
    - ``'CUDA'``           -- ``gpucub::DeviceScan::InclusiveScan`` / ``ExclusiveScan``. No ``AFFINE``.
    - ``'pure'``           -- portable single-loop fallback.

    The libnode is contractually pure: no aliasing between ``in`` and ``out`` is required
    (and not assumed), and no other state is read or written.
    """

    INPUT_CONNECTOR_NAME = INPUT_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = OUTPUT_CONNECTOR_NAME
    #: Optional scalar input connector; wire to a length-1 / scalar read to make
    #: the inclusive scan fold an explicit init value in (``out[k] = init OP in[0]
    #: OP ... OP in[k]``). Lets LoopToScan skip its seed-add Map.
    INIT_CONNECTOR_NAME = INIT_CONNECTOR_NAME

    op = EnumProperty(dtype=ScanOp, default=ScanOp.SUM, desc="Associative binary op for the scan.")
    exclusive = Property(dtype=bool, default=False, desc="If True, output an exclusive scan (out[0] = identity).")
    identity = Property(dtype=object,
                        default=None,
                        allow_none=True,
                        desc="Exclusive-scan identity element. Required for MIN/MAX exclusive scans.")
    chains = Property(dtype=int,
                      default=1,
                      desc="Number of INDEPENDENT scan chains carried by this node. Chain ``c > 0`` "
                      "uses the suffixed connectors ``_scan_in_c`` / ``_scan_out_c`` / ``_scan_init_c``; "
                      "chain 0 keeps the bare names. All chains share op, exclusivity, stride and "
                      "element count, and the parallel CPU expansion lowers them as list items of ONE "
                      "``reduction(inscan, op: ...)`` clause -- K carry chains, one fork/join, one pass "
                      "over the index space. Unit stride only.")

    stride = Property(dtype=object,
                      default=1,
                      allow_none=False,
                      desc="Per-element stride for the scan recurrence. Default ``1`` is the "
                      "contiguous case (``out[i+1] = out[i] OP in[i]``). Values ``s > 1`` express "
                      "``out[i+s] = out[i] OP in[i]``: the ``s`` residue classes mod ``s`` form "
                      "independent scans. The parallel CPU expansion splits the residue classes "
                      "across ONE parallel region and walks the strided space in place (no packed "
                      "copy and no region per class). "
                      "The expansion emits a runtime ``s > 0`` ``std::abort()`` check; passing a "
                      "non-positive stride at runtime terminates the program before the scan "
                      "starts. Exclusive strided scans (``exclusive=True`` with ``stride > 1``) "
                      "are not yet supported.")

    implementations = {
        "CPU": ExpandCPU,
        "CUDA": ExpandCUDA,
        "CUDA (block)": ExpandCUDABlock,
        "pure": ExpandPure,
    }
    default_implementation = 'CPU'

    def __init__(self,
                 name: str = 'Scan',
                 op: ScanOp = ScanOp.SUM,
                 exclusive: bool = False,
                 identity=None,
                 chains: int = 1,
                 *args,
                 **kwargs):
        # ``_scan_coef`` is part of the node's shape, not an optional extra like ``_scan_init``:
        # an affine scan with no coefficients is not a scan of some other kind, it is unwired.
        conns = {in_connector(c): None for c in range(chains)}
        if op is ScanOp.AFFINE:
            conns.update({coef_connector(c): None for c in range(chains)})
        super().__init__(name, *args, inputs=conns, outputs={out_connector(c): None for c in range(chains)}, **kwargs)
        self.op = op
        self.exclusive = exclusive
        self.identity = identity
        # Every Scan carries the same device trade, whichever rewrite produced it, so this is set
        # once here rather than at each of LoopToScan's construction sites. A scan does more work
        # than the sequential loop it replaces: it wins where there are threads to spare and loses
        # where the loop was already the cheap way to spend a core. Canonicalization takes the
        # parallel form and records the reverse for a specializing pass to consider.
        self.specialization_hint = ('parallel scan; canonicalization takes the parallel form.\n'
                                    'Alternative: a sequential loop over parallel maps.\n'
                                    'CPU: the loop is worth trying -- the scan does more work, and the loop '
                                    'may already saturate the memory system.\n'
                                    'GPU: the scan is usually the better of the two.\n'
                                    'Both are correct. Measure before choosing.')
        self.chains = chains

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        _validate_inputs_and_outputs(self, state, sdfg)
