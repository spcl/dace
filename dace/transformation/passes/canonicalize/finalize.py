# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Post-canonicalization target finalization for performance.

:func:`~dace.transformation.passes.canonicalize.pipeline.canonicalize` leaves
an SDFG in a clean, vectorizer-friendly form: ``Reduce`` / ``MatMul`` library
nodes are kept un-expanded (``implementation=None``) so later passes see one
shape per computation. That form codegens to the naive library expansion,
which is materially slower than the tiled/fast implementations ``auto_optimize``
selects.

This module supplies the optimization tail to run *after* canonicalization when
the goal is raw runtime (not vectorization or equivalence checking): it picks
the fast library implementation per node and moves small/independent transients
to registers and persistent storage. The library nodes themselves stay
un-expanded -- ``compile()`` expands each one exactly once at codegen, using the
implementation selected here -- so one shape per computation survives all the way
to the backend. It mirrors ``auto_optimize``'s library-and-storage finalization
(everything fusion-related is already done by the canonicalize pipeline), so
``canonicalize(s); finalize_for_target(s)`` is the perf-path counterpart to
``auto_optimize(s)``.
"""
from dace import SDFG, dtypes
from dace.sdfg import infer_types, nodes
from dace.transformation.auto.auto_optimize import (make_transients_persistent, move_small_arrays_to_stack,
                                                    set_fast_implementations)
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars

#: Map the canonicalize target string to the codegen device type.
_TARGET_DEVICE = {'cpu': dtypes.DeviceType.CPU, 'gpu': dtypes.DeviceType.GPU}

#: Per-dimension matmul extent at or below which canonicalization picks the inlined
#: ``'pure'`` expansion (a fusible/vectorizable sequential loop nest) over a BLAS call.
_SMALL_MATMUL_DIM = 256

#: The row-wise (ikj) ``'pure'`` variant is selected only for *tiny* matmuls -- every
#: dimension at most this. Its win (a vectorizable row update with a sequential K loop) is
#: a register/cache-blocking effect that only pays off at very small sizes; a larger
#: "small" matmul keeps the plain ``'pure'`` nest.
_ROWWISE_MATMUL_DIM = 64


def _all_matmul_extents_small(state, node, limit: int) -> bool:
    """True iff every operand/output extent of a matmul library ``node`` is a constant
    at most ``limit`` (the matmul is known-small). A symbolic extent -> not known-small."""
    saw = False
    for e in list(state.in_edges(node)) + list(state.out_edges(node)):
        if e.data is None or e.data.subset is None:
            continue
        for ext in e.data.subset.size():
            saw = True
            try:
                if int(ext) > limit:
                    return False
            except (TypeError, ValueError):
                return False  # symbolic extent -> unknown size, treat as not-small
    return saw


def canonicalize_set_fast_implementations(sdfg: SDFG, device: dtypes.DeviceType, small_dim: int = _SMALL_MATMUL_DIM):
    """Select library-node implementations for the canonicalize perf tail.

    Delegates to :func:`~dace.transformation.auto.auto_optimize.set_fast_implementations`
    for the general (fast BLAS) case, then OVERRIDES any GEMM/MatMul whose every
    dimension is a known constant at most ``small_dim`` to the inlined ``'pure'``
    expansion. A tiny matmul's BLAS/cuBLAS call is pure overhead, and -- unlike an opaque
    library call -- a sequential loop nest is fusible/vectorizable and keeps a loop of
    small matmuls sequential instead of issuing serialized library calls. Symbolic- or
    large-dimensioned matmuls keep the fast BLAS implementation.
    """
    set_fast_implementations(sdfg, device)
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, nodes.LibraryNode):
            continue
        impls = type(node).implementations
        if 'pure' not in impls:
            continue
        if _all_matmul_extents_small(state, node, small_dim):
            # Prefer the row-wise (ikj) pure expansion for TINY GEMMs (every dim <= 64): a
            # vectorizable row update with a sequential K accumulation. Larger small matmuls,
            # and nodes without a 'rowwise' impl (e.g. MatMul), keep the plain 'pure' nest.
            if 'rowwise' in impls and _all_matmul_extents_small(state, node, _ROWWISE_MATMUL_DIM):
                node.implementation = 'rowwise'
            else:
                node.implementation = 'pure'


def finalize_transient_storage(sdfg: SDFG, device: dtypes.DeviceType) -> None:
    """Finalize the storage of a canonicalized (or vectorized) SDFG's transients, in place.

    The single home for transient storage finalization -- a value-preserving perf tail that any
    downstream producer (canonicalize's ``finalize_for_target``, or a caller that has additionally
    vectorized the SDFG) runs to allocate temporaries well. Three steps, mirroring
    ``auto_optimize``'s storage tail:

    1. **Length-1 transient arrays -> scalars** (:class:`ConvertLengthOneArraysToScalars`,
       ``transient_only=True``): a single internal value belongs in a scalar, not a heap array.
       Non-transient length-1 arrays (SDFG-external returns / opaque handles) are left as arrays.
    2. **Small constant-size scratch -> registers** (:func:`move_small_arrays_to_stack`).
    3. **Independent top-level transients -> ``Persistent`` lifetime**
       (:func:`make_transients_persistent`, ``toplevel_only=True``): a state-struct member
       allocated once in ``__dace_init`` / freed in ``__dace_exit`` instead of a per-call
       ``malloc``/``free``. ``toplevel_only`` + ``get_parent_map``'s walk up across every
       nested-SDFG boundary excludes any per-thread buffer inside a parallel map body, so it is
       never collapsed to one shared copy; on GPU it also resets non-atomic WCR edges.

    A persistent size-1 WCR accumulator would land as ``__state->x`` -- not a valid OpenMP
    ``reduction(op:var)`` lvalue, so the parallel reduction fails to compile. Revert each promoted
    size-1 transient to a scope-lifetime register (a stack scalar).

    :param sdfg: SDFG whose transient storage is finalized in place.
    :param device: codegen device type (selects GPU WCR-reset / storage rules).
    """
    ConvertLengthOneArraysToScalars(recursive=True, transient_only=True).apply_pass(sdfg, {})
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    move_small_arrays_to_stack(sdfg)
    made_persistent = make_transients_persistent(sdfg, device)
    cfg_by_id = {sd.cfg_id: sd for sd in sdfg.all_sdfgs_recursive()}
    for cfg_id, names in made_persistent.items():
        sd = cfg_by_id.get(cfg_id)
        if sd is None:
            continue
        for name in names:
            desc = sd.arrays.get(name)
            if desc is not None and desc.total_size == 1:
                desc.lifetime = dtypes.AllocationLifetime.Scope
                desc.storage = dtypes.StorageType.Register
    sdfg.reset_cfg_list()


def finalize_for_target(sdfg: SDFG, target: str = 'cpu', validate: bool = True) -> SDFG:
    """Apply the performance finalization tail to a canonicalized ``sdfg``.

    Selects fast library implementations (leaving the nodes un-expanded for
    codegen to lower), then moves small constant-size transients to the stack and
    independent transients to persistent allocation. Operates in place.

    :param sdfg: A canonicalized SDFG.
    :param target: ``'cpu'`` or ``'gpu'`` (selects the fast-library priority).
    :param validate: Validate the SDFG once at the end.
    :returns: The same ``sdfg`` instance, finalized.
    """
    if target not in _TARGET_DEVICE:
        raise ValueError(f"target must be one of {sorted(_TARGET_DEVICE)}; got {target!r}")
    device = _TARGET_DEVICE[target]

    canonicalize_set_fast_implementations(sdfg, device)
    # Select the fast implementation per library node but DO NOT expand here: a library
    # node is expanded exactly once, at codegen (``compile()`` auto-expands using the
    # ``implementation`` chosen above). Expanding in canonicalization would re-introduce
    # the per-implementation shapes (BLAS scratch, reduction accumulators) into a form the
    # rest of the toolchain must then re-canonicalize; keeping one shape per computation
    # until codegen is the invariant every downstream pass relies on.
    infer_types.infer_connector_types(sdfg)
    finalize_transient_storage(sdfg, device)

    if validate:
        sdfg.validate()
    return sdfg
