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
the fast library implementation per node, expands the library nodes, and moves
small/independent transients to registers and persistent storage. It mirrors
``auto_optimize``'s library-and-storage finalization (everything fusion-related
is already done by the canonicalize pipeline), so ``canonicalize(s); finalize_for_target(s)``
is the perf-path counterpart to ``auto_optimize(s)``.
"""
from dace import SDFG, dtypes
from dace.sdfg import infer_types
from dace.transformation.auto.auto_optimize import (make_transients_persistent, move_small_arrays_to_stack,
                                                    set_fast_implementations)

#: Map the canonicalize target string to the codegen device type.
_TARGET_DEVICE = {'cpu': dtypes.DeviceType.CPU, 'gpu': dtypes.DeviceType.GPU}


def finalize_for_target(sdfg: SDFG, target: str = 'cpu', validate: bool = True) -> SDFG:
    """Apply the performance finalization tail to a canonicalized ``sdfg``.

    Selects fast library implementations, expands the library nodes, then moves
    small constant-size transients to the stack and independent transients to
    persistent allocation. Operates in place.

    :param sdfg: A canonicalized SDFG.
    :param target: ``'cpu'`` or ``'gpu'`` (selects the fast-library priority).
    :param validate: Validate the SDFG once at the end.
    :returns: The same ``sdfg`` instance, finalized.
    """
    if target not in _TARGET_DEVICE:
        raise ValueError(f"target must be one of {sorted(_TARGET_DEVICE)}; got {target!r}")
    device = _TARGET_DEVICE[target]

    set_fast_implementations(sdfg, device)
    # infer_types before expansion: a library node may expand into other library
    # nodes whose connector types/schedules must be resolved first.
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.expand_library_nodes()

    move_small_arrays_to_stack(sdfg)
    make_transients_persistent(sdfg, device)

    sdfg.reset_cfg_list()
    if validate:
        sdfg.validate()
    return sdfg
