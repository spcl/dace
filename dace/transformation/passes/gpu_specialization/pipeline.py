# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU specialization stage, the counterpart of ``cpu_specialize``, in two bands around the offload.

The GPU pipeline is ``canonicalize(s, target='gpu')`` -> ``gpu_specialize(s)`` -> ``offload_to_gpu(s)`` (or a
caller's own offload) -> ``finalize_for_target(s, 'gpu')``; the caller runs each step, none hides inside another.

1. :func:`gpu_specialize` runs BEFORE the offload: it restructures, changing which maps exist. The offload decides
   everything that follows from the structure -- which map is a kernel, where each container lives (a container a
   host guard reads stays on the host), which copies and host states to insert -- so it must see the result.
   Interchanging after it would move those host reads and copies into a kernel.
2. :func:`gpu_specialize_offloaded` runs AFTER the offload (``finalize_for_target``): it resolves the schedules of
   the device maps the offload created.
"""
from dace import SDFG
from dace.transformation.passes.gpu_specialization.contiguous_axis_to_threads import ContiguousAxisToThreads
from dace.transformation.passes.gpu_specialization.gpu_loop_interchange import GPULoopInterchange
from dace.transformation.passes.gpu_specialization.sequentialize_nested_device_scopes import (
    SequentializeNestedDeviceScopes)


def gpu_specialize(sdfg: SDFG, validate: bool = True) -> SDFG:
    """Specialize a canonicalized, not yet offloaded ``sdfg`` for the GPU, in place.

    :param sdfg: A canonicalized SDFG, before the device move.
    :param validate: Validate the SDFG once at the end.
    :returns: The same ``sdfg`` instance.
    """
    GPULoopInterchange().apply_pass(sdfg, {})
    if validate:
        sdfg.validate()
    return sdfg


def gpu_specialize_offloaded(sdfg: SDFG) -> SDFG:
    """Resolve the device schedules of an offloaded ``sdfg``, in place.

    :param sdfg: An offloaded SDFG.
    :returns: The same ``sdfg`` instance.
    """
    # ``map JK { work; map JL }`` makes JL a thread dimension first: pinned sequential below, it would not be one.
    ContiguousAxisToThreads().apply_pass(sdfg, {})
    SequentializeNestedDeviceScopes().apply_pass(sdfg, {})
    return sdfg
