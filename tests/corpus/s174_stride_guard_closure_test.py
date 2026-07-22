# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression: ``LoopToScan`` must close the ``stride >= 1`` guard on TSVC ``s174`` by
lifting to a bare ``Map`` -- never by lifting to an unconditional ``Scan``.

``s174`` is ``a[i + M] = a[i] + b[i]`` over ``i in [0, M)``. ``LoopToScan``'s matcher reads
that as a residue-class scan of symbolic stride ``M`` and, unable to prove ``M >= 1``,
specializes it into ``if (M >= 1) { scan } else { pinned sequential loop }``.

The guard is redundant: a recurrence step needs two in-range iterations ``M`` apart, and the
span ``[0, M)`` is only ``M`` wide, so no such pair exists -- the write region ``a[M..2M)`` is
disjoint from the read region ``a[0..M)`` and the loop is DOALL for every ``M``.
``_stride_guard_is_statically_dischargeable`` proves exactly that (``iter_end <= iter_start +
stride - 1``), so the fallback can go.

But the proof discharges ``stride <= 0`` via "then the span is ``<= 0``, so the loop is
empty". That is vacuous for a ``Map`` and FALSE for a ``Scan``:
``dace::scan::strided_inclusive_sum`` opens with ``if (s <= 0) std::abort();``, before it
ever looks at the element count. Dropping the guard while keeping the scan form therefore
turns the legal, reference-is-a-no-op input ``M == 0`` into a SIGABRT. Hence the closed form
must be a bare ``Map``, which is simply empty at ``M == 0``.

``test_s174_zero_trip_count_does_not_abort`` is the load-bearing one: it pins the ``M == 0``
boundary the guard used to cover, and fails (SIGABRT) on an unconditional-``Scan`` closure.
``test_s171_symbolic_stride_guard_is_kept`` pins the other direction -- a stride guard whose
predicate is NOT provable must survive untouched.
"""
import copy
import os

import numpy as np
import pytest

from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.libraries.standard.nodes.scan import Scan
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from tests.corpus.measure_parallelization import cpu_params, guarded_fallback_loops
from tests.corpus.tsvc import tsvc as _TS
from tests.corpus.tsvc.tsvc_numpy import REFERENCES as _TS_REF

_ZERO_TRIP_LEN = 64


def _canonicalized(name: str, tag: str):
    """The CPU-canonicalized SDFG for one TSVC kernel, plus its ``(arrays, call_kwargs)``."""
    kernel = _TS.collect(name=name)[0]
    sdfg = _TS.to_sdfg(kernel, tag=tag, simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, **cpu_params(4))
    return sdfg, _TS.make_inputs(kernel, seed=1234)


def _structure(sdfg):
    """``(loops, maps, scans)`` -- the counts that decide whether the guard is closed."""
    loops = sum(1 for cfr in sdfg.all_control_flow_regions() if isinstance(cfr, LoopRegion))
    maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry))
    scans = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan))
    return loops, maps, scans


def test_s174_stride_guard_closes_to_bare_map():
    """The provable ``M >= 1`` guard is discharged: no conditional, no pinned fallback, and the
    parallel form is a plain Map rather than an (abort-on-``stride <= 0``) Scan."""
    sdfg, _ = _canonicalized('s174_d_single', 'guardclose_struct')
    loops, maps, scans = _structure(sdfg)
    assert guarded_fallback_loops(sdfg) == 0, "guarded sequential fallback survived"
    assert loops == 0, "a sequential LoopRegion survived the closure"
    assert maps == 1, "the loop did not become a Map"
    assert scans == 0, "closed form must be a Map, not an unconditional (abort-prone) Scan"


def test_s174_closed_map_is_bit_exact():
    """Canonicalize 10x in-process (the cross-process canon flake is order-dependent): the
    closed form stays a bare Map and is BIT-exact against the numpy oracle every time."""
    kernel = _TS.collect(name='s174_d_single')[0]
    arrays, call_kwargs = _TS.make_inputs(kernel, seed=1234)
    ref = {n: a.copy() for n, a in arrays.items()}
    _TS_REF['s174_d_single'](**ref, **call_kwargs)

    for trial in range(10):
        sdfg = _TS.to_sdfg(kernel, tag=f'guardclose_exact_{trial}', simplify=True)
        canonicalize(sdfg, validate=True, validate_all=False, **cpu_params(4))
        assert guarded_fallback_loops(sdfg) == 0
        assert _structure(sdfg) == (0, 1, 0)
        fin = finalize_for_target(copy.deepcopy(sdfg), 'cpu')
        fin.name = f'{fin.name}_gc_exact_{trial}'
        got = {n: a.copy() for n, a in arrays.items()}
        fin.compile()(**got, **call_kwargs)
        # No reduction and no reassociation -- an elementwise DOALL Map must match exactly.
        for name in arrays:
            assert np.array_equal(got[name], ref[name]), f'trial {trial}: {name} not bit-exact'


def test_s174_zero_trip_count_does_not_abort():
    """``M == 0`` -- the value the dropped guard used to divert to the sequential fallback.

    The reference is a no-op there. An unconditional residue-class ``Scan`` would instead hit
    ``if (s <= 0) std::abort();`` in ``dace::scan::strided_inclusive_sum`` and die on SIGABRT,
    so this asserts the closure is the Map form. Run in a forked child: a regression aborts the
    process, which must not take the test session down with it.
    """
    sdfg, _ = _canonicalized('s174_d_single', 'guardclose_zero')
    fin = finalize_for_target(copy.deepcopy(sdfg), 'cpu')
    fin.name = f'{fin.name}_gc_zero'
    csdfg = fin.compile()

    a = np.arange(_ZERO_TRIP_LEN, dtype=np.float64)
    b = np.ones(_ZERO_TRIP_LEN, dtype=np.float64)

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: never return into pytest, and never raise past os._exit
        code = 1
        try:
            work = a.copy()
            csdfg(a=work, b=b.copy(), M=0, LEN_1D=_ZERO_TRIP_LEN)
            code = 0 if np.array_equal(work, a) else 2  # M == 0 must leave `a` untouched
        except BaseException:  # noqa: BLE001 -- report as an exit code, never as an exception
            code = 3
        finally:
            os.write(write_fd, bytes([code]))
            os._exit(code)

    os.close(write_fd)
    _, status = os.waitpid(pid, 0)
    os.close(read_fd)

    assert not os.WIFSIGNALED(status), (
        f'zero-trip M=0 killed by signal {os.WTERMSIG(status) if os.WIFSIGNALED(status) else "?"} '
        '(SIGABRT=6 means the guard was dropped onto an unconditional strided Scan)')
    assert os.WEXITSTATUS(status) != 2, 'M=0 must be a no-op, but the kernel wrote to `a`'
    assert os.WEXITSTATUS(status) == 0, f'zero-trip M=0 run failed (exit {os.WEXITSTATUS(status)})'


def test_s171_symbolic_stride_guard_is_kept():
    """The counterpart soundness bound: ``s171`` (``a[i * inc] = ...``) has a stride guard whose
    predicate is NOT statically provable, so the guarded fallback must SURVIVE. Closing it would
    be a miscompile, so this pins that the discharge stays narrow."""
    sdfg, _ = _canonicalized('s171_d_single', 'guardclose_kept')
    assert guarded_fallback_loops(sdfg) == 1, 's171 unprovable stride guard was wrongly dropped'
