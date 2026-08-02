# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Run a compiled SDFG in a throwaway process, so a faulting kernel cannot take pytest down.

``os.fork()`` cannot do this, and reaching for it here hangs the suite. A DaCe CPU program is
OpenMP, so the first compiled kernel any test in the process runs leaves libgomp holding a team of
``nproc - 1`` workers. ``fork()`` keeps only the calling thread, yet the child inherits the
heap-resident team barrier that still records the whole team -- so the child's own parallel region
waits on a futex for threads that do not exist in it. 0 CPU, forever, and the parent blocks in
``waitpid`` behind it.

Measured on this tree, one variable at a time: a parent that has run one compiled kernel -> child
hangs; the same parent under ``OMP_NUM_THREADS=1``, which builds no team -> no hang; a child that
compiles but never calls the kernel -> no hang. So the deadlock is the OpenMP runtime, not the
compiler, and it arms itself as soon as any earlier test in the process runs a kernel.

``spawn`` starts the child from a fresh interpreter, which builds its own team. It costs an
interpreter start plus a re-import of dace per call, so spend it only where the containment is
load-bearing: a would-be out-of-bounds access, or a kernel expected to ``abort``. Where it is not,
run in-process and compare directly.
"""
import multiprocessing as mp
import sys
from typing import Any, Dict, Sequence, Tuple

import numpy as np

import dace

#: Child exit code for "the kernel ran, but an output did not match its reference".
MISMATCH = 3

#: Tolerance for the inexact comparison. g++ defaults to ``-ffp-contract=fast``, so a compiled
#: ``a[i] + b[i]*c[i]`` contracts to one FMA while a numpy oracle rounds twice -- a ~1 ulp delta
#: that is correct lowering, not a bug. Callers pinning a bit-exact lowering pass ``exact=True``.
RTOL = 1e-12
ATOL = 1e-12


def own_arrays(kwargs: Dict[str, Any],
               outputs: Sequence[Tuple[Any, Any]]) -> Tuple[Dict[str, Any], Sequence[Tuple[Any, Any]]]:
    """Re-copy every unpickled array so it OWNS its buffer, sharing one copy per original.

    Unpickling hands back an ndarray whose ``base`` is the pickle buffer, and DaCe rejects any
    argument with a non-``None`` ``base`` as a view it cannot analyse. ``np.copy`` clears it. The
    identity map is what makes this safe to do: a buffer that is both a call argument and an output
    to check must stay ONE array here, or the comparison would read a copy the kernel never wrote.
    """
    owned: Dict[int, np.ndarray] = {}

    def own(value: Any) -> Any:
        if not isinstance(value, np.ndarray):
            return value
        if id(value) not in owned:
            owned[id(value)] = np.copy(value)
        return owned[id(value)]

    return {name: own(value) for name, value in kwargs.items()}, [(own(buf), ref) for buf, ref in outputs]


def compare_in_child(sdfg: dace.SDFG, kwargs: Dict[str, Any], outputs: Sequence[Tuple[Any, Any]], exact: bool) -> None:
    """Child body: compile, run, compare, and report the verdict as an exit code.

    Compares HERE rather than shipping buffers back, because the child mutates its own copies and
    the parent's arrays never see the result. ``kwargs`` and ``outputs`` cross as one pickled
    argument tuple, which preserves identity -- so a buffer passed both as a call argument and as
    an output to check is still one array in the child, and :func:`own_arrays` keeps it that way.

    Exceptions are deliberately not caught: multiprocessing prints the traceback and exits 1, which
    says far more than a swallowed error code.

    :param sdfg: SDFG to compile and call.
    :param kwargs: call arguments (arrays + symbols); output buffers are mutated in place.
    :param outputs: ``(buffer, reference)`` pairs to compare after the call.
    :param exact: compare bit-exact instead of within :data:`RTOL` / :data:`ATOL`.
    """
    kwargs, outputs = own_arrays(kwargs, outputs)
    sdfg.compile()(**kwargs)
    for index, (buffer, reference) in enumerate(outputs):
        if np.array_equal(buffer, reference) if exact else np.allclose(buffer, reference, rtol=RTOL, atol=ATOL):
            continue
        delta = np.nanmax(np.abs(np.asarray(buffer, dtype=np.float64) - np.asarray(reference, dtype=np.float64)))
        print(f'output {index}: max|diff| = {delta:.3e}', file=sys.stderr)
        sys.exit(MISMATCH)


def exit_code(sdfg: dace.SDFG,
              kwargs: Dict[str, Any],
              outputs: Sequence[Tuple[Any, Any]] = (),
              *,
              exact: bool = False,
              timeout: float = 900.0) -> int:
    """Exit code of a spawned child that compiled, called and checked ``sdfg``.

    Negative is a signal, ``-signal.SIGABRT`` for a kernel that trapped -- the verdict the runtime
    guard tests assert on. Use :func:`run_isolated` when only a clean run counts as a pass.
    """
    process = mp.get_context('spawn').Process(target=compare_in_child, args=(sdfg, dict(kwargs), list(outputs), exact))
    process.start()
    process.join(timeout)
    if process.exitcode is None:
        process.kill()
        process.join()
        raise AssertionError(f'isolated kernel did not finish within {timeout}s')
    return process.exitcode


def run_isolated(sdfg: dace.SDFG,
                 kwargs: Dict[str, Any],
                 outputs: Sequence[Tuple[Any, Any]] = (),
                 *,
                 exact: bool = False) -> None:
    """Compile + call ``sdfg`` in a spawned child; assert every output matched its reference."""
    code = exit_code(sdfg, kwargs, outputs, exact=exact)
    assert code >= 0, f'isolated kernel died on signal {-code}'
    assert code != MISMATCH, 'isolated kernel ran, but an output did not match its reference'
    assert code == 0, f'isolated kernel failed with exit code {code}'
