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

Two shapes are offered. :func:`run_isolated` / :func:`exit_code` compile and call an SDFG and report
a VERDICT, comparing in the child. :func:`call_in_child` evaluates an arbitrary picklable callable
and hands its RESULT back, for the cases whose comparison is domain-specific and lives in the
parent. ``spawn`` pickles the callable by qualified name, so it must be importable at module level
-- a lambda or a closure cannot cross, which is the one API difference from the ``os.fork`` helpers
this module replaces.
"""
import multiprocessing as mp
import sys
from typing import Any, Callable, Dict, Sequence, Tuple

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
    return exit_code_of(compare_in_child, sdfg, dict(kwargs), list(outputs), exact, timeout=timeout)


def exit_code_of(target: Callable[..., Any], *args: Any, timeout: float = 900.0) -> int:
    """Exit code of a spawned child that ran picklable ``target(*args)``; negative is a signal.

    For a child whose verdict is richer than pass/fail, or whose comparison has to stay in the
    child because it is the test's own -- ``target`` ends by calling :func:`sys.exit` with the code
    the caller will assert on.
    """
    process = mp.get_context('spawn').Process(target=target, args=args)
    process.start()
    process.join(timeout)
    if process.exitcode is None:
        process.kill()
        process.join()
        raise AssertionError(f'isolated child did not finish within {timeout}s')
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


def send_result(connection: Any, target: Callable[..., Any], args: Tuple[Any, ...]) -> None:
    """Child body for :func:`call_in_child`: evaluate ``target(*args)`` and ship the result home.

    An exception is deliberately left to propagate: multiprocessing prints the child traceback and
    exits non-zero, and the closed connection tells the parent there is no result coming.
    """
    try:
        connection.send(target(*args))
    finally:
        connection.close()


def call_in_child(target: Callable[..., Any], *args: Any, timeout: float = 900.0) -> Any:
    """Value of picklable ``target(*args)`` evaluated in a spawned child.

    For the callers whose comparison is domain-specific and stays in the parent. ``target`` is
    pickled by qualified name, so it must be a module-level function; its arguments and its result
    are pickled by value. A child that dies -- a segfault in generated code, or an exception, whose
    traceback multiprocessing has already printed -- closes the connection without a result and is
    reported as an ``AssertionError`` rather than a hang.

    NB the child mutates its own copies of any array in ``args``; only the returned value comes
    back. Anything the parent must inspect has to be part of that return value.
    """
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=send_result, args=(sender, target, args))
    process.start()
    sender.close()  # drop the parent's copy, or the child's EOF is never observed
    try:
        if not receiver.poll(timeout):
            process.kill()
            raise AssertionError(f'isolated call did not finish within {timeout}s')
        try:
            return receiver.recv()
        except EOFError:
            raise AssertionError('isolated call died before producing a result') from None
    finally:
        receiver.close()
        process.join(timeout)
