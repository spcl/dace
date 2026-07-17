# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Run a freshly-compiled candidate in a forked child, so a **segfault** or a **runaway loop** in
generated code cannot take down the layout sweep (GLOBAL_LAYOUT_DESIGN.md: "a segfaulting candidate
must not kill the campaign").

``os.fork`` shares the parent's memory copy-on-write, so no numpy array, compiled-SDFG handle or
closure is pickled -- the child just calls ``work_fn`` (which reuses the parent-compiled ``.so``) and
returns a small JSON-able verdict. A crash, timeout, or malformed result comes back as an
``{"error": ...}`` sentinel instead, and the parent always survives.

The one hazard fork introduces is the OpenMP thread pool: ``fork()`` duplicates only the calling
thread, so a child that enters a parallel region while the parent's pool is live blocks forever on
pool threads that no longer exist. libgomp -- gcc's default runtime, so dace's default -- deadlocks
exactly this way and installs no ``pthread_atfork`` handler. :func:`pause_openmp_pools` tears the
pool down first (OpenMP 5.0 ``omp_pause_resource_all``); the pool is a cache, so the parent's next
region simply respins it.
"""
import ctypes
import json
import os
import select
import signal
import time
import warnings
from typing import Callable, Dict

#: OpenMP runtimes whose thread pool must be torn down before a fork. Probed by soname; only the ones
#: actually loaded in this process are touched (RTLD_NOLOAD).
OMP_RUNTIME_SONAMES = ("libgomp.so.1", "libomp.so.5", "libomp.so", "libiomp5.so", "libnvomp.so")

#: ``omp_pause_resource_t`` (OpenMP 5.0). ``soft`` tears the pool down but keeps threadprivate data;
#: ``hard`` frees everything. ``soft`` is the default -- the weaker claim that still buys fork safety.
#: Measured on libgomp (dace's default): soft drops the pool 16->1 thread, i.e. a genuine tear-down.
OMP_PAUSE_SOFT = 1
OMP_PAUSE_HARD = 2
OMP_PAUSE_MODES = {"soft": OMP_PAUSE_SOFT, "hard": OMP_PAUSE_HARD}


def pause_openmp_pools(mode: int = OMP_PAUSE_SOFT) -> None:
    """Tear down the thread pool of every OpenMP runtime ALREADY loaded here, so the coming fork is
    safe. Best effort by construction (a pre-5.0 runtime, or a call from inside a parallel region,
    is passed over) but never SILENT -- an unhardened fork is a real deadlock hazard, so a skip is
    WARNED, not swallowed."""
    for soname in OMP_RUNTIME_SONAMES:
        try:
            lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)  # only ever ask a runtime already mapped
        except OSError:
            continue  # not loaded in this process: nothing to pause
        try:
            pause = lib.omp_pause_resource_all
        except AttributeError:
            warnings.warn(f"{soname}: no omp_pause_resource_all (pre-OpenMP-5.0 runtime); its thread "
                          f"pool was NOT torn down before the fork -- fork safety now rests on its own "
                          f"pthread_atfork handler, if it installs one (libgomp installs none).")
            continue
        pause.argtypes = [ctypes.c_int]
        pause.restype = ctypes.c_int
        if pause(mode) != 0:  # e.g. called from within a parallel region: the pool was NOT torn down
            warnings.warn(f"{soname}: omp_pause_resource_all(mode={mode}) returned non-zero; its "
                          f"thread pool was NOT torn down before the fork.")


def quiet_fatal_signals() -> None:
    """In the forked child, drop any inherited faulthandler: when generated code segfaults, that
    would dump a (misleading, parent-stack) Python traceback into the captured output. The parent
    already reports the crash via the child's exit signal, so let the child die quietly."""
    try:
        import faulthandler
        faulthandler.disable()
    except Exception:
        pass


def run_isolated(work_fn: Callable[[], Dict], timeout: float = 900.0) -> Dict:
    """Run ``work_fn`` in a forked child and return its JSON-able dict, or an ``{"error": ...}``
    sentinel on crash / timeout / malformed output. The parent always survives.

    ``work_fn`` must return a small JSON-able dict; anything that is not fork-safe (a live CUDA
    context in particular) must not be inside it -- this is a CPU-only guard.
    """
    pause_openmp_pools()  # a pool live across the fork deadlocks the child's first parallel region
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
        quiet_fatal_signals()
        try:
            payload = json.dumps(work_fn())
        except BaseException as e:  # a Python-level failure comes back as an error (a segfault does not)
            payload = json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"})
        try:
            os.write(w, payload.encode())
        finally:
            os.close(w)
            os._exit(0)
    os.close(w)
    start, buf, timed_out = time.perf_counter(), b"", True
    try:
        while True:
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0:
                break  # deadline hit -> timed_out stays True
            ready, _, _ = select.select([r], [], [], remaining)
            if not ready:
                break
            chunk = os.read(r, 65536)
            if not chunk:  # EOF: the child closed the pipe (finished writing, or died)
                timed_out = False
                break
            buf += chunk
    finally:
        os.close(r)
    if timed_out:
        reaped, status = os.waitpid(pid, os.WNOHANG)
        if reaped == 0:  # genuinely still running -> runaway; kill and reap
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            return {"error": f"timeout after {timeout:.0f}s (runaway kernel)"}
    else:
        _, status = os.waitpid(pid, 0)  # EOF seen: the child is exiting -> a blocking reap is safe
    if os.WIFSIGNALED(status):  # a segfault etc. never reached os.write, so buf is empty
        return {"error": f"crashed (signal {os.WTERMSIG(status)})"}
    try:
        return json.loads(buf) if buf else {"error": "child produced no result"}
    except json.JSONDecodeError:
        return {"error": "child produced malformed result (crashed mid-write)"}
