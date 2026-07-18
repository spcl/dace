# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Run a freshly-compiled candidate in a forked child, so a segfault or runaway loop in generated code cannot take down the layout sweep; also pauses OpenMP thread pools first to avoid a fork deadlock."""
import ctypes
import json
import os
import select
import signal
import time
import warnings
from typing import Callable, Dict

#: OpenMP runtimes whose thread pool must be torn down before a fork; only ones already loaded (RTLD_NOLOAD) are touched.
OMP_RUNTIME_SONAMES = ("libgomp.so.1", "libomp.so.5", "libomp.so", "libiomp5.so", "libnvomp.so")

#: omp_pause_resource_t (OpenMP 5.0): soft keeps threadprivate data, hard frees everything; soft suffices for fork safety.
OMP_PAUSE_SOFT = 1
OMP_PAUSE_HARD = 2
OMP_PAUSE_MODES = {"soft": OMP_PAUSE_SOFT, "hard": OMP_PAUSE_HARD}


def pause_openmp_pools(mode: int = OMP_PAUSE_SOFT) -> None:
    """Tear down the thread pool of every OpenMP runtime already loaded, so the coming fork is safe. Best effort; warns rather than silently skipping when it can't."""
    for soname in OMP_RUNTIME_SONAMES:
        try:
            lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)  # only touch runtimes already mapped
        except OSError:
            continue  # not loaded: nothing to pause
        try:
            pause = lib.omp_pause_resource_all
        except AttributeError:
            warnings.warn(f"{soname}: no omp_pause_resource_all (pre-OpenMP-5.0 runtime); its thread "
                          f"pool was NOT torn down before the fork -- fork safety now rests on its own "
                          f"pthread_atfork handler, if it installs one (libgomp installs none).")
            continue
        pause.argtypes = [ctypes.c_int]
        pause.restype = ctypes.c_int
        if pause(mode) != 0:  # e.g. inside a parallel region: pool not torn down
            warnings.warn(f"{soname}: omp_pause_resource_all(mode={mode}) returned non-zero; its "
                          f"thread pool was NOT torn down before the fork.")


def quiet_fatal_signals() -> None:
    """In the forked child, disable faulthandler so a segfault dies quietly instead of dumping a misleading traceback."""
    try:
        import faulthandler
        faulthandler.disable()
    except Exception:
        pass


def run_isolated(work_fn: Callable[[], Dict], timeout: float = 900.0) -> Dict:
    """Run ``work_fn`` in a forked child; returns its JSON-able dict, or an ``{"error": ...}`` sentinel on crash/timeout/malformed output. ``work_fn`` must be fork-safe (no live CUDA context)."""
    pause_openmp_pools()  # a live pool across fork deadlocks the child's first parallel region
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
        quiet_fatal_signals()
        try:
            payload = json.dumps(work_fn())
        except BaseException as e:  # Python exception -> error; a segfault never reaches here
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
                break  # deadline hit
            ready, _, _ = select.select([r], [], [], remaining)
            if not ready:
                break
            chunk = os.read(r, 65536)
            if not chunk:  # EOF: child closed the pipe (done or died)
                timed_out = False
                break
            buf += chunk
    finally:
        os.close(r)
    if timed_out:
        reaped, status = os.waitpid(pid, os.WNOHANG)
        if reaped == 0:  # still running -> runaway; kill and reap
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            return {"error": f"timeout after {timeout:.0f}s (runaway kernel)"}
    else:
        _, status = os.waitpid(pid, 0)  # EOF seen: child exiting, blocking reap is safe
    if os.WIFSIGNALED(status):  # segfault etc. never reached os.write, so buf is empty
        return {"error": f"crashed (signal {os.WTERMSIG(status)})"}
    try:
        return json.loads(buf) if buf else {"error": "child produced no result"}
    except json.JSONDecodeError:
        return {"error": "child produced malformed result (crashed mid-write)"}
