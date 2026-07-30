# CMake hangs under `srun`/`mpirun` — blocked `SIGCHLD`

## Symptom

Under an MPI/Slurm launcher, compilation wedges. `ps` shows a live `cmake` with **defunct
(zombie) children**. Nothing is compiling; CPU is idle. The same build runs fine outside the
launcher. On daint this looked like "the compile hangs", but no compiler was ever running.

## Cause

It is a blocked **signal mask**, not a signal handler.

`srun` and `mpirun` start their tasks with `SIGCHLD` **blocked**. A signal mask is inherited across
both `fork()` and `exec()`, so every descendant of the launched task inherits the block — Python,
the `cmake` it spawns, and everything below.

CMake (KWSys) learns that the helpers it spawns during *configure* — `uname`, the compiler-id and
ABI test binaries, later `make`/`ninja` — have exited by **receiving `SIGCHLD`**. Its process
loop parks in `select()` and expects the signal to wake it.

With `SIGCHLD` blocked:

1. The helper exits and becomes a zombie (it stays reaped-pending, so the parent can still `wait`).
2. The kernel makes `SIGCHLD` *pending* rather than delivering it.
3. `select()` is never interrupted, so CMake never calls `waitpid`.
4. It spins forever. The child is finished; nobody notices.

Blocked is the operative word. `SIG_IGN` would auto-reap and produce a *different* failure
(`waitpid` → `ECHILD`); blocking produces a silent stall with zombies parked in the table.

Confirmed under `srun`: every task's `/proc/self/status` shows `SigBlk` with the `SIGCHLD` bit set,
and a trivial `project()` configure hangs until the mask is cleared.

## Fix

`build_subprocess_sigmask()` in `dace/codegen/compiler.py` unblocks `SIGCHLD` **around the fork
only**:

```python
signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
try:
    yield          # subprocess.Popen() happens here
finally:
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})
```

Why this is sufficient and safe:

- A child inherits the **forking thread's** mask, and `Popen` does not reset it. Unblocking just
  across the fork is therefore enough — the spawned `cmake` starts with `SIGCHLD` deliverable and
  reaps its helpers normally.
- `pthread_sigmask` is **per-thread**, so it never disturbs another thread or the process's
  steady-state mask. The launcher's own `SIGCHLD` handling is untouched.
- The mask is restored in `finally`, so the block is reinstated even if the spawn raises.
- It early-returns when `SIGCHLD` is already deliverable (the common non-launcher case) and where
  `pthread_sigmask`/`SIGCHLD` do not exist (Windows).

## The second, separate safeguard

`build_subprocess_env()` strips MPI rank variables (`MPI_RANK_ENV_PREFIXES`) from the compile
subprocess's environment. Otherwise the build tool inherits an MPI identity and can itself try to
behave like a rank, which surfaces as the same stuck-`cmake`-with-defunct-children picture.
Compilation never needs an MPI identity; everything else (PATH, compiler flags, MCA tuning) is
preserved.

The two failures look identical from the outside, so both safeguards are applied together.

## Where they are applied

Both live at the single fork point, `_run_liveoutput()` in `dace/codegen/compiler.py` — CMake
configure/build and the native backend's compile/link lines all fork there. Deliberately not at
each call site, where a new caller would silently reintroduce the hang.

## Diagnosing a recurrence

```bash
grep SigBlk /proc/<pid>/status     # decode the mask; bit 17 (0x10000) is SIGCHLD
ps -o pid,stat,wchan,cmd -p <pid> --ppid <pid>   # parent in select(), children <defunct>
```

`SigBlk` with the `SIGCHLD` bit set on the `cmake` process is the signature.
