# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared helpers for the bulk TSVC vectorization test blocks.

Hosts the single copy of the Lever-1 ``_kernel_has_branch`` predicate
used by ``tsvc_vectorization_test_block{1,2,3,4}`` and
``tsvc_2d test files`` to skip the duplicate ``fp_factor``
parametrization for branchless kernels (merge vs fp_factor produce an
identical SDFG when there is no ``if`` to lower).
"""
import ast
import inspect
import textwrap

_BRANCH_CACHE = {}


def kernel_has_branch(prog) -> bool:
    """Return ``True`` iff the kernel body contains an ``if``.

    ``merge`` vs ``fp_factor`` branch-lowering only differ when there is
    a conditional to lower, so a branchless kernel produces an identical
    SDFG under both and the ``fp_factor`` run is pure duplication.

    :param prog: A ``@dace.program`` whose ``.f`` is the source function.
    :return: ``True`` if an ``ast.If`` is present (or the source cannot
        be parsed — conservatively keep both modes then).
    """
    key = getattr(prog, "name", id(prog))
    if key in _BRANCH_CACHE:
        return _BRANCH_CACHE[key]
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(prog.f)))
        has = any(isinstance(n, ast.If) for n in ast.walk(tree))
    except Exception:  # noqa: BLE001 - cannot classify -> keep both modes
        has = True
    _BRANCH_CACHE[key] = has
    return has


def build_tsvc_matrix(kernels, lens):
    """Pre-filtered (kernel, ..., remainder, branch, LEN) parametrize list.

    Generates only the *meaningful* combos so redundant ones are never
    collected (instead of being collected then ``pytest.skip``-ped):

    - Lever 1: a branchless kernel's ``fp_factor`` SDFG == its ``merge``
      SDFG (no ``if`` to lower) — emit ``merge`` only.
    - Lever 2: ``LEN %% W == 0`` ⇒ P2 emits no remainder ⇒ ``masked``
      SDFG == ``scalar`` SDFG — emit ``scalar`` only.

    Note: ``fp_factor`` + ``masked`` is now allowed on the tile path (the
    iter_mask gates stores; the ``c*x + (1-c)*y`` arithmetic runs on every
    lane unchanged). The legacy ``VectorizeCPU`` 1D path still rejects it,
    so the harness skips that combo for the legacy arm only.

    :param kernels: list of tuples whose ``[0]`` is the ``@dace.program``
        (remaining elements, e.g. argnames/argspec/params, are passed
        through unchanged).
    :param lens: iterable of LEN values (e.g. ``(64, 65)``).
    :return: ``(params, ids)`` for a single ``@pytest.mark.parametrize``.
    """
    params, ids = [], []
    for kt in kernels:
        prog = kt[0]
        has_branch = kernel_has_branch(prog)
        for L in lens:
            for rem in ("scalar", "masked"):
                for br in ("merge", "fp_factor"):
                    if br == "fp_factor" and not has_branch:
                        continue
                    if rem == "masked" and L % 8 == 0:
                        continue
                    params.append((*kt, rem, br, L))
                    ids.append(f"{prog.name}-{rem}-{br}-{L}")
    return params, ids
