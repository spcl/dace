# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared helpers for the bulk TSVC vectorization test blocks.

Hosts the single copy of the Lever-1 ``_kernel_has_branch`` predicate
used by ``tsvc_vectorization_test_block{1,2,3,4}`` and
``tsvc_vectorization_test_2d`` to skip the duplicate ``fp_factor``
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
