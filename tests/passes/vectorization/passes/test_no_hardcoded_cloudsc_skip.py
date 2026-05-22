# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tripwire: cloudsc-specific array names must not be hardcoded into the generic
``add_copies_before_and_after_nsdfg`` helper. Any user SDFG with arrays named
``zqxfg`` / ``zsolqb`` / ``zsolqa`` would otherwise have the copy-in/copy-out
classification silently skipped.

The skip is now exposed as a ``Vectorize.user_skip_nsdfg_arrays`` property
(default: empty); any benchmark that needs it should pass it explicitly. This
test xpasses since that change landed.
"""
import inspect

from dace.transformation.passes.vectorization.utils.nsdfg_reshape import add_copies_before_and_after_nsdfg


def test_no_cloudsc_array_names_in_helper_body():
    src = inspect.getsource(add_copies_before_and_after_nsdfg)
    for name in ("zqxfg", "zsolqb", "zsolqa"):
        assert f'"{name}"' not in src and f"'{name}'" not in src, (
            f"hardcoded cloudsc array name {name!r} still present in "
            f"add_copies_before_and_after_nsdfg")
