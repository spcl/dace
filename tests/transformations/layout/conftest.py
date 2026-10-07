# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The layout passes rewrite array descriptors through nested SDFGs on the assumption that a nested SDFG receives
the whole array, which is what simplification leaves. Left unsimplified, the frontend hands a nested SDFG one-element
slices (``__tmp_*_r``) of the arrays the tests lay out, so the tests run under the simplified frontend whatever the
configuration of the run."""

import pytest

import dace


@pytest.fixture(autouse=True)
def simplified_frontend():
    with dace.config.set_temporary("optimizer", "automatic_simplification", value=True):
        yield
