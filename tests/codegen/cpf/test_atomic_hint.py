# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The comment above a conflicting accumulation names its operator, its container and why iterations collide."""

from dace.codegen.targets.cpu import atomic_hint
from dace.memlet import Memlet


# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_an_atomic_names_its_operator_and_why_iterations_collide():
    """Review F27: a max fold said ``add to`` and named the whole scattered range."""
    scatter = Memlet("hist[0:256]")
    scatter.dynamic = True
    assert atomic_hint(scatter, "max").startswith(
        "// atomic update: iterations may fold max into an element of hist (the index may repeat across iterations)"
    )
    assert atomic_hint(Memlet("s[0]"), "*").startswith(
        "// atomic update: iterations may multiply into s[0] (every iteration writes that element)"
    )


if __name__ == "__main__":
    test_an_atomic_names_its_operator_and_why_iterations_collide()
