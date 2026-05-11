# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Regression tests for chained ``if/elif/else`` access-node wiring in the
Python frontend.

Before this fix, every elif/else arm that read an array element which an
earlier arm had written resolved to the earlier arm's write-side access
node, because ``_add_read_access`` returned the ``'w'`` entry of
``self.accesses`` if one existed. That is correct for straight-line code
(read-after-write semantics), but inside a conditional branch each arm
must see the pre-conditional state of the array. The visible symptom was
a numerical drift in the chained-if pattern below (TSVC s441 / s441_v2),
the structural symptom is that the elif/else arm's tasklet has its read
edge sourcing from the if-arm's write-side access node instead of an
independent read-side access node.

These tests pin both the structural and numerical contract.
"""
import numpy as np

import dace


@dace.program
def chained_if_kernel(a: dace.float64[8], b: dace.float64[8], c: dace.float64[8], d: dace.float64[8]):
    for i in dace.map[0:8:1]:
        if d[i] < 0.0:
            a[i] = a[i] + b[i] * c[i]
        elif d[i] == 0.0:
            a[i] = a[i] + b[i] * b[i]
        else:
            a[i] = a[i] + c[i] * c[i]


def _arm_states_for(sdfg: dace.SDFG):
    """Returns the three compute states (one per arm) that contain the
    final ``_Add_`` tasklet of each arm. Identifies them by the
    ``_Add_`` tasklet code marker plus the writeback's ``__inp`` source."""
    arm_states = {}
    for n, g in sdfg.all_nodes_recursive():
        if not isinstance(n, dace.nodes.Tasklet):
            continue
        if n.code.as_string == "__out = (__in1 + __in2)":
            arm_states[g.label] = (n, g)
    return arm_states


def test_chained_if_else_read_edges_route_to_read_side_access_nodes():
    """Structural: every elif/else arm's read of ``a[i]`` must source from
    a read-side access node (data name ending in ``_r``), not from an
    earlier arm's write-side access node (ending in ``_w``)."""
    sdfg = chained_if_kernel.to_sdfg(simplify=False)
    arm_states = _arm_states_for(sdfg)
    assert len(arm_states) == 3, sorted(arm_states.keys())

    misrouted = []
    for state_label, (tasklet, state) in arm_states.items():
        for e in state.in_edges(tasklet):
            src_data = getattr(e.src, "data", None)
            if src_data is None:
                continue
            # The arm-add tasklet reads two values; the ``__in1`` one is
            # the carried ``a[i]`` read. It must come from a read-side
            # access node, never a write-side one.
            if e.dst_conn == "__in1" and src_data.endswith("_w"):
                misrouted.append((state_label, src_data))
    assert not misrouted, (f"elif/else arms read a[i] from a write-side access node "
                           f"(should be read-side): {misrouted}")


def test_chained_if_else_numerical_matches_python_reference():
    """End-to-end: compile + run and compare against the plain-Python
    chained-if evaluation for inputs that exercise every arm."""
    sdfg = chained_if_kernel.to_sdfg(simplify=True)

    rng = np.random.default_rng(seed=0)
    a = rng.standard_normal(8).astype(np.float64)
    b = rng.standard_normal(8).astype(np.float64)
    c = rng.standard_normal(8).astype(np.float64)
    # Force every arm to be exercised across the lane vector.
    d = np.array([-1.0, 0.0, 1.0, -2.0, 0.0, 3.0, -0.5, 4.0], dtype=np.float64)

    expected = a.copy()
    for i in range(8):
        if d[i] < 0.0:
            expected[i] = expected[i] + b[i] * c[i]
        elif d[i] == 0.0:
            expected[i] = expected[i] + b[i] * b[i]
        else:
            expected[i] = expected[i] + c[i] * c[i]

    sdfg(a=a, b=b, c=c, d=d)
    np.testing.assert_allclose(a, expected, err_msg=f"a={a}\nexpected={expected}")
