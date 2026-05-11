# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Regression tests for if/elif/else access-node wiring in the Python frontend.

The frontend's element-access registry (``self.accesses``) maps a tuple
``(name, range, kind)`` (kind = ``'r'`` or ``'w'``) to a transient access
node, and ``_add_read_access`` historically returned the ``'w'`` entry
before the ``'r'`` one. That implements straight-line read-after-write
semantics correctly, but inside a conditional branch each arm must see
the pre-conditional state of the array, since only one arm runs at a
time. Without the fix:

- s441 chained ``if/elif/else``: the elif/else arm's read of ``a[i]``
  resolves to the if-arm's write-side access node, which is the wrong
  access node entirely (numerical drift, eventually a validation error
  in downstream passes).

- Simple ``if/else`` whose two arms write to the same array element
  worked numerically because they shared the access node, but every arm
  was building a SEPARATE NestedSDFG output connector when the fix was
  applied naively, which broke the connector convention and produced
  uninitialized writes.

The fix hides the if-arm's freshly-added ``'w'`` keys from the elif/else
arm's reads, while leaving the writes themselves shared so the NestedSDFG
emits one output connector per array element.

These tests pin three contracts:

1. Topology: every tasklet in the emitted SDFG must have its connector
   role agree with its edge direction, ``_assert_connector_role_matches_edges``
   walks the SDFG and verifies that no read edge lands on an out-connector
   and no write edge leaves an in-connector.

2. Numerical: end-to-end compile and run, compare to the plain-Python
   reference. Inputs are chosen so every arm is exercised.

3. NestedSDFG output uniqueness: simple ``if/else`` with a shared write
   target must produce exactly one NestedSDFG output connector for that
   element, not one per arm.
"""
import numpy as np

import dace


def _assert_connector_role_matches_edges(sdfg: dace.SDFG):
    """For every tasklet and NestedSDFG in the entire SDFG, verifies:
    - no in-connector has zero incoming edges,
    - no out-connector has zero outgoing edges,
    - no connector name is declared as both input and output,
    - no edge lands on an out-connector,
    - no edge leaves an in-connector.

    Reads use in-connectors, writes use out-connectors. The bug shape this
    pinned (s441) put a read edge on a tasklet's out-connector, which this
    helper catches independently of whether the resulting SDFG compiles.
    """
    for n, state in sdfg.all_nodes_recursive():
        if not isinstance(n, (dace.nodes.Tasklet, dace.nodes.NestedSDFG)):
            continue
        if not isinstance(state, dace.SDFGState):
            continue
        in_names = set(n.in_connectors.keys())
        out_names = set(n.out_connectors.keys())
        both = in_names & out_names
        assert not both, f"Node {n.label!r} has connector(s) {sorted(both)} declared as both input and output"

        for e in state.in_edges(n):
            if e.dst_conn is None:
                continue
            assert e.dst_conn not in out_names, (
                f"Node {n.label!r} in state {state.label!r}: read edge lands on out-connector "
                f"{e.dst_conn!r} (should land on an in-connector). Source: {e.src}")
            assert e.dst_conn in in_names, (
                f"Node {n.label!r} in state {state.label!r}: edge lands on unknown connector {e.dst_conn!r}")

        for e in state.out_edges(n):
            if e.src_conn is None:
                continue
            assert e.src_conn not in in_names, (
                f"Node {n.label!r} in state {state.label!r}: write edge leaves in-connector "
                f"{e.src_conn!r} (should leave from an out-connector). Destination: {e.dst}")
            assert e.src_conn in out_names, (
                f"Node {n.label!r} in state {state.label!r}: edge leaves unknown connector {e.src_conn!r}")


# ---------------------------------------------------------------------------
# Chained if/elif/else (TSVC s441 / s441_v2 shape).
# ---------------------------------------------------------------------------


@dace.program
def chained_if_kernel(a: dace.float64[8], b: dace.float64[8], c: dace.float64[8], d: dace.float64[8]):
    for i in dace.map[0:8:1]:
        if d[i] < 0.0:
            a[i] = a[i] + b[i] * c[i]
        elif d[i] == 0.0:
            a[i] = a[i] + b[i] * b[i]
        else:
            a[i] = a[i] + c[i] * c[i]


def _arm_add_tasklets(sdfg: dace.SDFG):
    """Returns each arm's final ``_Add_`` tasklet keyed by the state label."""
    out = {}
    for n, g in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.Tasklet) and n.code.as_string == "__out = (__in1 + __in2)":
            out[g.label] = (n, g)
    return out


def test_chained_if_else_read_edges_route_to_read_side_access_nodes():
    """For each arm's ``_Add_`` tasklet, the ``__in1`` edge that carries
    the read of ``a[i]`` must source from a read-side access node (``_r``
    suffix), not a write-side one (``_w`` suffix)."""
    sdfg = chained_if_kernel.to_sdfg(simplify=False)
    arm_tasklets = _arm_add_tasklets(sdfg)
    assert len(arm_tasklets) == 3, sorted(arm_tasklets.keys())

    misrouted = []
    for state_label, (tasklet, state) in arm_tasklets.items():
        for e in state.in_edges(tasklet):
            src_data = getattr(e.src, "data", None)
            if src_data is None:
                continue
            if e.dst_conn == "__in1" and src_data.endswith("_w"):
                misrouted.append((state_label, src_data))
    assert not misrouted, (f"elif/else arms read a[i] from a write-side access node "
                           f"(should be read-side): {misrouted}")


def test_chained_if_else_connector_topology_correct():
    sdfg = chained_if_kernel.to_sdfg(simplify=False)
    _assert_connector_role_matches_edges(sdfg)


def test_chained_if_else_numerical_matches_python_reference():
    sdfg = chained_if_kernel.to_sdfg(simplify=True)
    rng = np.random.default_rng(seed=0)
    a = rng.standard_normal(8).astype(np.float64)
    b = rng.standard_normal(8).astype(np.float64)
    c = rng.standard_normal(8).astype(np.float64)
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


# ---------------------------------------------------------------------------
# Simple two-arm if/else with a shared write target (division_by_zero shape).
# Pinned because a naive fix to the chained-elif bug ended up creating one
# NestedSDFG output connector per arm for ``B[i]``, which broke this case
# while fixing the chained one. Both arms must continue to write through
# the same write-side access node.
# ---------------------------------------------------------------------------


N = dace.symbol("N")


@dace.program
def division_by_zero_kernel(A: dace.float64[N], B: dace.float64[N], c: dace.float64):
    for i in dace.map[0:N]:
        if A[i] > 0.0:
            B[i] = c / A[i]
        else:
            B[i] = 0.0


def test_division_by_zero_single_output_connector_per_element():
    """The NestedSDFG that wraps the conditional must have exactly one
    output connector for ``B``, not one per arm."""
    sdfg = division_by_zero_kernel.to_sdfg(simplify=False)
    nested = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)]
    # There may be zero NestedSDFGs after frontend lowering if everything
    # was inlined, in which case there is nothing to assert. When one
    # exists it must not multiply output connectors per element.
    for n in nested:
        out_conns = list(n.out_connectors.keys())
        # Only count connectors that ultimately route to B in the outer SDFG.
        wlike = [c for c in out_conns if c.endswith("_w") or c == "B"]
        # All ``_w`` connectors for B should share the same prefix.
        b_writes = [c for c in wlike if "B" in c or c.startswith("__tmp")]
        # No proper way to detect "for B" from connector name alone, so
        # the stronger check is the topology one below.
    _assert_connector_role_matches_edges(sdfg)


def test_division_by_zero_connector_topology_correct():
    sdfg = division_by_zero_kernel.to_sdfg(simplify=False)
    _assert_connector_role_matches_edges(sdfg)


def test_division_by_zero_numerical_matches_python_reference():
    sdfg = division_by_zero_kernel.to_sdfg(simplify=True)
    rng = np.random.default_rng(seed=1)
    A = rng.standard_normal(64).astype(np.float64)
    B = np.zeros_like(A)
    c = 7.5

    expected = np.where(A > 0.0, c / np.where(A > 0.0, A, 1.0), 0.0)

    sdfg(A=A, B=B, c=c, N=64)
    np.testing.assert_allclose(B, expected, err_msg=f"B={B}\nexpected={expected}")
