# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Making the LogGP model price a GPU mid-flight relayout: the coalescing penalty is a LATENCY-term
effect, not a bandwidth-term one, so it only shows under FINITE device MLP.

A single nest reads ``A`` transposed (``A[j,i]``) under a schedule pinned straight by ``P[i,j]``. When
``A`` is stored row-major (identity) that read is uncoalesced/strided; storing it column-major (perm10)
makes it contiguous. The penalty is ``t(identity) / t(perm10)`` -- exactly the per-nest saving the
mid-flight transpose buys. The model (``count_loop_nest`` + ``nest_memory_time``) is evaluated under the
illustrative CPU and A100-class parameter sets in each memory regime:

    CPU: line == sector, so bytes and messages scale together -> the penalty is regime-invariant.
    GPU: sector (32B) < line (128B). Under the BANDWIDTH regime the small sector makes a strided read
         RELATIVELY CHEAPER than on CPU's 64B line. The real coalescing penalty is the transaction
         count (messages), which binds only in the LATENCY regime (messages * L / concurrency) -- i.e.
         under finite MLP. There it is far larger than CPU.

Takeaway (RELAYOUT_AUDIT.md): to make the LogGP model value a GPU relayout correctly, feed it GPU
params AND a FINITE concurrency (n_cores * core_mlp), never the parallel-schedule ``inf`` default --
otherwise it prices only bytes and UNDER-values the transpose, missing the coalescing win entirely.
"""
import pytest

import dace
from dace.transformation.layout.assignment_costs import EXAMPLE_CPU, EXAMPLE_GPU
from dace.transformation.layout.cost_model.loggp import nest_memory_time
from dace.transformation.layout.cost_model.logp_analysis import count_loop_nest
from dace.transformation.layout.externalize import externalize_nest, nest_entries
from dace.transformation.layout.line_graph import kernel_per_state, line_graph
from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.prepare import prepare_for_layout

N = dace.symbol("N")


@dace.program
def transposed_read(A: dace.float64[N, N], P: dace.float64[N, N], O: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        O[i, j] = A[j, i] + P[i, j]  # A read transposed; P straight pins the schedule to (i, j)


def a_cost(p, perm, concurrency, n=256):
    """``(time, messages)`` attributed to ``A`` in the nest under params ``p``, storage ``perm``
    (None=identity/row-major, [1,0]=column-major), and ``concurrency`` outstanding requests."""
    sdfg = transposed_read.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    kernel_per_state(sdfg)
    kernel = line_graph(sdfg)[0]
    ext = externalize_nest(kernel.state, kernel.map_entry, name=f"probe_{perm}_{id(p)}_{concurrency}")
    if perm is not None:
        PermuteDimensions(permute_map={"A": perm}, add_permute_maps=False).apply_pass(ext, {})
    state = next(iter(ext.states()))
    counts = count_loop_nest(state, nest_entries(state)[0], line_bytes=p.line_bytes, sector_bytes=p.sector_bytes)
    a = counts.arrays["A"]
    subs = {dace.symbol("N"): n}
    messages = float((a.messages_per_iter * counts.total_iters).subs(subs))
    bytes_moved = float((a.bytes_moved_per_iter * counts.total_iters).subs(subs))
    return float(nest_memory_time(p, bytes_moved, messages, concurrency)), messages


def penalty(p, concurrency):
    """t(row-major, read transposed = strided) / t(column-major = contiguous) for A."""
    return a_cost(p, None, concurrency)[0] / a_cost(p, [1, 0], concurrency)[0]


def test_gpu_coalescing_penalty_is_a_latency_effect():
    inf = float("inf")
    cpu_bw, cpu_lat = penalty(EXAMPLE_CPU, inf), penalty(EXAMPLE_CPU, EXAMPLE_CPU.core_mlp)
    gpu_bw, gpu_lat = penalty(EXAMPLE_GPU, inf), penalty(EXAMPLE_GPU, EXAMPLE_GPU.core_mlp)

    # CPU: line == sector, bytes and messages scale together -> penalty is the same in both regimes.
    assert cpu_bw == pytest.approx(cpu_lat, rel=0.05)

    # GPU bandwidth regime: the small 32B sector makes the strided read RELATIVELY cheaper than CPU.
    assert gpu_bw < cpu_bw

    # GPU latency regime (finite MLP): the transaction count binds -> the penalty explodes...
    assert gpu_lat > 2.0 * gpu_bw
    # ...and now exceeds CPU: the mid-flight transpose pays MOST on a latency-bound GPU nest.
    assert gpu_lat > cpu_lat

    # The GPU latency penalty IS the transaction-count ratio (messages), not the byte ratio.
    m_id = a_cost(EXAMPLE_GPU, None, EXAMPLE_GPU.core_mlp)[1]
    m_cm = a_cost(EXAMPLE_GPU, [1, 0], EXAMPLE_GPU.core_mlp)[1]
    assert gpu_lat == pytest.approx(m_id / m_cm, rel=0.02)


def test_model_costs_needs_n_cores_to_see_the_gpu_penalty():
    """The takeaway above, enforced on the PROVIDER. ``model_costs`` prices a nest at
    ``exposed_concurrency(...)``, and for any parallel schedule that is ``inf`` unless ``n_cores`` is
    given -- which zeroes the latency term the whole coalescing penalty lives in. Without the parameter
    the provider could only ever report the byte ratio, i.e. exactly the under-valuation this file exists
    to warn about, with no way for a caller to override it.

    ``EXAMPLE_GPU.core_mlp`` is DEVICE-wide MLP (~500 warp slots), so ``n_cores=1`` is the whole device;
    feeding an SM count on top would multiply the concurrency back into the bandwidth regime."""
    from dace.transformation.layout.assignment_costs import model_costs

    sdfg = transposed_read.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    kernel_per_state(sdfg)
    kernels = line_graph(sdfg)
    kernels[0].map_entry.map.schedule = dace.dtypes.ScheduleType.GPU_Device

    def ratio(**kwargs):
        costs = model_costs(sdfg, kernels, {"N": 256}, p=EXAMPLE_GPU, **kwargs)
        return costs.node_cost[("A", 0, "identity")] / costs.node_cost[("A", 0, "perm10")]

    unbounded, device = ratio(), ratio(n_cores=1)
    assert unbounded == pytest.approx(penalty(EXAMPLE_GPU, float("inf")), rel=0.02)  # bytes only
    assert device == pytest.approx(penalty(EXAMPLE_GPU, EXAMPLE_GPU.core_mlp), rel=0.02)  # transactions
    assert device > 3.0 * unbounded  # the penalty the unbounded pricing silently drops
    # it really is the transaction-count ratio, not the byte ratio
    messages_identity = a_cost(EXAMPLE_GPU, None, EXAMPLE_GPU.core_mlp)[1]
    messages_colmajor = a_cost(EXAMPLE_GPU, [1, 0], EXAMPLE_GPU.core_mlp)[1]
    assert device == pytest.approx(messages_identity / messages_colmajor, rel=0.02)


if __name__ == "__main__":
    test_gpu_coalescing_penalty_is_a_latency_effect()
    test_model_costs_needs_n_cores_to_see_the_gpu_penalty()
    print("gpu relayout cost-model test PASS")
