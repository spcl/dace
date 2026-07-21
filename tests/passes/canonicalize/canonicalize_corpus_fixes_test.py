# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for the (numerically verified) canonicalization/frontend fixes
landed for the npbench+polybench corpus.

Only the fixes confirmed value-preserving are pinned here:

* ``_find_new_name`` is connector-aware (sdfg.py) -- fixed symm.
* bare-name dtype casts (``f32(0)`` where ``f32 = dace.float32``) parse
  (newast.py) -- fixed azimint_naive/resnet parsing.

NOTE: three earlier fix attempts (expand_nested_sdfg_inputs WCR handling,
split_tasklets symbol-only-substatement anchoring, isolate_nested_sdfg re-clone
guard) were REVERTED -- they let canonicalization *complete* but produced silent
miscompiles (atax/cholesky/correlation/covariance numerically wrong, correlation
-inf). Loud failure beats a silent wrong answer; those kernels canon-fail again
until a value-preserving fix is found. Do not re-add their tests without a numeric
(canon-output == baseline-output) assertion.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np

import dace


def test_find_new_name_avoids_tasklet_connector():
    """``_find_new_name`` must not return a name already used as a tasklet
    connector (regression: ``add_scalar('tmp', find_new_name=True)`` collided with
    a ``tmp`` connector -> 'Connector already used as a symbol' in symm)."""
    sdfg = dace.SDFG("find_new_name_conn")
    state = sdfg.add_state()
    tasklet = state.add_tasklet("t", set(), {"tmp"}, "tmp = 0.0")
    sdfg.add_scalar("out", dace.float64, transient=True)
    state.add_edge(tasklet, "tmp", state.add_access("out"), None, dace.Memlet("out[0]"))

    new_name = sdfg.add_scalar("tmp", dace.float64, transient=True, find_new_name=True)
    assert new_name != "tmp"
    assert "tmp" not in sdfg.arrays


def test_bare_name_dtype_cast_parses():
    """A dtype typeclass bound to a plain name and used as a cast (``f32(0)``)
    must parse like the attribute form ``dace.float32(0)`` (regression: npbench
    ``dc_float(0)`` -> 'Unexpected call expression type Constant: float')."""
    N = dace.symbol("N")
    f32 = dace.float32

    @dace.program
    def caster(a: dace.float64[N]):
        return a + f32(0)

    sdfg = caster.to_sdfg()  # used to raise TypeError during parsing
    a = np.ones(8, dtype=np.float64)
    assert np.allclose(sdfg(a=a, N=8), a)


def test_finalize_does_not_persist_reduction_scalar():
    """``finalize_for_target`` must not leave a scalar / length-1 array in persistent
    storage. A size-1 WCR accumulator made persistent lands in the state struct
    (``__state->x``) and breaks the OpenMP ``reduction(op:var)`` clause -- the go_fast
    ``trace`` failure (``reduction(+:__state->__0_trace)`` did not compile). The
    accumulator must stay a non-persistent (register) local."""
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    Nsym = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def reduce_prog(a: dace.float64[Nsym]):
        s = 0.0
        for i in range(Nsym):
            s += a[i]
        return a + s

    sdfg = reduce_prog.to_sdfg(simplify=False)
    sdfg = canonicalize(sdfg,
                        validate=True,
                        target="cpu",
                        peel_limit=4,
                        break_anti_dependence=True,
                        interchange_carry_with_map=True,
                        scatter_to_guarded_maps=True)
    finalize_for_target(sdfg, "cpu")

    persistent_scalars = [
        f"{sd.label}:{nm}" for sd in sdfg.all_sdfgs_recursive() for nm, desc in sd.arrays.items()
        if desc.transient and desc.total_size == 1 and desc.lifetime == dace.dtypes.AllocationLifetime.Persistent
    ]
    assert not persistent_scalars, f"finalize left size-1 transient(s) persistent: {persistent_scalars}"

    # And it compiles: the reduction accumulator is a local lvalue, not __state->.
    csdfg = sdfg.compile()
    n = 256
    rng = np.random.default_rng(0)
    a = rng.random(n)
    assert np.allclose(csdfg(a=a, N=n), a + a.sum())


def test_finalize_selects_openmp_for_reduce():
    """``finalize_for_target`` must select the ``OpenMP`` implementation for a lifted ``Reduce``
    library node on CPU -- never the default ``pure`` (which builds a single-scalar WCR map that
    lowers to a serialized ``omp critical`` for min/max or a contended ``omp atomic`` for sum,
    100-3000x slower). The generated code must carry a real ``reduction(op:var)`` clause and no
    ``omp atomic`` / ``omp critical`` on the accumulator, and stay bit-exact vs numpy."""
    import numpy as np
    from dace.libraries.standard.nodes.reduce import Reduce
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    Nsym = dace.symbol("N", dtype=dace.int64)

    for op, npfn, clause in (("sum", np.sum, "reduction(+:"), ("max", np.max, "reduction(max:")):

        @dace.program
        def reducer(a: dace.float64[Nsym]):
            return npfn(a)

        sdfg = reducer.to_sdfg(simplify=False)
        sdfg = canonicalize(sdfg,
                            validate=True,
                            target="cpu",
                            peel_limit=4,
                            break_anti_dependence=True,
                            interchange_carry_with_map=True,
                            scatter_to_guarded_maps=True)
        finalize_for_target(sdfg, "cpu")

        reduces = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
        assert reduces, f"{op}: canonicalize should lift the loop-reduction to a Reduce library node"
        assert all(n.implementation == "OpenMP" for n in reduces), \
            f"{op}: Reduce must be OpenMP, got {[n.implementation for n in reduces]}"

        code = sdfg.generate_code()[0].clean_code
        assert clause.replace(" ", "") in code.replace(" ", ""), f"{op}: missing {clause}...) clause"
        assert "#pragma omp atomic" not in code, f"{op}: reduction must not fall back to omp atomic"
        assert "#pragma omp critical" not in code, f"{op}: reduction must not fall back to omp critical"

        rng = np.random.default_rng(0)
        a = rng.random(4096)
        got = np.asarray(sdfg.compile()(a=a, N=4096)).reshape(())
        assert np.allclose(got, npfn(a)), f"{op}: reduction not bit-exact vs numpy"


def test_finalize_selects_openmp_scan_for_prefix_scan():
    """``finalize_for_target`` must lower a lifted prefix ``Scan`` to the parallel ``CPU``
    expansion (OpenMP 5.0 ``reduction(inscan, ...)`` + ``#pragma omp scan``), never the serial
    ``pure`` loop. Bit-exact vs ``np.cumsum``."""
    import numpy as np
    from dace.libraries.standard.nodes.scan import Scan
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    Nsym = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def prefix_sum(a: dace.float64[Nsym], out: dace.float64[Nsym]):
        out[0] = a[0]
        for i in range(1, Nsym):
            out[i] = out[i - 1] + a[i]

    sdfg = prefix_sum.to_sdfg(simplify=False)
    sdfg = canonicalize(sdfg,
                        validate=True,
                        target="cpu",
                        peel_limit=4,
                        break_anti_dependence=True,
                        interchange_carry_with_map=True,
                        scatter_to_guarded_maps=True)
    finalize_for_target(sdfg, "cpu")

    scans = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan)]
    assert scans, "canonicalize should lift the prefix-sum loop to a Scan library node"
    assert all(n.implementation == "CPU" for n in scans), \
        f"Scan must use the parallel CPU (OpenMP-scan) expansion, got {[n.implementation for n in scans]}"

    rng = np.random.default_rng(0)
    a = rng.random(4096)
    out = np.zeros(4096)
    sdfg.compile()(a=a, out=out, N=4096)
    assert np.allclose(out, np.cumsum(a)), "prefix scan not bit-exact vs np.cumsum"


def test_finalize_nested_reduction_stays_sequential():
    """A reduction NESTED inside a parallel map (per-row sum) must lower to the sequential ``pure``
    accumulate -- adhering to its ``Sequential`` schedule -- NOT ``OpenMP``. An ``OpenMP`` reduction
    here would open a nested ``#pragma omp parallel`` per outer iteration (the "constant parallel
    reductions" slowdown). The generated code must contain exactly one parallel region (the outer
    map), and the result stays bit-exact."""
    import numpy as np
    from dace.libraries.standard.nodes.reduce import Reduce
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    Nsym = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def rowsum(A: dace.float64[Nsym, Nsym], y: dace.float64[Nsym]):
        for i in range(Nsym):
            acc = 0.0
            for j in range(Nsym):
                acc += A[i, j]
            y[i] = acc

    sdfg = rowsum.to_sdfg(simplify=False)
    sdfg = canonicalize(sdfg,
                        validate=True,
                        target="cpu",
                        peel_limit=4,
                        break_anti_dependence=True,
                        interchange_carry_with_map=True,
                        scatter_to_guarded_maps=True)
    finalize_for_target(sdfg, "cpu")

    reduces = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert all(n.implementation != "OpenMP" for n in reduces), \
        f"a nested (Sequential-scheduled) Reduce must not be OpenMP, got {[n.implementation for n in reduces]}"
    code = sdfg.generate_code()[0].clean_code
    assert code.count("#pragma omp parallel") == 1, \
        f"expected one parallel region (outer map), got {code.count('#pragma omp parallel')} (nested reduction?)"

    rng = np.random.default_rng(0)
    A = rng.random((256, 256))
    y = np.zeros(256)
    sdfg.compile()(A=A, y=y, N=256)
    assert np.allclose(y, A.sum(axis=1)), "nested row-sum not bit-exact"


def test_reduction_in_sequential_loop_is_not_parallelized():
    """The k-reduction nested in a doubly-carried nest must not itself become a parallel WCR-map.

    Lifting the reduction re-enters a map once per outer iteration, so the OpenMP fork/join
    dominates the tiny inner reduction: this is the nussinov ``table[i,j] = max(..., table[i,k] +
    table[k+1,j])`` k-reduction, which measured ~340x slower than the sequential baseline
    ``auto_optimize`` keeps. Both original loops are therefore ``pinned_sequential``.

    That pin is on the LOOPS, not on the program: ``WavefrontSkew`` may still skew the (i, j) nest
    and parallelize the resulting wavefront axis, which forks once per wavefront step rather than
    once per (i, j) pair. So this asserts the pin holds and the result is bit-exact, rather than
    counting parallel regions -- a region belonging to a skewed wavefront is a different (and
    legitimate) shape from the fork-per-outer-iteration one this test exists to forbid."""
    import numpy as np
    from dace.sdfg.state import LoopRegion
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    Nsym = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def nested_seq_reduction(A: dace.float64[Nsym, Nsym], out: dace.float64[Nsym, Nsym]):
        for i in range(1, Nsym):
            for j in range(1, Nsym):
                out[i, j] = out[i, j - 1] + out[i - 1, j]  # carried on BOTH i and j -> both sequential
                for k in range(Nsym):
                    # in-place compute-then-accumulate max-reduction over k (nussinov's shape)
                    out[i, j] = max(out[i, j], A[i, k] + A[k, j])

    sdfg = nested_seq_reduction.to_sdfg(simplify=False)
    sdfg = canonicalize(sdfg,
                        validate=True,
                        target="cpu",
                        peel_limit=4,
                        break_anti_dependence=True,
                        interchange_carry_with_map=True,
                        scatter_to_guarded_maps=True)
    finalize_for_target(sdfg, "cpu")

    surviving = [c for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion)]
    assert surviving, "the nest collapsed entirely; this test no longer exercises the pin"
    assert all(c.pinned_sequential for c in surviving), \
        (f"every surviving loop of a doubly-carried nest must stay pinned sequential, got "
         f"{[(c.label, c.pinned_sequential) for c in surviving]}")

    code = sdfg.generate_code()[0].clean_code
    # No WCR-map over the reduction axis: the k-reduction must not be the thing that parallelizes.
    assert "omp parallel for" not in code or "_skew" in code, \
        "the k-reduction was lifted to a parallel WCR-map (fork-per-outer-iteration)"

    rng = np.random.default_rng(0)
    N = 24
    A = rng.random((N, N))
    out = rng.random((N, N))
    ref = out.copy()
    for i in range(1, N):
        for j in range(1, N):
            ref[i, j] = ref[i, j - 1] + ref[i - 1, j]
            for k in range(N):
                ref[i, j] = max(ref[i, j], A[i, k] + A[k, j])
    got = out.copy()
    sdfg.compile()(A=A, out=got, N=N)
    assert np.array_equal(got, ref), "nested sequential reduction not bit-exact"


def test_finalize_never_selects_mkl_prefers_openblas():
    """The canonicalize perf tail must never pick ``MKL`` -- it prefers OpenBLAS (and OpenMP /
    HPTT / cuBLAS). A large matmul lowers to OpenBLAS, and no library node is left on ``MKL``."""
    from dace.sdfg import nodes
    from dace.transformation.passes.canonicalize.finalize import (finalize_for_target,
                                                                  canonicalize_fast_library_priority)

    assert 'MKL' not in canonicalize_fast_library_priority(dace.dtypes.DeviceType.CPU)

    Nsym = dace.symbol("N")

    @dace.program
    def gemm(a: dace.float64[Nsym, Nsym], b: dace.float64[Nsym, Nsym], c: dace.float64[Nsym, Nsym]):
        c[:] = a @ b

    sdfg = gemm.to_sdfg(simplify=True)
    finalize_for_target(sdfg, "cpu", validate=False)
    lib_impls = [n.implementation for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]
    assert 'MKL' not in lib_impls, f"MKL must never be selected, got {lib_impls}"
    assert 'OpenBLAS' in lib_impls, f"large matmul should lower to OpenBLAS, got {lib_impls}"


def test_finalize_gpu_offloads_and_sets_domain_matched_block():
    """``finalize_for_target(sdfg, 'gpu')`` must offload to the device and choose a thread-block
    matching the iteration domain (an ``N x N`` map -> a ``16x16`` block, not the ``32,1,1``
    default), then generate CUDA. Codegen-only (no device needed); the offload is idempotent."""
    from dace.sdfg import nodes
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    Nsym = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def madd(a: dace.float64[Nsym, Nsym], b: dace.float64[Nsym, Nsym], c: dace.float64[Nsym, Nsym]):
        for i in range(Nsym):
            for j in range(Nsym):
                c[i, j] = a[i, j] + b[i, j]

    sdfg = madd.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True, target="gpu")
    finalize_for_target(sdfg, "gpu", validate=True)

    gpu_maps = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
    ]
    assert gpu_maps, "finalize(gpu) should offload the loop nest to a GPU_Device map"
    assert any(n.map.gpu_block_size == [16, 16, 1] for n in gpu_maps), \
        f"N x N map should get a 16x16 block, got {[n.map.gpu_block_size for n in gpu_maps]}"
    titles = [c.title for c in sdfg.generate_code()]
    assert any("cuda" in t.lower() for t in titles), f"expected CUDA codegen, got {titles}"

    # Idempotent: a second finalize (already offloaded) does not double-offload / crash.
    finalize_for_target(sdfg, "gpu", validate=True)


def test_finalize_transient_storage_converts_len1_transient_array_to_scalar():
    """``finalize_transient_storage`` converts every length-1 *transient* array to a Scalar
    (a single internal value belongs in a scalar, not a heap array), while leaving a
    length-1 *non-transient* array (an SDFG-external return / handle) as an Array."""
    from dace import data as ddata
    from dace.transformation.passes.canonicalize.finalize import finalize_transient_storage

    sdfg = dace.SDFG("fin_len1")
    sdfg.add_array("A", [4], dace.float64)  # non-transient input
    sdfg.add_array("keep", [1], dace.float64)  # non-transient len-1 -> stays an Array
    sdfg.add_array("acc", [1], dace.float64, transient=True)  # transient len-1 -> becomes a Scalar
    st = sdfg.add_state()
    a, acc, keep = st.add_access("A"), st.add_access("acc"), st.add_access("keep")
    t = st.add_tasklet("t", {"x"}, {"y"}, "y = x")
    st.add_edge(a, None, t, "x", dace.Memlet("A[0]"))
    st.add_edge(t, "y", acc, None, dace.Memlet("acc[0]"))
    t2 = st.add_tasklet("t2", {"x"}, {"y"}, "y = x")
    st.add_edge(acc, None, t2, "x", dace.Memlet("acc[0]"))
    st.add_edge(t2, "y", keep, None, dace.Memlet("keep[0]"))
    sdfg.validate()

    finalize_transient_storage(sdfg, dace.dtypes.DeviceType.CPU)

    assert isinstance(sdfg.arrays["acc"], ddata.Scalar), "len-1 transient array must become a Scalar"
    assert isinstance(sdfg.arrays["keep"], ddata.Array), "non-transient len-1 array must stay an Array"
    leftover = [
        f"{sd.label}:{n}" for sd in sdfg.all_sdfgs_recursive() for n, d in sd.arrays.items()
        if d.transient and isinstance(d, ddata.Array) and d.total_size == 1
    ]
    assert not leftover, f"len-1 transient arrays left unconverted: {leftover}"
    sdfg.validate()
