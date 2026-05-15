# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import copy
import numpy as np
import pytest
from dace import Union
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes import eliminate_branches
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

LEN_1D = dace.symbol("LEN_1D")
ITERATIONS = dace.symbol("ITERATIONS")


@pytest.fixture(params=["scalar", "masked"])
def remainder_strategy(request) -> str:
    """Parametrise TSVC tests over the remainder-handling strategies wired
    into ``VectorizeCPU``. Matches the per-directory ``conftest.py`` fixture
    defined under ``tests/passes/vectorization/`` (this file lives one level
    up so it needs its own fixture). There is no ``divides_evenly``
    strategy — P2 proves divisibility itself and skips the split."""
    return request.param


@pytest.fixture(params=["merge"])
def branch_mode(request) -> str:
    """TSVC kernels here have no branches; pinned to ``merge`` (the default)
    rather than the full ``[fp_factor, merge]`` matrix to avoid pointless
    fan-out. Tests that exercise branches can declare a local parametrize
    to override."""
    return request.param


def run_vectorization_test(dace_func: Union[dace.SDFG, callable],
                           arrays,
                           params,
                           vector_width=8,
                           simplify=True,
                           skip_simplify=None,
                           sdfg_name=None,
                           fuse_overlapping_loads=False,
                           insert_copies=True,
                           filter_map=-1,
                           cleanup=False,
                           from_sdfg=False,
                           no_inline=False,
                           exact=None,
                           apply_loop_to_map=False,
                           remainder_strategy: str = "scalar",
                           branch_mode: str = "merge",
                           lower_to_intrinsics: bool = False):

    # Create copies for comparison
    arrays_orig = {k: copy.deepcopy(v) for k, v in arrays.items()}
    arrays_vec = {k: copy.deepcopy(v) for k, v in arrays.items()}

    # Original SDFG
    # Suffix sdfg_name with parametrisation keys so parallel pytest workers
    # don't race on a shared .dacecache/<name>/ build dir when the same
    # kernel is exercised under multiple variants.
    if sdfg_name is not None:
        sdfg_name = f"{sdfg_name}_{branch_mode}_{remainder_strategy}"

    if not from_sdfg:
        sdfg: dace.SDFG = dace_func.to_sdfg(simplify=False)
        sdfg.name = sdfg_name
        if simplify:
            sdfg.simplify(validate=True, validate_all=True, skip=skip_simplify or set())
    else:
        sdfg: dace.SDFG = dace_func
        if sdfg_name is not None:
            sdfg.name = sdfg_name

    if apply_loop_to_map:
        sdfg.apply_transformations_repeated(LoopToMap())
        sdfg.simplify()

    c_sdfg = sdfg.compile()

    # Vectorized SDFG
    copy_sdfg: dace.SDFG = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_vectorized"

    if cleanup:
        for e, g in copy_sdfg.all_edges_recursive():
            if isinstance(g, dace.SDFGState):
                if (isinstance(e.src, dace.nodes.AccessNode) and isinstance(e.dst, dace.nodes.AccessNode)
                        and isinstance(g.sdfg.arrays[e.dst.data], dace.data.Scalar)
                        and e.data.other_subset is not None):
                    # Add assignment taskelt
                    src_data = e.src.data
                    src_subset = e.data.subset if e.data.data == src_data else e.data.other_subset
                    dst_data = e.dst.data
                    dst_subset = e.data.subset if e.data.data == dst_data else e.data.other_subset
                    g.remove_edge(e)
                    t = g.add_tasklet(name=f"assign_dst_{dst_data}_from_{src_data}",
                                      code="_out = _in",
                                      inputs={"_in"},
                                      outputs={"_out"})
                    g.add_edge(e.src, e.src_conn, t, "_in",
                               dace.memlet.Memlet(data=src_data, subset=copy.deepcopy(src_subset)))
                    g.add_edge(t, "_out", e.dst, e.dst_conn,
                               dace.memlet.Memlet(data=dst_data, subset=copy.deepcopy(dst_subset)))
        copy_sdfg.validate()

    if filter_map != -1:
        map_labels = [n.map.label for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
        filter_map_labels = map_labels[0:filter_map]
        filter_map = filter_map_labels
    else:
        filter_map = None

    if branch_mode == "fp_factor":
        branch_kwargs = dict(use_fp_factor=True, branch_normalization=False)
    elif branch_mode == "merge":
        branch_kwargs = dict(use_fp_factor=False, branch_normalization=True)
    else:
        raise ValueError(f"branch_mode must be 'fp_factor' or 'merge', got {branch_mode!r}")

    VectorizeCPU(vector_width=vector_width,
                 fuse_overlapping_loads=fuse_overlapping_loads,
                 insert_copies=insert_copies,
                 apply_on_maps=filter_map,
                 no_inline=no_inline,
                 fail_on_unvectorizable=True,
                 remainder_strategy=remainder_strategy,
                 lower_to_intrinsics=lower_to_intrinsics,
                 **branch_kwargs).apply_pass(copy_sdfg, {})
    copy_sdfg.validate()

    c_copy_sdfg = copy_sdfg.compile()

    # Run both
    c_sdfg(**arrays_orig, **params)

    c_copy_sdfg(**arrays_vec, **params)

    # Compare results
    for name in arrays.keys():
        assert np.allclose(arrays_orig[name], arrays_vec[name], rtol=1e-32), \
            f"{name} Diff: {arrays_orig[name] - arrays_vec[name]}"
        if exact is not None:
            diff = arrays_vec[name] - exact
            assert np.allclose(arrays_vec[name], exact, rtol=0, atol=1e-300), \
                f"{name} Diff: max abs diff = {np.max(np.abs(diff))}"
    return copy_sdfg


def initialise_arrays():
    # Create array handles equivalent to the globals in C
    # Adjust shapes to match your actual code.
    a = np.zeros(LEN_1D, dtype=np.float64)
    b = np.zeros(LEN_1D, dtype=np.float64)
    c = np.zeros(LEN_1D, dtype=np.float64)
    d = np.zeros(LEN_1D, dtype=np.float64)
    e = np.zeros(LEN_1D, dtype=np.float64)
    aa = np.zeros(LEN_1D, dtype=np.float64)
    bb = np.zeros(LEN_1D, dtype=np.float64)
    cc = np.zeros(LEN_1D, dtype=np.float64)
    return a, b, c, d, e, aa, bb, cc


@dace.program
def dace_s317(q: dace.float64[1]):
    for nl in range(50):
        q[0] = 1.0
        # Inner reduction: q *= 0.99 repeated LEN_1D/2 times
        # Equivalent to: q = 0.99**(LEN_1D/2) but we follow the loop exactly.
        for i in range(LEN_1D // 2):
            q[0] *= 0.99


def test_s317(remainder_strategy, branch_mode):
    q = np.zeros(1, dtype=np.float64)
    run_vectorization_test(dace_func=dace_s317,
                           arrays={"q": q},
                           params={"LEN_1D": 64},
                           sdfg_name="dace_s317",
                           apply_loop_to_map=True,
                           remainder_strategy=remainder_strategy,
                           branch_mode=branch_mode)


@dace.program
def dace_s3251(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
               e: dace.float64[LEN_1D]):
    for nl in range(10):
        for i in range(LEN_1D - 1):
            a[i + 1] = b[i] + c[i]
            b[i] = c[i] * e[i]
            d[i] = a[i] * e[i]


@dace.program
def dace_s491(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D],
              ip: dace.int32[LEN_1D]):
    for nl in range(10):
        for i in range(LEN_1D):
            a[ip[i]] = b[i] + c[i] * d[i]


def test_s491(remainder_strategy, branch_mode):
    LEN_1D_val = 64

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.rand(LEN_1D_val).astype(np.float64)

    ip = np.random.permutation(LEN_1D_val).astype(np.int32)

    # s491 has indirect access (a[ip[i]]) so masked-remainder requires
    # lower_to_intrinsics=True per the locked Option B contract.
    run_vectorization_test(
        dace_func=dace_s491,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "ip": ip
        },
        params={"LEN_1D": LEN_1D_val},
        sdfg_name="dace_s491",
        apply_loop_to_map=True,
        remainder_strategy=remainder_strategy,
        branch_mode=branch_mode,
        lower_to_intrinsics=(remainder_strategy == "masked"),
    )

    return a


@dace.program
def dace_s293(a: dace.float64[LEN_1D]):
    for nl in range(50):
        # loop peeling structure preserved
        a0 = a[0]
        for i in range(LEN_1D):
            a[i] = a0


def test_s293(remainder_strategy, branch_mode):
    LEN_1D_val = 64  # example, adjust as needed

    # Allocate array a
    a = np.random.rand(LEN_1D_val).astype(np.float64)

    # Run DaCe test harness — InsertAssignTaskletsAtMapBoundary normalizes the
    # ``a[0] = a0`` slice (AccessNode → AccessNode with other_subset) into a
    # plain assign-tasklet chain, so the vectorize pipeline's
    # ``no_other_subset`` invariant holds.
    run_vectorization_test(
        dace_func=dace_s293,
        arrays={"a": a},
        params={"LEN_1D": LEN_1D_val},  # or your iteration count
        sdfg_name="dace_s293",
        apply_loop_to_map=True,
        remainder_strategy=remainder_strategy,
        branch_mode=branch_mode,
    )

    return a


def test_s3251(remainder_strategy, branch_mode):
    LEN_1D_val = 64

    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.rand(LEN_1D_val).astype(np.float64)
    e = np.random.rand(LEN_1D_val).astype(np.float64)

    run_vectorization_test(
        dace_func=dace_s3251,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e
        },
        params={"LEN_1D": LEN_1D_val},
        sdfg_name="dace_s3251",
        apply_loop_to_map=True,
        remainder_strategy=remainder_strategy,
        branch_mode=branch_mode,
    )

    return a, b, c, d, e


@dace.program
def dace_s441(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for nl in range(ITERATIONS):  # or iterations parameter
        for i in range(LEN_1D):
            if d[i] < 0.0:
                a[i] = a[i] + b[i] * c[i]
            elif d[i] == 0.0:
                a[i] = a[i] + b[i] * b[i]
            else:
                a[i] = a[i] + c[i] * c[i]


@dace.program
def dace_s441_v2(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D], d: dace.float64[LEN_1D]):
    for i in dace.map[0:LEN_1D:1]:
        if d[i] < 0.0:
            a[i] = a[i] + b[i] * c[i]
        elif d[i] == 0.0:
            a[i] = a[i] + b[i] * b[i]
        else:
            a[i] = a[i] + c[i] * c[i]


def test_s441(remainder_strategy):
    LEN_1D_val = 64  # example length

    # Allocate random inputs
    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.randn(LEN_1D_val).astype(np.float64)  # includes negative, zero, positive

    sdfg = dace_s441.to_sdfg()
    eliminate_branches.EliminateBranches().apply_pass(sdfg, {})
    branches = {n for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(branches) == 0, f"EliminateBranches left {len(branches)} ConditionalBlock(s) in dace_s441"

    # s441 has EliminateBranches applied manually → use_fp_factor=True is
    # the matching branch-lowering knob in VectorizeCPU. masked-remainder is
    # not compatible with fp_factor (locked plan decision), so pin to
    # branch_mode='fp_factor' and skip the masked variant.
    if remainder_strategy == "masked":
        pytest.skip("s441 uses EliminateBranches → fp_factor; masked-remainder requires branch_normalization (locked plan rule)")
    run_vectorization_test(
        dace_func=dace_s441,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={
            "LEN_1D": LEN_1D_val,
            "ITERATIONS": 1
        },
        sdfg_name="dace_s441",
        apply_loop_to_map=True,
        remainder_strategy=remainder_strategy,
        branch_mode="fp_factor",
    )

    return a


def test_s441_v2(remainder_strategy):
    LEN_1D_val = 64  # example length

    # Allocate random inputs
    a = np.random.rand(LEN_1D_val).astype(np.float64)
    b = np.random.rand(LEN_1D_val).astype(np.float64)
    c = np.random.rand(LEN_1D_val).astype(np.float64)
    d = np.random.randn(LEN_1D_val).astype(np.float64)  # includes negative, zero, positive

    sdfg = dace_s441_v2.to_sdfg()
    eliminate_branches.EliminateBranches().apply_pass(sdfg, {})
    branches = {n for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
    assert len(branches) == 0, f"EliminateBranches left {len(branches)} ConditionalBlock(s) in dace_s441_v2"

    if remainder_strategy == "masked":
        pytest.skip("s441_v2 uses EliminateBranches → fp_factor; masked-remainder requires branch_normalization")
    run_vectorization_test(
        dace_func=dace_s441_v2,
        arrays={
            "a": a,
            "b": b,
            "c": c,
            "d": d
        },
        params={"LEN_1D": LEN_1D_val},
        sdfg_name="dace_s441_v2",
        apply_loop_to_map=True,
        remainder_strategy=remainder_strategy,
        branch_mode="fp_factor",
    )

    return a
