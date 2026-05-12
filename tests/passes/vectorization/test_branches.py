# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import math
import copy
from typing import Tuple
import dace
import pytest
import numpy
from dace import InterstateEdge
from dace import Union
from dace.properties import CodeBlock
from dace.sdfg import ControlFlowRegion
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import branch_elimination
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import (
    ReplaceSTDExpWithDaCeExp, ReplaceSTDLogWithDaCeLog, ReplaceSTDPowWithDaCePow,
)
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.passes.vectorization._harness import (
    run_vectorization_test,
    N, S, S1, S2, klev, kidia, kfdia, n, m, nnz,
    KLON, KLEV, NCLDQL, NCLDQI, ssym, X, Y, C,
    log, exp, pow,
    _get_disjoint_chain_sdfg, _get_disjoint_chain_sdfg_two,
    _get_cloudsc_snippet_three, _get_cloudsc_snippet_four,
    _get_map_inside_nested_map,
    _get_dependency_edge_to_unary_symbol_sdfg,
    _get_unstructured_access_cloudsc_sdfg,
)

@dace.program
def unsupported_op(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        A[i, j] = math.exp(B[i, j])


@dace.program
def division_by_zero(A: dace.float64[N], B: dace.float64[N], c: dace.float64):
    for i in dace.map[
            0:N,
    ]:
        if A[i] > 0.0:
            B[i] = c / A[i]
        else:
            B[i] = 0.0


@dace.program
def tasklets_in_if(
    a: dace.float64[S, S],
    b: dace.float64[S, S],
    d: dace.float64[S, S],
    c: dace.float64,
):
    for i in dace.map[0:S:1]:
        for j in dace.map[0:S:1]:
            if a[i, j] > c:
                b[i, j] = b[i, j] + d[i, j]
            else:
                b[i, j] = b[i, j] - d[i, j]
            b[i, j] = (1 - a[i, j]) * c


@dace.program
def tasklets_in_if_two(
    a: dace.float64[S, S],
    b: dace.float64,
    c: dace.float64[S, S],
    d: dace.float64[S, S],
    e: dace.float64[S, S],
    f: dace.float64,
):
    for i in dace.map[0:S:1]:
        for j in dace.map[0:S:1]:
            if a[i, j] < b:
                e[i, j] = (c[i, j] * f * a[i, j] * 2.0) - a[i, j]


def test_division_by_zero_cpu(branch_mode):
    N = 256
    A = numpy.random.random((N, ))
    B = numpy.random.random((N, ))

    run_vectorization_test(
        dace_func=division_by_zero,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'N': N,
            "c": 8.9
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="division_by_zero",
        branch_mode=branch_mode,
    )


def test_unsupported_op(branch_mode):
    A = numpy.random.random((64, 64))
    B = numpy.random.random((64, 64))

    run_vectorization_test(dace_func=unsupported_op,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': 64},
                           vector_width=4,
                           skip_simplify={"ScalarToSymbolPromotion"},
                           save_sdfgs=True,
                           sdfg_name="unsupported_op",
                           branch_mode=branch_mode)


def test_unsupported_op_two(branch_mode):
    A = numpy.random.random((64, 64))
    B = numpy.random.random((64, 64))

    run_vectorization_test(dace_func=unsupported_op,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'N': 64},
                           vector_width=4,
                           save_sdfgs=True,
                           sdfg_name="unsupported_op_two",
                           branch_mode=branch_mode)


def test_tasklets_in_if(branch_mode):
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((_S, _S))
    D = numpy.random.random((_S, _S))
    C = numpy.random.random((1, ))

    arrays_orig = {'a': copy.deepcopy(A), 'b': copy.deepcopy(B), 'c': copy.deepcopy(C), 'd': copy.deepcopy(D)}
    arrays_vec = {'a': copy.deepcopy(A), 'b': copy.deepcopy(B), 'c': copy.deepcopy(C), 'd': copy.deepcopy(D)}

    sdfg = tasklets_in_if.to_sdfg()
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_vectorized"
    sdfg.save("nested_tasklets_in_if.sdfg")
    branch_kwargs = {"use_fp_factor": False, "branch_normalization": True} if branch_mode == "merge" else {}
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True, **branch_kwargs).apply_pass(copy_sdfg, {})
    copy_sdfg.save("nested_tasklets_in_if_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=arrays_orig['a'], b=arrays_orig['b'], c=arrays_orig['c'][0], d=arrays_orig['d'], S=_S)
    c_copy_sdfg(a=arrays_vec['a'], b=arrays_vec['b'], c=arrays_vec['c'][0], d=arrays_vec['d'], S=_S)

    assert numpy.allclose(arrays_orig['a'], arrays_vec['a'],
                          atol=1e-12), f"A Diff: {arrays_orig['a'] - arrays_vec['a']}"
    assert numpy.allclose(arrays_orig['b'], arrays_vec['b'],
                          atol=1e-12), f"B Diff: {arrays_orig['b'] - arrays_vec['b']}"
    assert numpy.allclose(arrays_orig['c'], arrays_vec['c'],
                          atol=1e-12), f"C Diff: {arrays_orig['c'] - arrays_vec['c']}"
    assert numpy.allclose(arrays_orig['d'], arrays_vec['d'],
                          atol=1e-12), f"D Diff: {arrays_orig['d'] - arrays_vec['d']}"


def test_tasklets_in_if_two(branch_mode):
    _S = 64
    A = numpy.random.random((_S, _S))
    B = numpy.random.random((1, ))
    C = numpy.random.random((_S, _S))
    D = numpy.random.random((_S, _S))
    E = numpy.random.random((_S, _S))
    F = numpy.random.random((1, ))

    arrays_orig = {
        'a': copy.deepcopy(A),
        'b': copy.deepcopy(B),
        'c': copy.deepcopy(C),
        'd': copy.deepcopy(D),
        'e': copy.deepcopy(E),
        'f': copy.deepcopy(F)
    }
    arrays_vec = {
        'a': copy.deepcopy(A),
        'b': copy.deepcopy(B),
        'c': copy.deepcopy(C),
        'd': copy.deepcopy(D),
        'e': copy.deepcopy(E),
        'f': copy.deepcopy(F)
    }

    sdfg = tasklets_in_if_two.to_sdfg(simplify=False)
    sdfg.simplify(skip=["ScalarToSymbolPromotion"])
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = copy_sdfg.name + "_vectorized"
    sdfg.save("nested_tasklets.sdfg")
    branch_kwargs = {"use_fp_factor": False, "branch_normalization": True} if branch_mode == "merge" else {}
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True, **branch_kwargs).apply_pass(copy_sdfg, {})
    copy_sdfg.save("nested_tasklets_vectorized.sdfg")

    c_sdfg = sdfg.compile()
    c_copy_sdfg = copy_sdfg.compile()

    c_sdfg(a=arrays_orig['a'],
           b=arrays_orig['b'][0],
           c=arrays_orig['c'],
           d=arrays_orig['d'],
           e=arrays_orig['e'],
           f=arrays_orig['f'][0],
           S=_S)
    c_copy_sdfg(a=arrays_vec['a'],
                b=arrays_vec['b'][0],
                c=arrays_vec['c'],
                d=arrays_vec['d'],
                e=arrays_vec['e'],
                f=arrays_vec['f'][0],
                S=_S)

    assert numpy.allclose(arrays_orig['a'], arrays_vec['a'], atol=1e-12), f"{arrays_orig['a'] - arrays_vec['a']}"
    assert numpy.allclose(arrays_orig['b'], arrays_vec['b'], atol=1e-12), f"{arrays_orig['b'] - arrays_vec['b']}"
    assert numpy.allclose(arrays_orig['c'], arrays_vec['c'], atol=1e-12), f"{arrays_orig['c'] - arrays_vec['c']}"
    assert numpy.allclose(arrays_orig['d'], arrays_vec['d'], atol=1e-12), f"{arrays_orig['d'] - arrays_vec['d']}"
    assert numpy.allclose(arrays_orig['e'], arrays_vec['e'], atol=1e-12), f"{arrays_orig['e'] - arrays_vec['e']}"
    assert numpy.allclose(arrays_orig['f'], arrays_vec['f'], atol=1e-12), f"{arrays_orig['f'] - arrays_vec['f']}"


@dace.program
def interstate_boolean_op_one(A: dace.float64[N, N], B: dace.float64[N, N], c0: dace.int64):
    for i, j in dace.map[0:N, 0:N]:
        c1 = i
        c2 = j
        c3 = (c1 > c0) or (c2 > c0)
        if c3:
            A[i, j] = A[i, j] + B[i, j]


@dace.program
def interstate_boolean_op_two(A: dace.float64[N, N], B: dace.float64[N, N], c0: dace.int64):
    for i, j in dace.map[0:N, 0:N]:
        c1 = i
        c2 = j
        c3 = (c1 > c0) or (c2 > c0)
        c4 = c3 or (A[i, j] > B[i, j])
        if not c4:
            A[i, j] = A[i, j] + B[i, j]


def test_interstate_boolean_op_one(branch_mode):
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.int64)
    c0 = numpy.int64(0)

    run_vectorization_test(
        dace_func=interstate_boolean_op_one,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'N': N,
            'c0': c0,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="interstate_boolean_op_one",
        branch_mode=branch_mode,
    )


def test_interstate_boolean_op_two(branch_mode):
    N = 64
    A = numpy.random.random((N, N)).astype(numpy.int64)
    B = numpy.random.random((N, N)).astype(numpy.int64)
    c0 = numpy.int64(0)

    run_vectorization_test(
        dace_func=interstate_boolean_op_two,
        arrays={
            'A': A,
            'B': B
        },
        params={
            'N': N,
            'c0': c0,
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name="interstate_boolean_op_two",
        branch_mode=branch_mode,
    )


def test_interstate_boolean_op_three(request, branch_mode):
    if branch_mode == "merge":
        request.applymarker(
            pytest.mark.xfail(reason=(
                "op_three adds an extra interstate edge AFTER the ConditionalBlock that reassigns "
                "symsym = __tmp0 or __tmp1, so __tmp0/__tmp1 have a downstream consumer in "
                "addition to the cb's branch condition. M3.1b's compound-cond lift currently "
                "deletes the upstream __tmp0/__tmp1 assignments after rerouting them into "
                "comparison tasklets; that leaves the downstream symsym = ... edge referencing "
                "symbols whose definitions are gone, and the SDFG validator flags 'Missing "
                "symbols on nested SDFG'. Fix is use-count tracking before deleting the upstream "
                "assignment, only delete when the symbol has no other consumer.")))
    sdfg = interstate_boolean_op_one.to_sdfg()
    sdfg.name = "interstate_boolean_op_three"
    nsdfg = {n for (n, g) in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG)}.pop()
    inner_sdfg: dace.SDFG = nsdfg.sdfg
    syms = inner_sdfg.symbols
    last_state = {s for s in inner_sdfg.nodes() if inner_sdfg.out_degree(s) == 0}.pop()
    last_last_state = inner_sdfg.add_state_after(last_state, "ssss", assignments={"symsym": "__tmp0 or __tmp1"})
    inner_sdfg.add_symbol("symsym", dace.int64)
    sdfg.validate()
    sdfg.save("interstate_boolean_op_three.sdfg")
    branch_kwargs = {"use_fp_factor": False, "branch_normalization": True} if branch_mode == "merge" else {}
    VectorizeCPU(vector_width=8,
                 fuse_overlapping_loads=True,
                 insert_copies=True,
                 apply_on_maps=None,
                 no_inline=False,
                 fail_on_unvectorizable=True,
                 **branch_kwargs).apply_pass(sdfg, {})
    sdfg.validate()
    sdfg.save("interstate_boolean_op_three_vectorized.sdfg")


@dace.program
def mid_sdfg(pap: dace.float64[N], ptsphy: dace.float64, r2es: dace.float64, r3ies: dace.float64, r4ies: dace.float64,
             rcldtopcf: dace.float64, rd: dace.float64, rdepliqrefdepth: dace.float64, rdepliqrefrate: dace.float64,
             rg: dace.float64, riceinit: dace.float64, rlmin: dace.float64, rlstt: dace.float64, rtt: dace.float64,
             rv: dace.float64, za: dace.float64[N], zdp: dace.float64[N], zfokoop: dace.float64[N],
             zicecld: dace.float64[N], zrho: dace.float64[N], ztp1: dace.float64[N], zcldtopdist: dace.float64[N],
             zicenuclei: dace.float64[N], zqxfg: dace.float64[N], zsolqa: dace.float64[N]):
    for it_47 in dace.map[
            0:N:1,
    ]:
        # Ice nucleation and deposition
        if ztp1[it_47] < rtt and zqxfg[it_47] > rlmin:
            # Calculate ice saturation vapor pressure
            tmp_arg_72 = (r3ies * (ztp1[it_47] - rtt)) / (ztp1[it_47] - r4ies)
            zicenuclei[it_47] = 2.0 * numpy.exp(tmp_arg_72)
            # Deposition calculation parameters
            zadd = (1.6666666666667 * rlstt * (rlstt / ztp1[it_47]))
            zbdd = (0.452488687782805 * pap[it_47] * rv * ztp1[it_47])
            # Update mixing ratios
            zqxfg[it_47] = zqxfg[it_47] + zadd
            zsolqa[it_47] = zqxfg[it_47] + zbdd


@dace.program
def huge_sdfg(pap: dace.float64[N], ptsphy: dace.float64, r2es: dace.float64, r3ies: dace.float64, r4ies: dace.float64,
              rcldtopcf: dace.float64, rd: dace.float64, rdepliqrefdepth: dace.float64, rdepliqrefrate: dace.float64,
              rg: dace.float64, riceinit: dace.float64, rlmin: dace.float64, rlstt: dace.float64, rtt: dace.float64,
              rv: dace.float64, za: dace.float64[N], zdp: dace.float64[N], zfokoop: dace.float64[N],
              zicecld: dace.float64[N], zrho: dace.float64[N], ztp1: dace.float64[N], zcldtopdist: dace.float64[N],
              zicenuclei: dace.float64[N], zqxfg: dace.float64[N], zsolqa: dace.float64[N]):
    for it_47 in dace.map[
            0:N:1,
    ]:
        # Check if crossing cloud top threshold
        if za[it_47] < rcldtopcf and za[it_47] >= rcldtopcf:
            zcldtopdist[it_47] = 0.0
        else:
            zcldtopdist[it_47] = zcldtopdist[it_47] + (zdp[it_47] / (rg * zrho[it_47]))

        # Ice nucleation and deposition
        if ztp1[it_47] < rtt and zqxfg[it_47] > rlmin:
            # Calculate ice saturation vapor pressure
            tmp_arg_72 = (r3ies * (ztp1[it_47] - rtt)) / (ztp1[it_47] - r4ies)
            tmp_call_47 = r2es * numpy.exp(tmp_arg_72)
            zvpice = (rv * tmp_call_47) / rd

            # Calculate liquid vapor pressure
            zvpliq = zfokoop[it_47] * numpy.log(zvpice)

            # Ice nuclei concentration
            tmp_arg_27 = -0.639 + ((-1.96 * zvpice + 1.96 * zvpliq) / zvpliq)
            zicenuclei[it_47] = 1000.0 * numpy.exp(tmp_arg_27)

            # Nucleation factor
            zinfactor = min(1.0, 6.66666666666667e-05 * zicenuclei[it_47])

            # Deposition calculation parameters
            zadd = (1.6666666666667 * rlstt * (rlstt / (rv * ztp1[it_47]) - 1.0)) / ztp1[it_47]
            zbdd = (0.452488687782805 * pap[it_47] * rv * ztp1[it_47]) / zvpice

            tmp_call_49 = (zicenuclei[it_47] / zrho[it_47])
            zcvds = (7.8 * tmp_call_49 * (zvpliq - zvpice)) / (zvpice * (zadd + zbdd))

            # Initial ice content
            zice0 = max(riceinit * zicenuclei[it_47] / zrho[it_47], zicecld[it_47])

            # New ice after deposition
            tmp_arg_30 = 0.666 * ptsphy * zcvds + zice0
            zinew = tmp_arg_30**1.5

            # Deposition amount
            zdepos1 = max(0.0, za[it_47] * (zinew - zice0))
            zdepos2 = min(zdepos1, 1.1)

            # Apply nucleation factor and cloud top distance factor
            tmp_arg_33 = zinfactor + (1.0 - zinfactor) * (rdepliqrefrate + zcldtopdist[it_47] / rdepliqrefdepth)
            zdepos3 = zdepos2 * min(1.0, tmp_arg_33)

            # Update mixing ratios
            zqxfg[it_47] = zqxfg[it_47] + zdepos3
            zsolqa[it_47] = zsolqa[it_47] + zdepos3


def test_huge_sdfg_with_log_exp_div(request, branch_mode):
    if branch_mode == "merge":
        request.applymarker(
            pytest.mark.xfail(reason="merge mode hits arm-local temp routing bug "
                              "(KeyError: 'ztp1_slice_minus_rtt'), pending follow-up"))
    """Generate test data for the loop body function"""
    eps_operator_type_for_log_and_div: str = "add"
    _N = 32
    data = {
        'ptsphy': numpy.float64(36.0),  # timestep (s)
        'r2es': numpy.float64(6.11),  # saturation vapor pressure constant (hPa)
        'r3ies': numpy.float64(12.0),  # ice saturation constant
        'r4ies': numpy.float64(15.5),  # ice saturation constant
        'rcldtopcf': numpy.float64(16.8),  # cloud top threshold
        'rd': numpy.float64(287.0),  # gas constant for dry air (J/kg/K)
        'rdepliqrefdepth': numpy.float64(20.0),  # reference depth
        'rdepliqrefrate': numpy.float64(17.3),  # reference rate
        'rg': numpy.float64(9.81),  # gravity (m/s²)
        'riceinit': numpy.float64(5.3),  # initial ice content (kg/m³)
        'rlmin': numpy.float64(3.9),  # minimum liquid water (kg/m³)
        'rlstt': numpy.float64(2.5e6),  # latent heat (J/kg)
        'rtt': numpy.float64(273.15),  # triple point temperature (K)
        'rv': numpy.float64(461.5),  # gas constant for water vapor (J/kg/K)
        'N': numpy.int64(_N),
    }

    # 1D arrays with safe ranges
    rng = numpy.random.default_rng(0)

    def safe_uniform(low, high, size):
        """Avoid near-zero or extreme values that could cause NaN in log/div."""
        return rng.uniform(low, high, size).astype(numpy.float64)

    # State variables (N = grid size)
    data['pap'] = safe_uniform(1.0, 2.0, (_N, ))  # pressure-like
    data['za'] = safe_uniform(0.9, 1.5, (_N, ))  # altitude/cloud-top
    data['ztp1'] = safe_uniform(260.0, 280.0, (_N, ))  # temperature near freezing
    data['zqxfg'] = safe_uniform(5.0, 11.0, (_N, ))  # mixing ratios
    data['zsolqa'] = safe_uniform(5.0, 11.0, (_N, ))  # ice tendencies

    data['zdp'] = safe_uniform(0.5, 2.0, (_N, ))  # layer depth
    data['zfokoop'] = safe_uniform(0.95, 1.05, (_N, ))  # correction factor
    data['zicecld'] = safe_uniform(10.0, 11.0, (_N, ))  # cloud ice
    data['zrho'] = safe_uniform(0.9, 1.2, (_N, ))  # density
    data['zcldtopdist'] = safe_uniform(0.1, 1.0, (_N, ))  # distance to cloud top
    data['zicenuclei'] = safe_uniform(1e2, 1e4, (_N, ))  # ice nuclei concentration

    sdfg = huge_sdfg.to_sdfg()
    sdfg.name = f"huge_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}"
    sdfg.validate()
    #it_23: dace.int64, it_47: dace.int64
    sdfg.validate()
    sdfg.auto_optimize(dace.dtypes.DeviceType.CPU, True, True)
    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in data.items()}
    sdfg(**out_no_fuse)
    sdfg.save(f"{sdfg.name}.sdfg")

    # Apply transformation
    copy_sdfg = copy.deepcopy(sdfg)
    branch_kwargs = {"use_fp_factor": False, "branch_normalization": True} if branch_mode == "merge" else {}
    VectorizeCPU(vector_width=8, insert_copies=False, **branch_kwargs).apply_pass(copy_sdfg, {})
    copy_sdfg.name = f"huge_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}_vectorized"
    copy_sdfg.save(f"{copy_sdfg.name}.sdfg")

    # Run SDFG version (with transformation)
    out_fused = {k: v.copy() for k, v in data.items()}

    copy_sdfg(**out_fused)

    # Compare all arrays
    for name in data.keys():
        print(name)
        print(out_fused[name] - out_no_fuse[name])
        numpy.testing.assert_allclose(out_fused[name], out_no_fuse[name], atol=1e-12)


def test_mid_sdfg_with_log_exp_div(request, branch_mode):
    if branch_mode == "merge":
        request.applymarker(
            pytest.mark.xfail(reason="merge mode hits arm-local temp routing bug "
                              "(KeyError: 'ztp1_slice_minus_rtt'), pending follow-up"))
    """Generate test data for the loop body function"""
    eps_operator_type_for_log_and_div = "add"
    _N = 32

    data = {
        'ptsphy': numpy.float64(36.0),  # timestep (s)
        'r2es': numpy.float64(6.11),  # saturation vapor pressure constant (hPa)
        'r3ies': numpy.float64(12.0),  # ice saturation constant
        'r4ies': numpy.float64(15.5),  # ice saturation constant
        'rcldtopcf': numpy.float64(16.8),  # cloud top threshold
        'rd': numpy.float64(287.0),  # gas constant for dry air (J/kg/K)
        'rdepliqrefdepth': numpy.float64(20.0),  # reference depth
        'rdepliqrefrate': numpy.float64(17.3),  # reference rate
        'rg': numpy.float64(9.81),  # gravity (m/s²)
        'riceinit': numpy.float64(5.3),  # initial ice content (kg/m³)
        'rlmin': numpy.float64(3.9),  # minimum liquid water (kg/m³)
        'rlstt': numpy.float64(2.5e6),  # latent heat (J/kg)
        'rtt': numpy.float64(273.15),  # triple point temperature (K)
        'rv': numpy.float64(461.5),  # gas constant for water vapor (J/kg/K)
        'N': numpy.int64(_N),
    }

    # 1D arrays with safe ranges
    rng = numpy.random.default_rng(0)

    def safe_uniform(low, high, size):
        """Avoid near-zero or extreme values that could cause NaN in log/div."""
        return rng.uniform(low, high, size).astype(numpy.float64)

    # State variables (N = grid size)
    data['pap'] = safe_uniform(1.0, 2.0, (_N, ))  # pressure-like
    data['za'] = safe_uniform(0.9, 1.5, (_N, ))  # altitude/cloud-top
    data['ztp1'] = safe_uniform(260.0, 280.0, (_N, ))  # temperature near freezing
    data['zqxfg'] = safe_uniform(5.0, 11.0, (_N, ))  # mixing ratios
    data['zsolqa'] = safe_uniform(5.0, 11.0, (_N, ))  # ice tendencies

    data['zdp'] = safe_uniform(0.5, 2.0, (_N, ))  # layer depth
    data['zfokoop'] = safe_uniform(0.95, 1.05, (_N, ))  # correction factor
    data['zicecld'] = safe_uniform(10.0, 11.0, (_N, ))  # cloud ice
    data['zrho'] = safe_uniform(0.9, 1.2, (_N, ))  # density
    data['zcldtopdist'] = safe_uniform(0.1, 1.0, (_N, ))  # distance to cloud top
    data['zicenuclei'] = safe_uniform(1e2, 1e4, (_N, ))  # ice nuclei concentration
    sdfg = mid_sdfg.to_sdfg()
    sdfg.name = f"mid_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}"
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = f"mid_sdfg_with_log_exp_div_operator_{eps_operator_type_for_log_and_div}_branch_eliminated"

    sdfg.validate()
    copy_sdfg.validate()

    #it_23: dace.int64, it_47: dace.int64

    sdfg.validate()
    sdfg.auto_optimize(dace.dtypes.DeviceType.CPU, True, True)
    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in data.items()}
    sdfg(**out_no_fuse)

    # Apply transformation
    branch_kwargs = {"use_fp_factor": False, "branch_normalization": True} if branch_mode == "merge" else {}
    VectorizeCPU(vector_width=8, insert_copies=False, **branch_kwargs).apply_pass(copy_sdfg, {})

    # Run SDFG version (with transformation)
    out_fused = {k: v.copy() for k, v in data.items()}

    copy_sdfg(**out_fused)

    # Compare all arrays
    for name in data.keys():
        print(name)
        print(out_fused[name] - out_no_fuse[name])
        numpy.testing.assert_allclose(out_fused[name], out_no_fuse[name], atol=1e-12)


@dace.program
def cloud_fraction_update(
        ZA: dace.float64[KLEV, KLON],
        ZQX: dace.float64[5, KLEV, KLON],  # last dim just example
        ZLI: dace.float64[KLEV, KLON],
        ZLIQFRAC: dace.float64[KLEV, KLON],
        ZICEFRAC: dace.float64[KLEV, KLON],
        RLMIN: dace.float64):
    # Loop over levels and horizontal domain
    for jk in dace.map[0:KLEV]:
        for jl in dace.map[0:KLON]:

            # 1. Clip ZA to [0, 1]
            ZA[jk, jl] = max(0.0, min(1.0, ZA[jk, jl]))

            # 2. Compute total liquid+ice
            ZLI[jk, jl] = (ZQX[NCLDQL, jk, jl] + ZQX[NCLDQI, jk, jl])

            if ZLI[jk, jl] > RLMIN:
                ZLIQFRAC[jk, jl] = (ZQX[NCLDQL, jk, jl] / ZLI[jk, jl])
                ZICEFRAC[jk, jl] = (1.0 - ZLIQFRAC[jk, jl])
            else:
                ZLIQFRAC[jk, jl] = 0.0
                ZICEFRAC[jk, jl] = 0.0


def test_cloud_fraction_update(request, branch_mode):
    if branch_mode == "merge":
        request.applymarker(
            pytest.mark.xfail(reason="merge mode hits a missing-symbols validation error "
                              "after the interstate cond is lowered, pending follow-up"))
    # Example sizes
    klev = 16
    klon = 32

    # Pick any valid indexes for QL/QI
    ncldql = 0
    ncldqi = 1

    # Create test arrays
    ZA = numpy.random.uniform(-0.2, 1.2, size=(klev, klon))
    ZQX = numpy.abs(numpy.random.randn(5, klev, klon)) * 1e-4
    ZLI = numpy.zeros((klev, klon))
    ZLIQFRAC = numpy.zeros((klev, klon))
    ZICEFRAC = numpy.zeros((klev, klon))

    RLMIN = 1e-12

    run_vectorization_test(
        dace_func=cloud_fraction_update,
        arrays={
            "ZA": ZA,
            "ZQX": ZQX,
            "ZLI": ZLI,
            "ZLIQFRAC": ZLIQFRAC,
            "ZICEFRAC": ZICEFRAC,
        },
        params={
            "RLMIN": RLMIN,
            "KLON": klon,
            "KLEV": klev,
            "NCLDQL": ncldql,
            "NCLDQI": ncldqi
        },
        vector_width=8,
        save_sdfgs=True,
        sdfg_name=f"cloud_fraction_update",
        branch_mode=branch_mode,
    )

