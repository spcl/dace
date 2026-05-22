# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Input-data generation and run-and-compare helpers for the inlined CloudSC
kernel in ``tests/corpus/cloudsc.py``, for end-to-end numerical tests.

The inlined ``cloudsc_py`` program requires no callbacks, so it compiles and
runs standalone. These helpers build a runnable SDFG, generate a
physically-realistic input set, and run two SDFGs on identical inputs to check
that a transformation (simplify, SymbolPropagation, vectorization, ...) is
numerically faithful to a non-transformed reference.

Typical use::

    ref = build_cloudsc_sdfg(simplify=False)
    sut = build_cloudsc_sdfg(simplify=False)
    sut.simplify()
    assert run_and_compare(ref, sut)

Data generation **follows the dwarf-p-cloudsc reference dataset**
(``config-files/input.h5`` of the upstream ECMWF dwarf): the YDCST/YDTHF/YRECLDP
physical constants in :data:`CLOUDSC_CONSTANTS` are the exact values from that
file, and every input array is filled with random values inside the ``[min,
max]`` range observed there (:data:`CLOUDSC_INPUT_RANGES`). The dwarf's problem
size (``klev = 137``, ``klon = 100``) and ``ncldtop = 15`` are used so the
vertical microphysics loop runs over a realistic depth. We do **not** load
``input.h5`` at runtime -- the harness is self-contained and only mirrors its
values -- so it stays runnable without the external dataset.

Filling thresholds and latent heats with the real constants (rather than
uniform ``[0, 1)`` noise) keeps the kernel out of its degenerate branch regimes,
which is what makes a transform-vs-reference comparison meaningful: a random
constant set sits on every ``MIN``/``MAX`` and ``< rlmin`` boundary, so a
harmless floating-point reassociation flips branches and masquerades as a bug.
"""
import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import sympy

import dace
from dace import dtypes
from dace.sdfg import nodes
from tests.corpus.cloudsc import cloudsc_py

#: Shape symbols and named integer index scalars. The cloud species ``ncldq*``
#: and ``ncldtop`` match the dwarf-p-cloudsc reference; the grid is kept small
#: (``klev = klon = 32``) for a fast compiled run -- the physical input ranges
#: are bounds, not vertical profiles, so they stay valid at any grid size, and
#: ``ncldtop = 15`` still leaves a meaningful vertical microphysics loop.
CLOUDSC_SYMBOLS: Dict[str, int] = {
    'klev': 32,
    'klon': 32,
    'nclv': 5,
    'ncldql': 1,  # liquid cloud water
    'ncldqi': 2,  # ice cloud water
    'ncldqr': 3,  # rain water
    'ncldqs': 4,  # snow
    'ncldqv': 5,  # vapour
    'kidia': 1,
    'kfdia': 32,
}

#: Exact YDCST/YDTHF/YRECLDP constants from the dwarf-p-cloudsc ``input.h5``
#: reference. The ``yrecldp_nssopt``/``ncldtop``/``laeri*`` entries are integer
#: scalars (cast on use); the rest are doubles. Mirrored here so the harness
#: needs no external dataset.
CLOUDSC_CONSTANTS: Dict[str, float] = {
    'ptsphy': 3600.0,
    'ydcst_rcpd': 1004.7088578330674,
    'ydcst_rd': 287.0596736665907,
    'ydcst_retv': 0.6077667316114637,
    'ydcst_rg': 9.80665,
    'ydcst_rlmlt': 333700.0,
    'ydcst_rlstt': 2834500.0,
    'ydcst_rlvtt': 2500800.0,
    'ydcst_rtt': 273.16,
    'ydcst_rv': 461.5249933083879,
    'ydthf_r2es': 380.1608703442847,
    'ydthf_r3ies': 22.587,
    'ydthf_r3les': 17.502,
    'ydthf_r4ies': -0.7,
    'ydthf_r4les': 32.19,
    'ydthf_r5alscp': 17451123.253362577,
    'ydthf_r5alvcp': 10497584.68169531,
    'ydthf_r5ies': 6185.67582,
    'ydthf_r5les': 4217.45694,
    'ydthf_ralfdcp': 332.1360187066693,
    'ydthf_ralsdcp': 2821.2152982440934,
    'ydthf_ralvdcp': 2489.0792795374246,
    'ydthf_rkoop1': 2.583,
    'ydthf_rkoop2': 0.0048116,
    'ydthf_rtice': 250.16000000000003,
    'ydthf_rticecu': 250.16000000000003,
    'ydthf_rtwat': 273.16,
    'ydthf_rtwat_rtice_r': 0.043478260869565216,
    'ydthf_rtwat_rticecu_r': 0.043478260869565216,
    'yrecldp_laericeauto': 0,
    'yrecldp_laericesed': 0,
    'yrecldp_laerliqautolsp': 0,
    'yrecldp_laerliqcoll': 0,
    'yrecldp_ncldtop': 15,
    'yrecldp_nssopt': 1,
    'yrecldp_ramid': 0.8,
    'yrecldp_ramin': 1e-08,
    'yrecldp_rccn': 125.0,
    'yrecldp_rcl_apb1': 714000000000.0,
    'yrecldp_rcl_apb2': 116000000.0,
    'yrecldp_rcl_apb3': 241.6,
    'yrecldp_rcl_cdenom1': 557000000000.0,
    'yrecldp_rcl_cdenom2': 103000000.0,
    'yrecldp_rcl_cdenom3': 204.0,
    'yrecldp_rcl_const1i': 3.6231880115136998e-06,
    'yrecldp_rcl_const1r': 1.382300767579509,
    'yrecldp_rcl_const1s': 3.6231880115136998e-06,
    'yrecldp_rcl_const2i': 6283185.307179586,
    'yrecldp_rcl_const2r': 2143.2299120517614,
    'yrecldp_rcl_const2s': 6283185.307179586,
    'yrecldp_rcl_const3i': 596.9998475835998,
    'yrecldp_rcl_const3r': 0.6349999999999998,
    'yrecldp_rcl_const3s': 596.9998475835998,
    'yrecldp_rcl_const4i': 0.6666666666666666,
    'yrecldp_rcl_const4r': -0.20000000000000018,
    'yrecldp_rcl_const4s': 0.6666666666666666,
    'yrecldp_rcl_const5i': 0.9211666666666667,
    'yrecldp_rcl_const5r': 8685252.965082133,
    'yrecldp_rcl_const5s': 0.9211666666666667,
    'yrecldp_rcl_const6i': 1.0000000948961185,
    'yrecldp_rcl_const6r': -4.8,
    'yrecldp_rcl_const6s': 1.0000000948961185,
    'yrecldp_rcl_const7s': 90363515.76351073,
    'yrecldp_rcl_const8s': 1.1756666666666666,
    'yrecldp_rcl_fac1': 4146.902789847063,
    'yrecldp_rcl_fac2': 0.5555555555555556,
    'yrecldp_rcl_fzrab': -0.66,
    'yrecldp_rcl_ka273': 0.024,
    'yrecldp_rcl_kk_cloud_num_land': 300.0,
    'yrecldp_rcl_kk_cloud_num_sea': 50.0,
    'yrecldp_rcl_kkaac': 67.0,
    'yrecldp_rcl_kkaau': 1350.0,
    'yrecldp_rcl_kkbac': 1.15,
    'yrecldp_rcl_kkbaun': -1.79,
    'yrecldp_rcl_kkbauq': 2.47,
    'yrecldp_rclcrit_land': 0.00055,
    'yrecldp_rclcrit_sea': 0.00025,
    'yrecldp_rcldiff': 3e-06,
    'yrecldp_rcldiff_convi': 7.0,
    'yrecldp_rcldtopcf': 0.01,
    'yrecldp_rcovpmin': 0.1,
    'yrecldp_rdensref': 1.0,
    'yrecldp_rdepliqrefdepth': 500.0,
    'yrecldp_rdepliqrefrate': 0.1,
    'yrecldp_riceinit': 1e-12,
    'yrecldp_rkconv': 0.00016666666666666666,
    'yrecldp_rkooptau': 10800.0,
    'yrecldp_rlcritsnow': 3e-05,
    'yrecldp_rlmin': 1e-08,
    'yrecldp_rnice': 0.027,
    'yrecldp_rpecons': 5.54725619859993e-05,
    'yrecldp_rprc1': 100.0,
    'yrecldp_rprecrhmax': 0.7,
    'yrecldp_rsnowlin1': 0.001,
    'yrecldp_rsnowlin2': 0.03,
    'yrecldp_rtaumel': 7200.0,
    'yrecldp_rthomo': 235.16000000000003,
    'yrecldp_rvice': 0.13,
    'yrecldp_rvrain': 4.0,
    'yrecldp_rvrfactor': 0.00509,
    'yrecldp_rvsnow': 1.0,
}

#: ``[min, max]`` range of each floating input array in the dwarf reference.
#: Arrays absent here are kernel outputs (or have no reference value) and are
#: zero-initialized. Several reference inputs are uniformly zero (``pmfd``,
#: ``pnice``, ``plsm``, ...); their ``(0.0, 0.0)`` range reproduces that.
CLOUDSC_INPUT_RANGES: Dict[str, Tuple[float, float]] = {
    'pa': (0.0, 1.0),
    'pap': (0.999923895048429, 101254.97084602337),
    'paph': (0.0, 101375.10057152908),
    'pccn': (0.0, 0.0),
    'pclv': (0.0, 4.0253768484705866e-05),
    'pdyna': (-0.0001187964540362494, 0.00013518935141888),
    'pdyni': (-2.140507622298193e-09, 1.5815637844680652e-09),
    'pdynl': (-4.134462057260392e-09, 2.9599904813560976e-09),
    'phrlw': (-0.00016678085208516417, 1.6941956516047845e-05),
    'phrsw': (-4.112083689734362e-20, 0.0),
    'picrit_aer': (0.0, 0.0),
    'plcrit_aer': (0.0, 0.0),
    'plsm': (0.0, 0.0),
    'plu': (0.0, 0.00028126794898995864),
    'plude': (0.0, 1.2765809819686113e-06),
    'pmfd': (0.0, 0.0),
    'pmfu': (0.0, 0.06243987753452692),
    'pnice': (0.0, 0.0),
    'pq': (1.030691002421725e-06, 0.0024358218045027608),
    'pre_ice': (0.0, 0.0),
    'psnde': (0.0, 0.0),
    'psupsat': (0.0, 4.762341884143876e-05),
    'pt': (196.49936539855418, 267.57212905220615),
    'pvervel': (-0.20627060482285262, 0.1776496848080718),
    'pvfa': (-0.0002305417072000441, 0.0002777777777777778),
    'pvfi': (-3.6683428537982023e-09, 1.2988008242524594e-08),
    'pvfl': (-2.193240268281019e-09, 6.007184256012529e-09),
    'tendency_tmp_a': (-0.00024253423245031549, 0.0002777777777777778),
    'tendency_tmp_cld': (-4.164243445880186e-09, 1.2724998598908134e-08),
    'tendency_tmp_q': (-6.92205735885683e-08, 4.265980175425775e-08),
    'tendency_tmp_t': (-0.000472018266728236, 0.0003986018193989591),
}

#: ``[min, max]`` integer range of each integer input array in the reference.
CLOUDSC_INT_RANGES: Dict[str, Tuple[int, int]] = {
    'ktype': (0, 3),
    'ldcum': (0, 1),
}


def build_cloudsc_sdfg(simplify: bool = False) -> dace.SDFG:
    """Build a fresh runnable SDFG from the inlined CloudSC program.

    :param simplify: Whether to run the simplify pipeline while building.
    :returns: A validated CloudSC SDFG.
    """
    sdfg = cloudsc_py.to_sdfg(simplify=simplify)
    sdfg.validate()
    return sdfg


def _instantiate_dim(dim) -> int:
    """Resolve one array-shape dimension to a concrete size via
    :data:`CLOUDSC_SYMBOLS`.

    :param dim: A shape dimension (int, ``dace.symbol``, or symbolic expr).
    :returns: The instantiated integer size.
    """
    if isinstance(dim, (int, sympy.Number)):
        return int(dim)
    if isinstance(dim, dace.symbol):
        return CLOUDSC_SYMBOLS[str(dim)]
    return int(sympy.sympify(dim).subs(CLOUDSC_SYMBOLS))


def generate_cloudsc_inputs(sdfg: dace.SDFG, seed: int = 0) -> Dict[str, Union[np.ndarray, int, float]]:
    """Generate a physically-realistic CloudSC input set for ``sdfg``.

    Every non-transient argument is filled following the dwarf reference (see
    the module docstring): named constants from :data:`CLOUDSC_CONSTANTS`,
    floating arrays uniform within their :data:`CLOUDSC_INPUT_RANGES` window
    (kernel outputs / unknown arrays zeroed), integer arrays uniform within
    :data:`CLOUDSC_INT_RANGES`, and the named index scalars / shape symbols from
    :data:`CLOUDSC_SYMBOLS`. Length-1 arrays are passed as scalars.

    :param sdfg: The CloudSC SDFG whose non-transient arrays are filled.
    :param seed: Seed for the random number generator (reproducible runs).
    :returns: A kwargs dict of arrays, scalars, and symbol values.
    """
    rng = np.random.default_rng(seed)
    arrays: Dict[str, np.ndarray] = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue
        dims: List[int] = [_instantiate_dim(d) for d in desc.shape]
        is_int = 'int' in str(desc.dtype)

        if name in CLOUDSC_CONSTANTS:
            value = CLOUDSC_CONSTANTS[name]
            arrays[name] = np.full(dims,
                                   int(value) if is_int else value,
                                   dtype=np.int32 if is_int else np.float64,
                                   order='F')
        elif is_int:
            if name in CLOUDSC_SYMBOLS:
                data = np.zeros(dims, order='F').astype(np.int32)
                data.flat[0] = CLOUDSC_SYMBOLS[name]
                arrays[name] = data
            else:
                lo, hi = CLOUDSC_INT_RANGES.get(name, (1, 1))
                arrays[name] = rng.integers(lo, hi + 1, size=dims).astype(np.int32, order='F')
        else:
            value_range: Optional[Tuple[float, float]] = CLOUDSC_INPUT_RANGES.get(name)
            if value_range is None:
                arrays[name] = np.zeros(dims, order='F')  # kernel output / no reference range
            else:
                lo, hi = value_range
                arrays[name] = (lo + (hi - lo) * rng.random(dims)).astype(np.float64, order='F')

    inputs: Dict[str, Union[np.ndarray, int, float]] = {
        name: (data.flat[0] if data.size == 1 else data)
        for name, data in arrays.items()
    }
    inputs.update(CLOUDSC_SYMBOLS)
    return inputs


def make_sequential(sdfg: dace.SDFG) -> Tuple[int, int]:
    """Force every map and library node to a sequential schedule, in place.

    A numerical-equivalence comparison must be deterministic: CloudSC's
    parallel (OpenMP) maps reorder floating-point reductions / accumulations
    run-to-run, so two separately-compiled SDFGs differ by ~1e-5 even when they
    are the *same* computation. Running sequentially removes that noise so a
    real numerical difference between a transform and its reference stands out.

    :param sdfg: The SDFG to make sequential (mutated in place).
    :returns: ``(maps, library_nodes)`` re-scheduled.
    """
    n_maps = n_lib = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            node.map.schedule = dtypes.ScheduleType.Sequential
            n_maps += 1
        elif isinstance(node, nodes.LibraryNode):
            if hasattr(node, 'schedule'):
                node.schedule = dtypes.ScheduleType.Sequential
            n_lib += 1
    return n_maps, n_lib


def run_and_compare(reference: dace.SDFG,
                    candidate: dace.SDFG,
                    seed: int = 0,
                    rtol: float = 1e-9,
                    atol: float = 1e-12,
                    sequential: bool = True,
                    verbose: bool = False) -> bool:
    """Run two CloudSC SDFGs on identical inputs and compare every shared
    non-transient output array.

    Both SDFGs are driven with the same generated inputs (a private deep copy
    each, since the call mutates the buffers in place). Only array outputs are
    compared; scalar and symbol inputs are not.

    :param reference: The reference SDFG (e.g. un-transformed).
    :param candidate: The SDFG under test.
    :param seed: Seed for input generation.
    :param rtol: Relative tolerance (``numpy.allclose``).
    :param atol: Absolute tolerance (``numpy.allclose``).
    :param sequential: Force both SDFGs sequential before running
        (:func:`make_sequential`) so the comparison is deterministic; leave on
        unless you specifically want to measure parallel behaviour.
    :param verbose: Print the max/avg difference of each mismatching array.
    :returns: ``True`` iff every shared output array matches within tolerance.
    """
    if sequential:
        make_sequential(reference)
        make_sequential(candidate)
    ref_inputs = generate_cloudsc_inputs(reference, seed)
    cand_inputs = copy.deepcopy(ref_inputs)
    reference(**ref_inputs)
    candidate(**cand_inputs)

    matches = True
    for name, ref_val in ref_inputs.items():
        if not isinstance(ref_val, np.ndarray):
            continue
        cand_val = cand_inputs[name]
        if not np.allclose(ref_val, cand_val, rtol=rtol, atol=atol, equal_nan=True):
            matches = False
            if verbose:
                diff = np.abs(ref_val - cand_val)
                print(f"{name}: max diff = {np.nanmax(diff):.6e}, avg diff = {np.nanmean(diff):.6e}")
    return matches
