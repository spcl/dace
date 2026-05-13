"""Registries + helpers for the full-CLOUDSC integration test.

Lifted near-verbatim from
``/home/primrose/Work/data_must_flow_artifacts/cloudsc/validate_cloudsc.py``
(the dict registries + ``get_inputs``/``get_outputs`` helpers that
drove the old ``dace.frontend.fortran`` Python frontend's
end-to-end test).  Reused here for the HLFIR bridge port  --  the
shapes and intent classifications are independent of which
frontend builds the SDFG.

Unified under one program key, ``cloudscexp2``, to match the
Fortran source's subroutine name ``CLOUDSCOUTER`` (entry =
``_QPcloudscouter``).  The original file had a typo where
``program_outputs`` keyed off ``'cloudscexp2_inner'`` while
``program_inputs`` / ``program_parameters`` keyed off
``'cloudscexp2_full_20230324'``; both consolidate to one key.
"""
from __future__ import annotations

from numbers import Integral, Number
from typing import Dict, Union

import numpy as np

# Small problem size so the test stays fast: NPROMA=1, KLEV=137,
# NBLOCKS=4 (~MB total memory, not a benchmark).
nbvalue = 4

PROGRAM = "cloudscexp2"

parameters = {
    'KLON': 1,  # equals NPROMA
    'KLEV': 137,
    'KIDIA': 1,
    'KFDIA': 137,
    'KFLDX': 1,
    'NCLV': 5,
    'NCLDQI': 2,
    'NCLDQL': 1,
    'NCLDQR': 3,
    'NCLDQS': 4,
    'NCLDQV': 5,
    'NCLDTOP': 15,
    'NSSOPT': 1,
    'NAECLBC': 1,
    'NAECLDU': 1,
    'NAECLOM': 1,
    'NAECLSS': 1,
    'NAECLSU': 1,
    'NCLDDIAG': 1,
    'NAERCLD': 1,
    'NBLOCKS': nbvalue,
    # Scalar LOGICAL parameters: keep as np.bool_ so the call-site
    # routing in test_cloudsc_full.py::_sdfg_call_args picks the
    # bridge's ``bool *`` ABI cleanly.  Casting to np.int32 here would
    # work for value=1 by accident (LSB matches) but silently corrupt
    # any value with bit-0 = 0 -- see ``test_bool_scalar_logical_pass_through``.
    'LDMAINCALL': np.bool_(True),
    'LDSLPHY': np.bool_(True),
    'LAERLIQAUTOCP': np.bool_(True),
    'LAERLIQAUTOCPB': np.bool_(True),
    'LAERLIQAUTOLSP': np.bool_(True),
    'LAERLIQCOLL': np.bool_(True),
    'LAERICESED': np.bool_(True),
    'LAERICEAUTO': np.bool_(True),
    'LCLDEXTRA': np.bool_(True),
    'LCLDBUDGET': np.bool_(True),
    'NGPBLKS': 10,
    'NUMOMP': 10,
    'NGPTOT': nbvalue * 1,
    'NGPTOTG': nbvalue * 1,
    'NPROMA': 1,
    'NBETA': nbvalue,
}

# Per-array shape + dtype registry.  ``(0,)`` marks a Fortran SCALAR
# (real(8) by default; dtype overridden via the
# ``[shape, dtype]`` list form).
data = {
    'NSHAPEP': (0, ),
    'NSHAPEQ': (0, ),
    'PTSPHY': (0, ),
    'R2ES': (0, ),
    'R3IES': (0, ),
    'R3LES': (0, ),
    'R4IES': (0, ),
    'R4LES': (0, ),
    'R5ALSCP': (0, ),
    'R5ALVCP': (0, ),
    'R5IES': (0, ),
    'R5LES': (0, ),
    'RALFDCP': (0, ),
    'RALSDCP': (0, ),
    'RALVDCP': (0, ),
    'RAMID': (0, ),
    'RAMIN': (0, ),
    'RBETA': (0, ),
    'RBETAP1': (0, ),
    'RCCN': (0, ),
    'RCCNOM': (0, ),
    'RCCNSS': (0, ),
    'RCCNSU': (0, ),
    'RCL_AI': (0, ),
    'RCL_BI': (0, ),
    'RCL_CI': (0, ),
    'RCL_DI': (0, ),
    'RCL_X1I': (0, ),
    'RCLCRIT': (0, ),
    'RCLCRIT_LAND': (0, ),
    'RCLCRIT_SEA': (0, ),
    'RCLDIFF': (0, ),
    'RCLDIFF_CONVI': (0, ),
    'RCLDMAX': (0, ),
    'RCLDTOPCF': (0, ),
    'RCLDTOPP': (0, ),
    'RCOVPMIN': (0, ),
    'RCPD': (0, ),
    'RD': (0, ),
    'RDEPLIQREFDEPTH': (0, ),
    'RDEPLIQREFRATE': (0, ),
    'RETV': (0, ),
    'RG': (0, ),
    'RICEHI1': (0, ),
    'RICEHI2': (0, ),
    'RICEINIT': (0, ),
    'RKCONV': (0, ),
    'RKOOP1': (0, ),
    'RKOOP2': (0, ),
    'RKOOPTAU': (0, ),
    'RLCRITSNOW': (0, ),
    'RLMIN': (0, ),
    'RLMLT': (0, ),
    'RLSTT': (0, ),
    'RLVTT': (0, ),
    'RNICE': (0, ),
    'RPECONS': (0, ),
    'RPRC1': (0, ),
    'RPRC2': (0, ),
    'RPRECRHMAX': (0, ),
    'RSNOWLIN1': (0, ),
    'RSNOWLIN2': (0, ),
    'RTAUMEL': (0, ),
    'RTHOMO': (0, ),
    'RTICE': (0, ),
    'RTICECU': (0, ),
    'RTT': (0, ),
    'RTWAT': (0, ),
    'RTWAT_RTICE_R': (0, ),
    'RTWAT_RTICECU_R': (0, ),
    'RV': (0, ),
    'RVICE': (0, ),
    'RVRAIN': (0, ),
    'RVRFACTOR': (0, ),
    'RVSNOW': (0, ),
    'RCL_KKAac': (0, ),
    'RCL_KKBac': (0, ),
    'RCL_KKAau': (0, ),
    'RCL_KKBauq': (0, ),
    'RCL_KKBaun': (0, ),
    'RCL_KK_CLOUD_NUM_SEA': (0, ),
    'RCL_KK_CLOUD_NUM_LAND': (0, ),
    'RCL_CONST1I': (0, ),
    'RCL_CONST2I': (0, ),
    'RCL_CONST3I': (0, ),
    'RCL_CONST4I': (0, ),
    'RCL_CONST5I': (0, ),
    'RCL_CONST6I': (0, ),
    'RCL_APB1': (0, ),
    'RCL_APB2': (0, ),
    'RCL_APB3': (0, ),
    'RCL_CONST1S': (0, ),
    'RCL_CONST2S': (0, ),
    'RCL_CONST3S': (0, ),
    'RCL_CONST4S': (0, ),
    'RCL_CONST5S': (0, ),
    'RCL_CONST6S': (0, ),
    'RCL_CONST7S': (0, ),
    'RCL_CONST8S': (0, ),
    'RDENSREF': (0, ),
    'RCL_KA273': (0, ),
    'RCL_CDENOM1': (0, ),
    'RCL_CDENOM2': (0, ),
    'RCL_CDENOM3': (0, ),
    'RCL_CONST1R': (0, ),
    'RCL_CONST2R': (0, ),
    'RCL_CONST3R': (0, ),
    'RCL_CONST4R': (0, ),
    'RCL_FAC1': (0, ),
    'RCL_FAC2': (0, ),
    'RCL_CONST5R': (0, ),
    'RCL_CONST6R': (0, ),
    'RCL_FZRAB': (0, ),
    'RCL_X2I': (0, ),
    'RCL_X3I': (0, ),
    'RCL_X4I': (0, ),
    'RCL_AS': (0, ),
    'RCL_BS': (0, ),
    'RCL_CS': (0, ),
    'RCL_DS': (0, ),
    'RCL_X1S': (0, ),
    'RCL_X2S': (0, ),
    'RCL_X3S': (0, ),
    'RCL_X4S': (0, ),
    'RDENSWAT': (0, ),
    'RCL_AR': (0, ),
    'RCL_BR': (0, ),
    'RCL_CR': (0, ),
    'RCL_DR': (0, ),
    'RCL_X1R': (0, ),
    'RCL_X2R': (0, ),
    'RCL_X4R': (0, ),
    'RCL_SCHMIDT': (0, ),
    'RCL_DYNVISC': (0, ),
    'RCL_FZRBB': (0, ),
    'IPHASE': (parameters['NCLV'], ),
    'KTYPE': [(parameters['KLON'], parameters['NBLOCKS']), np.int32],
    'LDCUM': [(parameters['KLON'], parameters['NBLOCKS']), np.bool_],
    'PA': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PAP': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PAPH': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PCCN': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PCLV': (parameters['KLON'], parameters['KLEV'], parameters['NCLV'], parameters['NBLOCKS']),
    'PCOVPTOT': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PDYNA': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PDYNI': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PDYNL': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PEXTRA': (parameters['KLON'], parameters['KLEV'], parameters['KFLDX'], parameters['NBLOCKS']),
    'PFCQLNG': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFCQNNG': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFCQRNG': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFCQSNG': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFHPSL': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFHPSN': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFPLSL': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFPLSN': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFSQIF': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFSQITUR': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFSQLF': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFSQLTUR': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFSQRF': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PFSQSF': (parameters['KLON'], parameters['KLEV'] + 1, parameters['NBLOCKS']),
    'PICRIT_AER': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PLCRIT_AER': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PHRLW': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PLSM': (parameters['KLON'], parameters['NBLOCKS']),
    'PHRSW': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PLU': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PLUDE': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PMFD': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PMFU': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PNICE': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PRAINFRAC_TOPRFZ': (parameters['KLON'], parameters['NBLOCKS']),
    'PQ': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PRE_ICE': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PSNDE': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PSUPSAT': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PT': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PVERVEL': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PVFA': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PVFI': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'PVFL': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_a': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV'], parameters['NBLOCKS']),
    'tendency_cml_o3': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_q': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_T': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_u': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_cml_v': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_a': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV'], parameters['NBLOCKS']),
    'tendency_loc_o3': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_q': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_T': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_u': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_loc_v': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_a': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV'], parameters['NBLOCKS']),
    'tendency_tmp_o3': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_q': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_T': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_u': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
    'tendency_tmp_v': (parameters['KLON'], parameters['KLEV'], parameters['NBLOCKS']),
}

# Scalar parameters that go on the SDFG / Fortran call signature.
program_parameters = (
    'NBLOCKS',
    'NGPBLKS',
    'NUMOMP',
    'NGPTOT',
    'NGPTOTG',
    'NPROMA',
    'KLON',
    'KLEV',
    'KFLDX',
    'LDSLPHY',
    'LDMAINCALL',
    'NCLV',
    'NCLDQL',
    'NCLDQI',
    'NCLDQR',
    'NCLDQS',
    'NCLDQV',
    'LAERLIQAUTOLSP',
    'LAERLIQAUTOCP',
    'LAERLIQAUTOCPB',
    'LAERLIQCOLL',
    'LAERICESED',
    'LAERICEAUTO',
    'LCLDEXTRA',
    'LCLDBUDGET',
    'NSSOPT',
    'NCLDTOP',
    'NAECLBC',
    'NAECLDU',
    'NAECLOM',
    'NAECLSS',
    'NAECLSU',
    'NCLDDIAG',
    'NAERCLD',
    'NBETA',
)

# Input arrays / scalars (every Fortran INTENT(IN) on the
# CLOUDSCOUTER signature).
program_inputs = (
    'PTSPHY',
    'PT',
    'PQ',
    'tendency_cml_a',
    'tendency_cml_cld',
    'tendency_cml_o3',
    'tendency_cml_q',
    'tendency_cml_T',
    'tendency_cml_u',
    'tendency_cml_v',
    'tendency_loc_a',
    'tendency_loc_cld',
    'tendency_loc_o3',
    'tendency_loc_q',
    'tendency_loc_T',
    'tendency_loc_u',
    'tendency_loc_v',
    'tendency_tmp_a',
    'tendency_tmp_cld',
    'tendency_tmp_o3',
    'tendency_tmp_q',
    'tendency_tmp_T',
    'tendency_tmp_u',
    'tendency_tmp_v',
    'PVFA',
    'PVFL',
    'PVFI',
    'PDYNA',
    'PDYNL',
    'PDYNI',
    'PHRSW',
    'PHRLW',
    'PVERVEL',
    'PAP',
    'PAPH',
    'PLSM',
    'LDCUM',
    'KTYPE',
    'PLU',
    'PSNDE',
    'PMFU',
    'PMFD',
    # prognostic
    'PA',
    'PCLV',
    'PSUPSAT',
    # aerosol-cloud interaction
    'PLCRIT_AER',
    'PICRIT_AER',
    'PRE_ICE',
    'PCCN',
    'PNICE',
    # physical constants (passed as scalars in the simplified flattened
    # variant; the verification_pipeline version bundles these as
    # `ydcst` / `ydthf` derived types).
    'RG',
    'RD',
    'RCPD',
    'RETV',
    'RLVTT',
    'RLSTT',
    'RLMLT',
    'RTT',
    'RV',
    'R2ES',
    'R3LES',
    'R3IES',
    'R4LES',
    'R4IES',
    'R5LES',
    'R5IES',
    'R5ALVCP',
    'R5ALSCP',
    'RALVDCP',
    'RALSDCP',
    'RALFDCP',
    'RTWAT',
    'RTICE',
    'RTICECU',
    'RTWAT_RTICE_R',
    'RTWAT_RTICECU_R',
    'RKOOP1',
    'RKOOP2',
    'RAMID',
    'RCLDIFF',
    'RCLDIFF_CONVI',
    'RCLCRIT',
    'RCLCRIT_SEA',
    'RCLCRIT_LAND',
    'RKCONV',
    'RPRC1',
    'RPRC2',
    'RCLDMAX',
    'RPECONS',
    'RVRFACTOR',
    'RPRECRHMAX',
    'RTAUMEL',
    'RAMIN',
    'RLMIN',
    'RKOOPTAU',
    'RCLDTOPP',
    'RLCRITSNOW',
    'RSNOWLIN1',
    'RSNOWLIN2',
    'RICEHI1',
    'RICEHI2',
    'RICEINIT',
    'RVICE',
    'RVRAIN',
    'RVSNOW',
    'RTHOMO',
    'RCOVPMIN',
    'RCCN',
    'RNICE',
    'RCCNOM',
    'RCCNSS',
    'RCCNSU',
    'RCLDTOPCF',
    'RDEPLIQREFRATE',
    'RDEPLIQREFDEPTH',
    'RCL_KKAac',
    'RCL_KKBac',
    'RCL_KKAau',
    'RCL_KKBauq',
    'RCL_KKBaun',
    'RCL_KK_CLOUD_NUM_SEA',
    'RCL_KK_CLOUD_NUM_LAND',
    'RCL_AI',
    'RCL_BI',
    'RCL_CI',
    'RCL_DI',
    'RCL_X1I',
    'RCL_X2I',
    'RCL_X3I',
    'RCL_X4I',
    'RCL_CONST1I',
    'RCL_CONST2I',
    'RCL_CONST3I',
    'RCL_CONST4I',
    'RCL_CONST5I',
    'RCL_CONST6I',
    'RCL_APB1',
    'RCL_APB2',
    'RCL_APB3',
    'RCL_AS',
    'RCL_BS',
    'RCL_CS',
    'RCL_DS',
    'RCL_X1S',
    'RCL_X2S',
    'RCL_X3S',
    'RCL_X4S',
    'RCL_CONST1S',
    'RCL_CONST2S',
    'RCL_CONST3S',
    'RCL_CONST4S',
    'RCL_CONST5S',
    'RCL_CONST6S',
    'RCL_CONST7S',
    'RCL_CONST8S',
    'RDENSWAT',
    'RDENSREF',
    'RCL_AR',
    'RCL_BR',
    'RCL_CR',
    'RCL_DR',
    'RCL_X1R',
    'RCL_X2R',
    'RCL_X4R',
    'RCL_KA273',
    'RCL_CDENOM1',
    'RCL_CDENOM2',
    'RCL_CDENOM3',
    'RCL_SCHMIDT',
    'RCL_DYNVISC',
    'RCL_CONST1R',
    'RCL_CONST2R',
    'RCL_CONST3R',
    'RCL_CONST4R',
    'RCL_FAC1',
    'RCL_FAC2',
    'RCL_CONST5R',
    'RCL_CONST6R',
    'RCL_FZRAB',
    'RCL_FZRBB',
    'NSHAPEP',
    'NSHAPEQ',
    'RBETA',
    'RBETAP1',
)

# Output arrays (every INTENT(OUT)/INTENT(INOUT) result we compare).
program_outputs = (
    'PLUDE',
    'PCOVPTOT',
    'PRAINFRAC_TOPRFZ',
    'PFSQLF',
    'PFSQIF',
    'PFCQNNG',
    'PFCQLNG',
    'PFSQRF',
    'PFSQSF',
    'PFCQRNG',
    'PFCQSNG',
    'PFSQLTUR',
    'PFSQITUR',
    'PFPLSL',
    'PFPLSN',
    'PFHPSL',
    'PFHPSN',
    'PEXTRA',
)


def get_inputs(rng: np.random.Generator) -> Dict[str, Union[Number, np.ndarray]]:
    """Build the input dict for one call to CLOUDSCOUTER.

    Includes both parameters (literal scalars from ``parameters``) and
    runtime input arrays/scalars (filled via the supplied RNG).  The
    returned dict's keys are the original Fortran identifier casing
    (uppercase for parameters and constants, mixed for ``tendency_*``).
    """
    inp_data = dict()
    for p in program_parameters:
        inp_data[p] = parameters[p]
    for inp in program_inputs:
        if inp not in data:
            raise KeyError(f"missing data entry for input {inp!r}")
        info = data[inp]
        if isinstance(info, list):
            shape, dtype = info
        else:
            shape = info
            dtype = np.float64
        if issubclass(dtype, Integral) or dtype is np.bool_:
            if dtype is np.bool_:
                # Generate ints in {0, 1} then cast to np.bool_  --  keep the
                # 1-byte boolean storage so the bridge's ``bool *`` SDFG
                # signature reads each element correctly.  Casting to
                # np.int32 here used to leave the 4-byte int32 array
                # under a ``bool *`` pointer; the SDFG then read byte
                # offsets 0..3 of one int32 as 4 separate "booleans",
                # which corrupted ``LDCUM`` and similar LOGICAL inputs.
                method = lambda s, d: rng.integers(0, 2, s, np.int32).astype(np.bool_)
            else:
                method = lambda s, d: rng.integers(0, 10, s, d)
        else:
            method = lambda s, d: rng.random(s, d)
        if shape == (0, ):
            inp_data[inp] = method(None, dtype)
        else:
            inp_data[inp] = np.asfortranarray(method(shape, dtype))
    return inp_data


def get_outputs(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Build the output dict for one call to CLOUDSCOUTER.

    Pre-allocated Fortran-order arrays per the ``program_outputs``
    list, filled with random data so the assertion catches anything
    the kernel forgets to overwrite.
    """
    out_data = dict()
    for out in program_outputs:
        if out not in data:
            raise KeyError(f"missing data entry for output {out!r}")
        info = data[out]
        if isinstance(info, list):
            shape, dtype = info
        else:
            shape = info
            dtype = np.float64
        if issubclass(dtype, Integral) or dtype is np.bool_:
            if dtype is np.bool_:
                method = lambda s, d: rng.integers(0, 2, s, d)
                dtype = np.int32
            else:
                method = lambda s, d: rng.integers(0, 10, s, d)
        else:
            method = lambda s, d: rng.random(s, d)
        if shape == (0, ):
            raise NotImplementedError(f"scalar outputs not supported: {out}")
        out_data[out] = np.asfortranarray(method(shape, dtype))
    return out_data
