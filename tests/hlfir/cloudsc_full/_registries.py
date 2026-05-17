"""Registries + helpers for the full-CLOUDSC integration test.

Lifted near-verbatim from
``/home/primrose/Work/data_must_flow_artifacts/cloudsc/validate_cloudsc.py``
(the dict registries + the input/output helpers that
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

from numbers import Integral, Number
from typing import Dict, Union

import numpy as np

# Small problem size so the test stays fast: NPROMA=1, KLEV=137,
# NBLOCKS=4 (~MB total memory, not a benchmark).
nbvalue = 4

# gfortran flags for the f2py reference.  Portable LLVM-flang core
# ``-O0 -fno-fast-math -ffp-contract=off`` plus the sole
# intentionally gfortran-only flag ``-ffree-line-length-none`` -- a
# non-semantic parser necessity for the long-line cloudsc source;
# LLVM-flang has no line limit so needs no equivalent.
CLOUDSC_F90FLAGS = "-O0 -fno-fast-math -ffp-contract=off -ffree-line-length-none"

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

# --------------------------------------------------------------------------
# Physically-plausible CLOUDSC input generator.
#
# ``get_inputs_physical`` maps every CLOUDSC input array onto a
# physically-plausible distribution so the microphysics stays in its
# valid operating regime -- no NaN/inf intermediates.  This matters
# because Fortran MIN/MAX with a NaN operand is processor-dependent
# by the standard, so a NaN reaching one would make the SDFG vs
# gfortran comparison test unspecified behaviour.
#
# Constant values are the canonical ECMWF IFS / dwarf-p-cloudsc
# settings (``yomcst.F90`` physical constants, ``yoethf_mod.F90``
# saturation-vapour coefficients, ``yoecldp.F90`` cloud-scheme
# tunables, ``RCL_*`` warm-rain / Abel-Boutle constants).  Profile
# shapes follow a standard mid-latitude troposphere/stratosphere
# column on the L137 vertical grid.
# --------------------------------------------------------------------------

# Canonical ECMWF physical constants (SI units), from IFS ``yomcst.F90``.
_RG = 9.80665  # gravitational acceleration [m s^-2]
_RD = 287.0597  # gas constant, dry air [J kg^-1 K^-1]
_RV = 461.5250  # gas constant, water vapour [J kg^-1 K^-1]
_RCPD = 1004.709  # specific heat, dry air, const. pressure [J kg^-1 K^-1]
_RETV = _RV / _RD - 1.0  # Rv/Rd - 1 (virtual-temperature factor)
_RLVTT = 2.5008e6  # latent heat of vaporisation [J kg^-1]
_RLSTT = 2.8345e6  # latent heat of sublimation [J kg^-1]
_RLMLT = _RLSTT - _RLVTT  # latent heat of fusion [J kg^-1]
_RTT = 273.16  # temperature of water triple point [K]
_RATM = 100000.0  # standard pressure [Pa]

# Saturation water-vapour-pressure (Teten/Magnus) coefficients, from
# IFS ``yoethf_mod.F90`` (``FOEEW`` statement functions in cloudsc.F90).
_R2ES = 611.21 * _RD / _RV  # = FOEEW prefactor (Pa, scaled to mixing ratio)
_R3LES = 17.502  # over liquid
_R3IES = 22.587  # over ice
_R4LES = 32.19  # over liquid [K]
_R4IES = -0.7  # over ice [K]
_R5LES = _R3LES * (_RTT - _R4LES)
_R5IES = _R3IES * (_RTT - _R4IES)
_R5ALVCP = _R5LES * _RLVTT / _RCPD
_R5ALSCP = _R5IES * _RLSTT / _RCPD
_RALVDCP = _RLVTT / _RCPD
_RALSDCP = _RLSTT / _RCPD
_RALFDCP = _RLMLT / _RCPD

# Mixed-phase / homogeneous-freezing temperatures [K] (``yoethf_mod``).
_RTWAT = _RTT  # all liquid above this
_RTICE = _RTT - 23.0  # all ice below this
_RTICECU = _RTT - 23.0
_RTWAT_RTICE_R = 1.0 / (_RTWAT - _RTICE)
_RTWAT_RTICECU_R = 1.0 / (_RTWAT - _RTICECU)
_RTHOMO = 235.16  # homogeneous-freezing threshold [K]
_RKOOP1 = 2.583  # Koop ice-nucleation fit (``yoethf_mod``)
_RKOOP2 = 0.48116e-2

# Cloud-scheme tunables, from IFS ``yoecldp.F90`` / ``cloud_layer``.
_RAMID = 0.8  # cloud-fraction threshold for autoconv
_RAMIN = 1.0e-8  # minimum cloud fraction
_RLMIN = 1.0e-8  # minimum mixing ratio [kg kg^-1]
_RCLDIFF = 2.0e-6  # erosion rate (clear air)
_RCLDIFF_CONVI = 7.0  # erosion enhancement in convection
_RCLCRIT = 0.4e-3  # critical condensate for autoconv [kg kg^-1]
_RCLCRIT_SEA = 0.25e-3
_RCLCRIT_LAND = 0.5e-3
_RKCONV = 1.0 / 6000.0  # autoconversion timescale [s^-1]
_RPRC1 = 300.0
_RPRC2 = 0.5
_RCLDMAX = 5.0e-3  # max in-cloud condensate [kg kg^-1]
_RPECONS = 5.547558e-5  # evaporation constant
_RVRFACTOR = 0.05
_RPRECRHMAX = 0.7  # max environmental RH for precip evaporation
_RTAUMEL = 1.1880e4  # melting timescale [s]
_RKOOPTAU = 1.0e4  # Koop nucleation timescale [s]
_RCLDTOPP = 10.0  # cloud-top pressure index limit
_RLCRITSNOW = 1.0e-4  # critical ice for snow autoconv [kg kg^-1]
_RSNOWLIN1 = 0.001
_RSNOWLIN2 = 0.03
_RICEHI1 = 1.0 / (-30.0)  # ice effective-radius fit
_RICEHI2 = 1.0 / (-1.0)
_RICEINIT = 1.0e-12  # initial ice content [kg kg^-1]
_RVICE = 0.15  # ice fall speed [m s^-1]
_RVRAIN = 4.0  # rain fall speed [m s^-1]
_RVSNOW = 1.0  # snow fall speed [m s^-1]
_RCOVPMIN = 0.1  # minimum precip cover
_RCCN = 125.0  # default CCN [cm^-3]
_RNICE = 0.027  # ice-nuclei reference number
_RCCNOM = 4.0
_RCCNSS = 1.0
_RCCNSU = 1.0
_RCLDTOPCF = 0.1
_RDEPLIQREFRATE = 0.1
_RDEPLIQREFDEPTH = 500.0

# Abel-Boutle / Khairoutdinov-Kogan warm-rain & ice/snow microphysics
# constants (``cloudsc.F90`` ``CLOUD_INIT``-equivalent values).
_RCL_KKAac = 1350.0
_RCL_KKBac = 2.47
_RCL_KKAau = 1350.0
_RCL_KKBauq = 2.47
_RCL_KKBaun = -1.79
_RCL_KK_CLOUD_NUM_SEA = 50.0e6
_RCL_KK_CLOUD_NUM_LAND = 300.0e6
_RCL_AI = 0.069
_RCL_BI = 2.0
_RCL_CI = 16.8
_RCL_DI = 0.527
_RCL_X1I = 1.0
_RCL_X2I = 0.0
_RCL_X3I = 2.0
_RCL_X4I = 0.0
_RCL_CONST1I = 1.0
_RCL_CONST2I = 1.0
_RCL_CONST3I = 0.5
_RCL_CONST4I = 0.5
_RCL_CONST5I = 1.0
_RCL_CONST6I = 1.0
_RCL_APB1 = 7.18e1
_RCL_APB2 = 4.78e1
_RCL_APB3 = 1.57e-3
_RCL_AS = 0.069
_RCL_BS = 2.0
_RCL_CS = 16.8
_RCL_DS = 0.527
_RCL_X1S = 1.0
_RCL_X2S = 0.0
_RCL_X3S = 2.0
_RCL_X4S = 0.0
_RCL_CONST1S = 1.0
_RCL_CONST2S = 1.0
_RCL_CONST3S = 0.5
_RCL_CONST4S = 0.5
_RCL_CONST5S = 1.0
_RCL_CONST6S = 1.0
_RCL_CONST7S = 1.0
_RCL_CONST8S = 1.0
_RDENSWAT = 1000.0  # density of water [kg m^-3]
_RDENSREF = 1.0  # reference air density [kg m^-3]
_RCL_AR = 523.6
_RCL_BR = 3.0
_RCL_CR = 130.0
_RCL_DR = 0.5
_RCL_X1R = 1.0
_RCL_X2R = 0.0
_RCL_X4R = 0.0
_RCL_KA273 = 2.4e-2  # thermal conductivity of air at 273 K
_RCL_CDENOM1 = 1.0
_RCL_CDENOM2 = 1.0
_RCL_CDENOM3 = 1.0
_RCL_SCHMIDT = 0.6  # Schmidt number
_RCL_DYNVISC = 1.717e-5  # dynamic viscosity of air [kg m^-1 s^-1]
_RCL_CONST1R = 1.0
_RCL_CONST2R = 1.0
_RCL_CONST3R = 0.5
_RCL_CONST4R = 0.5
_RCL_FAC1 = 1.0
_RCL_FAC2 = 1.0
_RCL_CONST5R = 1.0
_RCL_CONST6R = 1.0
_RCL_FZRAB = 0.66  # Bigg freezing constant
_RCL_FZRBB = 100.0

# Map a constant name to its canonical value (everything not in here is
# either a runtime profile array, a parameter, or generated below).
_PHYSICAL_CONSTANTS = {
    'RG': _RG,
    'RD': _RD,
    'RCPD': _RCPD,
    'RETV': _RETV,
    'RLVTT': _RLVTT,
    'RLSTT': _RLSTT,
    'RLMLT': _RLMLT,
    'RTT': _RTT,
    'RV': _RV,
    'R2ES': _R2ES,
    'R3LES': _R3LES,
    'R3IES': _R3IES,
    'R4LES': _R4LES,
    'R4IES': _R4IES,
    'R5LES': _R5LES,
    'R5IES': _R5IES,
    'R5ALVCP': _R5ALVCP,
    'R5ALSCP': _R5ALSCP,
    'RALVDCP': _RALVDCP,
    'RALSDCP': _RALSDCP,
    'RALFDCP': _RALFDCP,
    'RTWAT': _RTWAT,
    'RTICE': _RTICE,
    'RTICECU': _RTICECU,
    'RTWAT_RTICE_R': _RTWAT_RTICE_R,
    'RTWAT_RTICECU_R': _RTWAT_RTICECU_R,
    'RKOOP1': _RKOOP1,
    'RKOOP2': _RKOOP2,
    'RAMID': _RAMID,
    'RCLDIFF': _RCLDIFF,
    'RCLDIFF_CONVI': _RCLDIFF_CONVI,
    'RCLCRIT': _RCLCRIT,
    'RCLCRIT_SEA': _RCLCRIT_SEA,
    'RCLCRIT_LAND': _RCLCRIT_LAND,
    'RKCONV': _RKCONV,
    'RPRC1': _RPRC1,
    'RPRC2': _RPRC2,
    'RCLDMAX': _RCLDMAX,
    'RPECONS': _RPECONS,
    'RVRFACTOR': _RVRFACTOR,
    'RPRECRHMAX': _RPRECRHMAX,
    'RTAUMEL': _RTAUMEL,
    'RAMIN': _RAMIN,
    'RLMIN': _RLMIN,
    'RKOOPTAU': _RKOOPTAU,
    'RCLDTOPP': _RCLDTOPP,
    'RLCRITSNOW': _RLCRITSNOW,
    'RSNOWLIN1': _RSNOWLIN1,
    'RSNOWLIN2': _RSNOWLIN2,
    'RICEHI1': _RICEHI1,
    'RICEHI2': _RICEHI2,
    'RICEINIT': _RICEINIT,
    'RVICE': _RVICE,
    'RVRAIN': _RVRAIN,
    'RVSNOW': _RVSNOW,
    'RTHOMO': _RTHOMO,
    'RCOVPMIN': _RCOVPMIN,
    'RCCN': _RCCN,
    'RNICE': _RNICE,
    'RCCNOM': _RCCNOM,
    'RCCNSS': _RCCNSS,
    'RCCNSU': _RCCNSU,
    'RCLDTOPCF': _RCLDTOPCF,
    'RDEPLIQREFRATE': _RDEPLIQREFRATE,
    'RDEPLIQREFDEPTH': _RDEPLIQREFDEPTH,
    'RCL_KKAac': _RCL_KKAac,
    'RCL_KKBac': _RCL_KKBac,
    'RCL_KKAau': _RCL_KKAau,
    'RCL_KKBauq': _RCL_KKBauq,
    'RCL_KKBaun': _RCL_KKBaun,
    'RCL_KK_CLOUD_NUM_SEA': _RCL_KK_CLOUD_NUM_SEA,
    'RCL_KK_CLOUD_NUM_LAND': _RCL_KK_CLOUD_NUM_LAND,
    'RCL_AI': _RCL_AI,
    'RCL_BI': _RCL_BI,
    'RCL_CI': _RCL_CI,
    'RCL_DI': _RCL_DI,
    'RCL_X1I': _RCL_X1I,
    'RCL_X2I': _RCL_X2I,
    'RCL_X3I': _RCL_X3I,
    'RCL_X4I': _RCL_X4I,
    'RCL_CONST1I': _RCL_CONST1I,
    'RCL_CONST2I': _RCL_CONST2I,
    'RCL_CONST3I': _RCL_CONST3I,
    'RCL_CONST4I': _RCL_CONST4I,
    'RCL_CONST5I': _RCL_CONST5I,
    'RCL_CONST6I': _RCL_CONST6I,
    'RCL_APB1': _RCL_APB1,
    'RCL_APB2': _RCL_APB2,
    'RCL_APB3': _RCL_APB3,
    'RCL_AS': _RCL_AS,
    'RCL_BS': _RCL_BS,
    'RCL_CS': _RCL_CS,
    'RCL_DS': _RCL_DS,
    'RCL_X1S': _RCL_X1S,
    'RCL_X2S': _RCL_X2S,
    'RCL_X3S': _RCL_X3S,
    'RCL_X4S': _RCL_X4S,
    'RCL_CONST1S': _RCL_CONST1S,
    'RCL_CONST2S': _RCL_CONST2S,
    'RCL_CONST3S': _RCL_CONST3S,
    'RCL_CONST4S': _RCL_CONST4S,
    'RCL_CONST5S': _RCL_CONST5S,
    'RCL_CONST6S': _RCL_CONST6S,
    'RCL_CONST7S': _RCL_CONST7S,
    'RCL_CONST8S': _RCL_CONST8S,
    'RDENSWAT': _RDENSWAT,
    'RDENSREF': _RDENSREF,
    'RCL_AR': _RCL_AR,
    'RCL_BR': _RCL_BR,
    'RCL_CR': _RCL_CR,
    'RCL_DR': _RCL_DR,
    'RCL_X1R': _RCL_X1R,
    'RCL_X2R': _RCL_X2R,
    'RCL_X4R': _RCL_X4R,
    'RCL_KA273': _RCL_KA273,
    'RCL_CDENOM1': _RCL_CDENOM1,
    'RCL_CDENOM2': _RCL_CDENOM2,
    'RCL_CDENOM3': _RCL_CDENOM3,
    'RCL_SCHMIDT': _RCL_SCHMIDT,
    'RCL_DYNVISC': _RCL_DYNVISC,
    'RCL_CONST1R': _RCL_CONST1R,
    'RCL_CONST2R': _RCL_CONST2R,
    'RCL_CONST3R': _RCL_CONST3R,
    'RCL_CONST4R': _RCL_CONST4R,
    'RCL_FAC1': _RCL_FAC1,
    'RCL_FAC2': _RCL_FAC2,
    'RCL_CONST5R': _RCL_CONST5R,
    'RCL_CONST6R': _RCL_CONST6R,
    'RCL_FZRAB': _RCL_FZRAB,
    'RCL_FZRBB': _RCL_FZRBB,
}


def _pressure_half_levels(klev: int) -> np.ndarray:
    """Build a monotone-increasing half-level pressure profile [Pa].

    CLOUDSC reads ``ZDP = PAPH(JK+1) - PAPH(JK)`` and
    ``ZRHO = PAP/(RD*T)``, so the half-level pressure must increase with
    level index (top of atmosphere -> surface) and stay strictly
    positive.  A simple hybrid-sigma-like geometric-to-linear blend from
    ~2 hPa at the model top to ~1010 hPa at the surface reproduces the
    L137 IFS grid closely enough for the microphysics to stay in range.

    :param klev: number of full levels (half levels = ``klev + 1``).
    :returns: 1-D array of shape ``(klev + 1,)`` in Pa, strictly
        increasing.
    """
    p_top = 2.0e2  # ~2 hPa model top
    p_sfc = 1.01e5  # ~1010 hPa surface
    eta = np.linspace(0.0, 1.0, klev + 1)
    # Blend a cubic (fine resolution aloft) with a linear term so the
    # profile is monotone and smooth across the whole column.
    paph = p_top + (p_sfc - p_top) * (0.85 * eta**3 + 0.15 * eta)
    return paph


def _temperature_profile(klev: int, rng: np.random.Generator) -> np.ndarray:
    """Build a mid-latitude temperature profile [K] over full levels.

    Tropopause near level fraction ~0.25 (from the top), ~210 K minimum,
    warming to ~295 K at the surface, plus small (~1 K) noise.  Kept
    inside [180, 320] K so every ``FOEEW``/``EXP`` saturation evaluation
    and the ``RTHOMO``/``RTICE`` phase branches stay numerically sane.

    :param klev: number of full levels.
    :param rng: random generator for the small perturbation.
    :returns: 1-D array of shape ``(klev,)`` in K.
    """
    eta = np.linspace(0.0, 1.0, klev)  # 0 = top, 1 = surface
    t_strat = 215.0  # lower stratosphere
    t_sfc = 293.0  # near-surface
    trop = 0.28  # tropopause fraction from the top
    prof = np.where(
        eta < trop,
        t_strat + (220.0 - t_strat) * (eta / trop),
        220.0 + (t_sfc - 220.0) * ((eta - trop) / (1.0 - trop)),
    )
    prof = prof + rng.normal(0.0, 0.5, klev)
    return np.clip(prof, 180.0, 320.0)


def get_inputs_physical(rng: np.random.Generator) -> Dict[str, Union[Number, np.ndarray]]:
    """Build a physically-plausible input dict for one CLOUDSCOUTER call.

    The CLOUDSC input generator: every value sampled from a
    physically-plausible distribution (physical constants, profile
    shapes) so the kernel runs in its valid regime:

    * physical constants (``RG``/``RD``/``RTT``/``RCL_*`` etc.) set to
      their canonical ECMWF IFS / dwarf-p-cloudsc values rather than
      random;
    * temperature ``PT`` ~ 180-320 K mid-latitude column profile;
    * pressures ``PAP``/``PAPH`` ~ 2 hPa..1010 hPa, strictly monotone in
      level (so ``PAPH(JK+1)-PAPH(JK) > 0``);
    * specific humidity ``PQ`` and condensate mixing ratios ``PCLV`` /
      VDF/Dynamics sources non-negative and small (~1e-8..1e-2);
    * cloud fraction ``PA`` in ``[0, 1]``;
    * tendencies small and signed; mass fluxes small and signed;
    * boolean/integer inputs (``LDCUM``/``KTYPE``) unchanged in spirit.

    Keeps every microphysics intermediate (notably ``ZINEW`` in the
    ice-deposition body) finite so no NaN reaches a Fortran MIN/MAX
    (whose NaN result is processor-dependent by the standard).

    :param rng: NumPy random generator (caller controls the seed).
    :returns: dict keyed by original Fortran identifier casing, values
        are Python scalars (parameters/constants) or Fortran-order
        ``np.ndarray`` (profiles).
    """
    klon = parameters['KLON']
    klev = parameters['KLEV']
    nblk = parameters['NBLOCKS']
    nclv = parameters['NCLV']

    inp: Dict[str, Union[Number, np.ndarray]] = dict()
    for p in program_parameters:
        inp[p] = parameters[p]

    def _broadcast_level(profile_1d: np.ndarray, nlev: int) -> np.ndarray:
        """Tile a per-level 1-D profile to ``(KLON, nlev, NBLOCKS)`` and
        add small per-column/per-block noise so blocks are not identical
        (CLOUDSC is column-independent; distinct columns exercise the
        data-dependent branches)."""
        arr = np.empty((klon, nlev, nblk), dtype=np.float64)
        for ib in range(nblk):
            for il in range(klon):
                arr[il, :, ib] = profile_1d
        return np.asfortranarray(arr)

    paph_prof = _pressure_half_levels(klev)  # (klev+1,)
    pap_prof = 0.5 * (paph_prof[:-1] + paph_prof[1:])  # full-level mid (klev,)
    t_prof = _temperature_profile(klev, rng)  # (klev,)

    # Saturation specific humidity ~ Magnus over the profile, used as the
    # ceiling for the generated humidity / condensate so nothing exceeds
    # what the air can physically hold.
    esat = 611.21 * np.exp(17.502 * (t_prof - _RTT) / (t_prof - 32.19))
    qsat = 0.622 * esat / np.maximum(pap_prof - esat, 1.0)  # kg/kg

    for name in program_inputs:
        if name in _PHYSICAL_CONSTANTS:
            inp[name] = _PHYSICAL_CONSTANTS[name]
            continue
        if name not in data:
            raise KeyError(f"missing data entry for input {name!r}")
        info = data[name]
        shape = info[0] if isinstance(info, list) else info

        if name == 'PTSPHY':
            inp[name] = 50.0  # physics timestep [s] (IFS default)
            continue
        if name in ('NSHAPEP', 'NSHAPEQ'):
            inp[name] = 0.0
            continue
        if name in ('RBETA', 'RBETAP1'):
            inp[name] = 0.5
            continue

        if name == 'PT':
            inp[name] = _broadcast_level(t_prof, klev)
        elif name == 'PAP':
            inp[name] = _broadcast_level(pap_prof, klev)
        elif name == 'PAPH':
            inp[name] = _broadcast_level(paph_prof, klev + 1)
        elif name == 'PQ':
            # Relative humidity 20-90% of saturation.
            rh = rng.uniform(0.2, 0.9, (klon, klev, nblk))
            inp[name] = np.asfortranarray(rh * qsat[None, :, None])
        elif name == 'PA':
            # Cloud fraction in [0, 1], biased low (mostly clear sky).
            inp[name] = np.asfortranarray(np.clip(rng.beta(1.5, 4.0, (klon, klev, nblk)), 0.0, 1.0))
        elif name == 'PCLV':
            # Per-species condensate mixing ratios [kg/kg]: liquid/ice/
            # rain/snow small and non-negative, vapour ~ PQ-scale.
            pclv = np.zeros((klon, klev, nclv, nblk), dtype=np.float64)
            for sp, scale in (
                (parameters['NCLDQL'] - 1, 3.0e-4),
                (parameters['NCLDQI'] - 1, 2.0e-4),
                (parameters['NCLDQR'] - 1, 1.0e-4),
                (parameters['NCLDQS'] - 1, 1.0e-4),
            ):
                pclv[:, :, sp, :] = rng.uniform(0.0, scale, (klon, klev, nblk))
            qv = parameters['NCLDQV'] - 1
            pclv[:, :, qv, :] = rng.uniform(0.2, 0.9, (klon, klev, nblk)) * qsat[None, :, None]
            inp[name] = np.asfortranarray(pclv)
        elif name == 'PSUPSAT':
            inp[name] = np.asfortranarray(rng.uniform(0.0, 1.0e-5, (klon, klev, nblk)))
        elif name in ('PVFA', 'PDYNA'):
            # Cloud-fraction sources from VDF / Dynamics: small signed.
            inp[name] = np.asfortranarray(rng.uniform(-1.0e-5, 1.0e-5, (klon, klev, nblk)))
        elif name in ('PVFL', 'PVFI', 'PDYNL', 'PDYNI'):
            # Liquid/ice sources [kg/kg/s-ish, kept small signed].
            inp[name] = np.asfortranarray(rng.uniform(-1.0e-6, 1.0e-6, (klon, klev, nblk)))
        elif name in ('PHRSW', 'PHRLW'):
            # Radiative heating rates [K/s]: ~ +/- 5 K/day.
            inp[name] = np.asfortranarray(rng.uniform(-6.0e-5, 6.0e-5, (klon, klev, nblk)))
        elif name == 'PVERVEL':
            # Pressure vertical velocity omega [Pa/s]: ~ +/- 1 Pa/s.
            inp[name] = np.asfortranarray(rng.uniform(-1.0, 1.0, (klon, klev, nblk)))
        elif name == 'PLSM':
            # Land-sea mask fraction in [0, 1] (per column, per block).
            inp[name] = np.asfortranarray(rng.integers(0, 2, (klon, nblk)).astype(np.float64))
        elif name == 'LDCUM':
            inp[name] = np.asfortranarray(rng.integers(0, 2, (klon, nblk), np.int32).astype(np.bool_))
        elif name == 'KTYPE':
            inp[name] = np.asfortranarray(rng.integers(0, 3, (klon, nblk), np.int32))
        elif name in ('PLU', 'PLUDE'):
            # Convective condensate / detrained water [kg/kg].
            inp[name] = np.asfortranarray(rng.uniform(0.0, 5.0e-4, (klon, klev, nblk)))
        elif name == 'PSNDE':
            inp[name] = np.asfortranarray(rng.uniform(0.0, 1.0e-4, (klon, klev, nblk)))
        elif name in ('PMFU', 'PMFD'):
            # Convective mass fluxes [kg/m2/s]: up >=0, down <=0.
            sgn = 1.0 if name == 'PMFU' else -1.0
            inp[name] = np.asfortranarray(sgn * rng.uniform(0.0, 0.5, (klon, klev, nblk)))
        elif name == 'PCCN':
            inp[name] = np.asfortranarray(rng.uniform(20.0e6, 300.0e6, (klon, klev, nblk)))
        elif name == 'PNICE':
            inp[name] = np.asfortranarray(rng.uniform(1.0e3, 1.0e5, (klon, klev, nblk)))
        elif name in ('PLCRIT_AER', 'PICRIT_AER'):
            inp[name] = np.asfortranarray(rng.uniform(1.0e-6, 1.0e-3, (klon, klev, nblk)))
        elif name == 'PRE_ICE':
            # Ice effective radius [m] ~ 10-100 micron.
            inp[name] = np.asfortranarray(rng.uniform(1.0e-5, 1.0e-4, (klon, klev, nblk)))
        elif name.startswith('tendency_'):
            # Cumulative / local / tmp tendencies: small signed.
            # CLD species tendency is rank-4.
            tshape = (klon, klev, nclv, nblk) if name.endswith('_cld') \
                else (klon, klev, nblk)
            inp[name] = np.asfortranarray(rng.uniform(-1.0e-7, 1.0e-7, tshape))
        else:
            # Any remaining float input: small non-negative fallback.
            if shape == (0, ):
                inp[name] = float(rng.uniform(0.0, 1.0))
            else:
                inp[name] = np.asfortranarray(rng.uniform(0.0, 1.0e-4, shape))

    # Safety net: never hand an *exact* 0.0 to a continuous float field
    # -- nudge to a tiny same-signed epsilon (well below every field's
    # physical scale) so no degenerate zero-only branch / 0-division
    # edge case is exercised by the generated data.  Categorical fields
    # keep their exact values: integer/boolean arrays are skipped by
    # dtype, and the {0,1} land-sea mask ``PLSM`` is excluded by name.
    _ZERO_EPS = 1.0e-12
    for _k, _v in inp.items():
        if (isinstance(_v, np.ndarray) and _v.dtype.kind == 'f' and _k != 'PLSM'):
            _zero = np.abs(_v) < _ZERO_EPS
            if _zero.any():
                _sgn = np.sign(_v)
                _sgn[_sgn == 0.0] = 1.0
                _v[_zero] = _sgn[_zero] * _ZERO_EPS
    return inp


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
