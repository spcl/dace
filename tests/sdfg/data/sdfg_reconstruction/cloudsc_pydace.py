import numpy as np
import dace
from pathlib import Path

"""
# These need to be symbols
# number of microphysics variables
nclv = 5
# liquid cloud water
ncldql = 1
# ice cloud water
ncldqi = 2
# rain water
ncldqr = 3
# snow
ncldqs = 4
# vapour
ncldqv = 5
"""

klon = dace.symbol("klon", dtype=dace.int32)
klev = dace.symbol("klev", dtype=dace.int32)
nclv = dace.symbol("nclv", dtype=dace.int32)
ncldql = dace.symbol("ncldql", dtype=dace.int32)
ncldqi = dace.symbol("ncldqi", dtype=dace.int32)
ncldqr = dace.symbol("ncldqr", dtype=dace.int32)
ncldqs = dace.symbol("ncldqs", dtype=dace.int32)
ncldqv = dace.symbol("ncldqv", dtype=dace.int32)

def foedelta(ptare):
  return max(0.0, 1.0*np.sign(ptare - ydcst_rtt))

#                  FOEDELTA = 1    water
#                  FOEDELTA = 0    ice

#     THERMODYNAMICAL FUNCTIONS .

#     Pressure of water vapour at saturation
#        INPUT : PTARE = TEMPERATURE
def foeew(ptare):
  return ydthf_r2es*np.exp((ydthf_r3les*foedelta(ptare) + ydthf_r3ies*(1.0 - foedelta(ptare)))*(ptare - ydcst_rtt) / (ptare - (ydthf_r4les*foedelta(ptare) + ydthf_r4ies*(1.0 - foedelta(ptare)))))

def foede(ptare):
  return (foedelta(ptare)*ydthf_r5alvcp + (1.0 - foedelta(ptare))*ydthf_r5alscp) / (ptare - (ydthf_r4les*foedelta(ptare) + ydthf_r4ies*(1.0 - foedelta(ptare))))**2

def foedesu(ptare):
  return (foedelta(ptare)*ydthf_r5les + (1.0 - foedelta(ptare))*ydthf_r5ies) / (ptare - (ydthf_r4les*foedelta(ptare) + ydthf_r4ies*(1.0 - foedelta(ptare))))**2

def foelh(ptare):
  return foedelta(ptare)*ydcst_rlvtt + (1.0 - foedelta(ptare))*ydcst_rlstt

def foeldcp(ptare):
  return foedelta(ptare)*ydthf_ralvdcp + (1.0 - foedelta(ptare))*ydthf_ralsdcp

#     *****************************************************************

#           CONSIDERATION OF MIXED PHASES

#     *****************************************************************

#     FOEALFA is calculated to distinguish the three cases:

#                       FOEALFA=1            water phase
#                       FOEALFA=0            ice phase
#                       0 < FOEALFA < 1      mixed phase

#               INPUT : PTARE = TEMPERATURE
def foealfa(ptare):
  return min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ptare)) - ydthf_rtice)*ydthf_rtwat_rtice_r)**2)


#     Pressure of water vapour at saturation
#        INPUT : PTARE = TEMPERATURE
def foeewm(ptare):
  return ydthf_r2es*(foealfa(ptare)*np.exp(ydthf_r3les*(ptare - ydcst_rtt) / (ptare - ydthf_r4les)) + (1.0 - foealfa(ptare))*np.exp(ydthf_r3ies*(ptare - ydcst_rtt) / (ptare - ydthf_r4ies)))

def foe_dewm_dt(ptare):
  return ydthf_r2es*(ydthf_r3les*foealfa(ptare)*np.exp(ydthf_r3les*(ptare - ydcst_rtt) / (ptare - ydthf_r4les))*(ydcst_rtt - ydthf_r4les) / (ptare - ydthf_r4les)**2 + ydthf_r3ies*(1.0 - foealfa(ptare))*np.exp(ydthf_r3ies*(ptare - ydcst_rtt) / (ptare - ydthf_r4ies))*(ydcst_rtt - ydthf_r4ies) / 
    (ptare - ydthf_r4ies)**2)

def foedem(ptare):
  return foealfa(ptare)*ydthf_r5alvcp*(1.0 / (ptare - ydthf_r4les)**2) + (1.0 - foealfa(ptare))*ydthf_r5alscp*(1.0 / (ptare - ydthf_r4ies)**2)

def foeldcpm(ptare):
  return foealfa(ptare)*ydthf_ralvdcp + (1.0 - foealfa(ptare))*ydthf_ralsdcp

def foelhm(ptare):
  return foealfa(ptare)*ydcst_rlvtt + (1.0 - foealfa(ptare))*ydcst_rlstt


#     Temperature normalization for humidity background change of variable
#        INPUT : PTARE = TEMPERATURE
def foetb(ptare):
  return foealfa(ptare)*ydthf_r3les*(ydcst_rtt - ydthf_r4les)*(1.0 / (ptare - ydthf_r4les)**2) + (1.0 - foealfa(ptare))*ydthf_r3ies*(ydcst_rtt - ydthf_r4ies)*(1.0 / (ptare - ydthf_r4ies)**2)

# ============================================================
#  DIFFERENT MIXED PHASE FOR CONVECTION
# ============================================================
@dace.program
def foealfcu(
    ptare: dace.float64,
    rticecu: dace.float64,
    rtwat: dace.float64,
    rtwat_rticecu_r: dace.float64,
) -> dace.float64:
    return min(
        1.0,
        ((max(rticecu, min(rtwat, ptare)) - rticecu) * rtwat_rticecu_r) ** 2,
    )

@dace.program
def foeewmcu(
    ptare: dace.float64,
    rtt: dace.float64,
    r2es: dace.float64,
    r3les: dace.float64,
    r3ies: dace.float64,
    r4les: dace.float64,
    r4ies: dace.float64,
    rticecu: dace.float64,
    rtwat: dace.float64,
    rtwat_rticecu_r: dace.float64,
) -> dace.float64:
    alfa = foealfcu(ptare, rticecu, rtwat, rtwat_rticecu_r)
    return r2es * (
        alfa * np.exp(r3les * (ptare - rtt) / (ptare - r4les))
        + (1.0 - alfa) * np.exp(r3ies * (ptare - rtt) / (ptare - r4ies))
    )

@dace.program
def foedemcu(
    ptare: dace.float64,
    r4les: dace.float64,
    r4ies: dace.float64,
    r5alvcp: dace.float64,
    r5alscp: dace.float64,
    rticecu: dace.float64,
    rtwat: dace.float64,
    rtwat_rticecu_r: dace.float64,
) -> dace.float64:
    alfa = foealfcu(ptare, rticecu, rtwat, rtwat_rticecu_r)
    return (
        alfa * r5alvcp / (ptare - r4les) ** 2
        + (1.0 - alfa) * r5alscp / (ptare - r4ies) ** 2
    )

@dace.program
def foeldcpmcu(
    ptare: dace.float64,
    ralvdcp: dace.float64,
    ralsdcp: dace.float64,
    rticecu: dace.float64,
    rtwat: dace.float64,
    rtwat_rticecu_r: dace.float64,
) -> dace.float64:
    alfa = foealfcu(ptare, rticecu, rtwat, rtwat_rticecu_r)
    return alfa * ralvdcp + (1.0 - alfa) * ralsdcp

@dace.program
def foelhmcu(
    ptare: dace.float64,
    rlvtt: dace.float64,
    rlstt: dace.float64,
    rticecu: dace.float64,
    rtwat: dace.float64,
    rtwat_rticecu_r: dace.float64,
) -> dace.float64:
    alfa = foealfcu(ptare, rticecu, rtwat, rtwat_rticecu_r)
    return alfa * rlvtt + (1.0 - alfa) * rlstt


# ============================================================
#  WMO / SEPARATE ICE-LIQUID FUNCTIONS
# ============================================================
@dace.program
def foeewmo(
    ptare: dace.float64,
    rtt: dace.float64,
    r2es: dace.float64,
    r3les: dace.float64,
    r4les: dace.float64,
) -> dace.float64:
    """Saturation vapour pressure (WMO, always wrt water)."""
    return r2es * np.exp(r3les * (ptare - rtt) / (ptare - r4les))

@dace.program
def foeeliq(
    ptare: dace.float64,
    rtt: dace.float64,
    r2es: dace.float64,
    r3les: dace.float64,
    r4les: dace.float64,
) -> dace.float64:
    """Saturation vapour pressure always wrt liquid water."""
    return r2es * np.exp(r3les * (ptare - rtt) / (ptare - r4les))

@dace.program
def foeeice(
    ptare: dace.float64,
    rtt: dace.float64,
    r2es: dace.float64,
    r3ies: dace.float64,
    r4ies: dace.float64,
) -> dace.float64:
    """Saturation vapour pressure always wrt ice."""
    return r2es * np.exp(r3ies * (ptare - rtt) / (ptare - r4ies))

@dace.program
def foeles_v(
    ptare: dace.float64,
    rtt: dace.float64,
    r3les: dace.float64,
    r4les: dace.float64,
) -> dace.float64:
    return r3les * (ptare - rtt) / (ptare - r4les)

@dace.program
def foeies_v(
    ptare: dace.float64,
    rtt: dace.float64,
    r3ies: dace.float64,
    r4ies: dace.float64,
) -> dace.float64:
    return r3ies * (ptare - rtt) / (ptare - r4ies)

@dace.program
def foeewm_v(
    ptare: dace.float64,
    exp1: dace.float64,
    exp2: dace.float64,
    r2es: dace.float64,
    rtice: dace.float64,
    rtwat: dace.float64,
    rtwat_rtice_r: dace.float64,
) -> dace.float64:
    alfa = foealfa(ptare, rtice, rtwat, rtwat_rtice_r)
    return r2es * (alfa * exp1 + (1.0 - alfa) * exp2)

@dace.program
def foeewmcu_v(
    ptare: dace.float64,
    exp1: dace.float64,
    exp2: dace.float64,
    r2es: dace.float64,
    rticecu: dace.float64,
    rtwat: dace.float64,
    rtwat_rticecu_r: dace.float64,
) -> dace.float64:
    alfa = foealfcu(ptare, rticecu, rtwat, rtwat_rticecu_r)
    return r2es * (alfa * exp1 + (1.0 - alfa) * exp2)


# ============================================================
#  KOOP FORMULA (homogeneous nucleation of ice)
# ============================================================

@dace.program
def fokoop(
    ptare: dace.float64,
    rtt: dace.float64,
    r2es: dace.float64,
    r3les: dace.float64,
    r3ies: dace.float64,
    r4les: dace.float64,
    r4ies: dace.float64,
    rkoop1: dace.float64,
    rkoop2: dace.float64,
) -> dace.float64:
    return min(
        rkoop1 - rkoop2 * ptare,
        foeeliq(ptare, rtt, r2es, r3les, r4les)
        / foeeice(ptare, rtt, r2es, r3ies, r4ies),
    )

@dace.program
def cloudsc_py(
    # Scalar parameters
    kidia: dace.int32,
    kfdia: dace.int32,
    ptsphy: dace.float64,
    # State / tendency arrays (klev, klon)
    pt: dace.float64[klev, klon],
    pq: dace.float64[klev, klon],
    tendency_tmp_t: dace.float64[klev, klon],
    tendency_tmp_q: dace.float64[klev, klon],
    tendency_tmp_a: dace.float64[klev, klon],
    tendency_tmp_cld: dace.float64[nclv, klev, klon],
    tendency_loc_t: dace.float64[klev, klon],
    tendency_loc_q: dace.float64[klev, klon],
    tendency_loc_a: dace.float64[klev, klon],
    tendency_loc_cld: dace.float64[nclv, klev, klon],
    # Vertical diffusion fluxes (klev, klon)
    pvfa: dace.float64[klev, klon],
    pvfl: dace.float64[klev, klon],
    pvfi: dace.float64[klev, klon],
    # Dynamics tendencies (klev, klon)
    pdyna: dace.float64[klev, klon],
    pdynl: dace.float64[klev, klon],
    pdyni: dace.float64[klev, klon],
    # Radiation / forcing (klev, klon)
    phrsw: dace.float64[klev, klon],
    phrlw: dace.float64[klev, klon],
    pvervel: dace.float64[klev, klon],
    # Pressure (klev, klon) and half-levels (klev+1, klon)
    pap: dace.float64[klev, klon],
    paph: dace.float64[klev + 1, klon],
    # Surface / convection (klon)
    plsm: dace.float64[klon],
    ldcum: dace.int32[klon],
    ktype: dace.int32[klon],
    # Convection (klev, klon)
    plu: dace.float64[klev, klon],
    plude: dace.float64[klev, klon],
    psnde: dace.float64[klev, klon],
    pmfu: dace.float64[klev, klon],
    pmfd: dace.float64[klev, klon],
    # Cloud state (klev, klon) and (nclv, klev, klon)
    pa: dace.float64[klev, klon],
    pclv: dace.float64[nclv, klev, klon],
    psupsat: dace.float64[klev, klon],
    # Aerosol-related (klev, klon)
    plcrit_aer: dace.float64[klev, klon],
    picrit_aer: dace.float64[klev, klon],
    pre_ice: dace.float64[klev, klon],
    pccn: dace.float64[klev, klon],
    pnice: dace.float64[klev, klon],
    # Precipitation cover (klev, klon) and (klon)
    pcovptot: dace.float64[klev, klon],
    prainfrac_toprfz: dace.float64[klon],
    # Flux diagnostics (klev+1, klon)
    pfsqlf: dace.float64[klev + 1, klon],
    pfsqif: dace.float64[klev + 1, klon],
    pfcqnng: dace.float64[klev + 1, klon],
    pfcqlng: dace.float64[klev + 1, klon],
    pfsqrf: dace.float64[klev + 1, klon],
    pfsqsf: dace.float64[klev + 1, klon],
    pfcqrng: dace.float64[klev + 1, klon],
    pfcqsng: dace.float64[klev + 1, klon],
    pfsqltur: dace.float64[klev + 1, klon],
    pfsqitur: dace.float64[klev + 1, klon],
    # Precipitation fluxes (klev+1, klon)
    pfplsl: dace.float64[klev + 1, klon],
    pfplsn: dace.float64[klev + 1, klon],
    pfhpsl: dace.float64[klev + 1, klon],
    pfhpsn: dace.float64[klev + 1, klon],
    # --- YDCST (flattened) ---
    ydcst_rg: dace.float64,
    ydcst_rd: dace.float64,
    ydcst_rcpd: dace.float64,
    ydcst_retv: dace.float64,
    ydcst_rlvtt: dace.float64,
    ydcst_rlstt: dace.float64,
    ydcst_rlmlt: dace.float64,
    ydcst_rtt: dace.float64,
    ydcst_rv: dace.float64,
    # --- YDTHF (flattened) ---
    ydthf_r2es: dace.float64,
    ydthf_r3les: dace.float64,
    ydthf_r3ies: dace.float64,
    ydthf_r4les: dace.float64,
    ydthf_r4ies: dace.float64,
    ydthf_r5les: dace.float64,
    ydthf_r5ies: dace.float64,
    ydthf_r5alvcp: dace.float64,
    ydthf_r5alscp: dace.float64,
    ydthf_ralvdcp: dace.float64,
    ydthf_ralsdcp: dace.float64,
    ydthf_ralfdcp: dace.float64,
    ydthf_rtwat: dace.float64,
    ydthf_rtice: dace.float64,
    ydthf_rticecu: dace.float64,
    ydthf_rtwat_rtice_r: dace.float64,
    ydthf_rtwat_rticecu_r: dace.float64,
    ydthf_rkoop1: dace.float64,
    ydthf_rkoop2: dace.float64,
    # --- YRECLDP (flattened) ---
    yrecldp_ramid: dace.float64,
    yrecldp_rcldiff: dace.float64,
    yrecldp_rcldiff_convi: dace.float64,
    yrecldp_ramin: dace.float64,
    yrecldp_rlmin: dace.float64,
    yrecldp_rdensref: dace.float64,
    yrecldp_rtaumel: dace.float64,
    yrecldp_rvice: dace.float64,
    yrecldp_rvrain: dace.float64,
    yrecldp_rvsnow: dace.float64,
    yrecldp_rthomo: dace.float64,
    yrecldp_rcovpmin: dace.float64,
    yrecldp_rkooptau: dace.float64,
    yrecldp_rcldtopcf: dace.float64,
    yrecldp_rkconv: dace.float64,
    yrecldp_rclcrit_land: dace.float64,
    yrecldp_rclcrit_sea: dace.float64,
    yrecldp_rlcritsnow: dace.float64,
    yrecldp_rprecrhmax: dace.float64,
    yrecldp_rprc1: dace.float64,
    yrecldp_rvrfactor: dace.float64,
    yrecldp_rpecons: dace.float64,
    yrecldp_rnice: dace.float64,
    yrecldp_riceinit: dace.float64,
    yrecldp_rdepliqrefrate: dace.float64,
    yrecldp_rdepliqrefdepth: dace.float64,
    yrecldp_rsnowlin1: dace.float64,
    yrecldp_rsnowlin2: dace.float64,
    yrecldp_rccn: dace.float64,
    yrecldp_nssopt: dace.int32,
    yrecldp_ncldtop: dace.int32,
    yrecldp_laericesed: dace.int32,
    yrecldp_laerliqautolsp: dace.int32,
    yrecldp_laerliqcoll: dace.int32,
    yrecldp_laericeauto: dace.int32,
    # --- YRECLDP RCL_* microphysics constants ---
    yrecldp_rcl_kkaau: dace.float64,
    yrecldp_rcl_kkbauq: dace.float64,
    yrecldp_rcl_kkbaun: dace.float64,
    yrecldp_rcl_kkaac: dace.float64,
    yrecldp_rcl_kkbac: dace.float64,
    yrecldp_rcl_kk_cloud_num_land: dace.float64,
    yrecldp_rcl_kk_cloud_num_sea: dace.float64,
    yrecldp_rcl_fac1: dace.float64,
    yrecldp_rcl_fac2: dace.float64,
    yrecldp_rcl_fzrab: dace.float64,
    yrecldp_rcl_apb1: dace.float64,
    yrecldp_rcl_apb2: dace.float64,
    yrecldp_rcl_apb3: dace.float64,
    yrecldp_rcl_const1i: dace.float64,
    yrecldp_rcl_const2i: dace.float64,
    yrecldp_rcl_const3i: dace.float64,
    yrecldp_rcl_const4i: dace.float64,
    yrecldp_rcl_const5i: dace.float64,
    yrecldp_rcl_const6i: dace.float64,
    yrecldp_rcl_const1s: dace.float64,
    yrecldp_rcl_const2s: dace.float64,
    yrecldp_rcl_const3s: dace.float64,
    yrecldp_rcl_const4s: dace.float64,
    yrecldp_rcl_const5s: dace.float64,
    yrecldp_rcl_const6s: dace.float64,
    yrecldp_rcl_const7s: dace.float64,
    yrecldp_rcl_const8s: dace.float64,
    yrecldp_rcl_const1r: dace.float64,
    yrecldp_rcl_const2r: dace.float64,
    yrecldp_rcl_const3r: dace.float64,
    yrecldp_rcl_const4r: dace.float64,
    yrecldp_rcl_const5r: dace.float64,
    yrecldp_rcl_const6r: dace.float64,
    yrecldp_rcl_ka273: dace.float64,
    yrecldp_rcl_cdenom1: dace.float64,
    yrecldp_rcl_cdenom2: dace.float64,
    yrecldp_rcl_cdenom3: dace.float64,
):
  # USE YOMCST   , ONLY : RG, RD, RCPD, RETV, RLVTT, RLSTT, RLMLT, RTT, RV
  # USE YOETHF   , ONLY : R2ES, R3LES, R3IES, R4LES, R4IES, R5LES, R5IES, &
  #  & R5ALVCP, R5ALSCP, RALVDCP, RALSDCP, RALFDCP, RTWAT, RTICE, RTICECU, &
  #  & RTWAT_RTICE_R, RTWAT_RTICECU_R, RKOOP1, RKOOP2
  # USE YOECLDP  , ONLY : TECLDP, NCLDQV, NCLDQL, NCLDQR, NCLDQI, NCLDQS, NCLV
  # USE YOECLDP  , ONLY : TECLDP, NCLDQV, NCLDQL, NCLDQR, NCLDQI, NCLDQS, NCLV
  
  # USE FCTTRE_MOD, ONLY: FOEDELTA, FOEALFA, FOEEWM, FOEEICE, FOEELIQ, FOELDCP, FOELDCPM, FOEDEM
  # USE FCCLD_MOD, ONLY : FOKOOP

  #===============================================================================
  #  0.0     Beginning of timestep book-keeping
  #----------------------------------------------------------------------
    # --- 2D/3D work arrays ---
    # --- 1D work arrays (klon) ---
  zlcond1 = np.ndarray(shape=(klon,), dtype=np.float64)
  zlcond2 = np.ndarray(shape=(klon,), dtype=np.float64)
  zlevapl = np.ndarray(shape=(klon,), dtype=np.float64)
  zlevapi = np.ndarray(shape=(klon,), dtype=np.float64)
  zrainaut = np.ndarray(shape=(klon,), dtype=np.float64)
  zsnowaut = np.ndarray(shape=(klon,), dtype=np.float64)
  zliqcld = np.ndarray(shape=(klon,), dtype=np.float64)
  zicecld = np.ndarray(shape=(klon,), dtype=np.float64)
  zfokoop = np.ndarray(shape=(klon,), dtype=np.float64)
  zicenuclei = np.ndarray(shape=(klon,), dtype=np.float64)
  zlicld = np.ndarray(shape=(klon,), dtype=np.float64)
  zlfinalsum = np.ndarray(shape=(klon,), dtype=np.float64)
  zdqs = np.ndarray(shape=(klon,), dtype=np.float64)
  ztold = np.ndarray(shape=(klon,), dtype=np.float64)
  zqold = np.ndarray(shape=(klon,), dtype=np.float64)
  zdtgdp = np.ndarray(shape=(klon,), dtype=np.float64)
  zrdtgdp = np.ndarray(shape=(klon,), dtype=np.float64)
  ztrpaus = np.ndarray(shape=(klon,), dtype=np.float64)
  zcovpclr = np.ndarray(shape=(klon,), dtype=np.float64)
  zcovptot = np.ndarray(shape=(klon,), dtype=np.float64)
  zcovpmax = np.ndarray(shape=(klon,), dtype=np.float64)
  zqpretot = np.ndarray(shape=(klon,), dtype=np.float64)
  zldefr = np.ndarray(shape=(klon,), dtype=np.float64)
  zldifdt = np.ndarray(shape=(klon,), dtype=np.float64)
  zdtgdpf = np.ndarray(shape=(klon,), dtype=np.float64)
  zacust = np.ndarray(shape=(klon,), dtype=np.float64)
  zmf = np.ndarray(shape=(klon,), dtype=np.float64)
  zrho = np.ndarray(shape=(klon,), dtype=np.float64)
  ztmp1 = np.ndarray(shape=(klon,), dtype=np.float64)
  ztmp2 = np.ndarray(shape=(klon,), dtype=np.float64)
  ztmp3 = np.ndarray(shape=(klon,), dtype=np.float64)
  ztmp4 = np.ndarray(shape=(klon,), dtype=np.float64)
  ztmp5 = np.ndarray(shape=(klon,), dtype=np.float64)
  ztmp6 = np.ndarray(shape=(klon,), dtype=np.float64)
  ztmp7 = np.ndarray(shape=(klon,), dtype=np.float64)
  zalfawm = np.ndarray(shape=(klon,), dtype=np.float64)
  zsolab = np.ndarray(shape=(klon,), dtype=np.float64)
  zsolac = np.ndarray(shape=(klon,), dtype=np.float64)
  zanewm1 = np.ndarray(shape=(klon,), dtype=np.float64)
  zgdp = np.ndarray(shape=(klon,), dtype=np.float64)
  zda = np.ndarray(shape=(klon,), dtype=np.float64)
  zdp = np.ndarray(shape=(klon,), dtype=np.float64)
  zpaphd = np.ndarray(shape=(klon,), dtype=np.float64)
  zmin = np.ndarray(shape=(klon,), dtype=np.float64)
  zsupsat = np.ndarray(shape=(klon,), dtype=np.float64)
  zmeltmax = np.ndarray(shape=(klon,), dtype=np.float64)
  zfrzmax = np.ndarray(shape=(klon,), dtype=np.float64)
  zicetot = np.ndarray(shape=(klon,), dtype=np.float64)
  zdqsliqdt = np.ndarray(shape=(klon,), dtype=np.float64)
  zdqsicedt = np.ndarray(shape=(klon,), dtype=np.float64)
  zdqsmixdt = np.ndarray(shape=(klon,), dtype=np.float64)
  zcorqsliq = np.ndarray(shape=(klon,), dtype=np.float64)
  zcorqsice = np.ndarray(shape=(klon,), dtype=np.float64)
  zcorqsmix = np.ndarray(shape=(klon,), dtype=np.float64)
  zevaplimliq = np.ndarray(shape=(klon,), dtype=np.float64)
  zevaplimice = np.ndarray(shape=(klon,), dtype=np.float64)
  zevaplimmix = np.ndarray(shape=(klon,), dtype=np.float64)
  zcldtopdist = np.ndarray(shape=(klon,), dtype=np.float64)
  zrainacc = np.ndarray(shape=(klon,), dtype=np.float64)
  zraincld = np.ndarray(shape=(klon,), dtype=np.float64)
  zsnowrime = np.ndarray(shape=(klon,), dtype=np.float64)
  zsnowcld = np.ndarray(shape=(klon,), dtype=np.float64)
  zrg = np.ndarray(shape=(klon,), dtype=np.float64)
  psum_solqa = np.ndarray(shape=(klon,), dtype=np.float64)
  llflag = np.ndarray(shape=(klon,), dtype=np.float64)  # used as bool but stored as float
  llrainliq = np.ndarray(shape=(klon,), dtype=np.int32)

  # --- 1D work arrays (nclv) ---
  iphase = np.ndarray(shape=(nclv,), dtype=np.int32)
  imelt = np.ndarray(shape=(nclv,), dtype=np.int32)
  llfall = np.ndarray(shape=(nclv,), dtype=np.int32)
  zvqx = np.ndarray(shape=(nclv,), dtype=np.float64)
  zfoealfa = np.ndarray(shape=(klev + 1, klon), dtype=np.float64)
  ztp1 = np.ndarray(shape=(klev, klon), dtype=np.float64)
  zlcust = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zli = np.ndarray(shape=(klev, klon), dtype=np.float64)
  za = np.ndarray(shape=(klev, klon), dtype=np.float64)
  zaorig = np.ndarray(shape=(klev, klon), dtype=np.float64)
  llindex1 = np.ndarray(shape=(nclv, klon), dtype=np.int32)
  llindex3 = np.ndarray(shape=(nclv, nclv, klon), dtype=np.int32)
  iorder = np.ndarray(shape=(nclv, klon), dtype=np.int32)
  zliqfrac = np.ndarray(shape=(klev, klon), dtype=np.float64)
  zicefrac = np.ndarray(shape=(klev, klon), dtype=np.float64)
  zqx = np.ndarray(shape=(nclv, klev, klon), dtype=np.float64)
  zqx0 = np.ndarray(shape=(nclv, klev, klon), dtype=np.float64)
  zqxn = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zqxfg = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zqxnm1 = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zfluxq = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zpfplsx = np.ndarray(shape=(nclv, klev + 1, klon), dtype=np.float64)
  zlneg = np.ndarray(shape=(nclv, klev, klon), dtype=np.float64)
  zqxn2d = np.ndarray(shape=(nclv, klev, klon), dtype=np.float64)
  zqsmix = np.ndarray(shape=(klev, klon), dtype=np.float64)
  zqsliq = np.ndarray(shape=(klev, klon), dtype=np.float64)
  zqsice = np.ndarray(shape=(klev, klon), dtype=np.float64)
  zfoeewmt = np.ndarray(shape=(klev, klon), dtype=np.float64)
  zfoeew = np.ndarray(shape=(klev, klon), dtype=np.float64)
  zfoeeliqt = np.ndarray(shape=(klev, klon), dtype=np.float64)
  zsolqa = np.ndarray(shape=(nclv, nclv, klon), dtype=np.float64)
  zsolqb = np.ndarray(shape=(nclv, nclv, klon), dtype=np.float64)
  zqlhs = np.ndarray(shape=(nclv, nclv, klon), dtype=np.float64)
  zratio = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zsinksum = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zfallsink = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zfallsrce = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zconvsrce = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zconvsink = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  zpsupsatsrce = np.ndarray(shape=(nclv, klon), dtype=np.float64)
  
  #######################################################################
  #             0.  *** SET UP CONSTANTS ***
  #######################################################################
  # Numerical fit to wet bulb temperature
  ztw1 = 1329.31
  ztw2 = 0.0074615
  ztw3 = 0.85E5
  ztw4 = 40.637
  ztw5 = 275.0

  # ZEPSILON=100._JPRB*EPSILON(ZEPSILON)
  zepsilon = 1.E-14
  
  # ---------------------------------------------------------------------
  # Set version of warm-rain autoconversion/accretion
  # IWARMRAIN = 1 ! Sundquist
  # IWARMRAIN = 2 ! Khairoutdinov and Kogan (2000)
  # ---------------------------------------------------------------------
  iwarmrain = 2
  # ---------------------------------------------------------------------
  # Set version of rain evaporation
  # IEVAPRAIN = 1 ! Sundquist
  # IEVAPRAIN = 2 ! Abel and Boutle (2013)
  # ---------------------------------------------------------------------
  ievaprain = 2
  # ---------------------------------------------------------------------
  # Set version of snow evaporation
  # IEVAPSNOW = 1 ! Sundquist
  # IEVAPSNOW = 2 ! New
  # ---------------------------------------------------------------------
  ievapsnow = 1
  # ---------------------------------------------------------------------
  # Set version of ice deposition
  # IDEPICE = 1 ! Rotstayn (2001)
  # IDEPICE = 2 ! New
  # ---------------------------------------------------------------------
  idepice = 1
  
  # ---------------------
  # Some simple constants
  # ---------------------
  zqtmst = 1.0 / ptsphy
  zgdcp = ydcst_rg / ydcst_rcpd
  zrdcp = ydcst_rd / ydcst_rcpd
  zcons1a = ydcst_rcpd / (ydcst_rlmlt*ydcst_rg*yrecldp_rtaumel)
  zepsec = 1.E-14
  zrg_r = 1.0 / ydcst_rg
  zrldcp = 1.0 / (ydthf_ralsdcp - ydthf_ralvdcp)
  
  # Note: Defined in module/yoecldp.F90
  # NCLDQL=1    ! liquid cloud water
  # NCLDQI=2    ! ice cloud water
  # NCLDQR=3    ! rain water
  # NCLDQS=4    ! snow
  # NCLDQV=5    ! vapour
  
  # -----------------------------------------------
  # Define species phase, 0=vapour, 1=liquid, 2=ice
  # -----------------------------------------------
  iphase[ncldqv - 1] = 0
  iphase[ncldql - 1] = 1
  iphase[ncldqr - 1] = 1
  iphase[ncldqi - 1] = 2
  iphase[ncldqs - 1] = 2
  
  # ---------------------------------------------------
  # Set up melting/freezing index,
  # if an ice category melts/freezes, where does it go?
  # ---------------------------------------------------
  imelt[ncldqv - 1] = -99
  imelt[ncldql - 1] = ncldqi
  imelt[ncldqr - 1] = ncldqs
  imelt[ncldqi - 1] = ncldqr
  imelt[ncldqs - 1] = ncldqr
  
  # -----------------------------------------------
  # INITIALIZATION OF OUTPUT TENDENCIES
  # -----------------------------------------------
  for jk in range(1, klev + 1):
    for jl in range(kidia, kfdia + 1):
      tendency_loc_t[jk - 1, jl - 1] = 0.0
      tendency_loc_q[jk - 1, jl - 1] = 0.0
      tendency_loc_a[jk - 1, jl - 1] = 0.0
  for jm in range(1, nclv - 1 + 1):
    for jk in range(1, klev + 1):
      for jl in range(kidia, kfdia + 1):
        tendency_loc_cld[jm - 1, jk - 1, jl - 1] = 0.0
  
  #-- These were uninitialized : meaningful only when we compare error differences
  for jk in range(1, klev + 1):
    for jl in range(kidia, kfdia + 1):
      pcovptot[jk - 1, jl - 1] = 0.0
      tendency_loc_cld[nclv - 1, jk - 1, jl - 1] = 0.0
  
  # -------------------------
  # set up fall speeds in m/s
  # -------------------------
  zvqx[ncldqv - 1] = 0.0
  zvqx[ncldql - 1] = 0.0
  zvqx[ncldqi - 1] = yrecldp_rvice
  zvqx[ncldqr - 1] = yrecldp_rvrain
  zvqx[ncldqs - 1] = yrecldp_rvsnow
  llfall[:] = False
  for jm in range(1, nclv + 1):
    if zvqx[jm - 1] > 0.0:
      llfall[jm - 1] = True
    # falling species
  # Set LLFALL to false for ice (but ice still sediments!)
  # Need to rationalise this at some point
  llfall[ncldqi - 1] = False
  
  
  #######################################################################
  #             1.  *** INITIAL VALUES FOR VARIABLES ***
  #######################################################################
  
  
  # ----------------------
  # non CLV initialization
  # ----------------------
  for jk in range(1, klev + 1):
    for jl in range(kidia, kfdia + 1):
      ztp1[jk - 1, jl - 1] = pt[jk - 1, jl - 1] + ptsphy*tendency_tmp_t[jk - 1, jl - 1]
      zqx[ncldqv - 1, jk - 1, jl - 1] = pq[jk - 1, jl - 1] + ptsphy*tendency_tmp_q[jk - 1, jl - 1]
      zqx0[ncldqv - 1, jk - 1, jl - 1] = pq[jk - 1, jl - 1] + ptsphy*tendency_tmp_q[jk - 1, jl - 1]
      za[jk - 1, jl - 1] = pa[jk - 1, jl - 1] + ptsphy*tendency_tmp_a[jk - 1, jl - 1]
      zaorig[jk - 1, jl - 1] = pa[jk - 1, jl - 1] + ptsphy*tendency_tmp_a[jk - 1, jl - 1]
  
  # -------------------------------------
  # initialization for CLV family
  # -------------------------------------
  for jm in range(1, nclv - 1 + 1):
    for jk in range(1, klev + 1):
      for jl in range(kidia, kfdia + 1):
        zqx[jm - 1, jk - 1, jl - 1] = pclv[jm - 1, jk - 1, jl - 1] + ptsphy*tendency_tmp_cld[jm - 1, jk - 1, jl - 1]
        zqx0[jm - 1, jk - 1, jl - 1] = pclv[jm - 1, jk - 1, jl - 1] + ptsphy*tendency_tmp_cld[jm - 1, jk - 1, jl - 1]
  
  #-------------
  # zero arrays
  #-------------
  for jm in range(1, nclv + 1):
    for jk in range(1, klev + 1 + 1):
      for jl in range(kidia, kfdia + 1):
        zpfplsx[jm - 1, jk - 1, jl - 1] = 0.0          # precip fluxes
  
  for jm in range(1, nclv + 1):
    for jk in range(1, klev + 1):
      for jl in range(kidia, kfdia + 1):
        zqxn2d[jm - 1, jk - 1, jl - 1] = 0.0          # end of timestep values in 2D
        zlneg[jm - 1, jk - 1, jl - 1] = 0.0          # negative input check
  
  for jl in range(kidia, kfdia + 1):
    prainfrac_toprfz[jl - 1] = 0.0      # rain fraction at top of refreezing layer
  llrainliq[:] = True    # Assume all raindrops are liquid initially
  
  # ----------------------------------------------------
  # Tidy up very small cloud cover or total cloud water
  # ----------------------------------------------------
  for jk in range(1, klev + 1):
    for jl in range(kidia, kfdia + 1):
      if zqx[ncldql - 1, jk - 1, jl - 1] + zqx[ncldqi - 1, jk - 1, jl - 1] < yrecldp_rlmin or za[jk - 1, jl - 1] < yrecldp_ramin:
        
        # Evaporate small cloud liquid water amounts
        zlneg[ncldql - 1, jk - 1, jl - 1] = zlneg[ncldql - 1, jk - 1, jl - 1] + zqx[ncldql - 1, jk - 1, jl - 1]
        zqadj = zqx[ncldql - 1, jk - 1, jl - 1]*zqtmst
        tendency_loc_q[jk - 1, jl - 1] = tendency_loc_q[jk - 1, jl - 1] + zqadj
        tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] - ydthf_ralvdcp*zqadj
        zqx[ncldqv - 1, jk - 1, jl - 1] = zqx[ncldqv - 1, jk - 1, jl - 1] + zqx[ncldql - 1, jk - 1, jl - 1]
        zqx[ncldql - 1, jk - 1, jl - 1] = 0.0
        
        # Evaporate small cloud ice water amounts
        zlneg[ncldqi - 1, jk - 1, jl - 1] = zlneg[ncldqi - 1, jk - 1, jl - 1] + zqx[ncldqi - 1, jk - 1, jl - 1]
        zqadj = zqx[ncldqi - 1, jk - 1, jl - 1]*zqtmst
        tendency_loc_q[jk - 1, jl - 1] = tendency_loc_q[jk - 1, jl - 1] + zqadj
        tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] - ydthf_ralsdcp*zqadj
        zqx[ncldqv - 1, jk - 1, jl - 1] = zqx[ncldqv - 1, jk - 1, jl - 1] + zqx[ncldqi - 1, jk - 1, jl - 1]
        zqx[ncldqi - 1, jk - 1, jl - 1] = 0.0
        
        # Set cloud cover to zero
        za[jk - 1, jl - 1] = 0.0
        
  
  # ---------------------------------
  # Tidy up small CLV variables
  # ---------------------------------
  #DIR$ IVDEP
  for jm in range(1, nclv - 1 + 1):
    #DIR$ IVDEP
    for jk in range(1, klev + 1):
      #DIR$ IVDEP
      for jl in range(kidia, kfdia + 1):
        if zqx[jm - 1, jk - 1, jl - 1] < yrecldp_rlmin:
          zlneg[jm - 1, jk - 1, jl - 1] = zlneg[jm - 1, jk - 1, jl - 1] + zqx[jm - 1, jk - 1, jl - 1]
          zqadj = zqx[jm - 1, jk - 1, jl - 1]*zqtmst
          tendency_loc_q[jk - 1, jl - 1] = tendency_loc_q[jk - 1, jl - 1] + zqadj
          if iphase[jm - 1] == 1:
            tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] - ydthf_ralvdcp*zqadj
          if iphase[jm - 1] == 2:
            tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] - ydthf_ralsdcp*zqadj
          zqx[ncldqv - 1, jk - 1, jl - 1] = zqx[ncldqv - 1, jk - 1, jl - 1] + zqx[jm - 1, jk - 1, jl - 1]
          zqx[jm - 1, jk - 1, jl - 1] = 0.0
  
  
  # ------------------------------
  # Define saturation values
  # ------------------------------
  for jk in range(1, klev + 1):
    for jl in range(kidia, kfdia + 1):
      #----------------------------------------
      # old *diagnostic* mixed phase saturation
      #----------------------------------------
      zfoealfa[jk - 1, jl - 1] = foealfa(ztp1[jk - 1, jl - 1])
      zfoeewmt[jk - 1, jl - 1] = min(foeewm(ztp1[jk - 1, jl - 1]) / pap[jk - 1, jl - 1], 0.5)
      zqsmix[jk - 1, jl - 1] = zfoeewmt[jk - 1, jl - 1]
      zqsmix[jk - 1, jl - 1] = zqsmix[jk - 1, jl - 1] / (1.0 - ydcst_retv*zqsmix[jk - 1, jl - 1])
      
      #---------------------------------------------
      # ice saturation T<273K
      # liquid water saturation for T>273K
      #---------------------------------------------
      zalfa = foedelta(ztp1[jk - 1, jl - 1])
      zfoeew[jk - 1, jl - 1] = min((zalfa*foeeliq(ztp1[jk - 1, jl - 1]) + (1.0 - zalfa)*foeeice(ztp1[jk - 1, jl - 1])) / pap[jk - 1, jl - 1], 0.5)
      zfoeew[jk - 1, jl - 1] = min(0.5, zfoeew[jk - 1, jl - 1])
      zqsice[jk - 1, jl - 1] = zfoeew[jk - 1, jl - 1] / (1.0 - ydcst_retv*zfoeew[jk - 1, jl - 1])
      
      #----------------------------------
      # liquid water saturation
      #----------------------------------
      zfoeeliqt[jk - 1, jl - 1] = min(foeeliq(ztp1[jk - 1, jl - 1]) / pap[jk - 1, jl - 1], 0.5)
      zqsliq[jk - 1, jl - 1] = zfoeeliqt[jk - 1, jl - 1]
      zqsliq[jk - 1, jl - 1] = zqsliq[jk - 1, jl - 1] / (1.0 - ydcst_retv*zqsliq[jk - 1, jl - 1])
      
      #   !----------------------------------
      #   ! ice water saturation
      #   !----------------------------------
      #   ZFOEEICET(JL,JK)=MIN(FOEEICE(ZTP1(JL,JK))/PAP(JL,JK),0.5_JPRB)
      #   ZQSICE(JL,JK)=ZFOEEICET(JL,JK)
      #   ZQSICE(JL,JK)=ZQSICE(JL,JK)/(1.0_JPRB-RETV*ZQSICE(JL,JK))
    
  
  for jk in range(1, klev + 1):
    for jl in range(kidia, kfdia + 1):
      
      
      #------------------------------------------
      # Ensure cloud fraction is between 0 and 1
      #------------------------------------------
      za[jk - 1, jl - 1] = max(0.0, min(1.0, za[jk - 1, jl - 1]))
      
      #-------------------------------------------------------------------
      # Calculate liq/ice fractions (no longer a diagnostic relationship)
      #-------------------------------------------------------------------
      zli[jk - 1, jl - 1] = zqx[ncldql - 1, jk - 1, jl - 1] + zqx[ncldqi - 1, jk - 1, jl - 1]
      if zli[jk - 1, jl - 1] > yrecldp_rlmin:
        zliqfrac[jk - 1, jl - 1] = zqx[ncldql - 1, jk - 1, jl - 1] / zli[jk - 1, jl - 1]
        zicefrac[jk - 1, jl - 1] = 1.0 - zliqfrac[jk - 1, jl - 1]
      else:
        zliqfrac[jk - 1, jl - 1] = 0.0
        zicefrac[jk - 1, jl - 1] = 0.0
      
  
  #######################################################################
  #        2.       *** CONSTANTS AND PARAMETERS ***
  #######################################################################
  #  Calculate L in updrafts of bl-clouds
  #  Specify QS, P/PS for tropopause (for c2)
  #  And initialize variables
  #------------------------------------------
  
  #---------------------------------
  # Find tropopause level (ZTRPAUS)
  #---------------------------------
  for jl in range(kidia, kfdia + 1):
    ztrpaus[jl - 1] = 0.1
    zpaphd[jl - 1] = 1.0 / paph[klev + 1 - 1, jl - 1]
  for jk in range(1, klev - 1 + 1):
    for jl in range(kidia, kfdia + 1):
      zsig0 = pap[jk - 1, jl - 1]*zpaphd[jl - 1]
      if zsig0 > 0.1 and zsig0 < 0.4 and ztp1[jk - 1, jl - 1] > ztp1[jk + 1 - 1, jl - 1]:
        ztrpaus[jl - 1] = zsig0
  
  #-----------------------------
  # Reset single level variables
  #-----------------------------
  
  for jl in range(kidia, kfdia + 1):
    zanewm1[jl - 1] = 0.0
    zda[jl - 1] = 0.0
    zcovpclr[jl - 1] = 0.0
    zcovpmax[jl - 1] = 0.0
    zcovptot[jl - 1] = 0.0
    zcldtopdist[jl - 1] = 0.0
  
  #######################################################################
  #           3.       *** PHYSICS ***
  #######################################################################
  
  
  #----------------------------------------------------------------------
  #                       START OF VERTICAL LOOP
  #----------------------------------------------------------------------
  
  for jk in range(yrecldp_ncldtop, klev + 1):
    
    #----------------------------------------------------------------------
    # 3.0 INITIALIZE VARIABLES
    #----------------------------------------------------------------------
    
    #---------------------------------
    # First guess microphysics
    #---------------------------------
    for jm in range(1, nclv + 1):
      for jl in range(kidia, kfdia + 1):
        zqxfg[jm - 1, jl - 1] = zqx[jm - 1, jk - 1, jl - 1]
    
    #---------------------------------
    # Set KLON arrays to zero
    #---------------------------------
    
    for jl in range(kidia, kfdia + 1):
      zlicld[jl - 1] = 0.0
      zrainaut[jl - 1] = 0.0        # currently needed for diags
      zrainacc[jl - 1] = 0.0        # currently needed for diags
      zsnowaut[jl - 1] = 0.0        # needed
      zldefr[jl - 1] = 0.0
      zacust[jl - 1] = 0.0        # set later when needed
      zqpretot[jl - 1] = 0.0
      zlfinalsum[jl - 1] = 0.0
      
      # Required for first guess call
      zlcond1[jl - 1] = 0.0
      zlcond2[jl - 1] = 0.0
      zsupsat[jl - 1] = 0.0
      zlevapl[jl - 1] = 0.0
      zlevapi[jl - 1] = 0.0
      
      #-------------------------------------
      # solvers for cloud fraction
      #-------------------------------------
      zsolab[jl - 1] = 0.0
      zsolac[jl - 1] = 0.0
      
      zicetot[jl - 1] = 0.0
    
    #------------------------------------------
    # reset matrix so missing pathways are set
    #------------------------------------------
    for jm in range(1, nclv + 1):
      for jn in range(1, nclv + 1):
        for jl in range(kidia, kfdia + 1):
          zsolqb[jm - 1, jn - 1, jl - 1] = 0.0
          zsolqa[jm - 1, jn - 1, jl - 1] = 0.0
    
    #----------------------------------
    # reset new microphysics variables
    #----------------------------------
    for jm in range(1, nclv + 1):
      for jl in range(kidia, kfdia + 1):
        zfallsrce[jm - 1, jl - 1] = 0.0
        zfallsink[jm - 1, jl - 1] = 0.0
        zconvsrce[jm - 1, jl - 1] = 0.0
        zconvsink[jm - 1, jl - 1] = 0.0
        zpsupsatsrce[jm - 1, jl - 1] = 0.0
        zratio[jm - 1, jl - 1] = 0.0
    
    for jl in range(kidia, kfdia + 1):
      
      #-------------------------
      # derived variables needed
      #-------------------------
      
      zdp[jl - 1] = paph[jk + 1 - 1, jl - 1] - paph[jk - 1, jl - 1]        # dp
      zgdp[jl - 1] = ydcst_rg / zdp[jl - 1]        # g/dp
      zrho[jl - 1] = pap[jk - 1, jl - 1] / (ydcst_rd*ztp1[jk - 1, jl - 1])        # p/RT air density
      
      zdtgdp[jl - 1] = ptsphy*zgdp[jl - 1]        # dt g/dp
      zrdtgdp[jl - 1] = zdp[jl - 1]*(1.0 / (ptsphy*ydcst_rg))        # 1/(dt g/dp)
      
      if jk > 1:
        zdtgdpf[jl - 1] = ptsphy*ydcst_rg / (pap[jk - 1, jl - 1] - pap[jk - 1 - 1, jl - 1])
      
      #------------------------------------
      # Calculate dqs/dT correction factor
      #------------------------------------
      # Reminder: RETV=RV/RD-1
      
      # liquid
      zfacw0 = ydthf_r5les / ((ztp1[jk - 1, jl - 1] - ydthf_r4les)**2)
      zcor0 = 1.0 / (1.0 - ydcst_retv*zfoeeliqt[jk - 1, jl - 1])
      zdqsliqdt[jl - 1] = zfacw0*zcor*zqsliq[jk - 1, jl - 1]
      zcorqsliq[jl - 1] = 1.0 + ydthf_ralvdcp*zdqsliqdt[jl - 1]
      
      # ice
      zfaci0 = ydthf_r5ies / ((ztp1[jk - 1, jl - 1] - ydthf_r4ies)**2)
      zcor0 = 1.0 / (1.0 - ydcst_retv*zfoeew[jk - 1, jl - 1])
      zdqsicedt[jl - 1] = zfaci0*zcor0*zqsice[jk - 1, jl - 1]
      zcorqsice[jl - 1] = 1.0 + ydthf_ralsdcp*zdqsicedt[jl - 1]
      
      # diagnostic mixed
      zalfaw0 = zfoealfa[jk - 1, jl - 1]
      zalfawm[jl - 1] = zalfaw0
      zfac = zalfaw0*zfacw0 + (1.0 - zalfaw0)*zfaci0
      zcor = 1.0 / (1.0 - ydcst_retv*zfoeewmt[jk - 1, jl - 1])
      zdqsmixdt[jl - 1] = zfac*zcor*zqsmix[jk - 1, jl - 1]
      zcorqsmix[jl - 1] = 1.0 + foeldcpm(ztp1[jk - 1, jl - 1])*zdqsmixdt[jl - 1]
      
      # evaporation/sublimation limits
      zevaplimmix[jl - 1] = max((zqsmix[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1]) / zcorqsmix[jl - 1], 0.0)
      zevaplimliq[jl - 1] = max((zqsliq[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1]) / zcorqsliq[jl - 1], 0.0)
      zevaplimice[jl - 1] = max((zqsice[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1]) / zcorqsice[jl - 1], 0.0)
      
      #--------------------------------
      # in-cloud consensate amount
      #--------------------------------
      ztmpa = 1.0 / max(za[jk - 1, jl - 1], zepsec)
      zliqcld[jl - 1] = zqx[ncldql - 1, jk - 1, jl - 1]*ztmpa
      zicecld[jl - 1] = zqx[ncldqi - 1, jk - 1, jl - 1]*ztmpa
      zlicld[jl - 1] = zliqcld[jl - 1] + zicecld[jl - 1]
      
    
    #------------------------------------------------
    # Evaporate very small amounts of liquid and ice
    #------------------------------------------------
    for jl in range(kidia, kfdia + 1):
      
      if zqx[ncldql - 1, jk - 1, jl - 1] < yrecldp_rlmin:
        zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zqx[ncldql - 1, jk - 1, jl - 1]
        zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = -zqx[ncldql - 1, jk - 1, jl - 1]
      
      if zqx[ncldqi - 1, jk - 1, jl - 1] < yrecldp_rlmin:
        zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zqx[ncldqi - 1, jk - 1, jl - 1]
        zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = -zqx[ncldqi - 1, jk - 1, jl - 1]
      
    
    #---------------------------------------------------------------------
    #  3.1  ICE SUPERSATURATION ADJUSTMENT
    #---------------------------------------------------------------------
    # Note that the supersaturation adjustment is made with respect to
    # liquid saturation:  when T>0C
    # ice saturation:     when T<0C
    #                     with an adjustment made to allow for ice
    #                     supersaturation in the clear sky
    # Note also that the KOOP factor automatically clips the supersaturation
    # to a maximum set by the liquid water saturation mixing ratio
    # important for temperatures near to but below 0C
    #-----------------------------------------------------------------------
    
    #DIR$ NOFUSION
    for jl in range(kidia, kfdia + 1):
      
      #-----------------------------------
      # 3.1.1 Supersaturation limit (from Koop)
      #-----------------------------------
      # Needs to be set for all temperatures
      zfokoop[jl - 1] = fokoop(ztp1[jk - 1, jl - 1])
    for jl in range(kidia, kfdia + 1):
      
      if ztp1[jk - 1, jl - 1] >= ydcst_rtt or yrecldp_nssopt == 0:
        zfac = 1.0
        zfaci = 1.0
      else:
        zfac = za[jk - 1, jl - 1] + zfokoop[jl - 1]*(1.0 - za[jk - 1, jl - 1])
        zfaci = ptsphy / yrecldp_rkooptau
      
      #-------------------------------------------------------------------
      # 3.1.2 Calculate supersaturation wrt Koop including dqs/dT
      #       correction factor
      # [#Note: QSICE or QSLIQ]
      #-------------------------------------------------------------------
      
      # Calculate supersaturation to add to cloud
      if za[jk - 1, jl - 1] > 1.0 - yrecldp_ramin:
        zsupsat[jl - 1] = max((zqx[ncldqv - 1, jk - 1, jl - 1] - zfac*zqsice[jk - 1, jl - 1]) / zcorqsice[jl - 1], 0.0)
      else:
        # Calculate environmental humidity supersaturation
        zqp1env = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1]*zqsice[jk - 1, jl - 1]) / max(1.0 - za[jk - 1, jl - 1], zepsilon)
        #& SIGN(MAX(ABS(1.0_JPRB-ZA(JL,JK)),ZEPSILON),1.0_JPRB-ZA(JL,JK))
        zsupsat[jl - 1] = max((1.0 - za[jk - 1, jl - 1])*(zqp1env - zfac*zqsice[jk - 1, jl - 1]) / zcorqsice[jl - 1], 0.0)
      
      #-------------------------------------------------------------------
      # Here the supersaturation is turned into liquid water
      # However, if the temperature is below the threshold for homogeneous
      # freezing then the supersaturation is turned instantly to ice.
      #--------------------------------------------------------------------
      
      if zsupsat[jl - 1] > zepsec:
        
        if ztp1[jk - 1, jl - 1] > yrecldp_rthomo:
          # Turn supersaturation into liquid water
          zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = zsolqa[ncldqv - 1, ncldql - 1, jl - 1] + zsupsat[jl - 1]
          zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zsolqa[ncldql - 1, ncldqv - 1, jl - 1] - zsupsat[jl - 1]
          # Include liquid in first guess
          zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] + zsupsat[jl - 1]
        else:
          # Turn supersaturation into ice water
          zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] + zsupsat[jl - 1]
          zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] - zsupsat[jl - 1]
          # Add ice to first guess for deposition term
          zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zsupsat[jl - 1]
        
        # Increase cloud amount using RKOOPTAU timescale
        zsolac[jl - 1] = (1.0 - za[jk - 1, jl - 1])*zfaci
        
      
      #-------------------------------------------------------
      # 3.1.3 Include supersaturation from previous timestep
      # (Calculated in sltENDIF semi-lagrangian LDSLPHY=T)
      #-------------------------------------------------------
      if psupsat[jk - 1, jl - 1] > zepsec:
        if ztp1[jk - 1, jl - 1] > yrecldp_rthomo:
          # Turn supersaturation into liquid water
          zsolqa[ncldql - 1, ncldql - 1, jl - 1] = zsolqa[ncldql - 1, ncldql - 1, jl - 1] + psupsat[jk - 1, jl - 1]
          zpsupsatsrce[ncldql - 1, jl - 1] = psupsat[jk - 1, jl - 1]
          # Add liquid to first guess for deposition term
          zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] + psupsat[jk - 1, jl - 1]
          # Store cloud budget diagnostics if required
        else:
          # Turn supersaturation into ice water
          zsolqa[ncldqi - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqi - 1, jl - 1] + psupsat[jk - 1, jl - 1]
          zpsupsatsrce[ncldqi - 1, jl - 1] = psupsat[jk - 1, jl - 1]
          # Add ice to first guess for deposition term
          zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + psupsat[jk - 1, jl - 1]
          # Store cloud budget diagnostics if required
        
        # Increase cloud amount using RKOOPTAU timescale
        zsolac[jl - 1] = (1.0 - za[jk - 1, jl - 1])*zfaci
        # Store cloud budget diagnostics if required
      
    # on JL
    
    #---------------------------------------------------------------------
    #  3.2  DETRAINMENT FROM CONVECTION
    #---------------------------------------------------------------------
    # * Diagnostic T-ice/liq split retained for convection
    #    Note: This link is now flexible and a future convection
    #    scheme can detrain explicit seperate budgets of:
    #    cloud water, ice, rain and snow
    # * There is no (1-ZA) multiplier term on the cloud detrainment
    #    term, since is now written in mass-flux terms
    # [#Note: Should use ZFOEALFACU used in convection rather than ZFOEALFA]
    #---------------------------------------------------------------------
    if jk < klev and jk >= yrecldp_ncldtop:
      
      for jl in range(kidia, kfdia + 1):
        
        plude[jk - 1, jl - 1] = plude[jk - 1, jl - 1]*zdtgdp[jl - 1]
        
        if ldcum[jl - 1] and plude[jk - 1, jl - 1] > yrecldp_rlmin and plu[jk + 1 - 1, jl - 1] > zepsec:
          
          zsolac[jl - 1] = zsolac[jl - 1] + plude[jk - 1, jl - 1] / plu[jk + 1 - 1, jl - 1]
          # *diagnostic temperature split*
          zalfaw = zfoealfa[jk - 1, jl - 1]
          zconvsrce[ncldql - 1, jl - 1] = zalfaw*plude[jk - 1, jl - 1]
          zconvsrce[ncldqi - 1, jl - 1] = (1.0 - zalfaw)*plude[jk - 1, jl - 1]
          zsolqa[ncldql - 1, ncldql - 1, jl - 1] = zsolqa[ncldql - 1, ncldql - 1, jl - 1] + zconvsrce[ncldql - 1, jl - 1]
          zsolqa[ncldqi - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqi - 1, jl - 1] + zconvsrce[ncldqi - 1, jl - 1]
          
        else:
          
          plude[jk - 1, jl - 1] = 0.0
          
        # *convective snow detrainment source
        if ldcum[jl - 1]:
          zsolqa[ncldqs - 1, ncldqs - 1, jl - 1] = zsolqa[ncldqs - 1, ncldqs - 1, jl - 1] + psnde[jk - 1, jl - 1]*zdtgdp[jl - 1]
        
      
    # JK<KLEV
    
    #---------------------------------------------------------------------
    #  3.3  SUBSIDENCE COMPENSATING CONVECTIVE UPDRAUGHTS
    #---------------------------------------------------------------------
    # Three terms:
    # * Convective subsidence source of cloud from layer above
    # * Evaporation of cloud within the layer
    # * Subsidence sink of cloud to the layer below (Implicit solution)
    #---------------------------------------------------------------------
    
    #-----------------------------------------------
    # Subsidence source from layer above
    #               and
    # Evaporation of cloud within the layer
    #-----------------------------------------------
    if jk > yrecldp_ncldtop:
      
      for jl in range(kidia, kfdia + 1):
        zmf[jl - 1] = max(0.0, (pmfu[jk - 1, jl - 1] + pmfd[jk - 1, jl - 1])*zdtgdp[jl - 1])
        zacust[jl - 1] = zmf[jl - 1]*zanewm1[jl - 1]
      
      for jm in range(1, nclv + 1):
        if not llfall[jm - 1] and iphase[jm - 1] > 0:
          for jl in range(kidia, kfdia + 1):
            zlcust[jm - 1, jl - 1] = zmf[jl - 1]*zqxnm1[jm - 1, jl - 1]
            # record total flux for enthalpy budget:
            zconvsrce[jm - 1, jl - 1] = zconvsrce[jm - 1, jl - 1] + zlcust[jm - 1, jl - 1]
      
      # Now have to work out how much liquid evaporates at arrival point
      # since there is no prognostic memory for in-cloud humidity, i.e.
      # we always assume cloud is saturated.
      
      for jl in range(kidia, kfdia + 1):
        zdtdp = zrdcp*0.5*(ztp1[jk - 1 - 1, jl - 1] + ztp1[jk - 1, jl - 1]) / paph[jk - 1, jl - 1]
        zdtforc = zdtdp*(pap[jk - 1, jl - 1] - pap[jk - 1 - 1, jl - 1])
        #[#Note: Diagnostic mixed phase should be replaced below]
        zdqs[jl - 1] = zanewm1[jl - 1]*zdtforc*zdqsmixdt[jl - 1]
      
      for jm in range(1, nclv + 1):
        if not llfall[jm - 1] and iphase[jm - 1] > 0:
          for jl in range(kidia, kfdia + 1):
            zlfinal = max(0.0, zlcust[jm - 1, jl - 1] - zdqs[jl - 1])              #lim to zero
            # no supersaturation allowed incloud ---V
            zevap = min((zlcust[jm - 1, jl - 1] - zlfinal), zevaplimmix[jl - 1])
            #          ZEVAP=0.0_JPRB
            zlfinal = zlcust[jm - 1, jl - 1] - zevap
            zlfinalsum[jl - 1] = zlfinalsum[jl - 1] + zlfinal              # sum
            
            zsolqa[jm - 1, jm - 1, jl - 1] = zsolqa[jm - 1, jm - 1, jl - 1] + zlcust[jm - 1, jl - 1]              # whole sum
            zsolqa[jm - 1, ncldqv - 1, jl - 1] = zsolqa[jm - 1, ncldqv - 1, jl - 1] + zevap
            zsolqa[ncldqv - 1, jm - 1, jl - 1] = zsolqa[ncldqv - 1, jm - 1, jl - 1] - zevap
      
      #  Reset the cloud contribution if no cloud water survives to this level:
      for jl in range(kidia, kfdia + 1):
        if zlfinalsum[jl - 1] < zepsec:
          zacust[jl - 1] = 0.0
        zsolac[jl - 1] = zsolac[jl - 1] + zacust[jl - 1]
      
    # on  JK>NCLDTOP
    
    #---------------------------------------------------------------------
    # Subsidence sink of cloud to the layer below
    # (Implicit - re. CFL limit on convective mass flux)
    #---------------------------------------------------------------------
    
    for jl in range(kidia, kfdia + 1):
      
      if jk < klev:
        
        zmfdn = max(0.0, (pmfu[jk + 1 - 1, jl - 1] + pmfd[jk + 1 - 1, jl - 1])*zdtgdp[jl - 1])
        
        zsolab[jl - 1] = zsolab[jl - 1] + zmfdn
        zsolqb[ncldql - 1, ncldql - 1, jl - 1] = zsolqb[ncldql - 1, ncldql - 1, jl - 1] + zmfdn
        zsolqb[ncldqi - 1, ncldqi - 1, jl - 1] = zsolqb[ncldqi - 1, ncldqi - 1, jl - 1] + zmfdn
        
        # Record sink for cloud budget and enthalpy budget diagnostics
        zconvsink[ncldql - 1, jl - 1] = zmfdn
        zconvsink[ncldqi - 1, jl - 1] = zmfdn
        
      
    
    #----------------------------------------------------------------------
    # 3.4  EROSION OF CLOUDS BY TURBULENT MIXING
    #----------------------------------------------------------------------
    # NOTE: In default tiedtke scheme this process decreases the cloud
    #       area but leaves the specific cloud water content
    #       within clouds unchanged
    #----------------------------------------------------------------------
    
    # ------------------------------
    # Define turbulent erosion rate
    # ------------------------------
    for jl in range(kidia, kfdia + 1):
      zldifdt[jl - 1] = yrecldp_rcldiff*ptsphy        #original version
      #Increase by factor of 5 for convective points
      if ktype[jl - 1] > 0 and plude[jk - 1, jl - 1] > zepsec:
        zldifdt[jl - 1] = yrecldp_rcldiff_convi*zldifdt[jl - 1]
    
    # At the moment, works on mixed RH profile and partitioned ice/liq fraction
    # so that it is similar to previous scheme
    # Should apply RHw for liquid cloud and RHi for ice cloud separately
    for jl in range(kidia, kfdia + 1):
      if zli[jk - 1, jl - 1] > zepsec:
        # Calculate environmental humidity
        #      ZQE=(ZQX(JL,JK,NCLDQV)-ZA(JL,JK)*ZQSMIX(JL,JK))/&
        #    &      MAX(ZEPSEC,1.0_JPRB-ZA(JL,JK))
        #      ZE=ZLDIFDT(JL)*MAX(ZQSMIX(JL,JK)-ZQE,0.0_JPRB)
        ze = zldifdt[jl - 1]*max(zqsmix[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1], 0.0)
        zleros = za[jk - 1, jl - 1]*ze
        zleros = min(zleros, zevaplimmix[jl - 1])
        zleros = min(zleros, zli[jk - 1, jl - 1])
        zaeros = zleros / zlicld[jl - 1]          #if linear term
        
        # Erosion is -ve LINEAR in L,A
        zsolac[jl - 1] = zsolac[jl - 1] - zaeros          #linear
        
        zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zsolqa[ncldql - 1, ncldqv - 1, jl - 1] + zliqfrac[jk - 1, jl - 1]*zleros
        zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = zsolqa[ncldqv - 1, ncldql - 1, jl - 1] - zliqfrac[jk - 1, jl - 1]*zleros
        zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] + zicefrac[jk - 1, jl - 1]*zleros
        zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] - zicefrac[jk - 1, jl - 1]*zleros
        
    
    #----------------------------------------------------------------------
    # 3.4  CONDENSATION/EVAPORATION DUE TO DQSAT/DT
    #----------------------------------------------------------------------
    #  calculate dqs/dt
    #  Note: For the separate prognostic Qi and Ql, one would ideally use
    #  Qsat/DT wrt liquid/Koop here, since the physics is that new clouds
    #  forms by liquid droplets [liq] or when aqueous aerosols [Koop] form.
    #  These would then instantaneous freeze if T<-38C or lead to ice growth
    #  by deposition in warmer mixed phase clouds.  However, since we do
    #  not have a separate prognostic equation for in-cloud humidity or a
    #  statistical scheme approach in place, the depositional growth of ice
    #  in the mixed phase can not be modelled and we resort to supersaturation
    #  wrt ice instanteously converting to ice over one timestep
    #  (see Tompkins et al. QJRMS 2007 for details)
    #  Thus for the initial implementation the diagnostic mixed phase is
    #  retained for the moment, and the level of approximation noted.
    #----------------------------------------------------------------------
    
    for jl in range(kidia, kfdia + 1):
      zdtdp = zrdcp*ztp1[jk - 1, jl - 1] / pap[jk - 1, jl - 1]
      zdpmxdt = zdp[jl - 1]*zqtmst
      zmfdn = 0.0
      if jk < klev:
        zmfdn = pmfu[jk + 1 - 1, jl - 1] + pmfd[jk + 1 - 1, jl - 1]
      zwtot = pvervel[jk - 1, jl - 1] + 0.5*ydcst_rg*(pmfu[jk - 1, jl - 1] + pmfd[jk - 1, jl - 1] + zmfdn)
      zwtot = min(zdpmxdt, max(-zdpmxdt, zwtot))
      zzzdt = phrsw[jk - 1, jl - 1] + phrlw[jk - 1, jl - 1]
      zdtdiab = min(zdpmxdt*zdtdp, max(-zdpmxdt*zdtdp, zzzdt))*ptsphy + ydthf_ralfdcp*zldefr[jl - 1]
      # Note: ZLDEFR should be set to the difference between the mixed phase functions
      # in the convection and cloud scheme, but this is not calculated, so is zero and
      # the functions must be the same
      zdtforc = zdtdp*zwtot*ptsphy + zdtdiab
      zqold[jl - 1] = zqsmix[jk - 1, jl - 1]
      ztold[jl - 1] = ztp1[jk - 1, jl - 1]
      ztp1[jk - 1, jl - 1] = ztp1[jk - 1, jl - 1] + zdtforc
      ztp1[jk - 1, jl - 1] = max(ztp1[jk - 1, jl - 1], 160.0)
      llflag[jl - 1] = True
    
    # Formerly a call to CUADJTQ(..., ICALL=5)
    for jl in range(kidia, kfdia + 1):
      zqp = 1.0 / pap[jk - 1, jl - 1]
      zqsat = foeewm(ztp1[jk - 1, jl - 1])*zqp
      zqsat = min(0.5, zqsat)
      zcor = 1.0 / (1.0 - ydcst_retv*zqsat)
      zqsat = zqsat*zcor
      zcond = (zqsmix[jk - 1, jl - 1] - zqsat) / (1.0 + zqsat*zcor*foedem(ztp1[jk - 1, jl - 1]))
      ztp1[jk - 1, jl - 1] = ztp1[jk - 1, jl - 1] + foeldcpm(ztp1[jk - 1, jl - 1])*zcond
      zqsmix[jk - 1, jl - 1] = zqsmix[jk - 1, jl - 1] - zcond
      zqsat = foeewm(ztp1[jk - 1, jl - 1])*zqp
      zqsat = min(0.5, zqsat)
      zcor = 1.0 / (1.0 - ydcst_retv*zqsat)
      zqsat = zqsat*zcor
      zcond1 = (zqsmix[jk - 1, jl - 1] - zqsat) / (1.0 + zqsat*zcor*foedem(ztp1[jk - 1, jl - 1]))
      ztp1[jk - 1, jl - 1] = ztp1[jk - 1, jl - 1] + foeldcpm(ztp1[jk - 1, jl - 1])*zcond1
      zqsmix[jk - 1, jl - 1] = zqsmix[jk - 1, jl - 1] - zcond1
    
    for jl in range(kidia, kfdia + 1):
      zdqs[jl - 1] = zqsmix[jk - 1, jl - 1] - zqold[jl - 1]
      zqsmix[jk - 1, jl - 1] = zqold[jl - 1]
      ztp1[jk - 1, jl - 1] = ztold[jl - 1]
    
    #----------------------------------------------------------------------
    # 3.4a  ZDQS(JL) > 0:  EVAPORATION OF CLOUDS
    # ----------------------------------------------------------------------
    # Erosion term is LINEAR in L
    # Changed to be uniform distribution in cloud region
    
    for jl in range(kidia, kfdia + 1):
      
      # Previous function based on DELTA DISTRIBUTION in cloud:
      if zdqs[jl - 1] > 0.0:
        #    If subsidence evaporation term is turned off, then need to use updated
        #    liquid and cloud here?
        #    ZLEVAP = MAX(ZA(JL,JK)+ZACUST(JL),1.0_JPRB)*MIN(ZDQS(JL),ZLICLD(JL)+ZLFINALSUM(JL))
        zlevap = za[jk - 1, jl - 1]*min(zdqs[jl - 1], zlicld[jl - 1])
        zlevap = min(zlevap, zevaplimmix[jl - 1])
        zlevap = min(zlevap, max(zqsmix[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1], 0.0))
        
        # For first guess call
        zlevapl[jl - 1] = zliqfrac[jk - 1, jl - 1]*zlevap
        zlevapi[jl - 1] = zicefrac[jk - 1, jl - 1]*zlevap
        
        zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zsolqa[ncldql - 1, ncldqv - 1, jl - 1] + zliqfrac[jk - 1, jl - 1]*zlevap
        zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = zsolqa[ncldqv - 1, ncldql - 1, jl - 1] - zliqfrac[jk - 1, jl - 1]*zlevap
        
        zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] + zicefrac[jk - 1, jl - 1]*zlevap
        zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] - zicefrac[jk - 1, jl - 1]*zlevap
        
      
    
    #----------------------------------------------------------------------
    # 3.4b ZDQS(JL) < 0: FORMATION OF CLOUDS
    #----------------------------------------------------------------------
    # (1) Increase of cloud water in existing clouds
    for jl in range(kidia, kfdia + 1):
      if za[jk - 1, jl - 1] > zepsec and zdqs[jl - 1] <= -yrecldp_rlmin:
        
        zlcond1[jl - 1] = max(-zdqs[jl - 1], 0.0)          #new limiter
        
        #old limiter (significantly improves upper tropospheric humidity rms)
        if za[jk - 1, jl - 1] > 0.99:
          zcor = 1.0 / (1.0 - ydcst_retv*zqsmix[jk - 1, jl - 1])
          zcdmax = (zqx[ncldqv - 1, jk - 1, jl - 1] - zqsmix[jk - 1, jl - 1]) / (1.0 + zcor*zqsmix[jk - 1, jl - 1]*foedem(ztp1[jk - 1, jl - 1]))
        else:
          zcdmax = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1]*zqsmix[jk - 1, jl - 1]) / za[jk - 1, jl - 1]
        zlcond1[jl - 1] = max(min(zlcond1[jl - 1], zcdmax), 0.0)
        # end old limiter
        
        zlcond1[jl - 1] = za[jk - 1, jl - 1]*zlcond1[jl - 1]
        if zlcond1[jl - 1] < yrecldp_rlmin:
          zlcond1[jl - 1] = 0.0
        
        #-------------------------------------------------------------------------
        # All increase goes into liquid unless so cold cloud homogeneously freezes
        # Include new liquid formation in first guess value, otherwise liquid
        # remains at cold temperatures until next timestep.
        #-------------------------------------------------------------------------
        if ztp1[jk - 1, jl - 1] > yrecldp_rthomo:
          zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = zsolqa[ncldqv - 1, ncldql - 1, jl - 1] + zlcond1[jl - 1]
          zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zsolqa[ncldql - 1, ncldqv - 1, jl - 1] - zlcond1[jl - 1]
          zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] + zlcond1[jl - 1]
        else:
          zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] + zlcond1[jl - 1]
          zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] - zlcond1[jl - 1]
          zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zlcond1[jl - 1]
    
    # (2) Generation of new clouds (da/dt>0)
    
    for jl in range(kidia, kfdia + 1):
      
      if zdqs[jl - 1] <= -yrecldp_rlmin and za[jk - 1, jl - 1] < 1.0 - zepsec:
        
        #---------------------------
        # Critical relative humidity
        #---------------------------
        zsigk = pap[jk - 1, jl - 1] / paph[klev + 1 - 1, jl - 1]
        # Increase RHcrit to 1.0 towards the surface (eta>0.8)
        if zsigk > 0.8:
          zrhc = yrecldp_ramid + (1.0 - yrecldp_ramid)*((zsigk - 0.8) / 0.2)**2
        else:
            zrhc = yrecldp_ramid


        # Commented out for CY37R1 to reduce humidity in high trop and strat
        #      ! Increase RHcrit to 1.0 towards the tropopause (trop-0.2) and above
        #      ZBOTT=ZTRPAUS(JL)+0.2_JPRB
        #      IF(ZSIGK < ZBOTT) THEN
        #        ZRHC=RAMID+(1.0_JPRB-RAMID)*MIN(((ZBOTT-ZSIGK)/0.2_JPRB)**2,1.0_JPRB)
        #      ENDIF
        
        #---------------------------
        # Supersaturation options
        #---------------------------
        if yrecldp_nssopt == 0:
          # No scheme
          zqe = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1]*zqsice[jk - 1, jl - 1]) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
          zqe = max(0.0, zqe)
        elif yrecldp_nssopt == 1:
          # Tompkins
          zqe = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1]*zqsice[jk - 1, jl - 1]) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
          zqe = max(0.0, zqe)
        elif yrecldp_nssopt == 2:
          # Lohmann and Karcher
          zqe = zqx[ncldqv - 1, jk - 1, jl - 1]
        elif yrecldp_nssopt == 3:
          # Gierens
          zqe = zqx[ncldqv - 1, jk - 1, jl - 1] + zli[jk - 1, jl - 1]
        
        if ztp1[jk - 1, jl - 1] >= ydcst_rtt or yrecldp_nssopt == 0:
          # No ice supersaturation allowed
          zfac = 1.0
        else:
          # Ice supersaturation
          zfac = zfokoop[jl - 1]
        
        if zqe >= zrhc*zqsice[jk - 1, jl - 1]*zfac and zqe < zqsice[jk - 1, jl - 1]*zfac:
          # note: not **2 on 1-a term if ZQE is used.
          # Added correction term ZFAC to numerator 15/03/2010
          zacond = -(1.0 - za[jk - 1, jl - 1])*zfac*zdqs[jl - 1] / max(2.0*(zfac*zqsice[jk - 1, jl - 1] - zqe), zepsec)
          
          zacond = min(zacond, 1.0 - za[jk - 1, jl - 1])            #PUT THE LIMITER BACK
          
          # Linear term:
          # Added correction term ZFAC 15/03/2010
          zlcond2[jl - 1] = -zfac*zdqs[jl - 1]*0.5*zacond            #mine linear
          
          # new limiter formulation
          zzdl = 2.0*(zfac*zqsice[jk - 1, jl - 1] - zqe) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
          # Added correction term ZFAC 15/03/2010
          if zfac*zdqs[jl - 1] < -zzdl:
            # ZLCONDLIM=(ZA(JL,JK)-1.0_JPRB)*ZDQS(JL)-ZQSICE(JL,JK)+ZQX(JL,JK,NCLDQV)
            zlcondlim = (za[jk - 1, jl - 1] - 1.0)*zfac*zdqs[jl - 1] - zfac*zqsice[jk - 1, jl - 1] + zqx[ncldqv - 1, jk - 1, jl - 1]
            zlcond2[jl - 1] = min(zlcond2[jl - 1], zlcondlim)
          zlcond2[jl - 1] = max(zlcond2[jl - 1], 0.0)
          
          if zlcond2[jl - 1] < yrecldp_rlmin or (1.0 - za[jk - 1, jl - 1]) < zepsec:
            zlcond2[jl - 1] = 0.0
            zacond = 0.0
          if zlcond2[jl - 1] == 0.0:
            zacond = 0.0
          
          # Large-scale generation is LINEAR in A and LINEAR in L
          zsolac[jl - 1] = zsolac[jl - 1] + zacond            #linear
          
          #------------------------------------------------------------------------
          # All increase goes into liquid unless so cold cloud homogeneously freezes
          # Include new liquid formation in first guess value, otherwise liquid
          # remains at cold temperatures until next timestep.
          #------------------------------------------------------------------------
          if ztp1[jk - 1, jl - 1] > yrecldp_rthomo:
            zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = zsolqa[ncldqv - 1, ncldql - 1, jl - 1] + zlcond2[jl - 1]
            zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zsolqa[ncldql - 1, ncldqv - 1, jl - 1] - zlcond2[jl - 1]
            zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] + zlcond2[jl - 1]
          else:
            # homogeneous freezing
            zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] + zlcond2[jl - 1]
            zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] - zlcond2[jl - 1]
            zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zlcond2[jl - 1]
          
    
    #----------------------------------------------------------------------
    # 3.7 Growth of ice by vapour deposition
    #----------------------------------------------------------------------
    # Following Rotstayn et al. 2001:
    # does not use the ice nuclei number from cloudaer.F90
    # but rather a simple Meyers et al. 1992 form based on the
    # supersaturation and assuming clouds are saturated with
    # respect to liquid water (well mixed), (or Koop adjustment)
    # Growth considered as sink of liquid water if present so
    # Bergeron-Findeisen adjustment in autoconversion term no longer needed
    #----------------------------------------------------------------------
    
    #--------------------------------------------------------
    #-
    #- Ice deposition following Rotstayn et al. (2001)
    #-  (monodisperse ice particle size distribution)
    #-
    #--------------------------------------------------------
    if idepice == 1:
      
      for jl in range(kidia, kfdia + 1):
        
        #--------------------------------------------------------------
        # Calculate distance from cloud top
        # defined by cloudy layer below a layer with cloud frac <0.01
        # ZDZ = ZDP(JL)/(ZRHO(JL)*RG)
        #--------------------------------------------------------------
        
        if za[jk - 1 - 1, jl - 1] < yrecldp_rcldtopcf and za[jk - 1, jl - 1] >= yrecldp_rcldtopcf:
          zcldtopdist[jl - 1] = 0.0
        else:
          zcldtopdist[jl - 1] = zcldtopdist[jl - 1] + zdp[jl - 1] / (zrho[jl - 1]*ydcst_rg)
        
        #--------------------------------------------------------------
        # only treat depositional growth if liquid present. due to fact
        # that can not model ice growth from vapour without additional
        # in-cloud water vapour variable
        #--------------------------------------------------------------
        if ztp1[jk - 1, jl - 1] < ydcst_rtt and zqxfg[ncldql - 1, jl - 1] > yrecldp_rlmin:
          # T<273K
          
          zvpice = foeeice(ztp1[jk - 1, jl - 1])*ydcst_rv / ydcst_rd
          zvpliq = zvpice*zfokoop[jl - 1]
          zicenuclei[jl - 1] = 1000.0*np.exp(12.96*(zvpliq - zvpice) / zvpliq - 0.639)
          
          #------------------------------------------------
          #   2.4e-2 is conductivity of air
          #   8.8 = 700**1/3 = density of ice to the third
          #------------------------------------------------
          zadd = ydcst_rlstt*(ydcst_rlstt / (ydcst_rv*ztp1[jk - 1, jl - 1]) - 1.0) / (2.4E-2*ztp1[jk - 1, jl - 1])
          zbdd = ydcst_rv*ztp1[jk - 1, jl - 1]*pap[jk - 1, jl - 1] / (2.21*zvpice)
          zcvds = 7.8*(zicenuclei[jl - 1] / zrho[jl - 1])**0.666*(zvpliq - zvpice) / (8.87*(zadd + zbdd)*zvpice)
          
          #-----------------------------------------------------
          # RICEINIT=1.E-12_JPRB is initial mass of ice particle
          #-----------------------------------------------------
          zice0 = max(zicecld[jl - 1], zicenuclei[jl - 1]*yrecldp_riceinit / zrho[jl - 1])
          
          #------------------
          # new value of ice:
          #------------------
          zinew = (0.666*zcvds*ptsphy + zice0**0.666)**1.5
          
          #---------------------------
          # grid-mean deposition rate:
          #---------------------------
          zdepos = max(za[jk - 1, jl - 1]*(zinew - zice0), 0.0)
          
          #--------------------------------------------------------------------
          # Limit deposition to liquid water amount
          # If liquid is all frozen, ice would use up reservoir of water
          # vapour in excess of ice saturation mixing ratio - However this
          # can not be represented without a in-cloud humidity variable. Using
          # the grid-mean humidity would imply a large artificial horizontal
          # flux from the clear sky to the cloudy area. We thus rely on the
          # supersaturation check to clean up any remaining supersaturation
          #--------------------------------------------------------------------
          zdepos = min(zdepos, zqxfg[ncldql - 1, jl - 1])            # limit to liquid water amount
          
          #--------------------------------------------------------------------
          # At top of cloud, reduce deposition rate near cloud top to account for
          # small scale turbulent processes, limited ice nucleation and ice fallout
          #--------------------------------------------------------------------
          #      ZDEPOS = ZDEPOS*MIN(RDEPLIQREFRATE+ZCLDTOPDIST(JL)/RDEPLIQREFDEPTH,1.0_JPRB)
          # Change to include dependence on ice nuclei concentration
          # to increase deposition rate with decreasing temperatures
          zinfactor = min(zicenuclei[jl - 1] / 15000., 1.0)
          zdepos = zdepos*min(zinfactor + (1.0 - zinfactor)*(yrecldp_rdepliqrefrate + zcldtopdist[jl - 1] / yrecldp_rdepliqrefdepth), 1.0)
          
          #--------------
          # add to matrix
          #--------------
          zsolqa[ncldql - 1, ncldqi - 1, jl - 1] = zsolqa[ncldql - 1, ncldqi - 1, jl - 1] + zdepos
          zsolqa[ncldqi - 1, ncldql - 1, jl - 1] = zsolqa[ncldqi - 1, ncldql - 1, jl - 1] - zdepos
          zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zdepos
          zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] - zdepos
          
      
      #--------------------------------------------------------
      #-
      #- Ice deposition assuming ice PSD
      #-
      #--------------------------------------------------------
    elif idepice == 2:
      
      for jl in range(kidia, kfdia + 1):
        
        #--------------------------------------------------------------
        # Calculate distance from cloud top
        # defined by cloudy layer below a layer with cloud frac <0.01
        # ZDZ = ZDP(JL)/(ZRHO(JL)*RG)
        #--------------------------------------------------------------
        
        if za[jk - 1 - 1, jl - 1] < yrecldp_rcldtopcf and za[jk - 1, jl - 1] >= yrecldp_rcldtopcf:
          zcldtopdist[jl - 1] = 0.0
        else:
          zcldtopdist[jl - 1] = zcldtopdist[jl - 1] + zdp[jl - 1] / (zrho[jl - 1]*ydcst_rg)
        
        #--------------------------------------------------------------
        # only treat depositional growth if liquid present. due to fact
        # that can not model ice growth from vapour without additional
        # in-cloud water vapour variable
        #--------------------------------------------------------------
        if ztp1[jk - 1, jl - 1] < ydcst_rtt and zqxfg[ncldql - 1, jl - 1] > yrecldp_rlmin:
          # T<273K
          
          zvpice = foeeice(ztp1[jk - 1, jl - 1])*ydcst_rv / ydcst_rd
          zvpliq = zvpice*zfokoop[jl - 1]
          zicenuclei[jl - 1] = 1000.0*np.exp(12.96*(zvpliq - zvpice) / zvpliq - 0.639)
          
          #-----------------------------------------------------
          # RICEINIT=1.E-12_JPRB is initial mass of ice particle
          #-----------------------------------------------------
          zice0 = max(zicecld[jl - 1], zicenuclei[jl - 1]*yrecldp_riceinit / zrho[jl - 1])
          
          # Particle size distribution
          ztcg = 1.0
          zfacx1i = 1.0
          
          zaplusb = yrecldp_rcl_apb1*zvpice - yrecldp_rcl_apb2*zvpice*ztp1[jk - 1, jl - 1] + pap[jk - 1, jl - 1]*yrecldp_rcl_apb3*ztp1[jk - 1, jl - 1]**3.
          zcorrfac = (1.0 / zrho[jl - 1])**0.5
          zcorrfac2 = ((ztp1[jk - 1, jl - 1] / 273.0)**1.5)*(393.0 / (ztp1[jk - 1, jl - 1] + 120.0))
          
          zpr02 = zrho[jl - 1]*zice0*yrecldp_rcl_const1i / (ztcg*zfacx1i)
          
          zterm1 = (zvpliq - zvpice)*ztp1[jk - 1, jl - 1]**2.0*zvpice*zcorrfac2*ztcg*yrecldp_rcl_const2i*zfacx1i / (zrho[jl - 1]*zaplusb*zvpice)
          zterm2 = 0.65*yrecldp_rcl_const6i*zpr02**yrecldp_rcl_const4i + yrecldp_rcl_const3i*zcorrfac**0.5*zrho[jl - 1]**0.5*zpr02**yrecldp_rcl_const5i / zcorrfac2**0.5
          
          zdepos = max(za[jk - 1, jl - 1]*zterm1*zterm2*ptsphy, 0.0)
          
          #--------------------------------------------------------------------
          # Limit deposition to liquid water amount
          # If liquid is all frozen, ice would use up reservoir of water
          # vapour in excess of ice saturation mixing ratio - However this
          # can not be represented without a in-cloud humidity variable. Using
          # the grid-mean humidity would imply a large artificial horizontal
          # flux from the clear sky to the cloudy area. We thus rely on the
          # supersaturation check to clean up any remaining supersaturation
          #--------------------------------------------------------------------
          zdepos = min(zdepos, zqxfg[ncldql - 1, jl - 1])            # limit to liquid water amount
          
          #--------------------------------------------------------------------
          # At top of cloud, reduce deposition rate near cloud top to account for
          # small scale turbulent processes, limited ice nucleation and ice fallout
          #--------------------------------------------------------------------
          # Change to include dependence on ice nuclei concentration
          # to increase deposition rate with decreasing temperatures
          zinfactor = min(zicenuclei[jl - 1] / 15000., 1.0)
          zdepos = zdepos*min(zinfactor + (1.0 - zinfactor)*(yrecldp_rdepliqrefrate + zcldtopdist[jl - 1] / yrecldp_rdepliqrefdepth), 1.0)
          
          #--------------
          # add to matrix
          #--------------
          zsolqa[ncldql - 1, ncldqi - 1, jl - 1] = zsolqa[ncldql - 1, ncldqi - 1, jl - 1] + zdepos
          zsolqa[ncldqi - 1, ncldql - 1, jl - 1] = zsolqa[ncldqi - 1, ncldql - 1, jl - 1] - zdepos
          zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zdepos
          zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] - zdepos
      
    # on IDEPICE
    
    #######################################################################
    #              4  *** PRECIPITATION PROCESSES ***
    #######################################################################
    
    #----------------------------------
    # revise in-cloud consensate amount
    #----------------------------------
    for jl in range(kidia, kfdia + 1):
      ztmpa = 1.0 / max(za[jk - 1, jl - 1], zepsec)
      zliqcld[jl - 1] = zqxfg[ncldql - 1, jl - 1]*ztmpa
      zicecld[jl - 1] = zqxfg[ncldqi - 1, jl - 1]*ztmpa
      zlicld[jl - 1] = zliqcld[jl - 1] + zicecld[jl - 1]
    
    #----------------------------------------------------------------------
    # 4.2 SEDIMENTATION/FALLING OF *ALL* MICROPHYSICAL SPECIES
    #     now that rain, snow, graupel species are prognostic
    #     the precipitation flux can be defined directly level by level
    #     There is no vertical memory required from the flux variable
    #----------------------------------------------------------------------
    
    for jm in range(1, nclv + 1):
      if llfall[jm - 1] or jm == ncldqi:
        for jl in range(kidia, kfdia + 1):
          #------------------------
          # source from layer above
          #------------------------
          if jk > yrecldp_ncldtop:
            zfallsrce[jm - 1, jl - 1] = zpfplsx[jm - 1, jk - 1, jl - 1]*zdtgdp[jl - 1]
            zsolqa[jm - 1, jm - 1, jl - 1] = zsolqa[jm - 1, jm - 1, jl - 1] + zfallsrce[jm - 1, jl - 1]
            zqxfg[jm - 1, jl - 1] = zqxfg[jm - 1, jl - 1] + zfallsrce[jm - 1, jl - 1]
            # use first guess precip----------V
            zqpretot[jl - 1] = zqpretot[jl - 1] + zqxfg[jm - 1, jl - 1]
          #-------------------------------------------------
          # sink to next layer, constant fall speed
          #-------------------------------------------------
          # if aerosol effect then override
          #  note that for T>233K this is the same as above.
          if yrecldp_laericesed and jm == ncldqi:
            zre_ice = pre_ice[jk - 1, jl - 1]
            # The exponent value is from
            # Morrison et al. JAS 2005 Appendix
            zvqx[ncldqi - 1] = 0.002*zre_ice**1.0
          zfall = zvqx[jm - 1]*zrho[jl - 1]
          #-------------------------------------------------
          # modified by Heymsfield and Iaquinta JAS 2000
          #-------------------------------------------------
          # ZFALL = ZFALL*((PAP(JL,JK)*RICEHI1)**(-0.178_JPRB)) &
          #            &*((ZTP1(JL,JK)*RICEHI2)**(-0.394_JPRB))
          
          zfallsink[jm - 1, jl - 1] = zdtgdp[jl - 1]*zfall
          # Cloud budget diagnostic stored at end as implicit
        # jl
      # LLFALL
    # jm
    
    #---------------------------------------------------------------
    # Precip cover overlap using MAX-RAN Overlap
    # Since precipitation is now prognostic we must
    #   1) apply an arbitrary minimum coverage (0.3) if precip>0
    #   2) abandon the 2-flux clr/cld treatment
    #   3) Thus, since we have no memory of the clear sky precip
    #      fraction, we mimic the previous method by reducing
    #      ZCOVPTOT(JL), which has the memory, proportionally with
    #      the precip evaporation rate, taking cloud fraction
    #      into account
    #   #3 above leads to much smoother vertical profiles of
    #   precipitation fraction than the Klein-Jakob scheme which
    #   monotonically increases precip fraction and then resets
    #   it to zero in a step function once clear-sky precip reaches
    #   zero.
    #---------------------------------------------------------------
    for jl in range(kidia, kfdia + 1):
      if zqpretot[jl - 1] > zepsec:
        zcovptot[jl - 1] = 1.0 - ((1.0 - zcovptot[jl - 1])*(1.0 - max(za[jk - 1, jl - 1], za[jk - 1 - 1, jl - 1])) / (1.0 - min(za[jk - 1 - 1, jl - 1], 1.0 - 1.E-06)))
        zcovptot[jl - 1] = max(zcovptot[jl - 1], yrecldp_rcovpmin)
        zcovpclr[jl - 1] = max(0.0, zcovptot[jl - 1] - za[jk - 1, jl - 1])          # clear sky proportion
        zraincld[jl - 1] = zqxfg[ncldqr - 1, jl - 1] / zcovptot[jl - 1]
        zsnowcld[jl - 1] = zqxfg[ncldqs - 1, jl - 1] / zcovptot[jl - 1]
        zcovpmax[jl - 1] = max(zcovptot[jl - 1], zcovpmax[jl - 1])
      else:
        zraincld[jl - 1] = 0.0
        zsnowcld[jl - 1] = 0.0
        zcovptot[jl - 1] = 0.0          # no flux - reset cover
        zcovpclr[jl - 1] = 0.0          # reset clear sky proportion
        zcovpmax[jl - 1] = 0.0          # reset max cover for ZZRH calc
    
    #----------------------------------------------------------------------
    # 4.3a AUTOCONVERSION TO SNOW
    #----------------------------------------------------------------------
    for jl in range(kidia, kfdia + 1):
      
      if ztp1[jk - 1, jl - 1] <= ydcst_rtt:
        #-----------------------------------------------------
        #     Snow Autoconversion rate follow Lin et al. 1983
        #-----------------------------------------------------
        if zicecld[jl - 1] > zepsec:
          
          zzco = ptsphy*yrecldp_rsnowlin1*np.exp(yrecldp_rsnowlin2*(ztp1[jk - 1, jl - 1] - ydcst_rtt))
          
          if yrecldp_laericeauto:
            zlcrit = picrit_aer[jk - 1, jl - 1]
            # 0.3 = N**0.333 with N=0.027
            zzco = zzco*(yrecldp_rnice / pnice[jk - 1, jl - 1])**0.333
          else:
            zlcrit = yrecldp_rlcritsnow
          
          zsnowaut[jl - 1] = zzco*(1.0 - np.exp(-(zicecld[jl - 1] / zlcrit)**2))
          zsolqb[ncldqi - 1, ncldqs - 1, jl - 1] = zsolqb[ncldqi - 1, ncldqs - 1, jl - 1] + zsnowaut[jl - 1]
          
      
      #----------------------------------------------------------------------
      # 4.3b AUTOCONVERSION WARM CLOUDS
      #   Collection and accretion will require separate treatment
      #   but for now we keep this simple treatment
      #----------------------------------------------------------------------
      
      if zliqcld[jl - 1] > zepsec:
        
        #--------------------------------------------------------
        #-
        #- Warm-rain process follow Sundqvist (1989)
        #-
        #--------------------------------------------------------
        if iwarmrain == 1:
          
          zzco = yrecldp_rkconv*ptsphy
          
          if yrecldp_laerliqautolsp:
            zlcrit = plcrit_aer[jk - 1, jl - 1]
            # 0.3 = N**0.333 with N=125 cm-3
            zzco = zzco*(yrecldp_rccn / pccn[jk - 1, jl - 1])**0.333
          else:
            # Modify autoconversion threshold dependent on:
            #  land (polluted, high CCN, smaller droplets, higher threshold)
            #  sea  (clean, low CCN, larger droplets, lower threshold)
            if plsm[jl - 1] > 0.5:
              zlcrit = yrecldp_rclcrit_land                # land
            else:
              zlcrit = yrecldp_rclcrit_sea                # ocean
          
          #------------------------------------------------------------------
          # Parameters for cloud collection by rain and snow.
          # Note that with new prognostic variable it is now possible
          # to REPLACE this with an explicit collection parametrization
          #------------------------------------------------------------------
          zprecip = (zpfplsx[ncldqs - 1, jk - 1, jl - 1] + zpfplsx[ncldqr - 1, jk - 1, jl - 1]) / max(zepsec, zcovptot[jl - 1])
          zcfpr = 1.0 + yrecldp_rprc1*np.sqrt(max(zprecip, 0.0))
          #      ZCFPR=1.0_JPRB + RPRC1*SQRT(MAX(ZPRECIP,0.0_JPRB))*&
          #       &ZCOVPTOT(JL)/(MAX(ZA(JL,JK),ZEPSEC))
          
          if yrecldp_laerliqcoll:
            # 5.0 = N**0.333 with N=125 cm-3
            zcfpr = zcfpr*(yrecldp_rccn / pccn[jk - 1, jl - 1])**0.333
          
          zzco = zzco*zcfpr
          zlcrit = zlcrit / max(zcfpr, zepsec)
          
          if zliqcld[jl - 1] / zlcrit < 20.0:
            # Security for exp for some compilers
            zrainaut[jl - 1] = zzco*(1.0 - np.exp(-(zliqcld[jl - 1] / zlcrit)**2))
          else:
            zrainaut[jl - 1] = zzco
          
          # rain freezes instantly
          if ztp1[jk - 1, jl - 1] <= ydcst_rtt:
            zsolqb[ncldql - 1, ncldqs - 1, jl - 1] = zsolqb[ncldql - 1, ncldqs - 1, jl - 1] + zrainaut[jl - 1]
          else:
            zsolqb[ncldql - 1, ncldqr - 1, jl - 1] = zsolqb[ncldql - 1, ncldqr - 1, jl - 1] + zrainaut[jl - 1]
          
          #--------------------------------------------------------
          #-
          #- Warm-rain process follow Khairoutdinov and Kogan (2000)
          #-
          #--------------------------------------------------------
        elif iwarmrain == 2:
          
          if plsm[jl - 1] > 0.5:
            # land
            zconst = yrecldp_rcl_kk_cloud_num_land
            zlcrit = yrecldp_rclcrit_land
          else:
            # ocean
            zconst = yrecldp_rcl_kk_cloud_num_sea
            zlcrit = yrecldp_rclcrit_sea
          
          if zliqcld[jl - 1] > zlcrit:
            
            zrainaut[jl - 1] = 1.5*za[jk - 1, jl - 1]*ptsphy*yrecldp_rcl_kkaau*zliqcld[jl - 1]**yrecldp_rcl_kkbauq*zconst**yrecldp_rcl_kkbaun
            
            zrainaut[jl - 1] = min(zrainaut[jl - 1], zqxfg[ncldql - 1, jl - 1])
            if zrainaut[jl - 1] < zepsec:
              zrainaut[jl - 1] = 0.0
            
            zrainacc[jl - 1] = 2.0*za[jk - 1, jl - 1]*ptsphy*yrecldp_rcl_kkaac*(zliqcld[jl - 1]*zraincld[jl - 1])**yrecldp_rcl_kkbac
            
            zrainacc[jl - 1] = min(zrainacc[jl - 1], zqxfg[ncldql - 1, jl - 1])
            if zrainacc[jl - 1] < zepsec:
              zrainacc[jl - 1] = 0.0
            
          else:
            zrainaut[jl - 1] = 0.0
            zrainacc[jl - 1] = 0.0
          
          # If temperature < 0, then autoconversion produces snow rather than rain
          # Explicit
          if ztp1[jk - 1, jl - 1] <= ydcst_rtt:
            zsolqa[ncldql - 1, ncldqs - 1, jl - 1] = zsolqa[ncldql - 1, ncldqs - 1, jl - 1] + zrainaut[jl - 1]
            zsolqa[ncldql - 1, ncldqs - 1, jl - 1] = zsolqa[ncldql - 1, ncldqs - 1, jl - 1] + zrainacc[jl - 1]
            zsolqa[ncldqs - 1, ncldql - 1, jl - 1] = zsolqa[ncldqs - 1, ncldql - 1, jl - 1] - zrainaut[jl - 1]
            zsolqa[ncldqs - 1, ncldql - 1, jl - 1] = zsolqa[ncldqs - 1, ncldql - 1, jl - 1] - zrainacc[jl - 1]
          else:
            zsolqa[ncldql - 1, ncldqr - 1, jl - 1] = zsolqa[ncldql - 1, ncldqr - 1, jl - 1] + zrainaut[jl - 1]
            zsolqa[ncldql - 1, ncldqr - 1, jl - 1] = zsolqa[ncldql - 1, ncldqr - 1, jl - 1] + zrainacc[jl - 1]
            zsolqa[ncldqr - 1, ncldql - 1, jl - 1] = zsolqa[ncldqr - 1, ncldql - 1, jl - 1] - zrainaut[jl - 1]
            zsolqa[ncldqr - 1, ncldql - 1, jl - 1] = zsolqa[ncldqr - 1, ncldql - 1, jl - 1] - zrainacc[jl - 1]
          
        # on IWARMRAIN
        
      # on ZLIQCLD > ZEPSEC
    
    
    #----------------------------------------------------------------------
    # RIMING - COLLECTION OF CLOUD LIQUID DROPS BY SNOW AND ICE
    #      only active if T<0degC and supercooled liquid water is present
    #      AND if not Sundquist autoconversion (as this includes riming)
    #----------------------------------------------------------------------
    if iwarmrain > 1:
      
      for jl in range(kidia, kfdia + 1):
        if ztp1[jk - 1, jl - 1] <= ydcst_rtt and zliqcld[jl - 1] > zepsec:
          
          # Fallspeed air density correction
          zfallcorr = (yrecldp_rdensref / zrho[jl - 1])**0.4
          
          #------------------------------------------------------------------
          # Riming of snow by cloud water - implicit in lwc
          #------------------------------------------------------------------
          if zsnowcld[jl - 1] > zepsec and zcovptot[jl - 1] > 0.01:
            
            # Calculate riming term
            # Factor of liq water taken out because implicit
            zsnowrime[jl - 1] = 0.3*zcovptot[jl - 1]*ptsphy*yrecldp_rcl_const7s*zfallcorr*(zrho[jl - 1]*zsnowcld[jl - 1]*yrecldp_rcl_const1s)**yrecldp_rcl_const8s
            
            # Limit snow riming term
            zsnowrime[jl - 1] = min(zsnowrime[jl - 1], 1.0)
            
            zsolqb[ncldql - 1, ncldqs - 1, jl - 1] = zsolqb[ncldql - 1, ncldqs - 1, jl - 1] + zsnowrime[jl - 1]
            
          
          #------------------------------------------------------------------
          # Riming of ice by cloud water - implicit in lwc
          # NOT YET ACTIVE
          #------------------------------------------------------------------
          #      IF (ZICECLD(JL)>ZEPSEC .AND. ZA(JL,JK)>0.01_JPRB) THEN
          #
          #        ! Calculate riming term
          #        ! Factor of liq water taken out because implicit
          #        ZSNOWRIME(JL) = ZA(JL,JK)*PTSPHY*RCL_CONST7S*ZFALLCORR &
          #     &                  *(ZRHO(JL)*ZICECLD(JL)*RCL_CONST1S)**RCL_CONST8S
          #
          #        ! Limit ice riming term
          #        ZSNOWRIME(JL)=MIN(ZSNOWRIME(JL),1.0_JPRB)
          #
          #        ZSOLQB(JL,NCLDQI,NCLDQL) = ZSOLQB(JL,NCLDQI,NCLDQL) + ZSNOWRIME(JL)
          #
          #      ENDIF
      
    # on IWARMRAIN > 1
    
    
    #----------------------------------------------------------------------
    # 4.4a  MELTING OF SNOW and ICE
    #       with new implicit solver this also has to treat snow or ice
    #       precipitating from the level above... i.e. local ice AND flux.
    #       in situ ice and snow: could arise from LS advection or warming
    #       falling ice and snow: arrives by precipitation process
    #----------------------------------------------------------------------
    for jl in range(kidia, kfdia + 1):
      
      zicetot[jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zqxfg[ncldqs - 1, jl - 1]
      zmeltmax[jl - 1] = 0.0
      
      # If there are frozen hydrometeors present and dry-bulb temperature > 0degC
      if zicetot[jl - 1] > zepsec and ztp1[jk - 1, jl - 1] > ydcst_rtt:
        
        # Calculate subsaturation
        zsubsat = max(zqsice[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1], 0.0)
        
        # Calculate difference between dry-bulb (ZTP1) and the temperature
        # at which the wet-bulb=0degC (RTT-ZSUBSAT*....) using an approx.
        # Melting only occurs if the wet-bulb temperature >0
        # i.e. warming of ice particle due to melting > cooling
        # due to evaporation.
        ztdmtw0 = ztp1[jk - 1, jl - 1] - ydcst_rtt - zsubsat*(ztw1 + ztw2*(pap[jk - 1, jl - 1] - ztw3) - ztw4*(ztp1[jk - 1, jl - 1] - ztw5))
        # Not implicit yet...
        # Ensure ZCONS1 is positive so that ZMELTMAX=0 if ZTDMTW0<0
        zcons1 = abs(ptsphy*(1.0 + 0.5*ztdmtw0) / yrecldp_rtaumel)
        zmeltmax[jl - 1] = max(ztdmtw0*zcons1*zrldcp, 0.0)
    
    # Loop over frozen hydrometeors (ice, snow)
    for jm in range(1, nclv + 1):
      if iphase[jm - 1] == 2:
        jn = imelt[jm - 1]
        for jl in range(kidia, kfdia + 1):
          if zmeltmax[jl - 1] > zepsec and zicetot[jl - 1] > zepsec:
            # Apply melting in same proportion as frozen hydrometeor fractions
            zalfa2 = zqxfg[jm - 1, jl - 1] / zicetot[jl - 1]
            zmelt = min(zqxfg[jm - 1, jl - 1], zalfa2*zmeltmax[jl - 1])
            # needed in first guess
            # This implies that zqpretot has to be recalculated below
            # since is not conserved here if ice falls and liquid doesn't
            zqxfg[jm - 1, jl - 1] = zqxfg[jm - 1, jl - 1] - zmelt
            zqxfg[jn - 1, jl - 1] = zqxfg[jn - 1, jl - 1] + zmelt
            zsolqa[jm - 1, jn - 1, jl - 1] = zsolqa[jm - 1, jn - 1, jl - 1] + zmelt
            zsolqa[jn - 1, jm - 1, jl - 1] = zsolqa[jn - 1, jm - 1, jl - 1] - zmelt
    
    #----------------------------------------------------------------------
    # 4.4b  FREEZING of RAIN
    #----------------------------------------------------------------------
    for jl in range(kidia, kfdia + 1):
      
      # If rain present
      if zqx[ncldqr - 1, jk - 1, jl - 1] > zepsec:
        
        if ztp1[jk - 1, jl - 1] <= ydcst_rtt and ztp1[jk - 1 - 1, jl - 1] > ydcst_rtt:
          # Base of melting layer/top of refreezing layer so
          # store rain/snow fraction for precip type diagnosis
          # If mostly rain, then supercooled rain slow to freeze
          # otherwise faster to freeze (snow or ice pellets)
          zqpretot[jl - 1] = max(zqx[ncldqs - 1, jk - 1, jl - 1] + zqx[ncldqr - 1, jk - 1, jl - 1], zepsec)
          prainfrac_toprfz[jl - 1] = zqx[ncldqr - 1, jk - 1, jl - 1] / zqpretot[jl - 1]
          if prainfrac_toprfz[jl - 1] > 0.8:
            llrainliq[jl - 1] = True
          else:
            llrainliq[jl - 1] = False
        
        # If temperature less than zero
        if ztp1[jk - 1, jl - 1] < ydcst_rtt:
          
          if prainfrac_toprfz[jl - 1] > 0.8:
            
            # Majority of raindrops completely melted
            # Refreezing is by slow heterogeneous freezing
            
            # Slope of rain particle size distribution
            zlambda = (yrecldp_rcl_fac1 / (zrho[jl - 1]*zqx[ncldqr - 1, jk - 1, jl - 1]))**yrecldp_rcl_fac2
            
            # Calculate freezing rate based on Bigg(1953) and Wisner(1972)
            ztemp = yrecldp_rcl_fzrab*(ztp1[jk - 1, jl - 1] - ydcst_rtt)
            zfrz = ptsphy*(yrecldp_rcl_const5r / zrho[jl - 1])*(np.exp(ztemp) - 1.)*zlambda**yrecldp_rcl_const6r
            zfrzmax[jl - 1] = max(zfrz, 0.0)
            
          else:
            
            # Majority of raindrops only partially melted
            # Refreeze with a shorter timescale (reverse of melting...for now)
            
            zcons1 = abs(ptsphy*(1.0 + 0.5*(ydcst_rtt - ztp1[jk - 1, jl - 1])) / yrecldp_rtaumel)
            zfrzmax[jl - 1] = max((ydcst_rtt - ztp1[jk - 1, jl - 1])*zcons1*zrldcp, 0.0)
            
          
          if zfrzmax[jl - 1] > zepsec:
            zfrz = min(zqx[ncldqr - 1, jk - 1, jl - 1], zfrzmax[jl - 1])
            zsolqa[ncldqr - 1, ncldqs - 1, jl - 1] = zsolqa[ncldqr - 1, ncldqs - 1, jl - 1] + zfrz
            zsolqa[ncldqs - 1, ncldqr - 1, jl - 1] = zsolqa[ncldqs - 1, ncldqr - 1, jl - 1] - zfrz
        
      
    
    #----------------------------------------------------------------------
    # 4.4c  FREEZING of LIQUID
    #----------------------------------------------------------------------
    for jl in range(kidia, kfdia + 1):
      # not implicit yet...
      zfrzmax[jl - 1] = max((yrecldp_rthomo - ztp1[jk - 1, jl - 1])*zrldcp, 0.0)
    
    jm = ncldql
    jn = imelt[jm - 1]
    for jl in range(kidia, kfdia + 1):
      if zfrzmax[jl - 1] > zepsec and zqxfg[jm - 1, jl - 1] > zepsec:
        zfrz = min(zqxfg[jm - 1, jl - 1], zfrzmax[jl - 1])
        zsolqa[jm - 1, jn - 1, jl - 1] = zsolqa[jm - 1, jn - 1, jl - 1] + zfrz
        zsolqa[jn - 1, jm - 1, jl - 1] = zsolqa[jn - 1, jm - 1, jl - 1] - zfrz
    
    #----------------------------------------------------------------------
    # 4.5   EVAPORATION OF RAIN/SNOW
    #----------------------------------------------------------------------
    
    #----------------------------------------
    # Rain evaporation scheme from Sundquist
    #----------------------------------------
    if ievaprain == 1:
      
      # Rain
      
      for jl in range(kidia, kfdia + 1):
        
        zzrh = yrecldp_rprecrhmax + (1.0 - yrecldp_rprecrhmax)*zcovpmax[jl - 1] / max(zepsec, 1.0 - za[jk - 1, jl - 1])
        zzrh = min(max(zzrh, yrecldp_rprecrhmax), 1.0)
        
        zqe = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1]*zqsliq[jk - 1, jl - 1]) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
        #---------------------------------------------
        # humidity in moistest ZCOVPCLR part of domain
        #---------------------------------------------
        zqe = max(0.0, min(zqe, zqsliq[jk - 1, jl - 1]))
        llo1 = zcovpclr[jl - 1] > zepsec and zqxfg[ncldqr - 1, jl - 1] > zepsec and zqe < zzrh*zqsliq[jk - 1, jl - 1]
        
        if llo1:
          # note: zpreclr is a rain flux
          zpreclr = zqxfg[ncldqr - 1, jl - 1]*zcovpclr[jl - 1] / (max(abs(zcovptot[jl - 1]*zdtgdp[jl - 1]), zepsilon)*np.sign(zcovptot[jl - 1]*zdtgdp[jl - 1]))
          
          #--------------------------------------
          # actual microphysics formula in zbeta
          #--------------------------------------
          
          zbeta1 = np.sqrt(pap[jk - 1, jl - 1] / paph[klev + 1 - 1, jl - 1]) / yrecldp_rvrfactor*zpreclr / max(zcovpclr[jl - 1], zepsec)
          
          zbeta = ydcst_rg*yrecldp_rpecons*0.5*zbeta1**0.5777
          
          zdenom = 1.0 + zbeta*ptsphy*zcorqsliq[jl - 1]
          zdpr = zcovpclr[jl - 1]*zbeta*(zqsliq[jk - 1, jl - 1] - zqe) / zdenom*zdp[jl - 1]*zrg_r
          zdpevap = zdpr*zdtgdp[jl - 1]
          
          #---------------------------------------------------------
          # add evaporation term to explicit sink.
          # this has to be explicit since if treated in the implicit
          # term evaporation can not reduce rain to zero and model
          # produces small amounts of rainfall everywhere.
          #---------------------------------------------------------
          
          # Evaporate rain
          zevap = min(zdpevap, zqxfg[ncldqr - 1, jl - 1])
          
          zsolqa[ncldqr - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqr - 1, ncldqv - 1, jl - 1] + zevap
          zsolqa[ncldqv - 1, ncldqr - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqr - 1, jl - 1] - zevap
          
          #-------------------------------------------------------------
          # Reduce the total precip coverage proportional to evaporation
          # to mimic the previous scheme which had a diagnostic
          # 2-flux treatment, abandoned due to the new prognostic precip
          #-------------------------------------------------------------
          zcovptot[jl - 1] = max(yrecldp_rcovpmin, zcovptot[jl - 1] - max(0.0, (zcovptot[jl - 1] - za[jk - 1, jl - 1])*zevap / zqxfg[ncldqr - 1, jl - 1]))
          
          # Update fg field
          zqxfg[ncldqr - 1, jl - 1] = zqxfg[ncldqr - 1, jl - 1] - zevap
          
      
      
      #---------------------------------------------------------
      # Rain evaporation scheme based on Abel and Boutle (2013)
      #---------------------------------------------------------
    elif ievaprain == 2:
      
      for jl in range(kidia, kfdia + 1):
        
        #-----------------------------------------------------------------------
        # Calculate relative humidity limit for rain evaporation
        # to avoid cloud formation and saturation of the grid box
        #-----------------------------------------------------------------------
        # Limit RH for rain evaporation dependent on precipitation fraction
        zzrh = yrecldp_rprecrhmax + (1.0 - yrecldp_rprecrhmax)*zcovpmax[jl - 1] / max(zepsec, 1.0 - za[jk - 1, jl - 1])
        zzrh = min(max(zzrh, yrecldp_rprecrhmax), 1.0)
        
        # Critical relative humidity
        #ZRHC=RAMID
        #ZSIGK=PAP(JL,JK)/PAPH(JL,KLEV+1)
        # Increase RHcrit to 1.0 towards the surface (eta>0.8)
        #IF(ZSIGK > 0.8_JPRB) THEN
        #  ZRHC=RAMID+(1.0_JPRB-RAMID)*((ZSIGK-0.8_JPRB)/0.2_JPRB)**2
        #ENDIF
        #ZZRH = MIN(ZRHC,ZZRH)
        
        # Further limit RH for rain evaporation to 80% (RHcrit in free troposphere)
        zzrh = min(0.8, zzrh)
        
        zqe = max(0.0, min(zqx[ncldqv - 1, jk - 1, jl - 1], zqsliq[jk - 1, jl - 1]))
        
        llo1 = zcovpclr[jl - 1] > zepsec and zqxfg[ncldqr - 1, jl - 1] > zepsec and zqe < zzrh*zqsliq[jk - 1, jl - 1]
        
        if llo1:
          
          #-------------------------------------------
          # Abel and Boutle (2012) evaporation
          #-------------------------------------------
          # Calculate local precipitation (kg/kg)
          zpreclr = zqxfg[ncldqr - 1, jl - 1] / zcovptot[jl - 1]
          
          # Fallspeed air density correction
          zfallcorr = (yrecldp_rdensref / zrho[jl - 1])**0.4
          
          # Saturation vapour pressure with respect to liquid phase
          zesatliq = ydcst_rv / ydcst_rd*foeeliq(ztp1[jk - 1, jl - 1])
          
          # Slope of particle size distribution
          zlambda = (yrecldp_rcl_fac1 / (zrho[jl - 1]*zpreclr))**yrecldp_rcl_fac2            # ZPRECLR=kg/kg
          
          zevap_denom = yrecldp_rcl_cdenom1*zesatliq - yrecldp_rcl_cdenom2*ztp1[jk - 1, jl - 1]*zesatliq + yrecldp_rcl_cdenom3*ztp1[jk - 1, jl - 1]**3.*pap[jk - 1, jl - 1]
          
          # Temperature dependent conductivity
          zcorr2 = (ztp1[jk - 1, jl - 1] / 273.)**1.5*393. / (ztp1[jk - 1, jl - 1] + 120.)
          zka = yrecldp_rcl_ka273*zcorr2
          
          zsubsat = max(zzrh*zqsliq[jk - 1, jl - 1] - zqe, 0.0)
          
          zbeta = (0.5 / zqsliq[jk - 1, jl - 1])*ztp1[jk - 1, jl - 1]**2.*zesatliq*yrecldp_rcl_const1r*(zcorr2 / zevap_denom)*(0.78 / (zlambda**yrecldp_rcl_const4r) + yrecldp_rcl_const2r*(zrho[jl - 1]*zfallcorr)**0.5 / (zcorr2**0.5*zlambda**yrecldp_rcl_const3r))
          
          zdenom = 1.0 + zbeta*ptsphy            #*ZCORQSLIQ(JL)
          zdpevap = zcovpclr[jl - 1]*zbeta*ptsphy*zsubsat / zdenom
          
          #---------------------------------------------------------
          # Add evaporation term to explicit sink.
          # this has to be explicit since if treated in the implicit
          # term evaporation can not reduce rain to zero and model
          # produces small amounts of rainfall everywhere.
          #---------------------------------------------------------
          
          # Limit rain evaporation
          zevap = min(zdpevap, zqxfg[ncldqr - 1, jl - 1])
          
          zsolqa[ncldqr - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqr - 1, ncldqv - 1, jl - 1] + zevap
          zsolqa[ncldqv - 1, ncldqr - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqr - 1, jl - 1] - zevap
          
          #-------------------------------------------------------------
          # Reduce the total precip coverage proportional to evaporation
          # to mimic the previous scheme which had a diagnostic
          # 2-flux treatment, abandoned due to the new prognostic precip
          #-------------------------------------------------------------
          zcovptot[jl - 1] = max(yrecldp_rcovpmin, zcovptot[jl - 1] - max(0.0, (zcovptot[jl - 1] - za[jk - 1, jl - 1])*zevap / zqxfg[ncldqr - 1, jl - 1]))
          
          # Update fg field
          zqxfg[ncldqr - 1, jl - 1] = zqxfg[ncldqr - 1, jl - 1] - zevap
          
      
    # on IEVAPRAIN
    
    #----------------------------------------------------------------------
    # 4.5   EVAPORATION OF SNOW
    #----------------------------------------------------------------------
    # Snow
    if ievapsnow == 1:
      
      for jl in range(kidia, kfdia + 1):
        zzrh = yrecldp_rprecrhmax + (1.0 - yrecldp_rprecrhmax)*zcovpmax[jl - 1] / max(zepsec, 1.0 - za[jk - 1, jl - 1])
        zzrh = min(max(zzrh, yrecldp_rprecrhmax), 1.0)
        zqe = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1]*zqsice[jk - 1, jl - 1]) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
        
        #---------------------------------------------
        # humidity in moistest ZCOVPCLR part of domain
        #---------------------------------------------
        zqe = max(0.0, min(zqe, zqsice[jk - 1, jl - 1]))
        llo1 = zcovpclr[jl - 1] > zepsec and zqxfg[ncldqs - 1, jl - 1] > zepsec and zqe < zzrh*zqsice[jk - 1, jl - 1]
        
        if llo1:
          # note: zpreclr is a rain flux a
          zpreclr = zqxfg[ncldqs - 1, jl - 1]*zcovpclr[jl - 1] / (max(abs(zcovptot[jl - 1]*zdtgdp[jl - 1]), zepsilon)*np.sign(zcovptot[jl - 1]*zdtgdp[jl - 1]))
          
          #--------------------------------------
          # actual microphysics formula in zbeta
          #--------------------------------------
          
          zbeta1 = np.sqrt(pap[jk - 1, jl - 1] / paph[klev + 1 - 1, jl - 1]) / yrecldp_rvrfactor*zpreclr / max(zcovpclr[jl - 1], zepsec)
          
          zbeta = ydcst_rg*yrecldp_rpecons*zbeta1**0.5777
          
          zdenom = 1.0 + zbeta*ptsphy*zcorqsice[jl - 1]
          zdpr = zcovpclr[jl - 1]*zbeta*(zqsice[jk - 1, jl - 1] - zqe) / zdenom*zdp[jl - 1]*zrg_r
          zdpevap = zdpr*zdtgdp[jl - 1]
          
          #---------------------------------------------------------
          # add evaporation term to explicit sink.
          # this has to be explicit since if treated in the implicit
          # term evaporation can not reduce snow to zero and model
          # produces small amounts of snowfall everywhere.
          #---------------------------------------------------------
          
          # Evaporate snow
          zevap = min(zdpevap, zqxfg[ncldqs - 1, jl - 1])
          
          zsolqa[ncldqs - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqs - 1, ncldqv - 1, jl - 1] + zevap
          zsolqa[ncldqv - 1, ncldqs - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqs - 1, jl - 1] - zevap
          
          #-------------------------------------------------------------
          # Reduce the total precip coverage proportional to evaporation
          # to mimic the previous scheme which had a diagnostic
          # 2-flux treatment, abandoned due to the new prognostic precip
          #-------------------------------------------------------------
          zcovptot[jl - 1] = max(yrecldp_rcovpmin, zcovptot[jl - 1] - max(0.0, (zcovptot[jl - 1] - za[jk - 1, jl - 1])*zevap / zqxfg[ncldqs - 1, jl - 1]))
          
          #Update first guess field
          zqxfg[ncldqs - 1, jl - 1] = zqxfg[ncldqs - 1, jl - 1] - zevap
          
      #---------------------------------------------------------
    elif ievapsnow == 2:
      
      
      for jl in range(kidia, kfdia + 1):
        
        #-----------------------------------------------------------------------
        # Calculate relative humidity limit for snow evaporation
        #-----------------------------------------------------------------------
        zzrh = yrecldp_rprecrhmax + (1.0 - yrecldp_rprecrhmax)*zcovpmax[jl - 1] / max(zepsec, 1.0 - za[jk - 1, jl - 1])
        zzrh = min(max(zzrh, yrecldp_rprecrhmax), 1.0)
        zqe = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1]*zqsice[jk - 1, jl - 1]) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
        
        #---------------------------------------------
        # humidity in moistest ZCOVPCLR part of domain
        #---------------------------------------------
        zqe = max(0.0, min(zqe, zqsice[jk - 1, jl - 1]))
        llo1 = zcovpclr[jl - 1] > zepsec and zqx[ncldqs - 1, jk - 1, jl - 1] > zepsec and zqe < zzrh*zqsice[jk - 1, jl - 1]
        
        if llo1:
          
          # Calculate local precipitation (kg/kg)
          zpreclr = zqx[ncldqs - 1, jk - 1, jl - 1] / zcovptot[jl - 1]
          zvpice = foeeice(ztp1[jk - 1, jl - 1])*ydcst_rv / ydcst_rd
          
          # Particle size distribution
          # ZTCG increases Ni with colder temperatures - essentially a
          # Fletcher or Meyers scheme?
          ztcg = 1.0            #v1 EXP(RCL_X3I*(273.15_JPRB-ZTP1(JL,JK))/8.18_JPRB)
          # ZFACX1I modification is based on Andrew Barrett's results
          zfacx1s = 1.0            #v1 (ZICE0/1.E-5_JPRB)**0.627_JPRB
          
          zaplusb = yrecldp_rcl_apb1*zvpice - yrecldp_rcl_apb2*zvpice*ztp1[jk - 1, jl - 1] + pap[jk - 1, jl - 1]*yrecldp_rcl_apb3*ztp1[jk - 1, jl - 1]**3
          zcorrfac = (1.0 / zrho[jl - 1])**0.5
          zcorrfac2 = ((ztp1[jk - 1, jl - 1] / 273.0)**1.5)*(393.0 / (ztp1[jk - 1, jl - 1] + 120.0))
          
          zpr02 = zrho[jl - 1]*zpreclr*yrecldp_rcl_const1s / (ztcg*zfacx1s)
          
          zterm1 = (zqsice[jk - 1, jl - 1] - zqe)*ztp1[jk - 1, jl - 1]**2*zvpice*zcorrfac2*ztcg*yrecldp_rcl_const2s*zfacx1s / (zrho[jl - 1]*zaplusb*zqsice[jk - 1, jl - 1])
          zterm2 = 0.65*yrecldp_rcl_const6s*zpr02**yrecldp_rcl_const4s + yrecldp_rcl_const3s*zcorrfac**0.5*zrho[jl - 1]**0.5*zpr02**yrecldp_rcl_const5s / zcorrfac2**0.5
          
          zdpevap = max(zcovpclr[jl - 1]*zterm1*zterm2*ptsphy, 0.0)
          
          #--------------------------------------------------------------------
          # Limit evaporation to snow amount
          #--------------------------------------------------------------------
          zevap = min(zdpevap, zevaplimice[jl - 1])
          zevap = min(zevap, zqx[ncldqs - 1, jk - 1, jl - 1])
          
          
          zsolqa[ncldqs - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqs - 1, ncldqv - 1, jl - 1] + zevap
          zsolqa[ncldqv - 1, ncldqs - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqs - 1, jl - 1] - zevap
          
          #-------------------------------------------------------------
          # Reduce the total precip coverage proportional to evaporation
          # to mimic the previous scheme which had a diagnostic
          # 2-flux treatment, abandoned due to the new prognostic precip
          #-------------------------------------------------------------
          zcovptot[jl - 1] = max(yrecldp_rcovpmin, zcovptot[jl - 1] - max(0.0, (zcovptot[jl - 1] - za[jk - 1, jl - 1])*zevap / zqx[ncldqs - 1, jk - 1, jl - 1]))
          
          #Update first guess field
          zqxfg[ncldqs - 1, jl - 1] = zqxfg[ncldqs - 1, jl - 1] - zevap
          
      
    # on IEVAPSNOW
    
    #--------------------------------------
    # Evaporate small precipitation amounts
    #--------------------------------------
    for jm in range(1, nclv + 1):
      if llfall[jm - 1]:
        for jl in range(kidia, kfdia + 1):
          if zqxfg[jm - 1, jl - 1] < yrecldp_rlmin:
            zsolqa[jm - 1, ncldqv - 1, jl - 1] = zsolqa[jm - 1, ncldqv - 1, jl - 1] + zqxfg[jm - 1, jl - 1]
            zsolqa[ncldqv - 1, jm - 1, jl - 1] = zsolqa[ncldqv - 1, jm - 1, jl - 1] - zqxfg[jm - 1, jl - 1]
    
    #######################################################################
    #            5.0  *** SOLVERS FOR A AND L ***
    # now use an implicit solution rather than exact solution
    # solver is forward in time, upstream difference for advection
    #######################################################################
    
    #---------------------------
    # 5.1 solver for cloud cover
    #---------------------------
    for jl in range(kidia, kfdia + 1):
      zanew = (za[jk - 1, jl - 1] + zsolac[jl - 1]) / (1.0 + zsolab[jl - 1])
      zanew = min(zanew, 1.0)
      if zanew < yrecldp_ramin:
        zanew = 0.0
      zda[jl - 1] = zanew - zaorig[jk - 1, jl - 1]
      #---------------------------------
      # variables needed for next level
      #---------------------------------
      zanewm1[jl - 1] = zanew
    
    #--------------------------------
    # 5.2 solver for the microphysics
    #--------------------------------
    
    #--------------------------------------------------------------
    # Truncate explicit sinks to avoid negatives
    # Note: Species are treated in the order in which they run out
    # since the clipping will alter the balance for the other vars
    #--------------------------------------------------------------
    
    for jm in range(1, nclv + 1):
      for jn in range(1, nclv + 1):
        for jl in range(kidia, kfdia + 1):
          llindex3[jm - 1, jn - 1, jl - 1] = False
      for jl in range(kidia, kfdia + 1):
        zsinksum[jm - 1, jl - 1] = 0.0
    
    #----------------------------
    # collect sink terms and mark
    #----------------------------
    for jm in range(1, nclv + 1):
      for jn in range(1, nclv + 1):
        for jl in range(kidia, kfdia + 1):
          zsinksum[jm - 1, jl - 1] = zsinksum[jm - 1, jl - 1] - zsolqa[jn - 1, jm - 1, jl - 1]            # +ve total is bad
    
    #---------------------------------------
    # calculate overshoot and scaling factor
    #---------------------------------------
    for jm in range(1, nclv + 1):
      for jl in range(kidia, kfdia + 1):
        zmax = max(zqx[jm - 1, jk - 1, jl - 1], zepsec)
        zrat = max(zsinksum[jm - 1, jl - 1], zmax)
        zratio[jm - 1, jl - 1] = zmax / zrat
    
    #--------------------------------------------
    # scale the sink terms, in the correct order,
    # recalculating the scale factor each time
    #--------------------------------------------
    for jm in range(1, nclv + 1):
      for jl in range(kidia, kfdia + 1):
        zsinksum[jm - 1, jl - 1] = 0.0
    
    #----------------
    # recalculate sum
    #----------------
    for jm in range(1, nclv + 1):
      psum_solqa[:] = 0.0
      for jn in range(1, nclv + 1):
        for jl in range(kidia, kfdia + 1):
          psum_solqa[jl - 1] = psum_solqa[jl - 1] + zsolqa[jn - 1, jm - 1, jl - 1]
      for jl in range(kidia, kfdia + 1):
        # ZSINKSUM(JL,JM)=ZSINKSUM(JL,JM)-SUM(ZSOLQA(JL,JM,1:NCLV))
        zsinksum[jm - 1, jl - 1] = zsinksum[jm - 1, jl - 1] - psum_solqa[jl - 1]
      #---------------------------
      # recalculate scaling factor
      #---------------------------
      for jl in range(kidia, kfdia + 1):
        zmm = max(zqx[jm - 1, jk - 1, jl - 1], zepsec)
        zrr = max(zsinksum[jm - 1, jl - 1], zmm)
        zratio[jm - 1, jl - 1] = zmm / zrr
      #------
      # scale
      #------
      for jl in range(kidia, kfdia + 1):
        zzratio = zratio[jm - 1, jl - 1]
        #DIR$ IVDEP
        #DIR$ PREFERVECTOR
        for jn in range(1, nclv + 1):
          if zsolqa[jn - 1, jm - 1, jl - 1] < 0.0:
            zsolqa[jn - 1, jm - 1, jl - 1] = zsolqa[jn - 1, jm - 1, jl - 1]*zzratio
            zsolqa[jm - 1, jn - 1, jl - 1] = zsolqa[jm - 1, jn - 1, jl - 1]*zzratio
    
    #--------------------------------------------------------------
    # 5.2.2 Solver
    #------------------------
    
    #------------------------
    # set the LHS of equation
    #------------------------
    for jm in range(1, nclv + 1):
      for jn in range(1, nclv + 1):
        #----------------------------------------------
        # diagonals: microphysical sink terms+transport
        #----------------------------------------------
        if jn == jm:
          for jl in range(kidia, kfdia + 1):
            zqlhs[jm - 1, jn - 1, jl - 1] = 1.0 + zfallsink[jm - 1, jl - 1]
            for jo in range(1, nclv + 1):
              zqlhs[jm - 1, jn - 1, jl - 1] = zqlhs[jm - 1, jn - 1, jl - 1] + zsolqb[jn - 1, jo - 1, jl - 1]
          #------------------------------------------
          # non-diagonals: microphysical source terms
          #------------------------------------------
        else:
          for jl in range(kidia, kfdia + 1):
            zqlhs[jm - 1, jn - 1, jl - 1] = -zsolqb[jm - 1, jn - 1, jl - 1]              # here is the delta T - missing from doc.
    
    #------------------------
    # set the RHS of equation
    #------------------------
    for jm in range(1, nclv + 1):
      for jl in range(kidia, kfdia + 1):
        #---------------------------------
        # sum the explicit source and sink
        #---------------------------------
        zexplicit = 0.0
        for jn in range(1, nclv + 1):
          zexplicit = zexplicit + zsolqa[jn - 1, jm - 1, jl - 1]            # sum over middle index
        zqxn[jm - 1, jl - 1] = zqx[jm - 1, jk - 1, jl - 1] + zexplicit
    
    #-----------------------------------
    # *** solve by LU decomposition: ***
    #-----------------------------------
    
    # Note: This fast way of solving NCLVxNCLV system
    #       assumes a good behaviour (i.e. non-zero diagonal
    #       terms with comparable orders) of the matrix stored
    #       in ZQLHS. For the moment this is the case but
    #       be aware to preserve it when doing eventual
    #       modifications.
    
    # Non pivoting recursive factorization
    for jn in range(1, nclv - 1 + 1):
      # number of steps
      for jm in range(jn + 1, nclv + 1):
        # row index
        for jl in range(kidia, kfdia + 1):
          zqlhs[jn - 1, jm - 1, jl - 1] = zqlhs[jn - 1, jm - 1, jl - 1] / zqlhs[jn - 1, jn - 1, jl - 1]
        for ik in range(jn + 1, nclv + 1):
          # column index
          for jl in range(kidia, kfdia + 1):
            zqlhs[ik - 1, jm - 1, jl - 1] = zqlhs[ik - 1, jm - 1, jl - 1] - zqlhs[jn - 1, jm - 1, jl - 1]*zqlhs[ik - 1, jn - 1, jl - 1]
    
    # Backsubstitution
    #  step 1
    for jn in range(2, nclv + 1):
      for jm in range(1, jn - 1 + 1):
        for jl in range(kidia, kfdia + 1):
          zqxn[jn - 1, jl - 1] = zqxn[jn - 1, jl - 1] - zqlhs[jm - 1, jn - 1, jl - 1]*zqxn[jm - 1, jl - 1]
    #  step 2
    for jl in range(kidia, kfdia + 1):
      zqxn[nclv - 1, jl - 1] = zqxn[nclv - 1, jl - 1] / zqlhs[nclv - 1, nclv - 1, jl - 1]
    for jn in range(nclv - 1, 1 + -1, -1):
      for jm in range(jn + 1, nclv + 1):
        for jl in range(kidia, kfdia + 1):
          zqxn[jn - 1, jl - 1] = zqxn[jn - 1, jl - 1] - zqlhs[jm - 1, jn - 1, jl - 1]*zqxn[jm - 1, jl - 1]
      for jl in range(kidia, kfdia + 1):
        zqxn[jn - 1, jl - 1] = zqxn[jn - 1, jl - 1] / zqlhs[jn - 1, jn - 1, jl - 1]
    
    # Ensure no small values (including negatives) remain in cloud variables nor
    # precipitation rates.
    # Evaporate l,i,r,s to water vapour. Latent heating taken into account below
    for jn in range(1, nclv - 1 + 1):
      for jl in range(kidia, kfdia + 1):
        if zqxn[jn - 1, jl - 1] < zepsec:
          zqxn[ncldqv - 1, jl - 1] = zqxn[ncldqv - 1, jl - 1] + zqxn[jn - 1, jl - 1]
          zqxn[jn - 1, jl - 1] = 0.0
    
    #--------------------------------
    # variables needed for next level
    #--------------------------------
    for jm in range(1, nclv + 1):
      for jl in range(kidia, kfdia + 1):
        zqxnm1[jm - 1, jl - 1] = zqxn[jm - 1, jl - 1]
        zqxn2d[jm - 1, jk - 1, jl - 1] = zqxn[jm - 1, jl - 1]
    
    #------------------------------------------------------------------------
    # 5.3 Precipitation/sedimentation fluxes to next level
    #     diagnostic precipitation fluxes
    #     It is this scaled flux that must be used for source to next layer
    #------------------------------------------------------------------------
    
    for jm in range(1, nclv + 1):
      for jl in range(kidia, kfdia + 1):
        zpfplsx[jm - 1, jk + 1 - 1, jl - 1] = zfallsink[jm - 1, jl - 1]*zqxn[jm - 1, jl - 1]*zrdtgdp[jl - 1]
    
    # Ensure precipitation fraction is zero if no precipitation
    for jl in range(kidia, kfdia + 1):
      zqpretot[jl - 1] = zpfplsx[ncldqs - 1, jk + 1 - 1, jl - 1] + zpfplsx[ncldqr - 1, jk + 1 - 1, jl - 1]
    for jl in range(kidia, kfdia + 1):
      if zqpretot[jl - 1] < zepsec:
        zcovptot[jl - 1] = 0.0
    
    #######################################################################
    #              6  *** UPDATE TENDANCIES ***
    #######################################################################
    
    #--------------------------------
    # 6.1 Temperature and CLV budgets
    #--------------------------------
    
    for jm in range(1, nclv - 1 + 1):
      for jl in range(kidia, kfdia + 1):
        
        # calculate fluxes in and out of box for conservation of TL
        zfluxq[jm - 1, jl - 1] = zpsupsatsrce[jm - 1, jl - 1] + zconvsrce[jm - 1, jl - 1] + zfallsrce[jm - 1, jl - 1] - (zfallsink[jm - 1, jl - 1] + zconvsink[jm - 1, jl - 1])*zqxn[jm - 1, jl - 1]
      
      if iphase[jm - 1] == 1:
        for jl in range(kidia, kfdia + 1):
          tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] + ydthf_ralvdcp*(zqxn[jm - 1, jl - 1] - zqx[jm - 1, jk - 1, jl - 1] - zfluxq[jm - 1, jl - 1])*zqtmst
      
      if iphase[jm - 1] == 2:
        for jl in range(kidia, kfdia + 1):
          tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] + ydthf_ralsdcp*(zqxn[jm - 1, jl - 1] - zqx[jm - 1, jk - 1, jl - 1] - zfluxq[jm - 1, jl - 1])*zqtmst
      
      #----------------------------------------------------------------------
      # New prognostic tendencies - ice,liquid rain,snow
      # Note: CLV arrays use PCLV in calculation of tendency while humidity
      #       uses ZQX. This is due to clipping at start of cloudsc which
      #       include the tendency already in TENDENCY_LOC_T and TENDENCY_LOC_q. ZQX was reset
      #----------------------------------------------------------------------
      for jl in range(kidia, kfdia + 1):
        tendency_loc_cld[jm - 1, jk - 1, jl - 1] = tendency_loc_cld[jm - 1, jk - 1, jl - 1] + (zqxn[jm - 1, jl - 1] - zqx0[jm - 1, jk - 1, jl - 1])*zqtmst
      
    
    for jl in range(kidia, kfdia + 1):
      #----------------------
      # 6.2 Humidity budget
      #----------------------
      tendency_loc_q[jk - 1, jl - 1] = tendency_loc_q[jk - 1, jl - 1] + (zqxn[ncldqv - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1])*zqtmst
      
      #-------------------
      # 6.3 cloud cover
      #-----------------------
      tendency_loc_a[jk - 1, jl - 1] = tendency_loc_a[jk - 1, jl - 1] + zda[jl - 1]*zqtmst
    
    #--------------------------------------------------
    # Copy precipitation fraction into output variable
    #-------------------------------------------------
    for jl in range(kidia, kfdia + 1):
      pcovptot[jk - 1, jl - 1] = zcovptot[jl - 1]
    
  # on vertical level JK
  #----------------------------------------------------------------------
  #                       END OF VERTICAL LOOP
  #----------------------------------------------------------------------
  
  #######################################################################
  #              8  *** FLUX/DIAGNOSTICS COMPUTATIONS ***
  #######################################################################
  
  #--------------------------------------------------------------------
  # Copy general precip arrays back into PFP arrays for GRIB archiving
  # Add rain and liquid fluxes, ice and snow fluxes
  #--------------------------------------------------------------------
  for jk in range(1, klev + 1 + 1):
    for jl in range(kidia, kfdia + 1):
      pfplsl[jk - 1, jl - 1] = zpfplsx[ncldqr - 1, jk - 1, jl - 1] + zpfplsx[ncldql - 1, jk - 1, jl - 1]
      pfplsn[jk - 1, jl - 1] = zpfplsx[ncldqs - 1, jk - 1, jl - 1] + zpfplsx[ncldqi - 1, jk - 1, jl - 1]
  
  #--------
  # Fluxes:
  #--------
  for jl in range(kidia, kfdia + 1):
    pfsqlf[1 - 1, jl - 1] = 0.0
    pfsqif[1 - 1, jl - 1] = 0.0
    pfsqrf[1 - 1, jl - 1] = 0.0
    pfsqsf[1 - 1, jl - 1] = 0.0
    pfcqlng[1 - 1, jl - 1] = 0.0
    pfcqnng[1 - 1, jl - 1] = 0.0
    pfcqrng[1 - 1, jl - 1] = 0.0      #rain
    pfcqsng[1 - 1, jl - 1] = 0.0      #snow
    # fluxes due to turbulence
    pfsqltur[1 - 1, jl - 1] = 0.0
    pfsqitur[1 - 1, jl - 1] = 0.0
  
  for jk in range(1, klev + 1):
    for jl in range(kidia, kfdia + 1):
      
      zgdph_r = -zrg_r*(paph[jk + 1 - 1, jl - 1] - paph[jk - 1, jl - 1])*zqtmst
      pfsqlf[jk + 1 - 1, jl - 1] = pfsqlf[jk - 1, jl - 1]
      pfsqif[jk + 1 - 1, jl - 1] = pfsqif[jk - 1, jl - 1]
      pfsqrf[jk + 1 - 1, jl - 1] = pfsqlf[jk - 1, jl - 1]
      pfsqsf[jk + 1 - 1, jl - 1] = pfsqif[jk - 1, jl - 1]
      pfcqlng[jk + 1 - 1, jl - 1] = pfcqlng[jk - 1, jl - 1]
      pfcqnng[jk + 1 - 1, jl - 1] = pfcqnng[jk - 1, jl - 1]
      pfcqrng[jk + 1 - 1, jl - 1] = pfcqlng[jk - 1, jl - 1]
      pfcqsng[jk + 1 - 1, jl - 1] = pfcqnng[jk - 1, jl - 1]
      pfsqltur[jk + 1 - 1, jl - 1] = pfsqltur[jk - 1, jl - 1]
      pfsqitur[jk + 1 - 1, jl - 1] = pfsqitur[jk - 1, jl - 1]
      
      zalfaw = zfoealfa[jk - 1, jl - 1]
      
      # Liquid , LS scheme minus detrainment
      pfsqlf[jk + 1 - 1, jl - 1] = pfsqlf[jk + 1 - 1, jl - 1] + (zqxn2d[ncldql - 1, jk - 1, jl - 1] - zqx0[ncldql - 1, jk - 1, jl - 1] + pvfl[jk - 1, jl - 1]*ptsphy - zalfaw*plude[jk - 1, jl - 1])*zgdph_r
      # liquid, negative numbers
      pfcqlng[jk + 1 - 1, jl - 1] = pfcqlng[jk + 1 - 1, jl - 1] + zlneg[ncldql - 1, jk - 1, jl - 1]*zgdph_r
      
      # liquid, vertical diffusion
      pfsqltur[jk + 1 - 1, jl - 1] = pfsqltur[jk + 1 - 1, jl - 1] + pvfl[jk - 1, jl - 1]*ptsphy*zgdph_r
      
      # Rain, LS scheme
      pfsqrf[jk + 1 - 1, jl - 1] = pfsqrf[jk + 1 - 1, jl - 1] + (zqxn2d[ncldqr - 1, jk - 1, jl - 1] - zqx0[ncldqr - 1, jk - 1, jl - 1])*zgdph_r
      # rain, negative numbers
      pfcqrng[jk + 1 - 1, jl - 1] = pfcqrng[jk + 1 - 1, jl - 1] + zlneg[ncldqr - 1, jk - 1, jl - 1]*zgdph_r
      
      # Ice , LS scheme minus detrainment
      pfsqif[jk + 1 - 1, jl - 1] = pfsqif[jk + 1 - 1, jl - 1] + (zqxn2d[ncldqi - 1, jk - 1, jl - 1] - zqx0[ncldqi - 1, jk - 1, jl - 1] + pvfi[jk - 1, jl - 1]*ptsphy - (1.0 - zalfaw)*plude[jk - 1, jl - 1])*zgdph_r
      # ice, negative numbers
      pfcqnng[jk + 1 - 1, jl - 1] = pfcqnng[jk + 1 - 1, jl - 1] + zlneg[ncldqi - 1, jk - 1, jl - 1]*zgdph_r
      
      # ice, vertical diffusion
      pfsqitur[jk + 1 - 1, jl - 1] = pfsqitur[jk + 1 - 1, jl - 1] + pvfi[jk - 1, jl - 1]*ptsphy*zgdph_r
      
      # snow, LS scheme
      pfsqsf[jk + 1 - 1, jl - 1] = pfsqsf[jk + 1 - 1, jl - 1] + (zqxn2d[ncldqs - 1, jk - 1, jl - 1] - zqx0[ncldqs - 1, jk - 1, jl - 1])*zgdph_r
      # snow, negative numbers
      pfcqsng[jk + 1 - 1, jl - 1] = pfcqsng[jk + 1 - 1, jl - 1] + zlneg[ncldqs - 1, jk - 1, jl - 1]*zgdph_r
  
  #-----------------------------------
  # enthalpy flux due to precipitation
  #-----------------------------------
  for jk in range(1, klev + 1 + 1):
    for jl in range(kidia, kfdia + 1):
      pfhpsl[jk - 1, jl - 1] = -ydcst_rlvtt*pfplsl[jk - 1, jl - 1]
      pfhpsn[jk - 1, jl - 1] = -ydcst_rlstt*pfplsn[jk - 1, jl - 1]
  
  #===============================================================================
  #IF (LHOOK) CALL DR_HOOK('CLOUDSC',1,ZHOOK_HANDLE)
  return 
