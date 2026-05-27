import numpy as np
import dace
from pathlib import Path

from dace.transformation.passes import ScalarToSymbolPromotion
from dace.transformation.layout.split_array import SplitArray

klon = dace.symbol('klon', dtype=dace.int32)
klev = dace.symbol('klev', dtype=dace.int32)
nclv = dace.symbol('nclv', dtype=dace.int32)
ncldql = dace.symbol('ncldql', dtype=dace.int32)
ncldqi = dace.symbol('ncldqi', dtype=dace.int32)
ncldqr = dace.symbol('ncldqr', dtype=dace.int32)
ncldqs = dace.symbol('ncldqs', dtype=dace.int32)
ncldqv = dace.symbol('ncldqv', dtype=dace.int32)


@dace.program
def cloudsc_py(
    kidia: dace.int32,
    kfdia: dace.int32,
    ptsphy: dace.float64,
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
    pvfa: dace.float64[klev, klon],
    pvfl: dace.float64[klev, klon],
    pvfi: dace.float64[klev, klon],
    pdyna: dace.float64[klev, klon],
    pdynl: dace.float64[klev, klon],
    pdyni: dace.float64[klev, klon],
    phrsw: dace.float64[klev, klon],
    phrlw: dace.float64[klev, klon],
    pvervel: dace.float64[klev, klon],
    pap: dace.float64[klev, klon],
    paph: dace.float64[klev + 1, klon],
    plsm: dace.float64[klon],
    ldcum: dace.int32[klon],
    ktype: dace.int32[klon],
    plu: dace.float64[klev, klon],
    plude: dace.float64[klev, klon],
    psnde: dace.float64[klev, klon],
    pmfu: dace.float64[klev, klon],
    pmfd: dace.float64[klev, klon],
    pa: dace.float64[klev, klon],
    pclv: dace.float64[nclv, klev, klon],
    psupsat: dace.float64[klev, klon],
    plcrit_aer: dace.float64[klev, klon],
    picrit_aer: dace.float64[klev, klon],
    pre_ice: dace.float64[klev, klon],
    pccn: dace.float64[klev, klon],
    pnice: dace.float64[klev, klon],
    pcovptot: dace.float64[klev, klon],
    prainfrac_toprfz: dace.float64[klon],
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
    llflag = np.ndarray(shape=(klon,), dtype=np.float64)
    llrainliq = np.ndarray(shape=(klon,), dtype=np.int32)
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
    ztw1 = 1329.31
    ztw2 = 0.0074615
    ztw3 = 85000.0
    ztw4 = 40.637
    ztw5 = 275.0
    zepsilon = 1e-14
    iwarmrain = 2
    ievaprain = 2
    ievapsnow = 1
    idepice = 1
    zqtmst = 1.0 / ptsphy
    zgdcp = ydcst_rg / ydcst_rcpd
    zrdcp = ydcst_rd / ydcst_rcpd
    zcons1a = ydcst_rcpd / (ydcst_rlmlt * ydcst_rg * yrecldp_rtaumel)
    zepsec = 1e-14
    zrg_r = 1.0 / ydcst_rg
    zrldcp = 1.0 / (ydthf_ralsdcp - ydthf_ralvdcp)
    iphase[ncldqv - 1] = 0
    iphase[ncldql - 1] = 1
    iphase[ncldqr - 1] = 1
    iphase[ncldqi - 1] = 2
    iphase[ncldqs - 1] = 2
    imelt[ncldqv - 1] = -99
    imelt[ncldql - 1] = ncldqi
    imelt[ncldqr - 1] = ncldqs
    imelt[ncldqi - 1] = ncldqr
    imelt[ncldqs - 1] = ncldqr
    for jk in range(1, klev + 1):
        for jl in range(kidia, kfdia + 1):
            tendency_loc_t[jk - 1, jl - 1] = 0.0
            tendency_loc_q[jk - 1, jl - 1] = 0.0
            tendency_loc_a[jk - 1, jl - 1] = 0.0
    for jm in range(1, nclv - 1 + 1):
        for jk in range(1, klev + 1):
            for jl in range(kidia, kfdia + 1):
                tendency_loc_cld[jm - 1, jk - 1, jl - 1] = 0.0
    for jk in range(1, klev + 1):
        for jl in range(kidia, kfdia + 1):
            pcovptot[jk - 1, jl - 1] = 0.0
            tendency_loc_cld[nclv - 1, jk - 1, jl - 1] = 0.0
    zvqx[ncldqv - 1] = 0.0
    zvqx[ncldql - 1] = 0.0
    zvqx[ncldqi - 1] = yrecldp_rvice
    zvqx[ncldqr - 1] = yrecldp_rvrain
    zvqx[ncldqs - 1] = yrecldp_rvsnow
    for jm in range(1, nclv + 1):
        llfall[jm - 1] = False
    for jm in range(1, nclv + 1):
        if zvqx[jm - 1] > 0.0:
            llfall[jm - 1] = True
    llfall[ncldqi - 1] = False
    for jk in range(1, klev + 1):
        for jl in range(kidia, kfdia + 1):
            ztp1[jk - 1, jl - 1] = pt[jk - 1, jl - 1] + ptsphy * tendency_tmp_t[jk - 1, jl - 1]
            zqx[ncldqv - 1, jk - 1, jl - 1] = pq[jk - 1, jl - 1] + ptsphy * tendency_tmp_q[jk - 1, jl - 1]
            zqx0[ncldqv - 1, jk - 1, jl - 1] = pq[jk - 1, jl - 1] + ptsphy * tendency_tmp_q[jk - 1, jl - 1]
            za[jk - 1, jl - 1] = pa[jk - 1, jl - 1] + ptsphy * tendency_tmp_a[jk - 1, jl - 1]
            zaorig[jk - 1, jl - 1] = pa[jk - 1, jl - 1] + ptsphy * tendency_tmp_a[jk - 1, jl - 1]
    for jm in range(1, nclv - 1 + 1):
        for jk in range(1, klev + 1):
            for jl in range(kidia, kfdia + 1):
                zqx[jm - 1, jk - 1, jl - 1] = pclv[jm - 1, jk - 1, jl - 1] + ptsphy * tendency_tmp_cld[jm - 1, jk - 1, jl - 1]
                zqx0[jm - 1, jk - 1, jl - 1] = pclv[jm - 1, jk - 1, jl - 1] + ptsphy * tendency_tmp_cld[jm - 1, jk - 1, jl - 1]
    for jm in range(1, nclv + 1):
        for jk in range(1, klev + 1 + 1):
            for jl in range(kidia, kfdia + 1):
                zpfplsx[jm - 1, jk - 1, jl - 1] = 0.0
    for jm in range(1, nclv + 1):
        for jk in range(1, klev + 1):
            for jl in range(kidia, kfdia + 1):
                zqxn2d[jm - 1, jk - 1, jl - 1] = 0.0
                zlneg[jm - 1, jk - 1, jl - 1] = 0.0
    for jl in range(kidia, kfdia + 1):
        prainfrac_toprfz[jl - 1] = 0.0
    for jl in range(1, klon + 1):
        llrainliq[jl - 1] = True
    for jk in range(1, klev + 1):
        for jl in range(kidia, kfdia + 1):
            if zqx[ncldql - 1, jk - 1, jl - 1] + zqx[ncldqi - 1, jk - 1, jl - 1] < yrecldp_rlmin or za[jk - 1, jl - 1] < yrecldp_ramin:
                zlneg[ncldql - 1, jk - 1, jl - 1] = zlneg[ncldql - 1, jk - 1, jl - 1] + zqx[ncldql - 1, jk - 1, jl - 1]
                zqadj = zqx[ncldql - 1, jk - 1, jl - 1] * zqtmst
                tendency_loc_q[jk - 1, jl - 1] = tendency_loc_q[jk - 1, jl - 1] + zqadj
                tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] - ydthf_ralvdcp * zqadj
                zqx[ncldqv - 1, jk - 1, jl - 1] = zqx[ncldqv - 1, jk - 1, jl - 1] + zqx[ncldql - 1, jk - 1, jl - 1]
                zqx[ncldql - 1, jk - 1, jl - 1] = 0.0
                zlneg[ncldqi - 1, jk - 1, jl - 1] = zlneg[ncldqi - 1, jk - 1, jl - 1] + zqx[ncldqi - 1, jk - 1, jl - 1]
                zqadj = zqx[ncldqi - 1, jk - 1, jl - 1] * zqtmst
                tendency_loc_q[jk - 1, jl - 1] = tendency_loc_q[jk - 1, jl - 1] + zqadj
                tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] - ydthf_ralsdcp * zqadj
                zqx[ncldqv - 1, jk - 1, jl - 1] = zqx[ncldqv - 1, jk - 1, jl - 1] + zqx[ncldqi - 1, jk - 1, jl - 1]
                zqx[ncldqi - 1, jk - 1, jl - 1] = 0.0
                za[jk - 1, jl - 1] = 0.0
    for jm in range(1, nclv - 1 + 1):
        for jk in range(1, klev + 1):
            for jl in range(kidia, kfdia + 1):
                if zqx[jm - 1, jk - 1, jl - 1] < yrecldp_rlmin:
                    zlneg[jm - 1, jk - 1, jl - 1] = zlneg[jm - 1, jk - 1, jl - 1] + zqx[jm - 1, jk - 1, jl - 1]
                    zqadj = zqx[jm - 1, jk - 1, jl - 1] * zqtmst
                    tendency_loc_q[jk - 1, jl - 1] = tendency_loc_q[jk - 1, jl - 1] + zqadj
                    if iphase[jm - 1] == 1:
                        tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] - ydthf_ralvdcp * zqadj
                    if iphase[jm - 1] == 2:
                        tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] - ydthf_ralsdcp * zqadj
                    zqx[ncldqv - 1, jk - 1, jl - 1] = zqx[ncldqv - 1, jk - 1, jl - 1] + zqx[jm - 1, jk - 1, jl - 1]
                    zqx[jm - 1, jk - 1, jl - 1] = 0.0
    for jk in range(1, klev + 1):
        for jl in range(kidia, kfdia + 1):
            zfoealfa[jk - 1, jl - 1] = min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2)
            zfoeewmt[jk - 1, jl - 1] = min(ydthf_r2es * (min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2) * np.exp(ydthf_r3les * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4les)) + (1.0 - min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2)) * np.exp(ydthf_r3ies * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4ies))) / pap[jk - 1, jl - 1], 0.5)
            zqsmix[jk - 1, jl - 1] = zfoeewmt[jk - 1, jl - 1]
            zqsmix[jk - 1, jl - 1] = zqsmix[jk - 1, jl - 1] / (1.0 - ydcst_retv * zqsmix[jk - 1, jl - 1])
            zalfa = max(0.0, 1.0 * np.sign(ztp1[jk - 1, jl - 1] - ydcst_rtt))
            zfoeew[jk - 1, jl - 1] = min((zalfa * (ydthf_r2es * np.exp(ydthf_r3les * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4les))) + (1.0 - zalfa) * (ydthf_r2es * np.exp(ydthf_r3ies * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4ies)))) / pap[jk - 1, jl - 1], 0.5)
            zfoeew[jk - 1, jl - 1] = min(0.5, zfoeew[jk - 1, jl - 1])
            zqsice[jk - 1, jl - 1] = zfoeew[jk - 1, jl - 1] / (1.0 - ydcst_retv * zfoeew[jk - 1, jl - 1])
            zfoeeliqt[jk - 1, jl - 1] = min(ydthf_r2es * np.exp(ydthf_r3les * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4les)) / pap[jk - 1, jl - 1], 0.5)
            zqsliq[jk - 1, jl - 1] = zfoeeliqt[jk - 1, jl - 1]
            zqsliq[jk - 1, jl - 1] = zqsliq[jk - 1, jl - 1] / (1.0 - ydcst_retv * zqsliq[jk - 1, jl - 1])
    for jk in range(1, klev + 1):
        for jl in range(kidia, kfdia + 1):
            za[jk - 1, jl - 1] = max(0.0, min(1.0, za[jk - 1, jl - 1]))
            zli[jk - 1, jl - 1] = zqx[ncldql - 1, jk - 1, jl - 1] + zqx[ncldqi - 1, jk - 1, jl - 1]
            if zli[jk - 1, jl - 1] > yrecldp_rlmin:
                zliqfrac[jk - 1, jl - 1] = zqx[ncldql - 1, jk - 1, jl - 1] / zli[jk - 1, jl - 1]
                zicefrac[jk - 1, jl - 1] = 1.0 - zliqfrac[jk - 1, jl - 1]
            else:
                zliqfrac[jk - 1, jl - 1] = 0.0
                zicefrac[jk - 1, jl - 1] = 0.0
    for jl in range(kidia, kfdia + 1):
        ztrpaus[jl - 1] = 0.1
        zpaphd[jl - 1] = 1.0 / paph[klev + 1 - 1, jl - 1]
    for jk in range(1, klev - 1 + 1):
        for jl in range(kidia, kfdia + 1):
            zsig = pap[jk - 1, jl - 1] * zpaphd[jl - 1]
            if zsig > 0.1 and zsig < 0.4 and (ztp1[jk - 1, jl - 1] > ztp1[jk + 1 - 1, jl - 1]):
                ztrpaus[jl - 1] = zsig
    for jl in range(kidia, kfdia + 1):
        zanewm1[jl - 1] = 0.0
        zda[jl - 1] = 0.0
        zcovpclr[jl - 1] = 0.0
        zcovpmax[jl - 1] = 0.0
        zcovptot[jl - 1] = 0.0
        zcldtopdist[jl - 1] = 0.0
    for jk in range(yrecldp_ncldtop, klev + 1):
        for jm in range(1, nclv + 1):
            for jl in range(kidia, kfdia + 1):
                zqxfg[jm - 1, jl - 1] = zqx[jm - 1, jk - 1, jl - 1]
        for jl in range(kidia, kfdia + 1):
            zlicld[jl - 1] = 0.0
            zrainaut[jl - 1] = 0.0
            zrainacc[jl - 1] = 0.0
            zsnowaut[jl - 1] = 0.0
            zldefr[jl - 1] = 0.0
            zacust[jl - 1] = 0.0
            zqpretot[jl - 1] = 0.0
            zlfinalsum[jl - 1] = 0.0
            zlcond1[jl - 1] = 0.0
            zlcond2[jl - 1] = 0.0
            zsupsat[jl - 1] = 0.0
            zlevapl[jl - 1] = 0.0
            zlevapi[jl - 1] = 0.0
            zsolab[jl - 1] = 0.0
            zsolac[jl - 1] = 0.0
            zicetot[jl - 1] = 0.0
        for jm in range(1, nclv + 1):
            for jn in range(1, nclv + 1):
                for jl in range(kidia, kfdia + 1):
                    zsolqb[jm - 1, jn - 1, jl - 1] = 0.0
                    zsolqa[jm - 1, jn - 1, jl - 1] = 0.0
        for jm in range(1, nclv + 1):
            for jl in range(kidia, kfdia + 1):
                zfallsrce[jm - 1, jl - 1] = 0.0
                zfallsink[jm - 1, jl - 1] = 0.0
                zconvsrce[jm - 1, jl - 1] = 0.0
                zconvsink[jm - 1, jl - 1] = 0.0
                zpsupsatsrce[jm - 1, jl - 1] = 0.0
                zratio[jm - 1, jl - 1] = 0.0
        for jl in range(kidia, kfdia + 1):
            zdp[jl - 1] = paph[jk + 1 - 1, jl - 1] - paph[jk - 1, jl - 1]
            zgdp[jl - 1] = ydcst_rg / zdp[jl - 1]
            zrho[jl - 1] = pap[jk - 1, jl - 1] / (ydcst_rd * ztp1[jk - 1, jl - 1])
            zdtgdp[jl - 1] = ptsphy * zgdp[jl - 1]
            zrdtgdp[jl - 1] = zdp[jl - 1] * (1.0 / (ptsphy * ydcst_rg))
            if jk > 1:
                zdtgdpf[jl - 1] = ptsphy * ydcst_rg / (pap[jk - 1, jl - 1] - pap[jk - 1 - 1, jl - 1])
            zfacw = ydthf_r5les / (ztp1[jk - 1, jl - 1] - ydthf_r4les) ** 2
            zcor = 1.0 / (1.0 - ydcst_retv * zfoeeliqt[jk - 1, jl - 1])
            zdqsliqdt[jl - 1] = zfacw * zcor * zqsliq[jk - 1, jl - 1]
            zcorqsliq[jl - 1] = 1.0 + ydthf_ralvdcp * zdqsliqdt[jl - 1]
            zfaci = ydthf_r5ies / (ztp1[jk - 1, jl - 1] - ydthf_r4ies) ** 2
            zcor = 1.0 / (1.0 - ydcst_retv * zfoeew[jk - 1, jl - 1])
            zdqsicedt[jl - 1] = zfaci * zcor * zqsice[jk - 1, jl - 1]
            zcorqsice[jl - 1] = 1.0 + ydthf_ralsdcp * zdqsicedt[jl - 1]
            zalfaw = zfoealfa[jk - 1, jl - 1]
            zalfawm[jl - 1] = zalfaw
            zfac = zalfaw * zfacw + (1.0 - zalfaw) * zfaci
            zcor = 1.0 / (1.0 - ydcst_retv * zfoeewmt[jk - 1, jl - 1])
            zdqsmixdt[jl - 1] = zfac * zcor * zqsmix[jk - 1, jl - 1]
            zcorqsmix[jl - 1] = 1.0 + (min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2) * ydthf_ralvdcp + (1.0 - min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2)) * ydthf_ralsdcp) * zdqsmixdt[jl - 1]
            zevaplimmix[jl - 1] = max((zqsmix[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1]) / zcorqsmix[jl - 1], 0.0)
            zevaplimliq[jl - 1] = max((zqsliq[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1]) / zcorqsliq[jl - 1], 0.0)
            zevaplimice[jl - 1] = max((zqsice[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1]) / zcorqsice[jl - 1], 0.0)
            ztmpa = 1.0 / max(za[jk - 1, jl - 1], zepsec)
            zliqcld[jl - 1] = zqx[ncldql - 1, jk - 1, jl - 1] * ztmpa
            zicecld[jl - 1] = zqx[ncldqi - 1, jk - 1, jl - 1] * ztmpa
            zlicld[jl - 1] = zliqcld[jl - 1] + zicecld[jl - 1]
        for jl in range(kidia, kfdia + 1):
            if zqx[ncldql - 1, jk - 1, jl - 1] < yrecldp_rlmin:
                zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zqx[ncldql - 1, jk - 1, jl - 1]
                zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = -zqx[ncldql - 1, jk - 1, jl - 1]
            if zqx[ncldqi - 1, jk - 1, jl - 1] < yrecldp_rlmin:
                zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zqx[ncldqi - 1, jk - 1, jl - 1]
                zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = -zqx[ncldqi - 1, jk - 1, jl - 1]
        for jl in range(kidia, kfdia + 1):
            zfokoop[jl - 1] = min(ydthf_rkoop1 - ydthf_rkoop2 * ztp1[jk - 1, jl - 1], ydthf_r2es * np.exp(ydthf_r3les * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4les)) / (ydthf_r2es * np.exp(ydthf_r3ies * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4ies))))
        for jl in range(kidia, kfdia + 1):
            if ztp1[jk - 1, jl - 1] >= ydcst_rtt or yrecldp_nssopt == 0:
                zfac = 1.0
                zfaci = 1.0
            else:
                zfac = za[jk - 1, jl - 1] + zfokoop[jl - 1] * (1.0 - za[jk - 1, jl - 1])
                zfaci = ptsphy / yrecldp_rkooptau
            if za[jk - 1, jl - 1] > 1.0 - yrecldp_ramin:
                zsupsat[jl - 1] = max((zqx[ncldqv - 1, jk - 1, jl - 1] - zfac * zqsice[jk - 1, jl - 1]) / zcorqsice[jl - 1], 0.0)
            else:
                zqp1env = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1] * zqsice[jk - 1, jl - 1]) / max(1.0 - za[jk - 1, jl - 1], zepsilon)
                zsupsat[jl - 1] = max((1.0 - za[jk - 1, jl - 1]) * (zqp1env - zfac * zqsice[jk - 1, jl - 1]) / zcorqsice[jl - 1], 0.0)
            if zsupsat[jl - 1] > zepsec:
                if ztp1[jk - 1, jl - 1] > yrecldp_rthomo:
                    zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = zsolqa[ncldqv - 1, ncldql - 1, jl - 1] + zsupsat[jl - 1]
                    zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zsolqa[ncldql - 1, ncldqv - 1, jl - 1] - zsupsat[jl - 1]
                    zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] + zsupsat[jl - 1]
                else:
                    zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] + zsupsat[jl - 1]
                    zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] - zsupsat[jl - 1]
                    zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zsupsat[jl - 1]
                zsolac[jl - 1] = (1.0 - za[jk - 1, jl - 1]) * zfaci
            if psupsat[jk - 1, jl - 1] > zepsec:
                if ztp1[jk - 1, jl - 1] > yrecldp_rthomo:
                    zsolqa[ncldql - 1, ncldql - 1, jl - 1] = zsolqa[ncldql - 1, ncldql - 1, jl - 1] + psupsat[jk - 1, jl - 1]
                    zpsupsatsrce[ncldql - 1, jl - 1] = psupsat[jk - 1, jl - 1]
                    zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] + psupsat[jk - 1, jl - 1]
                else:
                    zsolqa[ncldqi - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqi - 1, jl - 1] + psupsat[jk - 1, jl - 1]
                    zpsupsatsrce[ncldqi - 1, jl - 1] = psupsat[jk - 1, jl - 1]
                    zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + psupsat[jk - 1, jl - 1]
                zsolac[jl - 1] = (1.0 - za[jk - 1, jl - 1]) * zfaci
        if jk < klev and jk >= yrecldp_ncldtop:
            for jl in range(kidia, kfdia + 1):
                plude[jk - 1, jl - 1] = plude[jk - 1, jl - 1] * zdtgdp[jl - 1]
                if ldcum[jl - 1] and plude[jk - 1, jl - 1] > yrecldp_rlmin and (plu[jk + 1 - 1, jl - 1] > zepsec):
                    zsolac[jl - 1] = zsolac[jl - 1] + plude[jk - 1, jl - 1] / plu[jk + 1 - 1, jl - 1]
                    zalfaw = zfoealfa[jk - 1, jl - 1]
                    zconvsrce[ncldql - 1, jl - 1] = zalfaw * plude[jk - 1, jl - 1]
                    zconvsrce[ncldqi - 1, jl - 1] = (1.0 - zalfaw) * plude[jk - 1, jl - 1]
                    zsolqa[ncldql - 1, ncldql - 1, jl - 1] = zsolqa[ncldql - 1, ncldql - 1, jl - 1] + zconvsrce[ncldql - 1, jl - 1]
                    zsolqa[ncldqi - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqi - 1, jl - 1] + zconvsrce[ncldqi - 1, jl - 1]
                else:
                    plude[jk - 1, jl - 1] = 0.0
                if ldcum[jl - 1]:
                    zsolqa[ncldqs - 1, ncldqs - 1, jl - 1] = zsolqa[ncldqs - 1, ncldqs - 1, jl - 1] + psnde[jk - 1, jl - 1] * zdtgdp[jl - 1]
        if jk > yrecldp_ncldtop:
            for jl in range(kidia, kfdia + 1):
                zmf[jl - 1] = max(0.0, (pmfu[jk - 1, jl - 1] + pmfd[jk - 1, jl - 1]) * zdtgdp[jl - 1])
                zacust[jl - 1] = zmf[jl - 1] * zanewm1[jl - 1]
            for jm in range(1, nclv + 1):
                if not llfall[jm - 1] and iphase[jm - 1] > 0:
                    for jl in range(kidia, kfdia + 1):
                        zlcust[jm - 1, jl - 1] = zmf[jl - 1] * zqxnm1[jm - 1, jl - 1]
                        zconvsrce[jm - 1, jl - 1] = zconvsrce[jm - 1, jl - 1] + zlcust[jm - 1, jl - 1]
            for jl in range(kidia, kfdia + 1):
                zdtdp = zrdcp * 0.5 * (ztp1[jk - 1 - 1, jl - 1] + ztp1[jk - 1, jl - 1]) / paph[jk - 1, jl - 1]
                zdtforc = zdtdp * (pap[jk - 1, jl - 1] - pap[jk - 1 - 1, jl - 1])
                zdqs[jl - 1] = zanewm1[jl - 1] * zdtforc * zdqsmixdt[jl - 1]
            for jm in range(1, nclv + 1):
                if not llfall[jm - 1] and iphase[jm - 1] > 0:
                    for jl in range(kidia, kfdia + 1):
                        zlfinal = max(0.0, zlcust[jm - 1, jl - 1] - zdqs[jl - 1])
                        zevap = min(zlcust[jm - 1, jl - 1] - zlfinal, zevaplimmix[jl - 1])
                        zlfinal = zlcust[jm - 1, jl - 1] - zevap
                        zlfinalsum[jl - 1] = zlfinalsum[jl - 1] + zlfinal
                        zsolqa[jm - 1, jm - 1, jl - 1] = zsolqa[jm - 1, jm - 1, jl - 1] + zlcust[jm - 1, jl - 1]
                        zsolqa[jm - 1, ncldqv - 1, jl - 1] = zsolqa[jm - 1, ncldqv - 1, jl - 1] + zevap
                        zsolqa[ncldqv - 1, jm - 1, jl - 1] = zsolqa[ncldqv - 1, jm - 1, jl - 1] - zevap
            for jl in range(kidia, kfdia + 1):
                if zlfinalsum[jl - 1] < zepsec:
                    zacust[jl - 1] = 0.0
                zsolac[jl - 1] = zsolac[jl - 1] + zacust[jl - 1]
        for jl in range(kidia, kfdia + 1):
            if jk < klev:
                zmfdn = max(0.0, (pmfu[jk + 1 - 1, jl - 1] + pmfd[jk + 1 - 1, jl - 1]) * zdtgdp[jl - 1])
                zsolab[jl - 1] = zsolab[jl - 1] + zmfdn
                zsolqb[ncldql - 1, ncldql - 1, jl - 1] = zsolqb[ncldql - 1, ncldql - 1, jl - 1] + zmfdn
                zsolqb[ncldqi - 1, ncldqi - 1, jl - 1] = zsolqb[ncldqi - 1, ncldqi - 1, jl - 1] + zmfdn
                zconvsink[ncldql - 1, jl - 1] = zmfdn
                zconvsink[ncldqi - 1, jl - 1] = zmfdn
        for jl in range(kidia, kfdia + 1):
            zldifdt[jl - 1] = yrecldp_rcldiff * ptsphy
            if ktype[jl - 1] > 0 and plude[jk - 1, jl - 1] > zepsec:
                zldifdt[jl - 1] = yrecldp_rcldiff_convi * zldifdt[jl - 1]
        for jl in range(kidia, kfdia + 1):
            if zli[jk - 1, jl - 1] > zepsec:
                ze = zldifdt[jl - 1] * max(zqsmix[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1], 0.0)
                zleros = za[jk - 1, jl - 1] * ze
                zleros = min(zleros, zevaplimmix[jl - 1])
                zleros = min(zleros, zli[jk - 1, jl - 1])
                zaeros = zleros / zlicld[jl - 1]
                zsolac[jl - 1] = zsolac[jl - 1] - zaeros
                zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zsolqa[ncldql - 1, ncldqv - 1, jl - 1] + zliqfrac[jk - 1, jl - 1] * zleros
                zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = zsolqa[ncldqv - 1, ncldql - 1, jl - 1] - zliqfrac[jk - 1, jl - 1] * zleros
                zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] + zicefrac[jk - 1, jl - 1] * zleros
                zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] - zicefrac[jk - 1, jl - 1] * zleros
        for jl in range(kidia, kfdia + 1):
            zdtdp = zrdcp * ztp1[jk - 1, jl - 1] / pap[jk - 1, jl - 1]
            zdpmxdt = zdp[jl - 1] * zqtmst
            zmfdn = 0.0
            if jk < klev:
                zmfdn = pmfu[jk + 1 - 1, jl - 1] + pmfd[jk + 1 - 1, jl - 1]
            zwtot = pvervel[jk - 1, jl - 1] + 0.5 * ydcst_rg * (pmfu[jk - 1, jl - 1] + pmfd[jk - 1, jl - 1] + zmfdn)
            zwtot = min(zdpmxdt, max(-zdpmxdt, zwtot))
            zzzdt = phrsw[jk - 1, jl - 1] + phrlw[jk - 1, jl - 1]
            zdtdiab = min(zdpmxdt * zdtdp, max(-zdpmxdt * zdtdp, zzzdt)) * ptsphy + ydthf_ralfdcp * zldefr[jl - 1]
            zdtforc = zdtdp * zwtot * ptsphy + zdtdiab
            zqold[jl - 1] = zqsmix[jk - 1, jl - 1]
            ztold[jl - 1] = ztp1[jk - 1, jl - 1]
            ztp1[jk - 1, jl - 1] = ztp1[jk - 1, jl - 1] + zdtforc
            ztp1[jk - 1, jl - 1] = max(ztp1[jk - 1, jl - 1], 160.0)
            llflag[jl - 1] = True
        for jl in range(kidia, kfdia + 1):
            zqp = 1.0 / pap[jk - 1, jl - 1]
            zqsat = ydthf_r2es * (min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2) * np.exp(ydthf_r3les * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4les)) + (1.0 - min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2)) * np.exp(ydthf_r3ies * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4ies))) * zqp
            zqsat = min(0.5, zqsat)
            zcor = 1.0 / (1.0 - ydcst_retv * zqsat)
            zqsat = zqsat * zcor
            zcond = (zqsmix[jk - 1, jl - 1] - zqsat) / (1.0 + zqsat * zcor * (min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2) * ydthf_r5alvcp * (1.0 / (ztp1[jk - 1, jl - 1] - ydthf_r4les) ** 2) + (1.0 - min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2)) * ydthf_r5alscp * (1.0 / (ztp1[jk - 1, jl - 1] - ydthf_r4ies) ** 2)))
            ztp1[jk - 1, jl - 1] = ztp1[jk - 1, jl - 1] + (min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2) * ydthf_ralvdcp + (1.0 - min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2)) * ydthf_ralsdcp) * zcond
            zqsmix[jk - 1, jl - 1] = zqsmix[jk - 1, jl - 1] - zcond
            zqsat = ydthf_r2es * (min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2) * np.exp(ydthf_r3les * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4les)) + (1.0 - min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2)) * np.exp(ydthf_r3ies * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4ies))) * zqp
            zqsat = min(0.5, zqsat)
            zcor = 1.0 / (1.0 - ydcst_retv * zqsat)
            zqsat = zqsat * zcor
            zcond1 = (zqsmix[jk - 1, jl - 1] - zqsat) / (1.0 + zqsat * zcor * (min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2) * ydthf_r5alvcp * (1.0 / (ztp1[jk - 1, jl - 1] - ydthf_r4les) ** 2) + (1.0 - min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2)) * ydthf_r5alscp * (1.0 / (ztp1[jk - 1, jl - 1] - ydthf_r4ies) ** 2)))
            ztp1[jk - 1, jl - 1] = ztp1[jk - 1, jl - 1] + (min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2) * ydthf_ralvdcp + (1.0 - min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2)) * ydthf_ralsdcp) * zcond1
            zqsmix[jk - 1, jl - 1] = zqsmix[jk - 1, jl - 1] - zcond1
        for jl in range(kidia, kfdia + 1):
            zdqs[jl - 1] = zqsmix[jk - 1, jl - 1] - zqold[jl - 1]
            zqsmix[jk - 1, jl - 1] = zqold[jl - 1]
            ztp1[jk - 1, jl - 1] = ztold[jl - 1]
        for jl in range(kidia, kfdia + 1):
            if zdqs[jl - 1] > 0.0:
                zlevap = za[jk - 1, jl - 1] * min(zdqs[jl - 1], zlicld[jl - 1])
                zlevap = min(zlevap, zevaplimmix[jl - 1])
                zlevap = min(zlevap, max(zqsmix[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1], 0.0))
                zlevapl[jl - 1] = zliqfrac[jk - 1, jl - 1] * zlevap
                zlevapi[jl - 1] = zicefrac[jk - 1, jl - 1] * zlevap
                zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zsolqa[ncldql - 1, ncldqv - 1, jl - 1] + zliqfrac[jk - 1, jl - 1] * zlevap
                zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = zsolqa[ncldqv - 1, ncldql - 1, jl - 1] - zliqfrac[jk - 1, jl - 1] * zlevap
                zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] + zicefrac[jk - 1, jl - 1] * zlevap
                zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] - zicefrac[jk - 1, jl - 1] * zlevap
        for jl in range(kidia, kfdia + 1):
            if za[jk - 1, jl - 1] > zepsec and zdqs[jl - 1] <= -yrecldp_rlmin:
                zlcond1[jl - 1] = max(-zdqs[jl - 1], 0.0)
                if za[jk - 1, jl - 1] > 0.99:
                    zcor = 1.0 / (1.0 - ydcst_retv * zqsmix[jk - 1, jl - 1])
                    zcdmax = (zqx[ncldqv - 1, jk - 1, jl - 1] - zqsmix[jk - 1, jl - 1]) / (1.0 + zcor * zqsmix[jk - 1, jl - 1] * (min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2) * ydthf_r5alvcp * (1.0 / (ztp1[jk - 1, jl - 1] - ydthf_r4les) ** 2) + (1.0 - min(1.0, ((max(ydthf_rtice, min(ydthf_rtwat, ztp1[jk - 1, jl - 1])) - ydthf_rtice) * ydthf_rtwat_rtice_r) ** 2)) * ydthf_r5alscp * (1.0 / (ztp1[jk - 1, jl - 1] - ydthf_r4ies) ** 2)))
                else:
                    zcdmax = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1] * zqsmix[jk - 1, jl - 1]) / za[jk - 1, jl - 1]
                zlcond1[jl - 1] = max(min(zlcond1[jl - 1], zcdmax), 0.0)
                zlcond1[jl - 1] = za[jk - 1, jl - 1] * zlcond1[jl - 1]
                if zlcond1[jl - 1] < yrecldp_rlmin:
                    zlcond1[jl - 1] = 0.0
                if ztp1[jk - 1, jl - 1] > yrecldp_rthomo:
                    zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = zsolqa[ncldqv - 1, ncldql - 1, jl - 1] + zlcond1[jl - 1]
                    zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zsolqa[ncldql - 1, ncldqv - 1, jl - 1] - zlcond1[jl - 1]
                    zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] + zlcond1[jl - 1]
                else:
                    zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] + zlcond1[jl - 1]
                    zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] - zlcond1[jl - 1]
                    zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zlcond1[jl - 1]
        for jl in range(kidia, kfdia + 1):
            if zdqs[jl - 1] <= -yrecldp_rlmin and za[jk - 1, jl - 1] < 1.0 - zepsec:
                zsigk = pap[jk - 1, jl - 1] / paph[klev + 1 - 1, jl - 1]
                if zsigk > 0.8:
                    zrhc = yrecldp_ramid + (1.0 - yrecldp_ramid) * ((zsigk - 0.8) / 0.2) ** 2
                else:
                    zrhc = yrecldp_ramid
                if yrecldp_nssopt == 0:
                    zqe = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1] * zqsice[jk - 1, jl - 1]) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
                    zqe = max(0.0, zqe)
                elif yrecldp_nssopt == 1:
                    zqe = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1] * zqsice[jk - 1, jl - 1]) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
                    zqe = max(0.0, zqe)
                elif yrecldp_nssopt == 2:
                    zqe = zqx[ncldqv - 1, jk - 1, jl - 1]
                elif yrecldp_nssopt == 3:
                    zqe = zqx[ncldqv - 1, jk - 1, jl - 1] + zli[jk - 1, jl - 1]
                if ztp1[jk - 1, jl - 1] >= ydcst_rtt or yrecldp_nssopt == 0:
                    zfac = 1.0
                else:
                    zfac = zfokoop[jl - 1]
                if zqe >= zrhc * zqsice[jk - 1, jl - 1] * zfac and zqe < zqsice[jk - 1, jl - 1] * zfac:
                    zacond = -(1.0 - za[jk - 1, jl - 1]) * zfac * zdqs[jl - 1] / max(2.0 * (zfac * zqsice[jk - 1, jl - 1] - zqe), zepsec)
                    zacond = min(zacond, 1.0 - za[jk - 1, jl - 1])
                    zlcond2[jl - 1] = -zfac * zdqs[jl - 1] * 0.5 * zacond
                    zzdl = 2.0 * (zfac * zqsice[jk - 1, jl - 1] - zqe) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
                    if zfac * zdqs[jl - 1] < -zzdl:
                        zlcondlim = (za[jk - 1, jl - 1] - 1.0) * zfac * zdqs[jl - 1] - zfac * zqsice[jk - 1, jl - 1] + zqx[ncldqv - 1, jk - 1, jl - 1]
                        zlcond2[jl - 1] = min(zlcond2[jl - 1], zlcondlim)
                    zlcond2[jl - 1] = max(zlcond2[jl - 1], 0.0)
                    if zlcond2[jl - 1] < yrecldp_rlmin or 1.0 - za[jk - 1, jl - 1] < zepsec:
                        zlcond2[jl - 1] = 0.0
                        zacond = 0.0
                    if zlcond2[jl - 1] == 0.0:
                        zacond = 0.0
                    zsolac[jl - 1] = zsolac[jl - 1] + zacond
                    if ztp1[jk - 1, jl - 1] > yrecldp_rthomo:
                        zsolqa[ncldqv - 1, ncldql - 1, jl - 1] = zsolqa[ncldqv - 1, ncldql - 1, jl - 1] + zlcond2[jl - 1]
                        zsolqa[ncldql - 1, ncldqv - 1, jl - 1] = zsolqa[ncldql - 1, ncldqv - 1, jl - 1] - zlcond2[jl - 1]
                        zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] + zlcond2[jl - 1]
                    else:
                        zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqi - 1, jl - 1] + zlcond2[jl - 1]
                        zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqi - 1, ncldqv - 1, jl - 1] - zlcond2[jl - 1]
                        zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zlcond2[jl - 1]
        if idepice == 1:
            for jl in range(kidia, kfdia + 1):
                if za[jk - 1 - 1, jl - 1] < yrecldp_rcldtopcf and za[jk - 1, jl - 1] >= yrecldp_rcldtopcf:
                    zcldtopdist[jl - 1] = 0.0
                else:
                    zcldtopdist[jl - 1] = zcldtopdist[jl - 1] + zdp[jl - 1] / (zrho[jl - 1] * ydcst_rg)
                if ztp1[jk - 1, jl - 1] < ydcst_rtt and zqxfg[ncldql - 1, jl - 1] > yrecldp_rlmin:
                    zvpice = ydthf_r2es * np.exp(ydthf_r3ies * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4ies)) * ydcst_rv / ydcst_rd
                    zvpliq = zvpice * zfokoop[jl - 1]
                    zicenuclei[jl - 1] = 1000.0 * np.exp(12.96 * (zvpliq - zvpice) / zvpliq - 0.639)
                    zadd = ydcst_rlstt * (ydcst_rlstt / (ydcst_rv * ztp1[jk - 1, jl - 1]) - 1.0) / (0.024 * ztp1[jk - 1, jl - 1])
                    zbdd = ydcst_rv * ztp1[jk - 1, jl - 1] * pap[jk - 1, jl - 1] / (2.21 * zvpice)
                    zcvds = 7.8 * (zicenuclei[jl - 1] / zrho[jl - 1]) ** 0.666 * (zvpliq - zvpice) / (8.87 * (zadd + zbdd) * zvpice)
                    zice0 = max(zicecld[jl - 1], zicenuclei[jl - 1] * yrecldp_riceinit / zrho[jl - 1])
                    zinew = (0.666 * zcvds * ptsphy + zice0 ** 0.666) ** 1.5
                    zdepos = max(za[jk - 1, jl - 1] * (zinew - zice0), 0.0)
                    zdepos = min(zdepos, zqxfg[ncldql - 1, jl - 1])
                    zinfactor = min(zicenuclei[jl - 1] / 15000.0, 1.0)
                    zdepos = zdepos * min(zinfactor + (1.0 - zinfactor) * (yrecldp_rdepliqrefrate + zcldtopdist[jl - 1] / yrecldp_rdepliqrefdepth), 1.0)
                    zsolqa[ncldql - 1, ncldqi - 1, jl - 1] = zsolqa[ncldql - 1, ncldqi - 1, jl - 1] + zdepos
                    zsolqa[ncldqi - 1, ncldql - 1, jl - 1] = zsolqa[ncldqi - 1, ncldql - 1, jl - 1] - zdepos
                    zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zdepos
                    zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] - zdepos
        elif idepice == 2:
            for jl in range(kidia, kfdia + 1):
                if za[jk - 1 - 1, jl - 1] < yrecldp_rcldtopcf and za[jk - 1, jl - 1] >= yrecldp_rcldtopcf:
                    zcldtopdist[jl - 1] = 0.0
                else:
                    zcldtopdist[jl - 1] = zcldtopdist[jl - 1] + zdp[jl - 1] / (zrho[jl - 1] * ydcst_rg)
                if ztp1[jk - 1, jl - 1] < ydcst_rtt and zqxfg[ncldql - 1, jl - 1] > yrecldp_rlmin:
                    zvpice = ydthf_r2es * np.exp(ydthf_r3ies * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4ies)) * ydcst_rv / ydcst_rd
                    zvpliq = zvpice * zfokoop[jl - 1]
                    zicenuclei[jl - 1] = 1000.0 * np.exp(12.96 * (zvpliq - zvpice) / zvpliq - 0.639)
                    zice0 = max(zicecld[jl - 1], zicenuclei[jl - 1] * yrecldp_riceinit / zrho[jl - 1])
                    ztcg = 1.0
                    zfacx1i = 1.0
                    zaplusb = yrecldp_rcl_apb1 * zvpice - yrecldp_rcl_apb2 * zvpice * ztp1[jk - 1, jl - 1] + pap[jk - 1, jl - 1] * yrecldp_rcl_apb3 * ztp1[jk - 1, jl - 1] ** 3.0
                    zcorrfac = (1.0 / zrho[jl - 1]) ** 0.5
                    zcorrfac2 = (ztp1[jk - 1, jl - 1] / 273.0) ** 1.5 * (393.0 / (ztp1[jk - 1, jl - 1] + 120.0))
                    zpr02 = zrho[jl - 1] * zice0 * yrecldp_rcl_const1i / (ztcg * zfacx1i)
                    zterm1 = (zvpliq - zvpice) * ztp1[jk - 1, jl - 1] ** 2.0 * zvpice * zcorrfac2 * ztcg * yrecldp_rcl_const2i * zfacx1i / (zrho[jl - 1] * zaplusb * zvpice)
                    zterm2 = 0.65 * yrecldp_rcl_const6i * zpr02 ** yrecldp_rcl_const4i + yrecldp_rcl_const3i * zcorrfac ** 0.5 * zrho[jl - 1] ** 0.5 * zpr02 ** yrecldp_rcl_const5i / zcorrfac2 ** 0.5
                    zdepos = max(za[jk - 1, jl - 1] * zterm1 * zterm2 * ptsphy, 0.0)
                    zdepos = min(zdepos, zqxfg[ncldql - 1, jl - 1])
                    zinfactor = min(zicenuclei[jl - 1] / 15000.0, 1.0)
                    zdepos = zdepos * min(zinfactor + (1.0 - zinfactor) * (yrecldp_rdepliqrefrate + zcldtopdist[jl - 1] / yrecldp_rdepliqrefdepth), 1.0)
                    zsolqa[ncldql - 1, ncldqi - 1, jl - 1] = zsolqa[ncldql - 1, ncldqi - 1, jl - 1] + zdepos
                    zsolqa[ncldqi - 1, ncldql - 1, jl - 1] = zsolqa[ncldqi - 1, ncldql - 1, jl - 1] - zdepos
                    zqxfg[ncldqi - 1, jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zdepos
                    zqxfg[ncldql - 1, jl - 1] = zqxfg[ncldql - 1, jl - 1] - zdepos
        for jl in range(kidia, kfdia + 1):
            ztmpa = 1.0 / max(za[jk - 1, jl - 1], zepsec)
            zliqcld[jl - 1] = zqxfg[ncldql - 1, jl - 1] * ztmpa
            zicecld[jl - 1] = zqxfg[ncldqi - 1, jl - 1] * ztmpa
            zlicld[jl - 1] = zliqcld[jl - 1] + zicecld[jl - 1]
        for jm in range(1, nclv + 1):
            if llfall[jm - 1] or jm == ncldqi:
                for jl in range(kidia, kfdia + 1):
                    if jk > yrecldp_ncldtop:
                        zfallsrce[jm - 1, jl - 1] = zpfplsx[jm - 1, jk - 1, jl - 1] * zdtgdp[jl - 1]
                        zsolqa[jm - 1, jm - 1, jl - 1] = zsolqa[jm - 1, jm - 1, jl - 1] + zfallsrce[jm - 1, jl - 1]
                        zqxfg[jm - 1, jl - 1] = zqxfg[jm - 1, jl - 1] + zfallsrce[jm - 1, jl - 1]
                        zqpretot[jl - 1] = zqpretot[jl - 1] + zqxfg[jm - 1, jl - 1]
                    if yrecldp_laericesed and jm == ncldqi:
                        zre_ice = pre_ice[jk - 1, jl - 1]
                        zvqx[ncldqi - 1] = 0.002 * zre_ice ** 1.0
                    zfall = zvqx[jm - 1] * zrho[jl - 1]
                    zfallsink[jm - 1, jl - 1] = zdtgdp[jl - 1] * zfall
        for jl in range(kidia, kfdia + 1):
            if zqpretot[jl - 1] > zepsec:
                zcovptot[jl - 1] = 1.0 - (1.0 - zcovptot[jl - 1]) * (1.0 - max(za[jk - 1, jl - 1], za[jk - 1 - 1, jl - 1])) / (1.0 - min(za[jk - 1 - 1, jl - 1], 1.0 - 1e-06))
                zcovptot[jl - 1] = max(zcovptot[jl - 1], yrecldp_rcovpmin)
                zcovpclr[jl - 1] = max(0.0, zcovptot[jl - 1] - za[jk - 1, jl - 1])
                zraincld[jl - 1] = zqxfg[ncldqr - 1, jl - 1] / zcovptot[jl - 1]
                zsnowcld[jl - 1] = zqxfg[ncldqs - 1, jl - 1] / zcovptot[jl - 1]
                zcovpmax[jl - 1] = max(zcovptot[jl - 1], zcovpmax[jl - 1])
            else:
                zraincld[jl - 1] = 0.0
                zsnowcld[jl - 1] = 0.0
                zcovptot[jl - 1] = 0.0
                zcovpclr[jl - 1] = 0.0
                zcovpmax[jl - 1] = 0.0
        for jl in range(kidia, kfdia + 1):
            if ztp1[jk - 1, jl - 1] <= ydcst_rtt:
                if zicecld[jl - 1] > zepsec:
                    zzco = ptsphy * yrecldp_rsnowlin1 * np.exp(yrecldp_rsnowlin2 * (ztp1[jk - 1, jl - 1] - ydcst_rtt))
                    if yrecldp_laericeauto:
                        zlcrit = picrit_aer[jk - 1, jl - 1]
                        zzco = zzco * (yrecldp_rnice / pnice[jk - 1, jl - 1]) ** 0.333
                    else:
                        zlcrit = yrecldp_rlcritsnow
                    zsnowaut[jl - 1] = zzco * (1.0 - np.exp(-(zicecld[jl - 1] / zlcrit) ** 2))
                    zsolqb[ncldqi - 1, ncldqs - 1, jl - 1] = zsolqb[ncldqi - 1, ncldqs - 1, jl - 1] + zsnowaut[jl - 1]
            if zliqcld[jl - 1] > zepsec:
                if iwarmrain == 1:
                    zzco = yrecldp_rkconv * ptsphy
                    if yrecldp_laerliqautolsp:
                        zlcrit = plcrit_aer[jk - 1, jl - 1]
                        zzco = zzco * (yrecldp_rccn / pccn[jk - 1, jl - 1]) ** 0.333
                    elif plsm[jl - 1] > 0.5:
                        zlcrit = yrecldp_rclcrit_land
                    else:
                        zlcrit = yrecldp_rclcrit_sea
                    zprecip = (zpfplsx[ncldqs - 1, jk - 1, jl - 1] + zpfplsx[ncldqr - 1, jk - 1, jl - 1]) / max(zepsec, zcovptot[jl - 1])
                    zcfpr = 1.0 + yrecldp_rprc1 * np.sqrt(max(zprecip, 0.0))
                    if yrecldp_laerliqcoll:
                        zcfpr = zcfpr * (yrecldp_rccn / pccn[jk - 1, jl - 1]) ** 0.333
                    zzco = zzco * zcfpr
                    zlcrit = zlcrit / max(zcfpr, zepsec)
                    if zliqcld[jl - 1] / zlcrit < 20.0:
                        zrainaut[jl - 1] = zzco * (1.0 - np.exp(-(zliqcld[jl - 1] / zlcrit) ** 2))
                    else:
                        zrainaut[jl - 1] = zzco
                    if ztp1[jk - 1, jl - 1] <= ydcst_rtt:
                        zsolqb[ncldql - 1, ncldqs - 1, jl - 1] = zsolqb[ncldql - 1, ncldqs - 1, jl - 1] + zrainaut[jl - 1]
                    else:
                        zsolqb[ncldql - 1, ncldqr - 1, jl - 1] = zsolqb[ncldql - 1, ncldqr - 1, jl - 1] + zrainaut[jl - 1]
                elif iwarmrain == 2:
                    if plsm[jl - 1] > 0.5:
                        zconst = yrecldp_rcl_kk_cloud_num_land
                        zlcrit = yrecldp_rclcrit_land
                    else:
                        zconst = yrecldp_rcl_kk_cloud_num_sea
                        zlcrit = yrecldp_rclcrit_sea
                    if zliqcld[jl - 1] > zlcrit:
                        zrainaut[jl - 1] = 1.5 * za[jk - 1, jl - 1] * ptsphy * yrecldp_rcl_kkaau * zliqcld[jl - 1] ** yrecldp_rcl_kkbauq * zconst ** yrecldp_rcl_kkbaun
                        zrainaut[jl - 1] = min(zrainaut[jl - 1], zqxfg[ncldql - 1, jl - 1])
                        if zrainaut[jl - 1] < zepsec:
                            zrainaut[jl - 1] = 0.0
                        zrainacc[jl - 1] = 2.0 * za[jk - 1, jl - 1] * ptsphy * yrecldp_rcl_kkaac * (zliqcld[jl - 1] * zraincld[jl - 1]) ** yrecldp_rcl_kkbac
                        zrainacc[jl - 1] = min(zrainacc[jl - 1], zqxfg[ncldql - 1, jl - 1])
                        if zrainacc[jl - 1] < zepsec:
                            zrainacc[jl - 1] = 0.0
                    else:
                        zrainaut[jl - 1] = 0.0
                        zrainacc[jl - 1] = 0.0
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
        if iwarmrain > 1:
            for jl in range(kidia, kfdia + 1):
                if ztp1[jk - 1, jl - 1] <= ydcst_rtt and zliqcld[jl - 1] > zepsec:
                    zfallcorr = (yrecldp_rdensref / zrho[jl - 1]) ** 0.4
                    if zsnowcld[jl - 1] > zepsec and zcovptot[jl - 1] > 0.01:
                        zsnowrime[jl - 1] = 0.3 * zcovptot[jl - 1] * ptsphy * yrecldp_rcl_const7s * zfallcorr * (zrho[jl - 1] * zsnowcld[jl - 1] * yrecldp_rcl_const1s) ** yrecldp_rcl_const8s
                        zsnowrime[jl - 1] = min(zsnowrime[jl - 1], 1.0)
                        zsolqb[ncldql - 1, ncldqs - 1, jl - 1] = zsolqb[ncldql - 1, ncldqs - 1, jl - 1] + zsnowrime[jl - 1]
        for jl in range(kidia, kfdia + 1):
            zicetot[jl - 1] = zqxfg[ncldqi - 1, jl - 1] + zqxfg[ncldqs - 1, jl - 1]
            zmeltmax[jl - 1] = 0.0
            if zicetot[jl - 1] > zepsec and ztp1[jk - 1, jl - 1] > ydcst_rtt:
                zsubsat = max(zqsice[jk - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1], 0.0)
                ztdmtw0 = ztp1[jk - 1, jl - 1] - ydcst_rtt - zsubsat * (ztw1 + ztw2 * (pap[jk - 1, jl - 1] - ztw3) - ztw4 * (ztp1[jk - 1, jl - 1] - ztw5))
                zcons1 = abs(ptsphy * (1.0 + 0.5 * ztdmtw0) / yrecldp_rtaumel)
                zmeltmax[jl - 1] = max(ztdmtw0 * zcons1 * zrldcp, 0.0)
        for jm in range(1, nclv + 1):
            if iphase[jm - 1] == 2:
                for jl in range(kidia, kfdia + 1):
                    if zmeltmax[jl - 1] > zepsec and zicetot[jl - 1] > zepsec:
                        zalfa2 = zqxfg[jm - 1, jl - 1] / zicetot[jl - 1]
                        zmelt = min(zqxfg[jm - 1, jl - 1], zalfa2 * zmeltmax[jl - 1])
                        zqxfg[jm - 1, jl - 1] = zqxfg[jm - 1, jl - 1] - zmelt
                        zqxfg[imelt[jm - 1] - 1, jl - 1] = zqxfg[imelt[jm - 1] - 1, jl - 1] + zmelt
                        zsolqa[jm - 1, imelt[jm - 1] - 1, jl - 1] = zsolqa[jm - 1, imelt[jm - 1] - 1, jl - 1] + zmelt
                        zsolqa[imelt[jm - 1] - 1, jm - 1, jl - 1] = zsolqa[imelt[jm - 1] - 1, jm - 1, jl - 1] - zmelt
        for jl in range(kidia, kfdia + 1):
            if zqx[ncldqr - 1, jk - 1, jl - 1] > zepsec:
                if ztp1[jk - 1, jl - 1] <= ydcst_rtt and ztp1[jk - 1 - 1, jl - 1] > ydcst_rtt:
                    zqpretot[jl - 1] = max(zqx[ncldqs - 1, jk - 1, jl - 1] + zqx[ncldqr - 1, jk - 1, jl - 1], zepsec)
                    prainfrac_toprfz[jl - 1] = zqx[ncldqr - 1, jk - 1, jl - 1] / zqpretot[jl - 1]
                    if prainfrac_toprfz[jl - 1] > 0.8:
                        llrainliq[jl - 1] = True
                    else:
                        llrainliq[jl - 1] = False
                if ztp1[jk - 1, jl - 1] < ydcst_rtt:
                    if prainfrac_toprfz[jl - 1] > 0.8:
                        zlambda = (yrecldp_rcl_fac1 / (zrho[jl - 1] * zqx[ncldqr - 1, jk - 1, jl - 1])) ** yrecldp_rcl_fac2
                        ztemp = yrecldp_rcl_fzrab * (ztp1[jk - 1, jl - 1] - ydcst_rtt)
                        zfrz = ptsphy * (yrecldp_rcl_const5r / zrho[jl - 1]) * (np.exp(ztemp) - 1.0) * zlambda ** yrecldp_rcl_const6r
                        zfrzmax[jl - 1] = max(zfrz, 0.0)
                    else:
                        zcons1 = abs(ptsphy * (1.0 + 0.5 * (ydcst_rtt - ztp1[jk - 1, jl - 1])) / yrecldp_rtaumel)
                        zfrzmax[jl - 1] = max((ydcst_rtt - ztp1[jk - 1, jl - 1]) * zcons1 * zrldcp, 0.0)
                    if zfrzmax[jl - 1] > zepsec:
                        zfrz = min(zqx[ncldqr - 1, jk - 1, jl - 1], zfrzmax[jl - 1])
                        zsolqa[ncldqr - 1, ncldqs - 1, jl - 1] = zsolqa[ncldqr - 1, ncldqs - 1, jl - 1] + zfrz
                        zsolqa[ncldqs - 1, ncldqr - 1, jl - 1] = zsolqa[ncldqs - 1, ncldqr - 1, jl - 1] - zfrz
        for jl in range(kidia, kfdia + 1):
            zfrzmax[jl - 1] = max((yrecldp_rthomo - ztp1[jk - 1, jl - 1]) * zrldcp, 0.0)

        for jl in range(kidia, kfdia + 1):
            if zfrzmax[jl - 1] > zepsec and zqxfg[ncldql - 1, jl - 1] > zepsec:
                zfrz = min(zqxfg[ncldql - 1, jl - 1], zfrzmax[jl - 1])
                zsolqa[ncldql - 1, imelt[ncldql - 1] - 1, jl - 1] = zsolqa[ncldql - 1, imelt[ncldql - 1] - 1, jl - 1] + zfrz
                zsolqa[imelt[ncldql - 1] - 1, ncldql - 1, jl - 1] = zsolqa[imelt[ncldql - 1] - 1, ncldql - 1, jl - 1] - zfrz
        if ievaprain == 1:
            for jl in range(kidia, kfdia + 1):
                zzrh = yrecldp_rprecrhmax + (1.0 - yrecldp_rprecrhmax) * zcovpmax[jl - 1] / max(zepsec, 1.0 - za[jk - 1, jl - 1])
                zzrh = min(max(zzrh, yrecldp_rprecrhmax), 1.0)
                zqe = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1] * zqsliq[jk - 1, jl - 1]) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
                zqe = max(0.0, min(zqe, zqsliq[jk - 1, jl - 1]))
                llo1 = zcovpclr[jl - 1] > zepsec and zqxfg[ncldqr - 1, jl - 1] > zepsec and (zqe < zzrh * zqsliq[jk - 1, jl - 1])
                if llo1:
                    zpreclr = zqxfg[ncldqr - 1, jl - 1] * zcovpclr[jl - 1] / (max(abs(zcovptot[jl - 1] * zdtgdp[jl - 1]), zepsilon) * np.sign(zcovptot[jl - 1] * zdtgdp[jl - 1]))
                    zbeta1 = np.sqrt(pap[jk - 1, jl - 1] / paph[klev + 1 - 1, jl - 1]) / yrecldp_rvrfactor * zpreclr / max(zcovpclr[jl - 1], zepsec)
                    zbeta = ydcst_rg * yrecldp_rpecons * 0.5 * zbeta1 ** 0.5777
                    zdenom = 1.0 + zbeta * ptsphy * zcorqsliq[jl - 1]
                    zdpr = zcovpclr[jl - 1] * zbeta * (zqsliq[jk - 1, jl - 1] - zqe) / zdenom * zdp[jl - 1] * zrg_r
                    zdpevap = zdpr * zdtgdp[jl - 1]
                    zevap = min(zdpevap, zqxfg[ncldqr - 1, jl - 1])
                    zsolqa[ncldqr - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqr - 1, ncldqv - 1, jl - 1] + zevap
                    zsolqa[ncldqv - 1, ncldqr - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqr - 1, jl - 1] - zevap
                    zcovptot[jl - 1] = max(yrecldp_rcovpmin, zcovptot[jl - 1] - max(0.0, (zcovptot[jl - 1] - za[jk - 1, jl - 1]) * zevap / zqxfg[ncldqr - 1, jl - 1]))
                    zqxfg[ncldqr - 1, jl - 1] = zqxfg[ncldqr - 1, jl - 1] - zevap
        elif ievaprain == 2:
            for jl in range(kidia, kfdia + 1):
                zzrh = yrecldp_rprecrhmax + (1.0 - yrecldp_rprecrhmax) * zcovpmax[jl - 1] / max(zepsec, 1.0 - za[jk - 1, jl - 1])
                zzrh = min(max(zzrh, yrecldp_rprecrhmax), 1.0)
                zzrh = min(0.8, zzrh)
                zqe = max(0.0, min(zqx[ncldqv - 1, jk - 1, jl - 1], zqsliq[jk - 1, jl - 1]))
                llo1 = zcovpclr[jl - 1] > zepsec and zqxfg[ncldqr - 1, jl - 1] > zepsec and (zqe < zzrh * zqsliq[jk - 1, jl - 1])
                if llo1:
                    zpreclr = zqxfg[ncldqr - 1, jl - 1] / zcovptot[jl - 1]
                    zfallcorr = (yrecldp_rdensref / zrho[jl - 1]) ** 0.4
                    zesatliq = ydcst_rv / ydcst_rd * (ydthf_r2es * np.exp(ydthf_r3les * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4les)))
                    zlambda = (yrecldp_rcl_fac1 / (zrho[jl - 1] * zpreclr)) ** yrecldp_rcl_fac2
                    zevap_denom = yrecldp_rcl_cdenom1 * zesatliq - yrecldp_rcl_cdenom2 * ztp1[jk - 1, jl - 1] * zesatliq + yrecldp_rcl_cdenom3 * ztp1[jk - 1, jl - 1] ** 3.0 * pap[jk - 1, jl - 1]
                    zcorr2 = (ztp1[jk - 1, jl - 1] / 273.0) ** 1.5 * 393.0 / (ztp1[jk - 1, jl - 1] + 120.0)
                    zka = yrecldp_rcl_ka273 * zcorr2
                    zsubsat = max(zzrh * zqsliq[jk - 1, jl - 1] - zqe, 0.0)
                    zbeta = 0.5 / zqsliq[jk - 1, jl - 1] * ztp1[jk - 1, jl - 1] ** 2.0 * zesatliq * yrecldp_rcl_const1r * (zcorr2 / zevap_denom) * (0.78 / zlambda ** yrecldp_rcl_const4r + yrecldp_rcl_const2r * (zrho[jl - 1] * zfallcorr) ** 0.5 / (zcorr2 ** 0.5 * zlambda ** yrecldp_rcl_const3r))
                    zdenom = 1.0 + zbeta * ptsphy
                    zdpevap = zcovpclr[jl - 1] * zbeta * ptsphy * zsubsat / zdenom
                    zevap = min(zdpevap, zqxfg[ncldqr - 1, jl - 1])
                    zsolqa[ncldqr - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqr - 1, ncldqv - 1, jl - 1] + zevap
                    zsolqa[ncldqv - 1, ncldqr - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqr - 1, jl - 1] - zevap
                    zcovptot[jl - 1] = max(yrecldp_rcovpmin, zcovptot[jl - 1] - max(0.0, (zcovptot[jl - 1] - za[jk - 1, jl - 1]) * zevap / zqxfg[ncldqr - 1, jl - 1]))
                    zqxfg[ncldqr - 1, jl - 1] = zqxfg[ncldqr - 1, jl - 1] - zevap
        if ievapsnow == 1:
            for jl in range(kidia, kfdia + 1):
                zzrh = yrecldp_rprecrhmax + (1.0 - yrecldp_rprecrhmax) * zcovpmax[jl - 1] / max(zepsec, 1.0 - za[jk - 1, jl - 1])
                zzrh = min(max(zzrh, yrecldp_rprecrhmax), 1.0)
                zqe = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1] * zqsice[jk - 1, jl - 1]) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
                zqe = max(0.0, min(zqe, zqsice[jk - 1, jl - 1]))
                llo1 = zcovpclr[jl - 1] > zepsec and zqxfg[ncldqs - 1, jl - 1] > zepsec and (zqe < zzrh * zqsice[jk - 1, jl - 1])
                if llo1:
                    zpreclr = zqxfg[ncldqs - 1, jl - 1] * zcovpclr[jl - 1] / (max(abs(zcovptot[jl - 1] * zdtgdp[jl - 1]), zepsilon) * np.sign(zcovptot[jl - 1] * zdtgdp[jl - 1]))
                    zbeta1 = np.sqrt(pap[jk - 1, jl - 1] / paph[klev + 1 - 1, jl - 1]) / yrecldp_rvrfactor * zpreclr / max(zcovpclr[jl - 1], zepsec)
                    zbeta = ydcst_rg * yrecldp_rpecons * zbeta1 ** 0.5777
                    zdenom = 1.0 + zbeta * ptsphy * zcorqsice[jl - 1]
                    zdpr = zcovpclr[jl - 1] * zbeta * (zqsice[jk - 1, jl - 1] - zqe) / zdenom * zdp[jl - 1] * zrg_r
                    zdpevap = zdpr * zdtgdp[jl - 1]
                    zevap = min(zdpevap, zqxfg[ncldqs - 1, jl - 1])
                    zsolqa[ncldqs - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqs - 1, ncldqv - 1, jl - 1] + zevap
                    zsolqa[ncldqv - 1, ncldqs - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqs - 1, jl - 1] - zevap
                    zcovptot[jl - 1] = max(yrecldp_rcovpmin, zcovptot[jl - 1] - max(0.0, (zcovptot[jl - 1] - za[jk - 1, jl - 1]) * zevap / zqxfg[ncldqs - 1, jl - 1]))
                    zqxfg[ncldqs - 1, jl - 1] = zqxfg[ncldqs - 1, jl - 1] - zevap
        elif ievapsnow == 2:
            for jl in range(kidia, kfdia + 1):
                zzrh = yrecldp_rprecrhmax + (1.0 - yrecldp_rprecrhmax) * zcovpmax[jl - 1] / max(zepsec, 1.0 - za[jk - 1, jl - 1])
                zzrh = min(max(zzrh, yrecldp_rprecrhmax), 1.0)
                zqe = (zqx[ncldqv - 1, jk - 1, jl - 1] - za[jk - 1, jl - 1] * zqsice[jk - 1, jl - 1]) / max(zepsec, 1.0 - za[jk - 1, jl - 1])
                zqe = max(0.0, min(zqe, zqsice[jk - 1, jl - 1]))
                llo1 = zcovpclr[jl - 1] > zepsec and zqx[ncldqs - 1, jk - 1, jl - 1] > zepsec and (zqe < zzrh * zqsice[jk - 1, jl - 1])
                if llo1:
                    zpreclr = zqx[ncldqs - 1, jk - 1, jl - 1] / zcovptot[jl - 1]
                    zvpice = ydthf_r2es * np.exp(ydthf_r3ies * (ztp1[jk - 1, jl - 1] - ydcst_rtt) / (ztp1[jk - 1, jl - 1] - ydthf_r4ies)) * ydcst_rv / ydcst_rd
                    ztcg = 1.0
                    zfacx1s = 1.0
                    zaplusb = yrecldp_rcl_apb1 * zvpice - yrecldp_rcl_apb2 * zvpice * ztp1[jk - 1, jl - 1] + pap[jk - 1, jl - 1] * yrecldp_rcl_apb3 * ztp1[jk - 1, jl - 1] ** 3
                    zcorrfac = (1.0 / zrho[jl - 1]) ** 0.5
                    zcorrfac2 = (ztp1[jk - 1, jl - 1] / 273.0) ** 1.5 * (393.0 / (ztp1[jk - 1, jl - 1] + 120.0))
                    zpr02 = zrho[jl - 1] * zpreclr * yrecldp_rcl_const1s / (ztcg * zfacx1s)
                    zterm1 = (zqsice[jk - 1, jl - 1] - zqe) * ztp1[jk - 1, jl - 1] ** 2 * zvpice * zcorrfac2 * ztcg * yrecldp_rcl_const2s * zfacx1s / (zrho[jl - 1] * zaplusb * zqsice[jk - 1, jl - 1])
                    zterm2 = 0.65 * yrecldp_rcl_const6s * zpr02 ** yrecldp_rcl_const4s + yrecldp_rcl_const3s * zcorrfac ** 0.5 * zrho[jl - 1] ** 0.5 * zpr02 ** yrecldp_rcl_const5s / zcorrfac2 ** 0.5
                    zdpevap = max(zcovpclr[jl - 1] * zterm1 * zterm2 * ptsphy, 0.0)
                    zevap = min(zdpevap, zevaplimice[jl - 1])
                    zevap = min(zevap, zqx[ncldqs - 1, jk - 1, jl - 1])
                    zsolqa[ncldqs - 1, ncldqv - 1, jl - 1] = zsolqa[ncldqs - 1, ncldqv - 1, jl - 1] + zevap
                    zsolqa[ncldqv - 1, ncldqs - 1, jl - 1] = zsolqa[ncldqv - 1, ncldqs - 1, jl - 1] - zevap
                    zcovptot[jl - 1] = max(yrecldp_rcovpmin, zcovptot[jl - 1] - max(0.0, (zcovptot[jl - 1] - za[jk - 1, jl - 1]) * zevap / zqx[ncldqs - 1, jk - 1, jl - 1]))
                    zqxfg[ncldqs - 1, jl - 1] = zqxfg[ncldqs - 1, jl - 1] - zevap
        for jm in range(1, nclv + 1):
            if llfall[jm - 1]:
                for jl in range(kidia, kfdia + 1):
                    if zqxfg[jm - 1, jl - 1] < yrecldp_rlmin:
                        zsolqa[jm - 1, ncldqv - 1, jl - 1] = zsolqa[jm - 1, ncldqv - 1, jl - 1] + zqxfg[jm - 1, jl - 1]
                        zsolqa[ncldqv - 1, jm - 1, jl - 1] = zsolqa[ncldqv - 1, jm - 1, jl - 1] - zqxfg[jm - 1, jl - 1]
        for jl in range(kidia, kfdia + 1):
            zanew = (za[jk - 1, jl - 1] + zsolac[jl - 1]) / (1.0 + zsolab[jl - 1])
            zanew = min(zanew, 1.0)
            if zanew < yrecldp_ramin:
                zanew = 0.0
            zda[jl - 1] = zanew - zaorig[jk - 1, jl - 1]
            zanewm1[jl - 1] = zanew
        for jm in range(1, nclv + 1):
            for jn in range(1, nclv + 1):
                for jl in range(kidia, kfdia + 1):
                    llindex3[jm - 1, jn - 1, jl - 1] = False
            for jl in range(kidia, kfdia + 1):
                zsinksum[jm - 1, jl - 1] = 0.0
        for jm in range(1, nclv + 1):
            for jn in range(1, nclv + 1):
                for jl in range(kidia, kfdia + 1):
                    zsinksum[jm - 1, jl - 1] = zsinksum[jm - 1, jl - 1] - zsolqa[jn - 1, jm - 1, jl - 1]
        for jm in range(1, nclv + 1):
            for jl in range(kidia, kfdia + 1):
                zmax = max(zqx[jm - 1, jk - 1, jl - 1], zepsec)
                zrat = max(zsinksum[jm - 1, jl - 1], zmax)
                zratio[jm - 1, jl - 1] = zmax / zrat
        for jm in range(1, nclv + 1):
            for jl in range(kidia, kfdia + 1):
                zsinksum[jm - 1, jl - 1] = 0.0
        for jm in range(1, nclv + 1):
            for jl in range(1, klon + 1):
                psum_solqa[jl - 1] = 0.0
            for jn in range(1, nclv + 1):
                for jl in range(kidia, kfdia + 1):
                    psum_solqa[jl - 1] = psum_solqa[jl - 1] + zsolqa[jn - 1, jm - 1, jl - 1]
            for jl in range(kidia, kfdia + 1):
                zsinksum[jm - 1, jl - 1] = zsinksum[jm - 1, jl - 1] - psum_solqa[jl - 1]
            for jl in range(kidia, kfdia + 1):
                zmm = max(zqx[jm - 1, jk - 1, jl - 1], zepsec)
                zrr = max(zsinksum[jm - 1, jl - 1], zmm)
                zratio[jm - 1, jl - 1] = zmm / zrr
            for jl in range(kidia, kfdia + 1):
                zzratio = zratio[jm - 1, jl - 1]
                for jn in range(1, nclv + 1):
                    if zsolqa[jn - 1, jm - 1, jl - 1] < 0.0:
                        zsolqa[jn - 1, jm - 1, jl - 1] = zsolqa[jn - 1, jm - 1, jl - 1] * zzratio
                        zsolqa[jm - 1, jn - 1, jl - 1] = zsolqa[jm - 1, jn - 1, jl - 1] * zzratio
        for jm in range(1, nclv + 1):
            for jn in range(1, nclv + 1):
                if jn == jm:
                    for jl in range(kidia, kfdia + 1):
                        zqlhs[jm - 1, jn - 1, jl - 1] = 1.0 + zfallsink[jm - 1, jl - 1]
                        for jo in range(1, nclv + 1):
                            zqlhs[jm - 1, jn - 1, jl - 1] = zqlhs[jm - 1, jn - 1, jl - 1] + zsolqb[jn - 1, jo - 1, jl - 1]
                else:
                    for jl in range(kidia, kfdia + 1):
                        zqlhs[jm - 1, jn - 1, jl - 1] = -zsolqb[jm - 1, jn - 1, jl - 1]
        for jm in range(1, nclv + 1):
            for jl in range(kidia, kfdia + 1):
                zexplicit = 0.0
                for jn in range(1, nclv + 1):
                    zexplicit = zexplicit + zsolqa[jn - 1, jm - 1, jl - 1]
                zqxn[jm - 1, jl - 1] = zqx[jm - 1, jk - 1, jl - 1] + zexplicit
        for jn in range(1, nclv - 1 + 1):
            for jm in range(jn + 1, nclv + 1):
                for jl in range(kidia, kfdia + 1):
                    zqlhs[jn - 1, jm - 1, jl - 1] = zqlhs[jn - 1, jm - 1, jl - 1] / zqlhs[jn - 1, jn - 1, jl - 1]
        for jn in range(1, nclv - 1 + 1):
            for jm in range(jn + 1, nclv + 1):
                for ik in range(jn + 1, nclv + 1):
                    for jl in range(kidia, kfdia + 1):
                        zqlhs[ik - 1, jm - 1, jl - 1] = zqlhs[ik - 1, jm - 1, jl - 1] - zqlhs[jn - 1, jm - 1, jl - 1] * zqlhs[ik - 1, jn - 1, jl - 1]
        for jn in range(2, nclv + 1):
            for jm in range(1, jn - 1 + 1):
                for jl in range(kidia, kfdia + 1):
                    zqxn[jn - 1, jl - 1] = zqxn[jn - 1, jl - 1] - zqlhs[jm - 1, jn - 1, jl - 1] * zqxn[jm - 1, jl - 1]
        for jl in range(kidia, kfdia + 1):
            zqxn[nclv - 1, jl - 1] = zqxn[nclv - 1, jl - 1] / zqlhs[nclv - 1, nclv - 1, jl - 1]
        for jn in range(nclv - 1, 1 + -1, -1):
            for jm in range(jn + 1, nclv + 1):
                for jl in range(kidia, kfdia + 1):
                    zqxn[jn - 1, jl - 1] = zqxn[jn - 1, jl - 1] - zqlhs[jm - 1, jn - 1, jl - 1] * zqxn[jm - 1, jl - 1]
            for jl in range(kidia, kfdia + 1):
                zqxn[jn - 1, jl - 1] = zqxn[jn - 1, jl - 1] / zqlhs[jn - 1, jn - 1, jl - 1]
        for jn in range(1, nclv - 1 + 1):
            for jl in range(kidia, kfdia + 1):
                if zqxn[jn - 1, jl - 1] < zepsec:
                    zqxn[ncldqv - 1, jl - 1] = zqxn[ncldqv - 1, jl - 1] + zqxn[jn - 1, jl - 1]
                    zqxn[jn - 1, jl - 1] = 0.0
        for jm in range(1, nclv + 1):
            for jl in range(kidia, kfdia + 1):
                zqxnm1[jm - 1, jl - 1] = zqxn[jm - 1, jl - 1]
                zqxn2d[jm - 1, jk - 1, jl - 1] = zqxn[jm - 1, jl - 1]
        for jm in range(1, nclv + 1):
            for jl in range(kidia, kfdia + 1):
                zpfplsx[jm - 1, jk + 1 - 1, jl - 1] = zfallsink[jm - 1, jl - 1] * zqxn[jm - 1, jl - 1] * zrdtgdp[jl - 1]
        for jl in range(kidia, kfdia + 1):
            zqpretot[jl - 1] = zpfplsx[ncldqs - 1, jk + 1 - 1, jl - 1] + zpfplsx[ncldqr - 1, jk + 1 - 1, jl - 1]
        for jl in range(kidia, kfdia + 1):
            if zqpretot[jl - 1] < zepsec:
                zcovptot[jl - 1] = 0.0
        for jm in range(1, nclv - 1 + 1):
            for jl in range(kidia, kfdia + 1):
                zfluxq[jm - 1, jl - 1] = zpsupsatsrce[jm - 1, jl - 1] + zconvsrce[jm - 1, jl - 1] + zfallsrce[jm - 1, jl - 1] - (zfallsink[jm - 1, jl - 1] + zconvsink[jm - 1, jl - 1]) * zqxn[jm - 1, jl - 1]
            if iphase[jm - 1] == 1:
                for jl in range(kidia, kfdia + 1):
                    tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] + ydthf_ralvdcp * (zqxn[jm - 1, jl - 1] - zqx[jm - 1, jk - 1, jl - 1] - zfluxq[jm - 1, jl - 1]) * zqtmst
            if iphase[jm - 1] == 2:
                for jl in range(kidia, kfdia + 1):
                    tendency_loc_t[jk - 1, jl - 1] = tendency_loc_t[jk - 1, jl - 1] + ydthf_ralsdcp * (zqxn[jm - 1, jl - 1] - zqx[jm - 1, jk - 1, jl - 1] - zfluxq[jm - 1, jl - 1]) * zqtmst
            for jl in range(kidia, kfdia + 1):
                tendency_loc_cld[jm - 1, jk - 1, jl - 1] = tendency_loc_cld[jm - 1, jk - 1, jl - 1] + (zqxn[jm - 1, jl - 1] - zqx0[jm - 1, jk - 1, jl - 1]) * zqtmst
        for jl in range(kidia, kfdia + 1):
            tendency_loc_q[jk - 1, jl - 1] = tendency_loc_q[jk - 1, jl - 1] + (zqxn[ncldqv - 1, jl - 1] - zqx[ncldqv - 1, jk - 1, jl - 1]) * zqtmst
            tendency_loc_a[jk - 1, jl - 1] = tendency_loc_a[jk - 1, jl - 1] + zda[jl - 1] * zqtmst
        for jl in range(kidia, kfdia + 1):
            pcovptot[jk - 1, jl - 1] = zcovptot[jl - 1]
    for jk in range(1, klev + 1 + 1):
        for jl in range(kidia, kfdia + 1):
            pfplsl[jk - 1, jl - 1] = zpfplsx[ncldqr - 1, jk - 1, jl - 1] + zpfplsx[ncldql - 1, jk - 1, jl - 1]
            pfplsn[jk - 1, jl - 1] = zpfplsx[ncldqs - 1, jk - 1, jl - 1] + zpfplsx[ncldqi - 1, jk - 1, jl - 1]
    for jl in range(kidia, kfdia + 1):
        pfsqlf[1 - 1, jl - 1] = 0.0
        pfsqif[1 - 1, jl - 1] = 0.0
        pfsqrf[1 - 1, jl - 1] = 0.0
        pfsqsf[1 - 1, jl - 1] = 0.0
        pfcqlng[1 - 1, jl - 1] = 0.0
        pfcqnng[1 - 1, jl - 1] = 0.0
        pfcqrng[1 - 1, jl - 1] = 0.0
        pfcqsng[1 - 1, jl - 1] = 0.0
        pfsqltur[1 - 1, jl - 1] = 0.0
        pfsqitur[1 - 1, jl - 1] = 0.0
    for jk in range(1, klev + 1):
        for jl in range(kidia, kfdia + 1):
            zgdph_r = -zrg_r * (paph[jk + 1 - 1, jl - 1] - paph[jk - 1, jl - 1]) * zqtmst
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
            pfsqlf[jk + 1 - 1, jl - 1] = pfsqlf[jk + 1 - 1, jl - 1] + (zqxn2d[ncldql - 1, jk - 1, jl - 1] - zqx0[ncldql - 1, jk - 1, jl - 1] + pvfl[jk - 1, jl - 1] * ptsphy - zalfaw * plude[jk - 1, jl - 1]) * zgdph_r
            pfcqlng[jk + 1 - 1, jl - 1] = pfcqlng[jk + 1 - 1, jl - 1] + zlneg[ncldql - 1, jk - 1, jl - 1] * zgdph_r
            pfsqltur[jk + 1 - 1, jl - 1] = pfsqltur[jk + 1 - 1, jl - 1] + pvfl[jk - 1, jl - 1] * ptsphy * zgdph_r
            pfsqrf[jk + 1 - 1, jl - 1] = pfsqrf[jk + 1 - 1, jl - 1] + (zqxn2d[ncldqr - 1, jk - 1, jl - 1] - zqx0[ncldqr - 1, jk - 1, jl - 1]) * zgdph_r
            pfcqrng[jk + 1 - 1, jl - 1] = pfcqrng[jk + 1 - 1, jl - 1] + zlneg[ncldqr - 1, jk - 1, jl - 1] * zgdph_r
            pfsqif[jk + 1 - 1, jl - 1] = pfsqif[jk + 1 - 1, jl - 1] + (zqxn2d[ncldqi - 1, jk - 1, jl - 1] - zqx0[ncldqi - 1, jk - 1, jl - 1] + pvfi[jk - 1, jl - 1] * ptsphy - (1.0 - zalfaw) * plude[jk - 1, jl - 1]) * zgdph_r
            pfcqnng[jk + 1 - 1, jl - 1] = pfcqnng[jk + 1 - 1, jl - 1] + zlneg[ncldqi - 1, jk - 1, jl - 1] * zgdph_r
            pfsqitur[jk + 1 - 1, jl - 1] = pfsqitur[jk + 1 - 1, jl - 1] + pvfi[jk - 1, jl - 1] * ptsphy * zgdph_r
            pfsqsf[jk + 1 - 1, jl - 1] = pfsqsf[jk + 1 - 1, jl - 1] + (zqxn2d[ncldqs - 1, jk - 1, jl - 1] - zqx0[ncldqs - 1, jk - 1, jl - 1]) * zgdph_r
            pfcqsng[jk + 1 - 1, jl - 1] = pfcqsng[jk + 1 - 1, jl - 1] + zlneg[ncldqs - 1, jk - 1, jl - 1] * zgdph_r
    for jk in range(1, klev + 1 + 1):
        for jl in range(kidia, kfdia + 1):
            pfhpsl[jk - 1, jl - 1] = -ydcst_rlvtt * pfplsl[jk - 1, jl - 1]
            pfhpsn[jk - 1, jl - 1] = -ydcst_rlstt * pfplsn[jk - 1, jl - 1]

if __name__ == '__main__':
    NAME_ORDER = ["ncldql", "ncldqi", "ncldqr", "ncldqs", "ncldqv"]
    NAME_MAP = {i: NAME_ORDER[i] for i in range(5)}
    # Split along the first dimension (size = nclv-1) into separate 2D arrays
    symbol_map = {
        "nclv": 5,
        "ncldql": 1,
        "ncldqi": 2,
        "ncldqr": 3,
        "ncldqs": 4,
        "ncldqv": 5,
    }
    name_map = {"nclv": NAME_ORDER}

    load_if_existing = False

    if load_if_existing and Path('./cloudsc_pydace_unsimplified.sdfgz').exists():
        print('SDFG already exists, skipping generation')
        sdfg = dace.SDFG.from_file('cloudsc_pydace_unsimplified.sdfgz')
        sdfg.validate()
    else:
        sdfg = cloudsc_py.to_sdfg(simplify=False)
        sdfg.save('cloudsc_pydace_unsimplified.sdfgz', compress=True)
        sdfg.validate()


    if load_if_existing and Path('./cloudsc_pydace_simplified_symbolic.sdfgz').exists():
        sdfg = dace.SDFG.from_file('cloudsc_pydace_simplified_symbolic.sdfgz')
        sdfg.validate()
    else:
        sdfg.simplify(skip={'ScalarToSymbolPromotion'})
        sdfg.save('cloudsc_pydace_simplified.sdfgz', compress=True)
        sdfg.validate()
        ScalarToSymbolPromotion().apply_pass(sdfg, {})
        sdfg.save('cloudsc_pydace_simplified_symbolic.sdfgz', compress=True)

    sdfg.validate()
    SplitArray(symbol_map=symbol_map, name_map=name_map).apply_pass(sdfg, {})
    sdfg.save("cloudsc_split.sdfgz", compress=True)