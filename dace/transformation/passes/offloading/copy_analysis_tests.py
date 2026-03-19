import dace

# ---------- symbols ----------
klev = dace.symbol('klev', dtype=dace.int32)
klon = dace.symbol('klon', dtype=dace.int32)
nclv = dace.symbol('nclv', dtype=dace.int32)
ncldql = dace.symbol('ncldql', dtype=dace.int32)
ncldqi = dace.symbol('ncldqi', dtype=dace.int32)
ncldqv = dace.symbol('ncldqv', dtype=dace.int32)

# ================================================================
# Variant 1: original 3D zsolqa[nclv, nclv, klon]
# ================================================================
@dace.program
def condense_3d(
    za: dace.float64[klev, klon],
    zdqs: dace.float64[klon],
    zqsmix: dace.float64[klev, klon],
    zqv: dace.float64[klev, klon],
    ztp1: dace.float64[klev, klon],
    zsolqa: dace.float64[nclv, nclv, klon],
    zqxfg: dace.float64[nclv, klon],
    retv: dace.float64, rtice: dace.float64, rtwat: dace.float64,
    rtwat_rtice_r: dace.float64, r5alvcp: dace.float64, r4les: dace.float64,
    r5alscp: dace.float64, r4ies: dace.float64, rthomo: dace.float64,
    rlmin: dace.float64,
):
    for jk in range(klev):
        for jl in range(klon):
            if za[jk, jl] > 1e-14:
                if zdqs[jl] <= -rlmin:
                    lc = max(-zdqs[jl], 0.0)
                    af = min(1.0, ((max(rtice, min(rtwat, ztp1[jk, jl])) - rtice) * rtwat_rtice_r) ** 2)
                    zcor = 1.0 / (1.0 - retv * zqsmix[jk, jl])
                    cdm_full = (zqv[jk, jl] - zqsmix[jk, jl]) / (1.0 + zcor * zqsmix[jk, jl] * (
                        af * r5alvcp / (ztp1[jk, jl] - r4les) ** 2
                        + (1.0 - af) * r5alscp / (ztp1[jk, jl] - r4ies) ** 2))
                    cdm_part = (zqv[jk, jl] - za[jk, jl] * zqsmix[jk, jl]) / za[jk, jl]
                    if za[jk, jl] > 0.99:
                        cdm = cdm_full
                    else:
                        cdm = cdm_part
                    lc = za[jk, jl] * max(min(lc, cdm), 0.0)
                    if lc >= rlmin:
                        if ztp1[jk, jl] > rthomo:
                            zsolqa[ncldqv - 1, ncldql - 1, jl] = zsolqa[ncldqv - 1, ncldql - 1, jl] + lc
                            zsolqa[ncldql - 1, ncldqv - 1, jl] = zsolqa[ncldql - 1, ncldqv - 1, jl] - lc
                            zqxfg[ncldql - 1, jl] = zqxfg[ncldql - 1, jl] + lc
                        else:
                            zsolqa[ncldqv - 1, ncldqi - 1, jl] = zsolqa[ncldqv - 1, ncldqi - 1, jl] + lc
                            zsolqa[ncldqi - 1, ncldqv - 1, jl] = zsolqa[ncldqi - 1, ncldqv - 1, jl] - lc
                            zqxfg[ncldqi - 1, jl] = zqxfg[ncldqi - 1, jl] + lc


# ================================================================
# Variant 2: split zsolqa_{src}_{dst}[klon]
# ================================================================
@dace.program
def condense_split(
    za: dace.float64[klev, klon],
    zdqs: dace.float64[klon],
    zqsmix: dace.float64[klev, klon],
    zqv: dace.float64[klev, klon],
    ztp1: dace.float64[klev, klon],
    zsolqa_ncldqv_ncldql: dace.float64[klon],
    zsolqa_ncldql_ncldqv: dace.float64[klon],
    zsolqa_ncldqv_ncldqi: dace.float64[klon],
    zsolqa_ncldqi_ncldqv: dace.float64[klon],
    zqxfg_ncldql: dace.float64[klon],
    zqxfg_ncldqi: dace.float64[klon],
    retv: dace.float64, rtice: dace.float64, rtwat: dace.float64,
    rtwat_rtice_r: dace.float64, r5alvcp: dace.float64, r4les: dace.float64,
    r5alscp: dace.float64, r4ies: dace.float64, rthomo: dace.float64,
    rlmin: dace.float64,
):
    for jk in range(klev):
        for jl in range(klon):
            if za[jk, jl] > 1e-14:
                if zdqs[jl] <= -rlmin:
                    lc = max(-zdqs[jl], 0.0)
                    af = min(1.0, ((max(rtice, min(rtwat, ztp1[jk, jl])) - rtice) * rtwat_rtice_r) ** 2)
                    zcor = 1.0 / (1.0 - retv * zqsmix[jk, jl])
                    cdm_full = (zqv[jk, jl] - zqsmix[jk, jl]) / (1.0 + zcor * zqsmix[jk, jl] * (
                        af * r5alvcp / (ztp1[jk, jl] - r4les) ** 2
                        + (1.0 - af) * r5alscp / (ztp1[jk, jl] - r4ies) ** 2))
                    cdm_part = (zqv[jk, jl] - za[jk, jl] * zqsmix[jk, jl]) / za[jk, jl]
                    if za[jk, jl] > 0.99:
                        cdm = cdm_full
                    else:
                        cdm = cdm_part
                    lc = za[jk, jl] * max(min(lc, cdm), 0.0)
                    if lc >= rlmin:
                        if ztp1[jk, jl] > rthomo:
                            zsolqa_ncldqv_ncldql[jl] = zsolqa_ncldqv_ncldql[jl] + lc
                            zsolqa_ncldql_ncldqv[jl] = zsolqa_ncldql_ncldqv[jl] - lc
                            zqxfg_ncldql[jl] = zqxfg_ncldql[jl] + lc
                        else:
                            zsolqa_ncldqv_ncldqi[jl] = zsolqa_ncldqv_ncldqi[jl] + lc
                            zsolqa_ncldqi_ncldqv[jl] = zsolqa_ncldqi_ncldqv[jl] - lc
                            zqxfg_ncldqi[jl] = zqxfg_ncldqi[jl] + lc

N = dace.symbol('N')
X = dace.symbol('X')
Y = dace.symbol('Y')
# Tasklet in NestedSDFGs Symbols
S1 = dace.symbol("S1")
S2 = dace.symbol("S2")
S = dace.symbol("S")
# CloudSC Symbols
klev = dace.symbol("klev")
kidia = dace.symbol("kidia")
kfdia = dace.symbol("kfdia")
# SpMV Symbols
n = dace.symbol('n')  # number of rows
m = dace.symbol('m')  # number of columns
nnz = dace.symbol('nnz')  # number of nonzeros

@dace.program
def memset_4d(A: dace.float64[N, N, N, N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:N]:
            for k in dace.map[0:N]:
                for m in dace.map[0:N]:
                    A[i, j, k, m] = 0.0

@dace.program
def interstate_boolean_op_two(A: dace.float64[N, N], B: dace.float64[N, N], c0: dace.int64):
    for i, j in dace.map[0:N, 0:N]:
        c1 = i
        c2 = j
        c3 = (c1 > c0) or (c2 > c0)
        c4 = c3 or (A[i, j] > B[i, j])
        if not c4:
            A[i, j] = A[i, j] + B[i, j]

@dace.program
def nested_matrix_gather_load(A: dace.float32[Y, X], B: dace.int32[Y, X], C: dace.float32[Y, X], scale: dace.float32):
    for i, j in dace.map[0:Y:1, 0:X:1]:
        C[i, j] = A[i, B[i, j]] * scale

@dace.program
def nested_matrix_gather_load_specialized(A: dace.float32[Y, X], B: dace.int32[Y, X], C: dace.float32[Y, X]):
    for i, j in dace.map[0:Y:1, 0:X:1]:
        C[i, j] = A[i, B[i, j]] * 2.0

@dace.program
def division_by_zero(A: dace.float64[N], B: dace.float64[N], c: dace.float64):
    for i in dace.map[
            0:N,
    ]:
        if A[i] > 0.0:
            B[i] = c / A[i]
        else:
            A[i] = 0.0


## tests
from dace.transformation.passes.offloading.OffloadToAccelerator import OffloadToAccelerator as OtA

def pretty_print(gpu_set, cpu_set):
    print("both GPU and CPU:")
    for name in gpu_set & cpu_set:
        print("\t", name)

    print("\nGPU only:")
    for name in gpu_set - cpu_set:
        print("\t", name)
    
    print("\nCPU only:")
    for name in cpu_set - gpu_set:
        print("\t", name)


sdfg = condense_3d.to_sdfg()
sdfg.view()
gpu_set, cpu_set = OtA().get_data_locations(sdfg)
pretty_print(gpu_set, cpu_set)

{"zdqs", "zdqs_index_0", "neg_zdqs_slice"}
{"rtwat", "min_rtwat_ztp1_slice", "rtice", "max_rtice_expr", "expr_minus_rtice", "rtwat_rtice_r", "expr_rtice_times_rtwat_rtice_r", "expr_rtice_rtwat_rtice_pow_2"}
{"zpt1", "ztp1_index", "ztp1_index_0", "ztp1_index_1", "r4les", "zpt1_slice_minus_r4les", "r4ies", "zpt1_slice_minus_r4ies", "zpt1_slice_r4ies_pow_2"}
{"r5alscp", "__tmp1", "__tmp2", "__tmp3", "__tmp4", "__tmp5", "af", "af_times_r5alscp", "af_r5alvcp_div_ztp1_slice_r4les_2", "af_r5alvcp_ztp1_slice_r4les_2_plus_1_0_af_r5alscp_ztp1_slice_r4ies_2"}
{"za", "za_index_0", "za_index_1", "zqsmix"}