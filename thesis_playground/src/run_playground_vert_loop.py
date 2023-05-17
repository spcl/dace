from argparse import ArgumentParser
import inspect
import numpy as np
import dace
from subprocess import run

from utils.general import optimize_sdfg, copy_to_device
from utils.ncu import get_action, get_achieved_performance, get_peak_performance

NBLOCKS = dace.symbol('NBLOCKS')
KLEV = 137
KLON = 1
NCLV = 10
KIDIA = 0
KFDIA = 1
NCLDQI = 2
NCLDQL = 3
NCLDQR = 4
NCLDQS = 5
NCLDQV = 6
NCLDTOP = 1


@dace.program
def inner_loops_7(
        PTSPHY: dace.float64,
        RLMIN: dace.float64,
        ZEPSEC: dace.float64,
        RG: dace.float64,
        RTHOMO: dace.float64,
        ZALFAW: dace.float64,
        PLU_NF: dace.float64[KLON, KLEV],
        LDCUM_NF: dace.int32[KLON],
        PSNDE_NF: dace.float64[KLON, KLEV],
        PAPH_NF: dace.float64[KLON, KLEV+1],
        PSUPSAT_NF: dace.float64[KLON, KLEV],
        PT_NF: dace.float64[KLON, KLEV],
        tendency_tmp_t_NF: dace.float64[KLON, KLEV],
        PLUDE_NF: dace.float64[KLON, KLEV]
        ):

    ZCONVSRCE = np.zeros([KLON, NCLV], dtype=np.float64)
    ZSOLQA = np.zeros([KLON, NCLV, NCLV], dtype=np.float64)
    ZDTGDP = np.zeros([KLON], dtype=np.float64)
    ZDP = np.zeros([KLON], dtype=np.float64)
    ZGDP = np.zeros([KLON], dtype=np.float64)
    ZTP1 = np.zeros([KLON], dtype=np.float64)

    for JK in range(NCLDTOP, KLEV):
        for JL in range(KIDIA, KFDIA):
            ZTP1[JL] = PT_NF[JL, JK] + PTSPHY * tendency_tmp_t_NF[JL, JK]
            if PSUPSAT_NF[JL, JK] > ZEPSEC:
                if ZTP1[JL] > RTHOMO:
                    ZSOLQA[JL, NCLDQL, NCLDQL] = ZSOLQA[JL, NCLDQL, NCLDQL] + PSUPSAT_NF[JL, JK]
                else:
                    ZSOLQA[JL, NCLDQI, NCLDQI] = ZSOLQA[JL, NCLDQI, NCLDQI] + PSUPSAT_NF[JL, JK]

        for JL in range(KIDIA, KFDIA):
            ZDP[JL] = PAPH_NF[JL, JK+1]-PAPH_NF[JL, JK]
            ZGDP[JL] = RG/ZDP[JL]
            ZDTGDP[JL] = PTSPHY*ZGDP[JL]

        for JL in range(KIDIA, KFDIA):
            PLUDE_NF[JL, JK] = PLUDE_NF[JL, JK]*ZDTGDP[JL]
            if LDCUM_NF[JL] and PLUDE_NF[JL, JK] > RLMIN and PLU_NF[JL, JK+1] > ZEPSEC:
                ZCONVSRCE[JL, NCLDQL] = ZALFAW*PLUDE_NF[JL, JK]
                ZCONVSRCE[JL, NCLDQI] = [1.0 - ZALFAW]*PLUDE_NF[JL, JK]
                ZSOLQA[JL, NCLDQL, NCLDQL] = ZSOLQA[JL, NCLDQL, NCLDQL]+ZCONVSRCE[JL, NCLDQL]
                ZSOLQA[JL, NCLDQI, NCLDQI] = ZSOLQA[JL, NCLDQI, NCLDQI]+ZCONVSRCE[JL, NCLDQI]
            else:
                PLUDE_NF[JL, JK] = 0.0

            if LDCUM_NF[JL]:
                ZSOLQA[JL, NCLDQS, NCLDQS] = ZSOLQA[JL, NCLDQS, NCLDQS] + PSNDE_NF[JL, JK]*ZDTGDP[JL]


@dace.program
def vert_loop_7(
            PTSPHY: dace.float64,
            RLMIN: dace.float64,
            ZEPSEC: dace.float64,
            RG: dace.float64,
            RTHOMO: dace.float64,
            ZALFAW: dace.float64,
            PLU_NF: dace.float64[KLON, KLEV, NBLOCKS],
            LDCUM_NF: dace.int32[KLON, NBLOCKS],
            PSNDE_NF: dace.float64[KLON, KLEV, NBLOCKS],
            PAPH_NF: dace.float64[KLON, KLEV+1, NBLOCKS],
            PSUPSAT_NF: dace.float64[KLON, KLEV, NBLOCKS],
            PT_NF: dace.float64[KLON, KLEV, NBLOCKS],
            tendency_tmp_t_NF: dace.float64[KLON, KLEV, NBLOCKS],
            PLUDE_NF: dace.float64[KLON, KLEV, NBLOCKS]
        ):

    for JN in dace.map[0:NBLOCKS:KLON]:
        inner_loops_7(
            KLON,  KLEV,  NCLV,  KIDIA,  KFDIA,  NCLDQS,  NCLDQI,  NCLDQL,  NCLDTOP,
            PTSPHY,  RLMIN,  ZEPSEC,  RG,  RTHOMO,  ZALFAW,  PLU_NF[:, :, JN],  LDCUM_NF[:, JN],  PSNDE_NF[:, :, JN],
            PAPH_NF[:, :, JN], PSUPSAT_NF[:, :, JN],  PT_NF[:, :, JN],  tendency_tmp_t_NF[:, :, JN], PLUDE_NF[:, :, JN])


@dace.program
def vert_loop_7_1(
            PTSPHY: dace.float64,
            RLMIN: dace.float64,
            ZEPSEC: dace.float64,
            RG: dace.float64,
            RTHOMO: dace.float64,
            ZALFAW: dace.float64,
            PLU_NF: dace.float64[KLON, KLEV, NBLOCKS],
            LDCUM_NF: dace.int32[KLON, NBLOCKS],
            PSNDE_NF: dace.float64[KLON, KLEV, NBLOCKS],
            PAPH_NF: dace.float64[KLON, KLEV+1, NBLOCKS],
            PSUPSAT_NF: dace.float64[KLON, KLEV, NBLOCKS],
            PT_NF: dace.float64[KLON, KLEV, NBLOCKS],
            tendency_tmp_t_NF: dace.float64[KLON, KLEV, NBLOCKS],
            PLUDE_NF: dace.float64[KLON, KLEV, NBLOCKS]
        ):

    for JN in dace.map[0:NBLOCKS:KLON]:
        ZCONVSRCE = np.zeros([KLON, NCLV], dtype=np.float64)
        ZSOLQA = np.zeros([KLON, NCLV, NCLV], dtype=np.float64)
        ZDTGDP = np.zeros([KLON], dtype=np.float64)
        ZDP = np.zeros([KLON], dtype=np.float64)
        ZGDP = np.zeros([KLON], dtype=np.float64)
        ZTP1 = np.zeros([KLON], dtype=np.float64)

        for JK in range(NCLDTOP, KLEV):
            for JL in range(KIDIA, KFDIA):
                ZTP1[JL] = PT_NF[JL, JK, JN] + PTSPHY * tendency_tmp_t_NF[JL, JK, JN]
                if PSUPSAT_NF[JL, JK, JN] > ZEPSEC:
                    if ZTP1[JL] > RTHOMO:
                        ZSOLQA[JL, NCLDQL, NCLDQL] = ZSOLQA[JL, NCLDQL, NCLDQL] + PSUPSAT_NF[JL, JK, JN]
                    else:
                        ZSOLQA[JL, NCLDQI, NCLDQI] = ZSOLQA[JL, NCLDQI, NCLDQI] + PSUPSAT_NF[JL, JK, JN]

            for JL in range(KIDIA, KFDIA):
                ZDP[JL] = PAPH_NF[JL, JK+1, JN]-PAPH_NF[JL, JK, JN]
                ZGDP[JL] = RG/ZDP[JL]
                ZDTGDP[JL] = PTSPHY*ZGDP[JL]

            for JL in range(KIDIA, KFDIA):
                PLUDE_NF[JL, JK, JN] = PLUDE_NF[JL, JK, JN]*ZDTGDP[JL]
                if LDCUM_NF[JL, JN] and PLUDE_NF[JL, JK, JN] > RLMIN and PLU_NF[JL, JK+1, JN] > ZEPSEC:
                    ZCONVSRCE[JL, NCLDQL] = ZALFAW*PLUDE_NF[JL, JK, JN]
                    ZCONVSRCE[JL, NCLDQI] = [1.0 - ZALFAW]*PLUDE_NF[JL, JK, JN]
                    ZSOLQA[JL, NCLDQL, NCLDQL] = ZSOLQA[JL, NCLDQL, NCLDQL] + ZCONVSRCE[JL, NCLDQL]
                    ZSOLQA[JL, NCLDQI, NCLDQI] = ZSOLQA[JL, NCLDQI, NCLDQI] + ZCONVSRCE[JL, NCLDQI]
                else:
                    PLUDE_NF[JL, JK, JN] = 0.0

                if LDCUM_NF[JL]:
                    ZSOLQA[JL, NCLDQS, NCLDQS] = ZSOLQA[JL, NCLDQS, NCLDQS] + PSNDE_NF[JL, JK, JN] * ZDTGDP[JL]


kernels = {
    'vert_loop_7': vert_loop_7,
    'vert_loop_7_1': vert_loop_7_1
}


def run_function_dace(f: dace.frontend.python.parser.DaceProgram, nblocks: int):
    rng = np.random.default_rng(42)
    arguments = {}
    for parameter in inspect.signature(f.f).parameters.values():
        # print(type(parameter.annotation))
        if isinstance(parameter.annotation, dace.dtypes.typeclass):
            arguments[parameter.name] = rng.random(dtype=parameter.annotation.dtype.as_numpy_dtype())
        elif isinstance(parameter.annotation, dace.data.Array):
            # types = {'double': np.float64, 'int': np.int32}
            shape = list(parameter.annotation.shape)
            for index, dim in enumerate(shape):
                if isinstance(dim, dace.symbolic.symbol) and dim.name == 'NBLOCKS':
                    shape[index] = nblocks

            dtype = parameter.annotation.dtype.as_numpy_dtype()
            if np.issubdtype(dtype, np.integer):
                arguments[parameter.name] = rng.integers(0, 2, shape, dtype=parameter.annotation.dtype.as_numpy_dtype())
            else:
                arguments[parameter.name] = rng.random(shape, dtype=parameter.annotation.dtype.as_numpy_dtype())

    sdfg = f.to_sdfg(validate=True, simplify=True)
    optimize_sdfg(sdfg, device=dace.DeviceType.GPU)
    csdfg = sdfg.compile()
    arguments_device = copy_to_device(arguments)
    csdfg(**arguments_device, NBLOCKS=nblocks)


def action_run(args):
    print(f"Run {args.program} with NBLOCKS={args.NBLOCKS}")
    run_function_dace(kernels[args.program], args.NBLOCKS)


def action_profile(args):
    cmds = ['ncu', '--set', 'full', '-f', '--export', '/tmp/profile.ncu-rep', 'python3', __file__, 'run',
            args.program, '--NBLOCKS', str(args.NBLOCKS)]
    run(cmds)
    action = get_action('/tmp/profile.ncu-rep')
    bw = get_achieved_performance(action)[1] / get_peak_performance(action)[1]
    print(f"BW: {bw}")


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Commands",
        help="See the help of the respective command")

    run_parser = subparsers.add_parser('run', description='Run a given kernel')
    run_parser.add_argument('program', type=str)
    run_parser.add_argument('--NBLOCKS', type=int)
    run_parser.set_defaults(func=action_run)

    profile_parser = subparsers.add_parser('profile', description='Profile a given kernel')
    profile_parser.add_argument('program', type=str)
    profile_parser.add_argument('--NBLOCKS', type=int)
    profile_parser.set_defaults(func=action_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
