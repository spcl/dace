from argparse import ArgumentParser
import inspect
from subprocess import run
from numbers import Number
import copy
import os
from typing import Dict, Union, Tuple
from tabulate import tabulate
import dace
import numpy as np
import cupy as cp
import pandas as pd
import sympy

from utils.general import optimize_sdfg, copy_to_device, use_cache, enable_debug_flags
from utils.ncu import get_all_actions_filtered, get_achieved_performance, get_peak_performance, get_achieved_bytes, \
                      get_runtime
from utils.paths import get_thesis_playground_root_dir, get_playground_results_dir

NBLOCKS = dace.symbol('NBLOCKS')
KLEV = dace.symbol('KLEV')
KLON = 1
NCLV = dace.symbol('NCLV')
# NCLV = 10
KIDIA = 0
KFDIA = 1
NCLDQI = 2
NCLDQL = 3
NCLDQR = 4
NCLDQS = 5
NCLDQV = 6
NCLDTOP = 0

params = {'KLEV': 137, 'NCLV': 10}
df_index_cols = ['program', 'NBLOCKS', 'description']


@dace.program
def kernel_1(inp1: dace.float64[KLEV, NBLOCKS], out1: dace.float64[KLEV, NBLOCKS]):
    tmp = np.zeros([2, NBLOCKS], dtype=np.float64)
    for i in dace.map[0:NBLOCKS]:
        for j in range(1, KLEV):
            tmp[j % 2, i] = (inp1[j, i] + inp1[j - 1, i])
            out1[j, i] = (tmp[j % 2, i] + tmp[(j-1) % 2, i])


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

    for JK in range(NCLDTOP, KLEV-1):
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

        for JK in range(NCLDTOP, KLEV-1):
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

                if LDCUM_NF[JL, JN]:
                    ZSOLQA[JL, NCLDQS, NCLDQS] = ZSOLQA[JL, NCLDQS, NCLDQS] + PSNDE_NF[JL, JK, JN] * ZDTGDP[JL]


@dace.program
def vert_loop_7_1_no_klon(
            PTSPHY: dace.float64,
            RLMIN: dace.float64,
            ZEPSEC: dace.float64,
            RG: dace.float64,
            RTHOMO: dace.float64,
            ZALFAW: dace.float64,
            PLU_NF: dace.float64[KLEV, NBLOCKS],
            LDCUM_NF: dace.int32[NBLOCKS],
            PSNDE_NF: dace.float64[KLEV, NBLOCKS],
            PAPH_NF: dace.float64[KLEV+1, NBLOCKS],
            PSUPSAT_NF: dace.float64[KLEV, NBLOCKS],
            PT_NF: dace.float64[KLEV, NBLOCKS],
            tendency_tmp_t_NF: dace.float64[KLEV, NBLOCKS],
            PLUDE_NF: dace.float64[KLEV, NBLOCKS]
        ):

    for JN in dace.map[0:NBLOCKS]:
        ZCONVSRCE = np.zeros([NCLV], dtype=np.float64)
        ZSOLQA = np.zeros([NCLV, NCLV], dtype=np.float64)
        ZDTGDP = 0.0
        ZDP = 0.0
        ZGDP = 0.0
        ZTP1 = 0.0

        for JK in range(NCLDTOP, KLEV-1):
            ZTP1 = PT_NF[JK, JN] + PTSPHY * tendency_tmp_t_NF[JK, JN]
            if PSUPSAT_NF[JK, JN] > ZEPSEC:
                if ZTP1 > RTHOMO:
                    ZSOLQA[NCLDQL, NCLDQL] = ZSOLQA[NCLDQL, NCLDQL] + PSUPSAT_NF[JK, JN]
                else:
                    ZSOLQA[NCLDQI, NCLDQI] = ZSOLQA[NCLDQI, NCLDQI] + PSUPSAT_NF[JK, JN]

            ZDP = PAPH_NF[JK+1, JN]-PAPH_NF[JK, JN]
            ZGDP = RG/ZDP
            ZDTGDP = PTSPHY*ZGDP

            PLUDE_NF[JK, JN] = PLUDE_NF[JK, JN]*ZDTGDP
            if LDCUM_NF[JN] and PLUDE_NF[JK, JN] > RLMIN and PLU_NF[JK+1, JN] > ZEPSEC:
                ZCONVSRCE[NCLDQL] = ZALFAW*PLUDE_NF[JK, JN]
                ZCONVSRCE[NCLDQI] = [1.0 - ZALFAW]*PLUDE_NF[JK, JN]
                ZSOLQA[NCLDQL, NCLDQL] = ZSOLQA[NCLDQL, NCLDQL] + ZCONVSRCE[NCLDQL]
                ZSOLQA[NCLDQI, NCLDQI] = ZSOLQA[NCLDQI, NCLDQI] + ZCONVSRCE[NCLDQI]
            else:
                PLUDE_NF[JK, JN] = 0.0

            if LDCUM_NF[JN]:
                ZSOLQA[NCLDQS, NCLDQS] = ZSOLQA[NCLDQS, NCLDQS] + PSNDE_NF[JK, JN] * ZDTGDP


@dace.program
def vert_loop_wip(
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

        for JK in range(NCLDTOP, KLEV-1):
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

                if LDCUM_NF[JL, JN]:
                    ZSOLQA[JL, NCLDQS, NCLDQS] = ZSOLQA[JL, NCLDQS, NCLDQS] + PSNDE_NF[JL, JK, JN] * ZDTGDP[JL]


kernels = {
    'vert_loop_7': vert_loop_7,
    'vert_loop_7_1': vert_loop_7_1,
    'vert_loop_7_1_no_klon': vert_loop_7_1_no_klon,
    'vert_loop_wip': vert_loop_wip,
    'kernel_1': kernel_1,
}


def eval_argument_shape(parameter: inspect.Parameter, params: Dict[str, int]) -> Tuple[int]:
    """
    Evaluate the shape of the given parameter/argument for an array

    :param parameter: The parameter/argument
    :type parameter: inspect.Parameter
    :param params: The parameters used to evaulate
    :type params: Dict[str, int]
    :return: The shape
    :rtype: Tuple[int]
    """
    shape = list(parameter.annotation.shape)
    for index, dim in enumerate(shape):
        if isinstance(dim, sympy.core.expr.Expr):
            shape[index] = int(dim.evalf(subs=params))
    return shape


def gen_arguments(f: dace.frontend.python.parser.DaceProgram,
                  params: Dict[str, int]) -> Dict[str, Union[np.ndarray, Number]]:
    """
    Generates the neccessary arguments to call the given function

    :param f: The DaceProgram
    :type f: dace.frontend.python.parser.DaceProgram
    :param params: Values for symbols
    :type params: Dict[str, int]
    :return: Dict, keys are argument names, values are argument values
    :rtype: Dict[str, Union[nd.array, Number]]
    """
    rng = np.random.default_rng(42)
    arguments = {}
    for parameter in inspect.signature(f.f).parameters.values():
        if isinstance(parameter.annotation, dace.dtypes.typeclass):
            arguments[parameter.name] = rng.random(dtype=parameter.annotation.dtype.as_numpy_dtype())
        elif isinstance(parameter.annotation, dace.data.Array):
            shape = eval_argument_shape(parameter, params)
            dtype = parameter.annotation.dtype.as_numpy_dtype()
            if np.issubdtype(dtype, np.integer):
                arguments[parameter.name] = rng.integers(0, 2, shape, dtype=parameter.annotation.dtype.as_numpy_dtype())
            else:
                arguments[parameter.name] = rng.random(shape, dtype=parameter.annotation.dtype.as_numpy_dtype())
    return arguments


def get_size_of_parameters(dace_f: dace.frontend.python.parser.DaceProgram, params: Dict[str, int]) -> int:
    size = 0
    for name, parameter in inspect.signature(dace_f.f).parameters.items():
        if isinstance(parameter.annotation, dace.dtypes.typeclass):
            size += 1
        elif isinstance(parameter.annotation, dace.data.Array):
            shape = eval_argument_shape(parameter, params)
            size += np.prod(shape)
            # print(f"{name:15} ({shape}) adds {np.prod(shape):12,} bytes. New size {size:12,}")
    return int(size * 8)


def run_function_dace(f: dace.frontend.python.parser.DaceProgram, params: Dict[str, int], save_graphs: bool = False):
    sdfg = f.to_sdfg(validate=True, simplify=True)
    if save_graphs:
        optimize_sdfg(sdfg, device=dace.DeviceType.GPU, verbose_name=f"py_{f.name}")
    else:
        optimize_sdfg(sdfg, device=dace.DeviceType.GPU)
    csdfg = sdfg.compile()
    arguments = gen_arguments(f, params)
    arguments_device = copy_to_device(arguments)
    csdfg(**arguments_device, **params)


def action_run(args):
    print(f"Run {args.program} with NBLOCKS={args.NBLOCKS:,}", end="")
    if args.debug:
        print(" in DEBUG mode", end="")
        enable_debug_flags()
    if args.cache:
        use_cache(dacecache_folder=args.program)
    print()
    params.update({'NBLOCKS': args.NBLOCKS})
    run_function_dace(kernels[args.program], params, args.save_graphs)


def action_profile(args):
    data = []
    params.update({'NBLOCKS': args.NBLOCKS})

    # collect data
    for program in args.programs:
        report_path = '/tmp/profile.ncu-rep'
        if args.ncu_report:
            report_path = os.path.join(get_thesis_playground_root_dir(), 'ncu-reports',
                                       f"report_py_{program}_{args.NBLOCKS:.1e}.ncu-rep")
            print(f"Save ncu report into {report_path}")
        cmds = ['ncu', '--set', 'full', '-f', '--export', report_path, 'python3', __file__, 'run',
                program, '--NBLOCKS', str(args.NBLOCKS)]
        if args.cache:
            cmds.append('--cache')
        run(cmds)
        actions = get_all_actions_filtered(report_path, '_numpy_full_')
        action = actions[0]
        if len(actions) > 1:
            print(f"WARNING: More than one action, taking first {action}")
        upper_Q = get_size_of_parameters(kernels[program], params)
        D = get_achieved_bytes(action)
        bw = get_achieved_performance(action)[1] / get_peak_performance(action)[1]
        T = get_runtime(action)
        data.append([program, D, bw, upper_Q, T])

    # Print data
    print(tabulate(data, headers=['program', 'D', 'BW [ratio]', 'upper limit Q', 'T [s]'], intfmt=',', floatfmt='.3f'))

    # save data
    if args.export is not None:
        datafile = os.path.join(get_playground_results_dir(), 'python', args.export)
        if not args.export.endswith('.csv'):
            datafile = f"{datafile}.csv"
        print(f"Save data into {datafile}")
        for row in data:
            row.append(args.NBLOCKS)
            if args.description is None:
                print("WARNING: No description set. It is recommended to do so when saving the results")
                row.append('N.A.')
            else:
                row.append(args.description)

        # create dataframe
        this_df = pd.DataFrame(data, columns=['program', 'D', 'bw_ratio', 'upper_Q', 'T', 'NBLOCKS', 'description'])
        this_df.set_index(df_index_cols, inplace=True)

        # create folder if neccessary
        if not os.path.exists(os.path.dirname(datafile)):
            os.makedirs(os.path.dirname(datafile))

        # read any existing data, append and write again
        if os.path.exists(datafile):
            df = pd.read_csv(datafile, index_col=df_index_cols)
            df = pd.concat([df, this_df])
        else:
            df = this_df
        df.to_csv(datafile)


def action_test(args):
    params.update({'NBLOCKS': 10, 'KLEV': 137})
    for program in args.programs:
        if args.cache:
            use_cache(dacecache_folder=program)
        dace_f = kernels[program]
        arguments = gen_arguments(dace_f, params)
        # Does not work because it can not convert the symbol NBLOCKS
        dace_f.f(**arguments)

        arguments_device = copy_to_device(copy.deepcopy(arguments))
        sdfg = dace_f.to_sdfg(validate=True, simplify=True)
        optimize_sdfg(sdfg, device=dace.DeviceType.GPU)
        csdfg = sdfg.compile()
        csdfg(**arguments, **params)

        assert cp.allclose(cp.asarray(arguments), arguments_device)


def action_print(args):
    joined_df = pd.DataFrame()
    for file in args.files:
        if not file.endswith('.csv'):
            file = f"{file}.csv"
        path = os.path.join(get_playground_results_dir(), 'python', file)
        df = pd.read_csv(path, index_col=df_index_cols)
        joined_df = pd.concat([joined_df, df])

    print(tabulate(joined_df.reset_index(), tablefmt='pipe', showindex=False, intfmt=',',
                   headers=['program', 'NBLOCKS', 'description', 'transferred bytes', 'Bandwidth [ratio]',
                            'Upper limit of Q', 'runtime [s]']))


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Commands",
        help="See the help of the respective command")

    run_parser = subparsers.add_parser('run', description='Run a given kernel')
    run_parser.add_argument('program', type=str)
    run_parser.add_argument('--NBLOCKS', type=int, default=int(1e5))
    run_parser.add_argument('--cache', action='store_true', default=False)
    run_parser.add_argument('--debug', action='store_true', default=False)
    run_parser.add_argument('--save-graphs', action='store_true', default=False)
    run_parser.set_defaults(func=action_run)

    profile_parser = subparsers.add_parser('profile', description='Profile given kernels')
    profile_parser.add_argument('programs', type=str, nargs='+')
    profile_parser.add_argument('--cache', action='store_true', default=False)
    profile_parser.add_argument('--NBLOCKS', type=int, default=int(1e5))
    profile_parser.add_argument('--ncu-report', action='store_true', default=False)
    profile_parser.add_argument('--export', default=None, type=str,
                                help=f"Save into the given filename in {get_playground_results_dir()}/python."
                                     f"Will append if File exists. Will store in csv format")
    profile_parser.add_argument('--description', default=None, type=str, help='Description of the specific run')
    profile_parser.set_defaults(func=action_profile)

    test_parser = subparsers.add_parser('test', description='Test given kernels')
    test_parser.add_argument('programs', type=str, nargs='+')
    test_parser.add_argument('--cache', action='store_true', default=False)
    test_parser.set_defaults(func=action_test)

    print_parser = subparsers.add_parser('print', description='Print saved results')
    print_parser.add_argument('files', type=str, nargs='+', help=f"Files in {get_playground_results_dir()}/python")
    print_parser.set_defaults(func=action_print)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
