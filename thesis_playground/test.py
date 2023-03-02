import numpy as np
from numpy import f2py
from numbers import Number
from typing import Dict, Union
import copy
from importlib import import_module
import os
import sys
import tempfile

import dace
from dace.frontend.fortran import fortran_parser
from dace.sdfg import utils
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes import RemoveUnusedSymbols, ScalarToSymbolPromotion


programs = {
    'cloudsc_class1_658': 'non_clv_init',
    'cloudsc_class1_670': 'clv_init',
    'cloudsc_class1_2783': 'copy_precipitation_flux',
    'cloudsc_class1_2857': 'enthalpy_flux_due_to_precipitation',
    'cloudsc_class2_781': 'liq_ice_fractions',
    'cloudsc_class2_1762': 'precipitation_cover_overlap',
    'cloudsc_class2_1001': 'ice_supersaturation_adjustment',
    'cloudsc_class2_1516': 'ice_growth_vapour_deposition',
    'cloudsc_class3_691': 'tidy_up_cloud_cover',
    'cloudsc_class3_965': 'evaporate_small_liquid_ice',
    'cloudsc_class3_1985': 'melting_snow_ice',
}


program_parameters = {
    'cloudsc_class1_658': ('KLON', 'KLEV', 'NCLV', 'KIDIA', 'KFDIA', 'NCLDQV'),
    'cloudsc_class1_670': ('KLON', 'KLEV', 'NCLV', 'KIDIA', 'KFDIA'),
    'cloudsc_class1_2783': ('KLON', 'KLEV', 'NCLV', 'KIDIA', 'KFDIA', 'NCLDQR', 'NCLDQS', 'NCLDQL', 'NCLDQI'),
    'cloudsc_class1_2857': ('KLON', 'KLEV', 'KIDIA', 'KFDIA'),
    'cloudsc_class2_781': ('KLON', 'KLEV', 'NCLV', 'KIDIA', 'KFDIA', 'NCLDQI', 'NCLDQL'),
    'cloudsc_class2_1762': ('KLON', 'KLEV', 'KIDIA', 'KFDIA', 'NCLV', 'NCLDTOP', 'NCLDQS', 'NCLDQR'),
    'cloudsc_class2_1001': ('KLON', 'KLEV', 'KIDIA', 'KFDIA', 'NCLV', 'NCLDTOP', 'NCLDQI', 'NCLDQL', 'NCLDQV',
        'NSSOPT'),
    'cloudsc_class2_1516': ('KLON', 'KLEV', 'KIDIA', 'KFDIA', 'NCLV', 'NCLDTOP', 'NCLDQL', 'NCLDQI'),
    'cloudsc_class3_691': ('KLON', 'KLEV', 'NCLV', 'KIDIA', 'KFDIA', 'NCLDQV', 'NCLDQI', 'NCLDQL'),
    'cloudsc_class3_965': ('KLON', 'KLEV', 'NCLV', 'KIDIA', 'KFDIA', 'NCLDQV', 'NCLDQI', 'NCLDQL', 'NCLDTOP'),
    'cloudsc_class3_1985': ('KLON', 'KLEV', 'NCLV', 'KIDIA', 'KFDIA', 'NCLDQV', 'NCLDQS','NCLDQI', 'NCLDQL', 'NCLDTOP'),
}


program_inputs = {
    'cloudsc_class1_658': ('PTSPHY', 'tendency_tmp_t', 'tendency_tmp_q', 'tendency_tmp_a', 'PT', 'PQ', 'PA'),
    'cloudsc_class1_670': ('PCLV', 'PTSPHY', 'tendency_tmp_cld'),
    'cloudsc_class1_2783': ('ZPFPLSX',),
    'cloudsc_class1_2857': ('RLVTT', 'RLSTT', 'PFPLSL', 'PFPLSN'),
    'cloudsc_class2_781': ('RLMIN', 'ZQX'),
    'cloudsc_class2_1762': ('RCOVPMIN', 'ZEPSEC', 'ZQXFG', 'ZA', 'ZQPRETOT'),
    'cloudsc_class2_1001': ('PTSPHY', 'RAMIN', 'RKOOP1', 'RKOOP2', 'RKOOPTAU', 'R2ES', 'R3IES', 'R3LES', 'R4IES',
        'R4LES', 'RTHOMO', 'RTT', 'ZEPSEC', 'ZEPSILON', 'PSUPSAT', 'ZA', 'ZCORQSICE', 'ZQSICE', 'ZQX', 'ZTP1'),
    'cloudsc_class2_1516': ('RTT', 'R2ES', 'R3IES', 'R4IES', 'RLMIN', 'RV', 'RD', 'RG', 'RLSTT', 'RDEPLIQREFRATE',
        'RDEPLIQREFDEPTH', 'RCLDTOPCF', 'PTSPHY', 'RICEINIT', 'ZA', 'ZCLDTOPDIST', 'ZDP', 'ZRHO', 'ZTP1', 'ZFOKOOP', 'PAP',
        'ZICECLD'),
    'cloudsc_class3_691': ('RLMIN', 'RAMIN', 'ZQTMST', 'RALVDCP', 'RALSDCP', 'ZA'),
    'cloudsc_class3_965': ('RLMIN', 'ZQX'),
    'cloudsc_class3_1985': ('ZEPSEC', 'RTT', 'PTSPHY', 'ZRLDCP', 'RTAUMEL', 'ZQSICE', 'ZQXFG', 'ZTP1', 'PAP', 'ZQX'),
}


program_outputs = {
    'cloudsc_class1_658': ('ZTP1', 'ZQX', 'ZQX0', 'ZA', 'ZAORIG'),
    'cloudsc_class1_670': ('ZQX', 'ZQX0'),
    'cloudsc_class1_2783': ('PFPLSL', 'PFPLSN'),
    'cloudsc_class1_2857': ('PFHPSL', 'PFHPSN'),
    'cloudsc_class2_781': ('ZA', 'ZLIQFRAC', 'ZICEFRAC', 'ZLI'),
    'cloudsc_class2_1762': ('ZCOVPTOT', 'ZCOVPCLR', 'ZCOVPMAX', 'ZRAINCLD', 'ZSNOWCLD'),
    'cloudsc_class2_1001': ('ZSOLQA', 'ZQXFG', 'ZSOLAC', 'ZFOKOOP', 'ZPSUPSATSRCE', 'ZSUPSAT'),
    'cloudsc_class2_1516': ('ZCOVPTOT', 'ZICENUCLEI', 'ZSOLQA', 'ZQXFG'),
    'cloudsc_class3_691': ('ZQX', 'ZLNEG', 'tendency_loc_q', 'tendency_loc_T'),
    'cloudsc_class3_965': ('ZSOLQA',),
    'cloudsc_class3_1985': ('ZICETOT', 'ZMELTMAX'),
}

# Copied from tests/fortran/cloudsc.py as well as the functions/dicts below
def read_source(filename: str, extension: str = 'f90') -> str:
    source = None
    with open(os.path.join(os.path.dirname(__file__), f'{filename}.{extension}'), 'r') as file:
        source = file.read()
    assert source
    return source


def get_fortran(source: str, program_name: str, subroutine_name: str, fortran_extension: str = '.f90'):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        f2py.compile(source, modulename=program_name, verbose=True, extension=fortran_extension)
        sys.path.append(tmp_dir)
        module = import_module(program_name)
        function = getattr(module, subroutine_name)
        os.chdir(cwd)
        return function


def get_sdfg(source: str, program_name: str, normalize_offsets: bool = False) -> dace.SDFG:

    intial_sdfg = fortran_parser.create_sdfg_from_string(source, program_name)
    
    # Find first NestedSDFG
    sdfg = None
    for state in intial_sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                sdfg = node.sdfg
                break
    if not sdfg:
        raise ValueError("SDFG not found.")

    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()

    if normalize_offsets:
        my_simplify = Pipeline([RemoveUnusedSymbols(), ScalarToSymbolPromotion()])
    else:
        my_simplify = Pipeline([RemoveUnusedSymbols()])
    my_simplify.apply_pass(sdfg, {})

    if normalize_offsets:
        utils.normalize_offsets(sdfg)

    return sdfg


def remove_custom_literal_constants(source: str, custom_type: str) -> str:
    """
    Removes all occurances of literal constants of the given custom type

    :param source: The source code where to remove the literal constants in
    :type source: str
    :return: The name of the custom type
    :rtype: str
    """
    return source.replace(f'_{custom_type}', '_<empty>')


parameters = {
    'KLON': 10,
    'KLEV': 10,
    'KIDIA': 2,
    'KFDIA': 8,
    'NCLV': 10,
    'NCLDQI': 3,
    'NCLDQL': 4,
    'NCLDQR': 5,
    'NCLDQS': 6,
    'NCLDQV': 7,
    'NCLDTOP': 1,
    'NSSOPT': 1,
}


data = {
    'PTSPHY': (0,),
    'R2ES': (0,),
    'R3IES': (0,),
    'R3LES': (0,),
    'R4IES': (0,),
    'R4LES': (0,),
    'RALSDCP': (0,),
    'RALVDCP': (0,),
    'RAMIN': (0,),
    'RCOVPMIN': (0,),
    'RCLDTOPCF': (0,),
    'RD': (0,),
    'RDEPLIQREFDEPTH': (0,),
    'RDEPLIQREFRATE': (0,),
    'RG': (0,),
    'RICEINIT': (0,),
    'RKOOP1': (0,),
    'RKOOP2': (0,),
    'RKOOPTAU': (0,),
    'RLMIN': (0,),
    'RLSTT': (0,),
    'RLVTT': (0,),
    'RPRECRHMAX': (0,),
    'RTAUMEL': (0,),
    'RTHOMO': (0,),
    'RTT': (0,),
    'RV': (0,),
    'RVRFACTOR': (0,),
    'ZEPSEC': (0,),
    'ZEPSILON': (0,),
    'ZRG_R': (0,),
    'ZRLDCP': (0,),
    'ZQTMST': (0,),
    'ZVPICE': (0,),
    'ZVPLIQ': (0,),
    'IPHASE': (parameters['NCLV'],),
    'PAPH': (parameters['KLON'], parameters['KLEV']+1),
    'PAP': (parameters['KLON'], parameters['KLEV']),
    'PCOVPTOT': (parameters['KLON'], parameters['KLEV']),
    'PFCQLNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFCQNNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFCQRNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFCQSNG': (parameters['KLON'], parameters['KLEV']+1),
    'PFHPSL': (parameters['KLON'], parameters['KLEV']+1),
    'PFHPSN': (parameters['KLON'], parameters['KLEV']+1),
    'PFPLSL': (parameters['KLON'], parameters['KLEV']+1),
    'PFPLSN': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQIF': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQITUR': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQLF': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQLTUR': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQRF': (parameters['KLON'], parameters['KLEV']+1),
    'PFSQSF': (parameters['KLON'], parameters['KLEV']+1),
    'PLUDE': (parameters['KLON'], parameters['KLEV']),
    'PSUPSAT': (parameters['KLON'], parameters['KLEV']),
    'PVFI': (parameters['KLON'], parameters['KLEV']),
    'PVFL': (parameters['KLON'], parameters['KLEV']),
    'tendency_loc_a': (parameters['KLON'], parameters['KLEV']),
    'tendency_loc_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'tendency_loc_q': (parameters['KLON'], parameters['KLEV']),
    'tendency_loc_T': (parameters['KLON'], parameters['KLEV']),
    'tendency_tmp_t': (parameters['KLON'], parameters['KLEV']),
    'tendency_tmp_q': (parameters['KLON'], parameters['KLEV']),
    'tendency_tmp_a': (parameters['KLON'], parameters['KLEV']),
    'tendency_tmp_cld': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZA': (parameters['KLON'], parameters['KLEV']),
    'ZAORIG': (parameters['KLON'], parameters['KLEV']),
    'ZCLDTOPDIST': (parameters['KLON'],),
    'ZCONVSINK': (parameters['KLON'], parameters['NCLV']),
    'ZCONVSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZCORQSICE': (parameters['KLON']),
    'ZCOVPTOT': (parameters['KLON'],),
    'ZCOVPCLR': (parameters['KLON'],),
    'ZCOVPMAX': (parameters['KLON'],),
    'ZDTGDP': (parameters['KLON'],),
    'ZICENUCLEI': (parameters['KLON'],),
    'ZRAINCLD': (parameters['KLON'],),
    'ZSNOWCLD': (parameters['KLON'],),
    'ZDA': (parameters['KLON'],),
    'ZDP': (parameters['KLON'],),
    'ZRHO': (parameters['KLON'],),
    'ZFALLSINK': (parameters['KLON'], parameters['NCLV']),
    'ZFALLSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZFOKOOP': (parameters['KLON'],),
    'ZFLUXQ': (parameters['KLON'], parameters['NCLV']),
    'ZFOEALFA': (parameters['KLON'], parameters['KLEV']+1),
    'ZICECLD': (parameters['KLON'],),
    'ZICEFRAC': (parameters['KLON'], parameters['KLEV']),
    'ZICETOT': (parameters['KLON'],),
    'ZLI': (parameters['KLON'], parameters['KLEV']),
    'ZLIQFRAC': (parameters['KLON'], parameters['KLEV']),
    'ZLNEG': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZPFPLSX': (parameters['KLON'], parameters['KLEV']+1, parameters['NCLV']),
    'ZPSUPSATSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZMELTMAX': (parameters['KLON'],),
    'ZQPRETOT': (parameters['KLON'],),
    'ZQSLIQ': (parameters['KLON'], parameters['KLEV']),
    'ZQSICE': (parameters['KLON'], parameters['KLEV']),
    'ZQX': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZQX0': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZQXFG': (parameters['KLON'], parameters['NCLV']),
    'ZQXN': (parameters['KLON'], parameters['NCLV']),
    'ZQXN2D': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZSOLAC': (parameters['KLON'],),
    'ZSOLQA': (parameters['KLON'], parameters['NCLV'], parameters['NCLV']),
    'ZSUPSAT': (parameters['KLON'],),
    'ZTP1': (parameters['KLON'], parameters['KLEV']),
    'PT': (parameters['KLON'], parameters['KLEV']),
    'PQ': (parameters['KLON'], parameters['KLEV']),
    'PA': (parameters['KLON'], parameters['KLEV']),
    'PCLV': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
}

# Copied from tests/fortran/cloudsc.py
def get_inputs(program: str, rng: np.random.Generator) -> Dict[str, Union[Number, np.ndarray]]:
    inp_data = dict()
    for p in program_parameters[program]:
        inp_data[p] = parameters[p]
    for inp in program_inputs[program]:
        shape = data[inp]
        if shape == (0,):  # Scalar
            inp_data[inp] = rng.random()
        else:
            inp_data[inp] = np.asfortranarray(rng.random(shape))
    return inp_data


# Copied from tests/fortran/cloudsc.py
# TODO: Init to 0?
def get_outputs(program: str, rng: np.random.Generator) -> Dict[str, Union[Number, np.ndarray]]:
    out_data = dict()
    for out in program_outputs[program]:
        shape = data[out]
        if shape == (0,):  # Scalar
            raise NotImplementedError
        else:
            out_data[out] = np.asfortranarray(rng.random(shape))
    return out_data


# Copied from tests/fortran/cloudsc.py
def test_program(program: str, device: dace.DeviceType, normalize_memlets: bool):

    fsource = read_source(program)
    program_name = programs[program]
    routine_name = f'{program_name}_routine'
    ffunc = get_fortran(fsource, program_name, routine_name)
    # sdfg = get_sdfg(remove_custom_literal_constants(fsource, 'JPRB'), program_name, normalize_memlets)
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    if device == dace.DeviceType.GPU:
        auto_optimize(sdfg, device)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng)
    outputs_f = get_outputs(program, rng)
    outputs_d = copy.deepcopy(outputs_f)
    sdfg.validate()
    sdfg.simplify(validate_all=True)
    sdfg.save("graph.sdfg")

    ffunc(**{k.lower(): v for k, v in inputs.items()}, **{k.lower(): v for k, v in outputs_f.items()})
    sdfg(**inputs, **outputs_d)

    print(program)
    for k in outputs_f.keys():
        farr = outputs_f[k]
        darr = outputs_f[k]
        assert np.allclose(farr, darr)
        print(f"variable {k:20} ", end="")
        print(f"Sum: {farr.sum():.2e}", end=", ")
        print(f"avg: {np.average(farr):.2e}", end=", ")
        print(f"median: {np.median(farr):.2e}", end=", ")
        print(f"nnz: {np.count_nonzero(farr)}", end=", ")
        print(f"#: {np.prod(farr.shape)}")
    print("Success")


if __name__=="__main__":
    # does currently not work, due to errors/problems when simplifying the SDFG
    # test_program('cloudsc_class1_658', dace.DeviceType.CPU, False)
    # test_program('cloudsc_class1_670', dace.DeviceType.CPU, False)
    # test_program('cloudsc_class1_2783', dace.DeviceType.CPU, False)
    # test_program('cloudsc_class1_2857', dace.DeviceType.CPU, False)
    # test_program('cloudsc_class2_781', dace.DeviceType.CPU, False)
    # test_program('cloudsc_class2_1762', dace.DeviceType.CPU, False)
    # test_program('cloudsc_class2_1001', dace.DeviceType.CPU, False)
    # test_program('cloudsc_class2_1516', dace.DeviceType.CPU, False)
    # test_program('cloudsc_class3_691', dace.DeviceType.CPU, False)
    # test_program('cloudsc_class3_965', dace.DeviceType.CPU, False)
    # test_program('cloudsc_class3_1985', dace.DeviceType.CPU, False)

    # does currently not work, due to errors/problems when simplifying the SDFG
    # test_program('cloudsc_class1_658', dace.DeviceType.CPU, False)
    # Does not work:
    # Fails to find /usr/local/cuda/include/cooperative_groups.h which is not present on this ault (06) node, but on the
    # login node
    # test_program('cloudsc_class1_670', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class1_2783', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class1_2857', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class2_781', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class2_1762', dace.DeviceType.GPU, False)
    # Does not work
    test_program('cloudsc_class2_1001', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class2_1516', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class3_691', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class3_965', dace.DeviceType.GPU, False)
    # test_program('cloudsc_class3_1985', dace.DeviceType.GPU, False)
