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
}


program_parameters = {
    'cloudsc_class1_658': ('KLON', 'KLEV', 'NCLV', 'KIDIA', 'KFDIA', 'NCLDQV'),
}


program_inputs = {
    'cloudsc_class1_658': ('PTSPHY', 'tendency_tmp_t', 'tendency_tmp_q', 'tendency_tmp_a', 'PT', 'PQ', 'PA'),
}


program_outputs = {
    'cloudsc_class1_658': ('ZTP1', 'ZQX', 'ZQX0', 'ZA', 'ZAORIG'),
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
    'RKOOP1': (0,),
    'RKOOP2': (0,),
    'RKOOPTAU': (0,),
    'RLMIN': (0,),
    'RLSTT': (0,),
    'RLVTT': (0,),
    'RTHOMO': (0,),
    'RTT': (0,),
    'ZEPSEC': (0,),
    'ZEPSILON': (0,),
    'ZRG_R': (0,),
    'ZQTMST': (0,),
    'IPHASE': (parameters['NCLV'],),
    'PAPH': (parameters['KLON'], parameters['KLEV']+1),
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
    'ZA': (parameters['KLON'], parameters['KLEV']),
    'ZAORIG': (parameters['KLON'], parameters['KLEV']),
    'ZCONVSINK': (parameters['KLON'], parameters['NCLV']),
    'ZCONVSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZCORQSICE': (parameters['KLON']),
    'ZCOVPTOT': (parameters['KLON'],),
    'ZDA': (parameters['KLON']),
    'ZFALLSINK': (parameters['KLON'], parameters['NCLV']),
    'ZFALLSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZFLUXQ': (parameters['KLON'], parameters['NCLV']),
    'ZFOEALFA': (parameters['KLON'], parameters['KLEV']+1),
    'ZICEFRAC': (parameters['KLON'], parameters['KLEV']),
    'ZLI': (parameters['KLON'], parameters['KLEV']),
    'ZLIQFRAC': (parameters['KLON'], parameters['KLEV']),
    'ZLNEG': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZPFPLSX': (parameters['KLON'], parameters['KLEV']+1, parameters['NCLV']),
    'ZPSUPSATSRCE': (parameters['KLON'], parameters['NCLV']),
    'ZSOLQA': (parameters['KLON'], parameters['NCLV'], parameters['NCLV']),
    'ZQSICE': (parameters['KLON'], parameters['KLEV']),
    'ZQX': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZQX0': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZQXFG': (parameters['KLON'], parameters['NCLV']),
    'ZQXN': (parameters['KLON'], parameters['NCLV']),
    'ZQXN2D': (parameters['KLON'], parameters['KLEV'], parameters['NCLV']),
    'ZTP1': (parameters['KLON'], parameters['KLEV']),
    'PT': (parameters['KLON'], parameters['KLEV']),
    'PQ': (parameters['KLON'], parameters['KLEV']),
    'PA': (parameters['KLON'], parameters['KLEV']),
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
    sdfg = get_sdfg(fsource, program_name, normalize_memlets)
    if device == dace.DeviceType.GPU:
        auto_optimize(sdfg, device)

    rng = np.random.default_rng(42)
    inputs = get_inputs(program, rng)
    outputs_f = get_outputs(program, rng)
    outputs_d = copy.deepcopy(outputs_f)
    sdfg.validate()
    sdfg.save("graph.sdfg")
    sdfg.simplify(validate_all=True)

    ffunc(**{k.lower(): v for k, v in inputs.items()}, **{k.lower(): v for k, v in outputs_f.items()})
    sdfg(**inputs, **outputs_d)

    for k in outputs_f.keys():
        farr = outputs_f[k]
        darr = outputs_f[k]
        assert np.allclose(farr, darr)


if __name__=="__main__":
    test_program('cloudsc_class1_658', dace.DeviceType.CPU, False)