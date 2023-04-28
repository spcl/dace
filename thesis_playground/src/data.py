import copy
from typing import Dict, Union, Tuple, List
import numpy as np
from numbers import Number
from print_utils import print_with_time

parameters = {
    'KLON': 10000,
    'KLEV': 10000,
    'KIDIA': 2,
    'KFDIA': 9998,
    'NCLV': 10,
    'NCLDQI': 3,
    'NCLDQL': 4,
    'NCLDQR': 5,
    'NCLDQS': 6,
    'NCLDQV': 7,
    'NCLDTOP': 2,
    'NSSOPT': 1,
    'NPROMA': 1,
    'NBLOCKS': 10000,
}

# changes from the parameters dict for certrain programs
custom_parameters = {
    'cloudsc_class1_658': {
        'KLON': 5000,
        'KLEV': 5000,
        'KFDIA': 4998,
    },
    'cloudsc_class1_670': {
        'KLON': 1000,
        'KLEV': 1000,
        'KFDIA': 998,
    },
    'cloudsc_class2_781': {
        'KLON': 5000,
        'KLEV': 5000,
        'KFDIA': 4998
    },
    'my_test': {
        'KLON': 100000000
    },
    'cloudsc_class2_1516':
    {
        'KLON': 3000,
        'KLEV': 3000,
        'KFDIA': 2998
    },
    'cloudsc_class3_691': {
        'KLON': 3000,
        'KLEV': 3000,
        'KFDIA': 2998
    },
    'cloudsc_class3_965': {
        'KLON': 3000,
        'KLEV': 3000,
        'KFDIA': 2998
    },
    'cloudsc_class3_1985': {
        'KLON': 3000,
        'KLEV': 3000,
        'KFDIA': 2998
    },
    'cloudsc_class3_2120': {
        'KLON': 3000,
        'KLEV': 3000,
        'KFDIA': 2998
    },
    'my_roofline_test':
    {
        'KLON': 10000,
        'KLEV': 10000,
        'KIDIA': 1,
        'KFDIA': 10000,
    },
    'cloudsc_vert_loop_2':
    {
        'KLEV': 137,
        'KLON': 1,
        'NPROMA': 1,
        'NBLOCKS': 10000
    }
}

# changes from the parameters dict for testing
testing_parameters = {'KLON': 10, 'KLEV': 10, 'KFDIA': 8}


def get_data(params: Dict[str, int]) -> Dict[str, Tuple]:
    """
    Returns the data dict given the parameters dict

    :param params: The parameters dict
    :type params: Dict[str, int]
    :return: The data dict
    :rtype: Dict[str, Tuple]
    """
    return {
        'PTSPHY': (0, ),
        'R2ES': (0, ),
        'R3IES': (0, ),
        'R3LES': (0, ),
        'R4IES': (0, ),
        'R4LES': (0, ),
        'RALSDCP': (0, ),
        'RALVDCP': (0, ),
        'RAMIN': (0, ),
        'RCOVPMIN': (0, ),
        'RCLDTOPCF': (0, ),
        'RD': (0, ),
        'RDEPLIQREFDEPTH': (0, ),
        'RDEPLIQREFRATE': (0, ),
        'RG': (0, ),
        'RICEINIT': (0, ),
        'RKOOP1': (0, ),
        'RKOOP2': (0, ),
        'RKOOPTAU': (0, ),
        'RLMIN': (0, ),
        'RLSTT': (0, ),
        'RLVTT': (0, ),
        'RPECONS': (0, ),
        'RPRECRHMAX': (0, ),
        'RTAUMEL': (0, ),
        'RTHOMO': (0, ),
        'RTT': (0, ),
        'RV': (0, ),
        'RVRFACTOR': (0, ),
        'ZEPSEC': (0, ),
        'ZEPSILON': (0, ),
        'ZRG_R': (0, ),
        'ZRLDCP': (0, ),
        'ZQTMST': (0, ),
        'ZVPICE': (0, ),
        'ZVPLIQ': (0, ),
        'ZALFAW': (0, ),
        'ARRAY_A': (params['KLON'], params['KLEV']),
        'ARRAY_B': (params['KLON'], params['KLEV']),
        'ARRAY_C': (params['KLON'], params['KLEV']),
        'IPHASE': (params['NCLV'], ),
        'LDCUM': (params['KLON'], params['NBLOCKS']),
        'PAPH': (params['KLON'], params['KLEV'] + 1),
        'PAPH_N': (params['KLON'], params['KLEV'] + 1, params['NBLOCKS']),
        'PAP': (params['KLON'], params['KLEV']),
        'PCOVPTOT': (params['KLON'], params['KLEV']),
        'PFCQLNG': (params['KLON'], params['KLEV'] + 1),
        'PFCQNNG': (params['KLON'], params['KLEV'] + 1),
        'PFCQRNG': (params['KLON'], params['KLEV'] + 1),
        'PFCQSNG': (params['KLON'], params['KLEV'] + 1),
        'PFHPSL': (params['KLON'], params['KLEV'] + 1),
        'PFHPSN': (params['KLON'], params['KLEV'] + 1),
        'PFPLSL': (params['KLON'], params['KLEV'] + 1),
        'PFPLSN': (params['KLON'], params['KLEV'] + 1),
        'PFSQIF': (params['KLON'], params['KLEV'] + 1),
        'PFSQITUR': (params['KLON'], params['KLEV'] + 1),
        'PFSQLF': (params['KLON'], params['KLEV'] + 1),
        'PFSQLTUR': (params['KLON'], params['KLEV'] + 1),
        'PFSQRF': (params['KLON'], params['KLEV'] + 1),
        'PFSQSF': (params['KLON'], params['KLEV'] + 1),
        'PLU': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'PLUDE': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'PSUPSAT': (params['KLON'], params['KLEV']),
        'PSUPSAT_N': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'PSNDE': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'PVFI': (params['KLON'], params['KLEV']),
        'PVFL': (params['KLON'], params['KLEV']),
        'tendency_loc_a': (params['KLON'], params['KLEV']),
        'tendency_loc_cld': (params['KLON'], params['KLEV'], params['NCLV']),
        'tendency_loc_q': (params['KLON'], params['KLEV']),
        'tendency_loc_T': (params['KLON'], params['KLEV']),
        'tendency_tmp_t': (params['KLON'], params['KLEV']),
        'tendency_tmp_t_N': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'tendency_tmp_q': (params['KLON'], params['KLEV']),
        'tendency_tmp_a': (params['KLON'], params['KLEV']),
        'tendency_tmp_cld': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZA': (params['KLON'], params['KLEV']),
        'ZAORIG': (params['KLON'], params['KLEV']),
        'ZCLDTOPDIST': (params['KLON'], ),
        'ZCLDTOPDIST2': (params['KLON'], params['KLEV']),
        'ZCONVSINK': (params['KLON'], params['NCLV']),
        'ZCONVSRCE': (params['KLON'], params['NCLV']),
        'ZCORQSICE': (params['KLON']),
        'ZCORQSLIQ': (params['KLON']),
        'ZCOVPTOT': (params['KLON'], ),
        'ZCOVPTOT2': (params['KLON'], params['KLEV']),
        'ZCOVPCLR': (params['KLON'], ),
        'ZCOVPCLR2': (params['KLON'], params['KLEV']),
        'ZCOVPMAX': (params['KLON'], ),
        'ZCOVPMAX2': (params['KLON'], params['KLEV']),
        'ZDTGDP': (params['KLON'], ),
        'ZICENUCLEI': (params['KLON'], ),
        'ZRAINCLD': (params['KLON'], ),
        'ZRAINCLD2': (params['KLON'], params['KLEV']),
        'ZSNOWCLD': (params['KLON'], ),
        'ZSNOWCLD2': (params['KLON'], params['KLEV']),
        'ZDA': (params['KLON'], ),
        'ZDP': (params['KLON'], ),
        'ZRHO': (params['KLON'], ),
        'ZFALLSINK': (params['KLON'], params['NCLV']),
        'ZFALLSRCE': (params['KLON'], params['NCLV']),
        'ZFOKOOP': (params['KLON'], ),
        'ZFLUXQ': (params['KLON'], params['NCLV']),
        'ZFOEALFA': (params['KLON'], params['KLEV'] + 1),
        'ZICECLD': (params['KLON'], ),
        'ZICEFRAC': (params['KLON'], params['KLEV']),
        'ZICETOT': (params['KLON'], ),
        'ZICETOT2': (params['KLON'], params['KLEV']),
        'ZLI': (params['KLON'], params['KLEV']),
        'ZLIQFRAC': (params['KLON'], params['KLEV']),
        'ZLNEG': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZPFPLSX': (params['KLON'], params['KLEV'] + 1, params['NCLV']),
        'ZPSUPSATSRCE': (params['KLON'], params['NCLV']),
        'ZMELTMAX': (params['KLON'], ),
        'ZMELTMAX2': (params['KLON'], params['KLEV']),
        'ZQPRETOT': (params['KLON'], ),
        'ZQPRETOT2': (params['KLON'], params['KLEV']),
        'ZQSLIQ': (params['KLON'], params['KLEV']),
        'ZQSICE': (params['KLON'], params['KLEV']),
        'ZQX': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZQX0': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZQXFG': (params['KLON'], params['NCLV']),
        'ZQXFG2': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZQXN': (params['KLON'], params['NCLV']),
        'ZQXN2D': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZSOLAC': (params['KLON'], ),
        'ZSOLQA': (params['KLON'], params['NCLV'], params['NCLV']),
        'ZSOLQA2': (params['KLON'], params['KLEV'], params['NCLV'], params['NCLV']),
        'ZSUPSAT': (params['KLON'], ),
        'ZTP1': (params['KLON'], params['KLEV']),
        'PT': (params['KLON'], params['KLEV']),
        'PT_N': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'PQ': (params['KLON'], params['KLEV']),
        'PA': (params['KLON'], params['KLEV']),
        'PCLV': (params['KLON'], params['KLEV'], params['NCLV']),
        'PCLV_N': (params['KLON'], params['KLEV'], params['NCLV'], params['NBLOCKS']),
    }


def get_iteration_ranges(params: Dict[str, int], program: str) -> List[Dict]:
    """
    Returns the iteration ranges for the returned variables given the program

    :param params: The parameters used
    :type params: Dict[str, int]
    :param program: The program name
    :type program: str
    :return: List of dictionaries. Each dictionary has an entry 'variables' listing all variable names which have the
    given ranges, an entry 'start' and 'stop' which are tuples with the start and stop indices (0-based) for the range
    to check/compare. Alternatively entries in start can also be lists of indices or a single index, the corresponding
    entries in stop must then be None
    :rtype: List[Dict]
    """
    # This ranges have inclusive starts and exclusive ends. Remember that for fortran loops start and end are inclusive
    # and that fortran arrays are 1-based but python are 0-based -> shift starting positions by -1
    ranges = {
            'cloudsc_class1_658': [
                {
                    'variables': ['ZTP1', 'ZA', 'ZAORIG'],
                    'start': (params['KIDIA']-1, 0),
                    'end': (params['KFDIA'], params['KLEV'])
                },
                {
                    'variables': ['ZQX', 'ZQX0'],
                    'start': (params['KIDIA']-1, 0, params['NCLDQV']-1),
                    'end': (params['KFDIA'], params['KLEV'], None)
                }
            ],
            'cloudsc_class1_670': [
                {
                    'variables': ['ZQX', 'ZQX0'],
                    'start': (params['KIDIA']-1, 0, 0),
                    'end': (params['KFDIA'], params['KLEV'], params['NCLV']-1)
                }
            ],
            'cloudsc_class1_2857': [
                {
                    'variables': ['PFHPSL', 'PFHPSN'],
                    'start': (params['KIDIA']-1, 0, 0),
                    'end': (params['KFDIA'], params['KLEV']+1)
                }
            ],
            'cloudsc_class1_2783': [
                {
                    'variables': ['PFPLSL', 'PFPLSN'],
                    'start': (params['KIDIA']-1, 0, 0),
                    'end': (params['KFDIA'], params['KLEV']+1)
                }
            ],
            'cloudsc_class2_1516': [
                {
                    'variables': ['ZSOLQA2'],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1, [params['NCLDQI'], params['NCLDQL']],
                              [params['NCLDQI'], params['NCLDQL']]),
                    'end': (params['KFDIA'], params['KLEV'], None, None)
                },
                {
                    'variables': ['ZQXFG2'],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1, [params['NCLDQI'], params['NCLDQL']]),
                    'end': (params['KFDIA'], params['KLEV'], None)
                },
                {
                    'variables': ['ZCLDTOPDIST2'],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_class2_1762': [
                {
                    'variables': ['ZCOVPTOT2', 'ZCOVPCLR2', 'ZCOVPMAX2', 'ZRAINCLD2', 'ZSNOWCLD2'],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_class2_781': [
                {
                    'variables': ["ZA", "ZLIQFRAC", "ZICEFRAC", "ZLI"],
                    'start': (params['KIDIA']-1, 0),
                    'end': (params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_class2_1001': [
                {
                    'variables': ["ZSOLQA", "ZSUPSAT"],
                    'start': (params['KIDIA']-1, [params['NCLDQI'], params['NCLDQL'], params['NCLDQV']]),
                    'end': (params['KFDIA'], None)
                },
                {
                    'variables': ["ZQXFG", "ZPSUPSATSRCE"],
                    'start': (params['KIDIA']-1, [params['NCLDQL'], params['NCLDQI']]),
                    'end': (params['KFDIA'], None)
                },
                {
                    'variables': ["ZSOLAC", "ZFOKOOP", "ZSUPSAT"],
                    'start': (params['KIDIA']-1),
                    'end': (params['KFDIA'])
                }
            ],
            'cloudsc_class3_1985': [
                {
                    'variables': ["ZICETOT2", "ZMELTMAX2"],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_class3_2120': [
                {
                    'variables': ["ZSOLQA2"],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1, 0, 0),
                    'end': (params['KFDIA'], params['KLEV'], params['NCLV'], params['NCLV'])
                },
                {
                    'variables': ["ZCOVPTOT2"],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['KFDIA'], params['KLEV'])
                },
                {
                    'variables': ["ZQXFG2"],
                    'start': (params['KIDIA']-1, params['NCLDTOP'], params['NCLDQR']),
                    'end': (params['KFDIA'], params['KLEV'], None)
                }
            ],
            'cloudsc_class3_691': [
                {
                    'variables': ["ZQX"],
                    'start': (params['KIDIA']-1, 0, [params['NCLDQV'], params['NCLDQL'], params['NCLDQI']]),
                    'end': (params['KFDIA'], params['KLEV'], None)
                },
                {
                    'variables': ["ZLNEG"],
                    'start': (params['KIDIA']-1, 0, [params['NCLDQL'], params['NCLDQI']]),
                    'end': (params['KFDIA'], params['KLEV'], None)
                },
                {
                    'variables': ["tendency_loc_q", "tendency_loc_T", "ZA"],
                    'start': (params['KIDIA']-1, 0),
                    'end': (params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_class3_965': [
                {
                    'variables': ["ZSOLQA2"],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1, 0, 0),
                    'end': (params['KFDIA'], params['KLEV'], params['NCLV'], params['NCLV'])
                }
            ],
            'my_test': [
                {
                    'variables': ['ARRAY_A'],
                    'start': (0,),
                    'end': (params['KLON'],)
                }
            ],
            'my_roofline_test': [
                {
                    'variables': ['ARRAY_A'],
                    'start': (params['KIDIA']-1, 0),
                    'end': (params['KFDIA'], params['KLEV'])
                }
            ],
    }
    return ranges[program]


def set_input_pattern(
        inputs: Dict[str, Union[Number, np.ndarray]],
        outputs: Dict[str, Union[Number, np.ndarray]],
        program: str, pattern: str):
    """
    Sets a specific pattern into the given input data depending on the program. Used to trigger certain pattern for
    if/else conditions inside the programs. Current patterns are:
        - const: Use the const branch in class 2 (setting to const value)
        - formula: Use the formula branch in class 2
        - worst: Alternate between each branch inbetween each thread in a block

    Builds on the assumption that the given input data is all in the range (0,1)

    :param inputs: The input data to manipulate
    :type inputs: Dict[str, Union[Number, np.ndarray]]
    :param inputs: The output data to manipulate, which sometimes also serves as input
    :type inputs: Dict[str, Union[Number, np.ndarray]]
    :param program: The program name
    :type program: str
    :param pattern: The pattern name
    :type pattern: str
    """
    print_with_time(f"Set input pattern {pattern} for {program}")
    params = get_program_parameters_data(program)['parameters']
    if pattern == 'const':
        if program == 'cloudsc_class2_781':
            inputs['RLMIN'] = 10.0
        elif program == 'cloudsc_class2_1762':
            inputs['ZEPSEC'] = 10.0
        elif program == 'cloudsc_class2_1516':
            print(f"WARNING: Pattern {pattern} not possible for cloudsc_class2_1516 for first loop, only possible for "
                  f"second")
            inputs['RTT'] = 0.0
            inputs['RLMIN'] = 10.0
        elif program == 'my_test_routine':
            inputs['ARRAY_B'] = np.zeros_like(inputs['ARRAY_B'])
        elif program == 'cloudsc_class3_691':
            inputs['RAMIN'] = 0.0
            inputs['RLMIN'] = 10.0
        elif program == 'cloudsc_class3_965':
            inputs['RLMIN'] = 0.0
        elif program == 'cloudsc_class3_1985':
            inputs['ZEPSEC'] = 10.0
            inputs['RTT'] = 10.0
        elif program == 'cloudsc_class3_2120':
            inputs['ZEPSEC'] = 10.0
        else:
            print(f"ERROR: Pattern {pattern} does not exists for {program}")
    elif pattern == 'formula':
        if program == 'cloudsc_class2_781':
            inputs['RLMIN'] = 0.0
        elif program == 'cloudsc_class2_1762':
            inputs['ZEPSEC'] = 0.0
        elif program == 'cloudsc_class2_1516':
            inputs['ZA'] = np.ones_like(inputs['ZA'])
            inputs['RTT'] = 10.0
            inputs['RLMIN'] = 0.0
        elif program == 'my_test_routine':
            inputs['ARRAY_B'] = np.ones_like(inputs['ARRAY_B'])
        elif program == 'cloudsc_class3_691':
            inputs['RAMIN'] = 10.0
        elif program == 'cloudsc_class3_965':
            inputs['RLMIN'] = 10.0
        elif program == 'cloudsc_class3_1985':
            inputs['ZEPSEC'] = 0.0
            inputs['RTT'] = 0.0
        elif program == 'cloudsc_class3_2120':
            inputs['ZEPSEC'] = 0.0
            inputs['RPRECRHMAX'] = 10000.0
        else:
            print(f"ERROR: Pattern {pattern} does not exists for {program}")
    elif pattern == 'worst':
        if program == 'cloudsc_class2_781':
            inputs['RLMIN'] = 1.0
            inputs['ZQX'] = np.zeros_like(inputs['ZQX'])
            inputs['ZQX'][0::2, :, params['NCLDQL']-1] = 100.0
        elif program == 'cloudsc_class2_1762':
            inputs['ZEPSEC'] = 1.0
            inputs['ZQPRETOT2'] = np.zeros_like(inputs['ZQPRETOT2'])
            inputs['ZQPRETOT2'][0::2, :] = 100.0
        elif program == 'cloudsc_class2_1516':
            inputs['ZA'] = np.zeros_like(inputs['ZA'])
            inputs['ZA'][0::2, :] = 1.0
            inputs['RCLDTOPCF'] = 0.5
        elif program == 'my_test_routine':
            inputs['ARRAY_B'] = np.ones_like(inputs['ARRAY_B'])
            inputs['ARRAY_B'][0::2] = 0.0
        elif program == 'cloudsc_class3_691':
            inputs['RLMIN'] = 0.0
            inputs['RAMIN'] = 0.5
            outputs['ZA'] = np.zeros_like(outputs['ZA'])
            outputs['ZA'][0::2, :] = 1.0
        elif program == 'my_test_routine':
            inputs['ZQX'] = np.ones_like(inputs['ZQX'])
            inputs['ZQX'][0::2] = 0.0
        else:
            print(f"ERROR: Pattern {pattern} does not exists for {program}")
    else:
        print(f"ERROR: Unknown pattern {pattern}")


def get_program_parameters_data(program: str) -> Dict[str, Dict[str, Union[int, Tuple]]]:
    """
    Gets program parameters and program data according for the specific program

    :param program: The name of the program/file
    :type program: str
    :return: Dict with two entries: 'parameters' and 'data'. Each containing a dict with the parameters/data
    :rtype: Dict[str, Dict[str, Union[int, Tuple]]]
    """
    global parameters, custom_parameters
    params_data = {}
    params_data['parameters'] = copy.deepcopy(parameters)

    if program in custom_parameters:
        for parameter in custom_parameters[program]:
            params_data['parameters'][parameter] = custom_parameters[program][parameter]

    params_data['data'] = get_data(params_data['parameters'])
    return params_data


def get_testing_parameters_data() -> Dict[str, Dict[str, Union[int, Tuple]]]:
    """
    Gets the parameters and data used for testing. Does not take custom_parameters into account.

    :return: Dict with two entries: 'parameters' and 'data'. Each containing a dict with the parameters/data
    :rtype: Dict[str, Dict[str, Union[int, Tuple]]]
    """
    global parameters
    params_data = {}
    params_data['parameters'] = copy.deepcopy(parameters)

    for parameter in testing_parameters:
        params_data['parameters'][parameter] = testing_parameters[parameter]

    params_data['data'] = get_data(params_data['parameters'])
    return params_data
