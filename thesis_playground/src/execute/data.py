from typing import Dict, Union, Tuple, List, Optional
import numpy as np
from numbers import Number
import logging
import copy

from execute.parameters import ParametersProvider

logger = logging.getLogger(__name__)


def get_data(params: ParametersProvider, program: Optional[str] = None) -> Dict[str, Tuple]:
    """
    Returns the data dict given the parameters dict. If given a program name, will change/add program specific values.

    :param params: The parameters dict
    :type params: ParameterProvider
    :param program: Name of program to get the data for, can be None
    :type program: Optional[str]
    :return: The data dict
    :rtype: Dict[str, Tuple]
    """
    default_data = {
        'PTSPHY': (0, ),
        'R2ES': (0, ),
        'R3IES': (0, ),
        'R3LES': (0, ),
        'R4IES': (0, ),
        'R4LES': (0, ),
        'R5IES': (0, ),
        'R5LES': (0, ),
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
        'LDCUM_NF': (params['NBLOCKS'], params['KLON']),
        'LDCUM_NFS': (params['NBLOCKS']),
        'PAPH': (params['KLON'], params['KLEV'] + 1),
        'PAPH_N': (params['KLON'], params['KLEV'] + 1, params['NBLOCKS']),
        'PAPH_NF': (params['NBLOCKS'], params['KLON'], params['KLEV'] + 1),
        'PAPH_NFS': (params['NBLOCKS'], params['KLEV'] + 1),
        'PAPH_NS': (params['KLEV'] + 1, params['NBLOCKS']),
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
        'PLU': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
        'PLU_NF': (params['NBLOCKS'], params['KLON'], params['KLEV']+1),
        'PLU_NFS': (params['NBLOCKS'], params['KLEV']+1),
        'PLUDE': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'PLUDE_NF': (params['NBLOCKS'], params['KLON'], params['KLEV']),
        'PLUDE_NFS': (params['NBLOCKS'], params['KLEV']),
        'PLUDE_NS': (params['KLEV'], params['NBLOCKS'], ),
        'PSUPSAT': (params['KLON'], params['KLEV']),
        'PSUPSAT_N': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'PSUPSAT_NF': (params['NBLOCKS'], params['KLON'], params['KLEV']),
        'PSUPSAT_NFS': (params['NBLOCKS'], params['KLEV']),
        'PSNDE': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'PSNDE_NF': (params['NBLOCKS'], params['KLON'], params['KLEV']),
        'PSNDE_NFS': (params['NBLOCKS'], params['KLEV']),
        'PVFI': (params['KLON'], params['KLEV']),
        'PVFL': (params['KLON'], params['KLEV']),
        'tendency_loc_a': (params['KLON'], params['KLEV']),
        'tendency_loc_cld': (params['KLON'], params['KLEV'], params['NCLV']),
        'tendency_loc_q': (params['KLON'], params['KLEV']),
        'tendency_loc_T': (params['KLON'], params['KLEV']),
        'tendency_tmp_t': (params['KLON'], params['KLEV']),
        'TENDENCY_TMP_T': (params['KLON'], params['KLEV']),
        'tendency_tmp_t_N': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'TENDENCY_TMP_T_N': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'tendency_tmp_t_NF': (params['NBLOCKS'], params['KLON'], params['KLEV']),
        'TENDENCY_TMP_T_NF': (params['NBLOCKS'], params['KLON'], params['KLEV']),
        'tendency_tmp_t_NFS': (params['NBLOCKS'], params['KLEV']),
        'TENDENCY_TMP_T_NFS': (params['NBLOCKS'], params['KLEV']),
        'tendency_tmp_q': (params['KLON'], params['KLEV']),
        'tendency_tmp_a': (params['KLON'], params['KLEV']),
        'tendency_tmp_cld': (params['KLON'], params['KLEV'], params['NCLV']),
        'tendency_tmp_cld_N': (params['KLON'], params['KLEV'], params['NCLV'], params['NBLOCKS']),
        'TENDENCY_TMP_CLD_N': (params['KLON'], params['KLEV'], params['NCLV'], params['NBLOCKS']),
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
        'ZSOLQA_N': (params['KLON'], params['NCLV'], params['NCLV'], params['NBLOCKS']),
        'ZSOLQA_NF': (params['NBLOCKS'],  params['KLON'], params['NCLV'], params['NCLV']),
        'ZSOLQA2': (params['KLON'], params['KLEV'], params['NCLV'], params['NCLV']),
        'ZSUPSAT': (params['KLON'], ),
        'ZTP1': (params['KLON'], params['KLEV']),
        'PT': (params['KLON'], params['KLEV']),
        'PT_N': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'PT_NF': (params['NBLOCKS'], params['KLON'], params['KLEV']),
        'PT_NFS': (params['NBLOCKS'], params['KLEV']),
        'PQ': (params['KLON'], params['KLEV']),
        'PA': (params['KLON'], params['KLEV']),
        'PCLV': (params['KLON'], params['KLEV'], params['NCLV']),
        'PCLV_N': (params['KLON'], params['KLEV'], params['NCLV'], params['NBLOCKS']),
        'INPUT': (params['KLEV'], params['NBLOCKS']),
        'INPUT_F': (params['NBLOCKS'], params['KLEV']),
        'OUTPUT': (params['KLEV'], params['NBLOCKS']),
        'OUTPUT_F': (params['NBLOCKS'], params['KLEV']),
        'INP1': (params['KLON'], params['KLEV'], params['NBLOCKS']),
        'INP3': (params['KLON'], params['KLEV'], params['NCLV'], params['NBLOCKS']),
        'OUT1': (params['KLON'], params['KLEV'], params['NBLOCKS'])
    }
    program_data = {
        'cloudscexp2': {
            'PT': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PQ': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_cml_u': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_cml_v': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_cml_T': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_cml_o3': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_cml_q': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_cml_a': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_cml_cld': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_tmp_u': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_tmp_v': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_tmp_T': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_tmp_o3': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_tmp_q': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_tmp_a': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_tmp_cld': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_loc_u': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_loc_v': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_loc_T': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_loc_o3': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_loc_q': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_loc_a': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'tendency_loc_cld': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PVFA': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PVFL': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PVFI': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PDYNA': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PDYNL': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PDYNI': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PHRSW': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PHRLW': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PVERVEL': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PAP': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PAPH': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PLSM': (params['KLON'], params['NBLOCKS']),
            'LDCUM': (params['KLON'], params['NBLOCKS']),
            'KTYPE': (params['KLON'], params['NBLOCKS']),
            'PLU': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PLUDE': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PSNDE': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PMFU': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PMFD': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PA': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PCLV': (params['KLON'], params['KLEV'], params['NCLV'], params['NBLOCKS']),
            'PSUPSAT': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PLCRIT_AER': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PICRIT_AER': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PRE_ICE': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PCCN': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PNICE': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PCOVPTOT': (params['KLON'], params['KLEV'], params['NBLOCKS']),
            'PRAINFRAC_TOPRFZ': (params['KLON'], params['NBLOCKS']),
            'PFSQLF': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFSQIF': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFCQLNG': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFCQNNG': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFSQRF': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFSQSF': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFCQRNG': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFCQSNG': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFSQLTUR': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFSQITUR': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFPLSL': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFPLSN': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFHPSL': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PFHPSN': (params['KLON'], params['KLEV']+1, params['NBLOCKS']),
            'PEXTRA': (params['KLON'], params['KLEV'], params['KFLDX'], params['NBLOCKS']),
            }
    }
    if program is not None and program in program_data:
        data = copy.deepcopy(default_data)
        logger.debug(f"Update data with program specific data for {program}")
        data.update(program_data[program])
        return data
    return default_data


def get_iteration_ranges(params: ParametersProvider, program: str) -> List[Dict]:
    """
    Returns the iteration ranges for the returned variables given the program

    :param params: The parameters used
    :type params: ParametersProvider
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
            'cloudsc_vert_loop_4': [
                {
                    'variables': ['PLUDE'],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1, 0),
                    'end': (params['KFDIA'], params['KLEV'], params['NBLOCKS'])
                }
            ],
            'cloudsc_vert_loop_4_ZSOLQA': [
                {
                    'variables': ['PLUDE'],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1, 0),
                    'end': (params['KFDIA'], params['KLEV'], params['NBLOCKS'])
                },
                {
                    'variables': ['ZSOLQA_N'],
                    'start': (params['KIDIA']-1, 0, 0, 0),
                    'end': (params['KFDIA'], params['NCLV'], params['NCLV'], params['NBLOCKS'])
                }
            ],
            'cloudsc_vert_loop_5': [
                {
                    'variables': ['PLUDE'],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1, 0),
                    'end': (params['KFDIA'], params['KLEV'], params['NBLOCKS'])
                }
            ],
            'cloudsc_vert_loop_6': [
                {
                    'variables': ['PLUDE_NF'],
                    'start': (0, params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_vert_loop_6_ZSOLQA': [
                {
                    'variables': ['PLUDE_NF'],
                    'start': (0, params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['KLEV'])
                },
                {
                    'variables': ['ZSOLQA_NF'],
                    'start': (0, params['KIDIA']-1, 0, 0),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['NCLV'], params['NCLV'])
                }
            ],
            'cloudsc_vert_loop_6_1': [
                {
                    'variables': ['PLUDE_NF'],
                    'start': (0, params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_vert_loop_6_1_ZSOLQA': [
                {
                    'variables': ['PLUDE_NF'],
                    'start': (0, params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['KLEV'])
                },
                {
                    'variables': ['ZSOLQA_NF'],
                    'start': (0, params['KIDIA']-1, 0, 0),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['NCLV'], params['NCLV'])
                }
            ],
            'cloudsc_vert_loop_7': [
                {
                    'variables': ['PLUDE_NF'],
                    'start': (0, params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_vert_loop_7_1': [
                {
                    'variables': ['PLUDE_NF'],
                    'start': (0, params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_vert_loop_7_2': [
                {
                    'variables': ['PLUDE_NF'],
                    'start': (0, params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_vert_loop_7_3': [
                {
                    'variables': ['PLUDE_NF'],
                    'start': (0, params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['KLEV'])
                },
                {
                    'variables': ['ZSOLQA_NF'],
                    'start': (0, params['KIDIA']-1, 0, 0),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['NCLV'], params['NCLV'])
                }
            ],
            'cloudsc_vert_loop_mwe': [
                {
                    'variables': ['PLUDE_NF'],
                    'start': (0, params['KIDIA']-1, params['NCLDTOP']-1),
                    'end': (params['NBLOCKS'], params['KFDIA'], params['KLEV'])
                }
            ],
            'cloudsc_vert_loop_mwe_no_klon': [
                {
                    'variables': ['PLUDE_NFS'],
                    'start': (0, params['NCLDTOP']-1),
                    'end': (params['NBLOCKS'], params['KLEV'])
                }
            ],
            'cloudsc_vert_loop_orig_mwe_no_klon': [
                {
                    'variables': ['PLUDE_NS'],
                    'start': (params['NCLDTOP']-1, 0),
                    'end': (params['KLEV'], params['NBLOCKS'])
                }
            ],
            'cloudsc_vert_loop_mwe_wip': [
                {'variables': ['OUTPUT_F'], 'start': (2, 0), 'end': (params['KLEV'], params['NBLOCKS'])}
            ],
            'microbenchmark_v1': [
                {'variables': ['OUTPUT'], 'start': (2, 0), 'end': (params['KLEV'], params['NBLOCKS'])}
            ],
            'microbenchmark_v3': [
                {'variables': ['OUTPUT_F'], 'start': (2, 0), 'end': (params['KLEV'], params['NBLOCKS'])}
            ],
            'cloudsc_vert_loop_10': [
                {
                    'variables': ['PLUDE'],
                    'start': (params['KIDIA']-1, params['NCLDTOP']-1, 0),
                    'end': (params['KFDIA'], params['KLEV'], params['NBLOCKS'])
                },
                {
                    'variables': ['ZSOLQA_N'],
                    'start': (params['KIDIA']-1, 0, 0, 0),
                    'end': (params['KFDIA'], params['NCLV'], params['NCLV'], params['NBLOCKS'])
                }
            ],
    }
    return ranges[program]


def set_input_pattern(
        inputs: Dict[str, Union[Number, np.ndarray]],
        outputs: Dict[str, Union[Number, np.ndarray]],
        params: ParametersProvider,
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
    :param params: The parameters used
    :type params: ParametersProvider
    :param pattern: The pattern name
    :type pattern: str
    """
    logger.info(f"Set input pattern {pattern} for {program}")
    if pattern == 'const':
        if program == 'cloudsc_class2_781':
            inputs['RLMIN'] = 10.0
        elif program == 'cloudsc_class2_1762':
            inputs['ZEPSEC'] = 10.0
        elif program == 'cloudsc_class2_1516':
            logger.info(f"WARNING: Pattern {pattern} not possible for cloudsc_class2_1516 for first loop, only "
                        f"possible for second")
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
            logger.error(f"Pattern {pattern} does not exists for {program}")
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
            logger.error(f"Pattern {pattern} does not exists for {program}")
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
            logger.error(f"Pattern {pattern} does not exists for {program}")
    else:
        logger.error(f"Unknown pattern {pattern}")


def is_integer(name: str) -> bool:
    """
    Return whether the given name should be treated as a integer variable

    :param name: The name to check
    :type name: str
    :return: True if to be treated as an integer
    :rtype: bool
    """
    integer_names = ['LDCUM', 'LDCUM_NF', 'LDCUM_NFS']
    return name in integer_names
