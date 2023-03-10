from typing import Dict, Union, Tuple

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
    'NCLDTOP': 1,
    'NSSOPT': 1
}

custom_parameters = {
    # 'cloudsc_class1_670': {
    #     'KLON': 1000,
    #     'KLEV': 10,
    #     'KFDIA': 998,
    #     },
}


def get_data(params: Dict[str, int]) -> Dict[str, Tuple]:
    """
    Returns the data dict given the parameters dict

    :param params: The parameters dict
    :type params: Dict[str, int]
    :return: The data dict
    :rtype: Dict[str, Tuple]
    """
    return {
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
        'RPECONS': (0,),
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
        'IPHASE': (params['NCLV'],),
        'PAPH': (params['KLON'], params['KLEV']+1),
        'PAP': (params['KLON'], params['KLEV']),
        'PCOVPTOT': (params['KLON'], params['KLEV']),
        'PFCQLNG': (params['KLON'], params['KLEV']+1),
        'PFCQNNG': (params['KLON'], params['KLEV']+1),
        'PFCQRNG': (params['KLON'], params['KLEV']+1),
        'PFCQSNG': (params['KLON'], params['KLEV']+1),
        'PFHPSL': (params['KLON'], params['KLEV']+1),
        'PFHPSN': (params['KLON'], params['KLEV']+1),
        'PFPLSL': (params['KLON'], params['KLEV']+1),
        'PFPLSN': (params['KLON'], params['KLEV']+1),
        'PFSQIF': (params['KLON'], params['KLEV']+1),
        'PFSQITUR': (params['KLON'], params['KLEV']+1),
        'PFSQLF': (params['KLON'], params['KLEV']+1),
        'PFSQLTUR': (params['KLON'], params['KLEV']+1),
        'PFSQRF': (params['KLON'], params['KLEV']+1),
        'PFSQSF': (params['KLON'], params['KLEV']+1),
        'PLUDE': (params['KLON'], params['KLEV']),
        'PSUPSAT': (params['KLON'], params['KLEV']),
        'PVFI': (params['KLON'], params['KLEV']),
        'PVFL': (params['KLON'], params['KLEV']),
        'tendency_loc_a': (params['KLON'], params['KLEV']),
        'tendency_loc_cld': (params['KLON'], params['KLEV'], params['NCLV']),
        'tendency_loc_q': (params['KLON'], params['KLEV']),
        'tendency_loc_T': (params['KLON'], params['KLEV']),
        'tendency_tmp_t': (params['KLON'], params['KLEV']),
        'tendency_tmp_q': (params['KLON'], params['KLEV']),
        'tendency_tmp_a': (params['KLON'], params['KLEV']),
        'tendency_tmp_cld': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZA': (params['KLON'], params['KLEV']),
        'ZAORIG': (params['KLON'], params['KLEV']),
        'ZCLDTOPDIST': (params['KLON'],),
        'ZCONVSINK': (params['KLON'], params['NCLV']),
        'ZCONVSRCE': (params['KLON'], params['NCLV']),
        'ZCORQSICE': (params['KLON']),
        'ZCORQSLIQ': (params['KLON']),
        'ZCOVPTOT': (params['KLON'],),
        'ZCOVPCLR': (params['KLON'],),
        'ZCOVPMAX': (params['KLON'],),
        'ZDTGDP': (params['KLON'],),
        'ZICENUCLEI': (params['KLON'],),
        'ZRAINCLD': (params['KLON'],),
        'ZSNOWCLD': (params['KLON'],),
        'ZDA': (params['KLON'],),
        'ZDP': (params['KLON'],),
        'ZRHO': (params['KLON'],),
        'ZFALLSINK': (params['KLON'], params['NCLV']),
        'ZFALLSRCE': (params['KLON'], params['NCLV']),
        'ZFOKOOP': (params['KLON'],),
        'ZFLUXQ': (params['KLON'], params['NCLV']),
        'ZFOEALFA': (params['KLON'], params['KLEV']+1),
        'ZICECLD': (params['KLON'],),
        'ZICEFRAC': (params['KLON'], params['KLEV']),
        'ZICETOT': (params['KLON'],),
        'ZLI': (params['KLON'], params['KLEV']),
        'ZLIQFRAC': (params['KLON'], params['KLEV']),
        'ZLNEG': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZPFPLSX': (params['KLON'], params['KLEV']+1, params['NCLV']),
        'ZPSUPSATSRCE': (params['KLON'], params['NCLV']),
        'ZMELTMAX': (params['KLON'],),
        'ZQPRETOT': (params['KLON'],),
        'ZQSLIQ': (params['KLON'], params['KLEV']),
        'ZQSICE': (params['KLON'], params['KLEV']),
        'ZQX': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZQX0': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZQXFG': (params['KLON'], params['NCLV']),
        'ZQXN': (params['KLON'], params['NCLV']),
        'ZQXN2D': (params['KLON'], params['KLEV'], params['NCLV']),
        'ZSOLAC': (params['KLON'],),
        'ZSOLQA': (params['KLON'], params['NCLV'], params['NCLV']),
        'ZSUPSAT': (params['KLON'],),
        'ZTP1': (params['KLON'], params['KLEV']),
        'PT': (params['KLON'], params['KLEV']),
        'PQ': (params['KLON'], params['KLEV']),
        'PA': (params['KLON'], params['KLEV']),
        'PCLV': (params['KLON'], params['KLEV'], params['NCLV']),
    }


def get_program_parameters_data(program: str) -> Dict[str, Dict[str, Union[int, Tuple]]]:
    """
    Gets program parameters and program data according for the specific program

    :param program: The name of the program/file
    :type program: str
    :return: Dict with two entries: 'parameters' and 'data'. Each containing a dict with the parameters/data
    :rtype: Dict[str, Dict[str, Union[int, Tuple]]]
    """
    params_data = {}
    params_data['parameters'] = parameters

    if program in custom_parameters:
        for parameter in custom_parameters[program]:
            params_data['parameters'][parameter] = custom_parameters[program][parameter]

    params_data['data'] = get_data(params_data['parameters'])
    return params_data

