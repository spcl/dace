from typing import Dict, Union, Tuple
from numbers import Number
import numpy as np
import copy
import json


class FlopCount:
    adds: int
    muls: int
    divs: int
    minmax: int
    abs: int

    def __init__(self, adds: int = 0, muls: int = 0, divs: int = 0, minmax: int = 0, abs: int = 0, powers: int = 0,
                 roots: int = 0):
        self.adds = adds
        self.muls = muls
        self.divs = divs
        self.minmax = minmax
        self.abs = abs
        self.powers = powers
        self.roots = roots

    def get_total_flops(self) -> int:
        return self.adds + self.muls + self.divs

    def __mul__(self, a: Number):
        return FlopCount(
            adds=self.adds * a,
            muls=self.muls * a,
            divs=self.divs * a,
            minmax=self.minmax * a,
            abs=self.abs * a,
            powers=self.powers * a,
            roots=self.roots * a)

    def __rmul__(self, a: Number):
        return self * a

    def __add__(self, other: 'FlopCount'):
        if not isinstance(other, type(self)):
            raise NotImplementedError
        else:
            return FlopCount(
                adds=self.adds + other.adds,
                muls=self.muls + other.muls,
                divs=self.divs + other.divs,
                minmax=self.minmax + other.minmax,
                abs=self.abs + other.abs,
                powers=self.powers + other.powers,
                roots=self.roots + other.roots)

    def to_dict(self) -> Dict[str, Number]:
        return {"adds": self.adds, "muls": self.muls, "divs": self.divs, "minmax": self.minmax, "abs": self.abs,
                "powers": self.powers, "roots": self.roots}


def save_roofline_data(data: Dict[str, Tuple[FlopCount, Number]], filename: str):
    all_dict = copy.deepcopy(data)
    for program in all_dict:
        all_dict[program] = (all_dict[program][0].to_dict(), all_dict[program][1])

    with open(filename, 'w') as file:
        print(f"Write file into {filename}")
        json.dump(all_dict, file)


def read_roofline_data(filename: str) -> Dict[str, Tuple[FlopCount, Number]]:
    with open(filename, 'r') as file:
        all_dict = json.load(file)
        for program in all_dict:
            all_dict[program] = (FlopCount(**all_dict[program][0]), all_dict[program][1])
        return all_dict


def get_number_of_flops(
        params: Dict[str, Number],
        inputs: Dict[str, Union[Number, np.ndarray]],
        outputs: Dict[str, Union[Number, np.ndarray]],
        program: str) -> FlopCount:
    KLEV = params['KLEV']
    NCLDTOP = params['NCLDTOP']
    KIDIA = params['KIDIA']
    KFDIA = params['KFDIA']
    if program == 'cloudsc_class3_691':
        zqx_ql_qi = outputs['ZQX'][KIDIA-1:KFDIA, 0:KLEV, params['NCLDQI']] + \
                    outputs['ZQX'][KIDIA-1:KFDIA, 0:KLEV, params['NCLDQL']]
        number_iterations = np.count_nonzero(
                (zqx_ql_qi < inputs['RLMIN']) | (outputs['ZA'][KIDIA-1:KFDIA, 0:KLEV] < inputs['RAMIN']))
        # print(f"{number_iterations:,} / {(KLEV) * (KFDIA-KIDIA+1):,}")
        return number_iterations * FlopCount(adds=8, muls=4)
    elif program == 'cloudsc_class3_965':
        number_of_iterations = np.count_nonzero(
                inputs['ZQX'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV, params['NCLDQL']] < inputs['RLMIN'])
        number_of_iterations += np.count_nonzero(
                inputs['ZQX'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV, params['NCLDQI']] < inputs['RLMIN'])
        # print(f"{number_of_iterations:,} / {2 * (KLEV-NCLDTOP+1) * (KFDIA-KIDIA+1):,}")
        return number_of_iterations * FlopCount(adds=1)
    elif program == 'cloudsc_class3_1985':
        number_if_iterations = np.count_nonzero(
                (outputs['ZICETOT2'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV] > inputs['ZEPSEC']) &
                (inputs['ZTP1'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV] > inputs['RTT']))
        # print(f"{number_if_iterations:,} / {(KLEV-NCLDTOP+1) * (KFDIA-KIDIA+1):,}")
        return (KLEV-NCLDTOP+1) * (KFDIA-KIDIA+1) * FlopCount(adds=1) + \
            number_if_iterations * FlopCount(adds=8, muls=7, divs=1, minmax=2, abs=1)
    elif program == 'cloudsc_class3_2120':
        zqe = (inputs['ZQX'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV, params['NCLDQV']]
               - inputs['ZA'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV]) \
                        * np.maximum(inputs['ZEPSEC'], 1.0 - inputs['ZA'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV])
        zqe = np.maximum(0.0, np.minimum(zqe, inputs['ZQSLIQ'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV]))
        zzrh = np.maximum(inputs['ZEPSEC'], 1.0 - inputs['ZA'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV])
        zzrh = inputs['ZCOVPMAX'][KIDIA-1:KFDIA] / zzrh.transpose()
        zzrh = inputs['RPRECRHMAX'] + (1.0 - inputs['RPRECRHMAX']) * zzrh.transpose()
        zzrh = np.minimum(zzrh, np.maximum(inputs['RPRECRHMAX'], 1.0))
        number_if_iterations = np.count_nonzero(
                (inputs['ZCOVPCLR'][KIDIA-1:KFDIA] > inputs['ZEPSEC']) &
                (outputs['ZQXFG2'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV, params['NCLDQR']] > inputs['ZEPSEC']).transpose() &
                (zqe < zzrh * inputs['ZQSLIQ'][KIDIA-1:KFDIA, NCLDTOP-1:KLEV]).transpose())
        # print(f"{number_if_iterations:,} / {(KLEV-NCLDTOP+1) * (KFDIA-KIDIA+1):,}")
        return (KLEV-NCLDTOP+1) * (KFDIA-KIDIA+1) * FlopCount(adds=5, muls=2, divs=2, minmax=6) + \
            number_if_iterations * FlopCount(adds=8, muls=15, divs=6, minmax=5, abs=1, powers=1, roots=1)


# Length of a double in bytes
BYTES_DOUBLE = 8


def get_number_of_bytes_rough(
        params: Dict[str, Number],
        inputs: Dict[str, Union[Number, np.ndarray]],
        outputs: Dict[str, Union[Number, np.ndarray]],
        program: str) -> Number:
    """
    Very rough calculation of bytes transfered/used. Does not take real iteration ranges into account. Counts output
    arrays twice (copy in and copyout) and input only as once

    :param params: The parameters used for the given program
    :type params: Dict[str, Number]
    :param inputs: The inputs used for the given program
    :type inputs: Dict[str, Union[Number, np.ndarray]]
    :param outputs: The outputs used for the given program
    :type outputs: Dict[str, Union[Number, np.ndarray]]
    :param program: The program name
    :type program: str
    :return: Number of bytes
    :rtype: Number
    """
    bytes = BYTES_DOUBLE * (len(params) + sum([np.prod(array.shape) if isinstance(array, np.ndarray) else 1 for array in inputs.values()]) +
                            sum([np.prod(array.shape) for array in outputs.values()]) * 2)
    return int(bytes)
