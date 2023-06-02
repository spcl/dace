from typing import Dict, Optional
from numbers import Number
import copy
from argparse import Namespace


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
    },
    'cloudsc_vert_loop_4':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
        # 'NBLOCKS': 3000,
        'NBLOCKS': 20000
    },
    'cloudsc_vert_loop_5':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
        'NBLOCKS': 200000
    },
    'cloudsc_vert_loop_6':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
        'NBLOCKS': 200000
    },
    'cloudsc_vert_loop_6_1':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
        'NBLOCKS': 200000
    },
    'cloudsc_vert_loop_7':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
        'NBLOCKS': 200000
    },
    'cloudsc_vert_loop_7_1':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
        'NBLOCKS': 200000
    },
    'cloudsc_vert_loop_7_no_klon':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
        'NBLOCKS': 200000
    },
    'cloudsc_vert_loop_orig_mwe_no_klon':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
    },
    'cloudsc_vert_loop_mwe_no_klon':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
    },
    'cloudsc_vert_loop_mwe':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
    },
    'cloudsc_vert_loop_mwe_wip':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
    },
    'microbenchmark_v1':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
    },
    'microbenchmark_v3':
    {
        'KLEV': 137,
        'KLON': 1,
        'KFDIA': 1,
        'KIDIA': 1,
    }
}


# changes from the parameters dict for testing
# testing_parameters = {'KLON': 1, 'KLEV': 4, 'KFDIA': 1, 'KIDIA': 1, 'NBLOCKS': 5}
# testing_parameters = {'KLON': 1, 'KLEV': 13, 'KFDIA': 1, 'KIDIA': 1, 'NBLOCKS': 20}
testing_parameters = {'KLON': 1, 'KLEV': 137, 'KFDIA': 1, 'KIDIA': 1, 'NBLOCKS': 100}


class ParametersProvider:
    parameters: Dict[str, Number]
    program: str

    def __init__(self, program: str, testing: bool = False, update: Optional[Dict[str, Number]] = None):
        self.program = program
        self.parameters = copy.deepcopy(parameters)
        if testing:
            self.parameters.update(testing_parameters)
        elif program in custom_parameters:
            self.parameters.update(custom_parameters[program])
        if update is not None:
            self.parameters.update(update)

    def update_from_args(self, args: Namespace):
        args_dict = vars(args)
        for key in args_dict:
            if key in self.parameters and args_dict[key] is not None:
                self.parameters[key] = args_dict[key]

    def __getitem__(self, key: str) -> Number:
        if key in self.parameters:
            return self.parameters[key]
        else:
            print(f"WARNING: key {key} not in parameters, returned 1 instead")
            return 1

    def __len__(self) -> int:
        return len(self.parameters)

    def __str__(self) -> str:
        return 'ParametersProvider with: ' + ' '.join([f"{key}: {value}" for key, value in self.parameters.items()])

    def __contains__(self, key: str) -> bool:
        return key in self.parameters

    def get_dict(self) -> Dict[str, Number]:
        return copy.deepcopy(self.parameters)

    @staticmethod
    def to_json(params: 'ParametersProvider') -> Dict:
        dict = copy.deepcopy(params.parameters)
        dict.update({'__ParametersProvider__': True, 'program': params.program})
        return dict

    @staticmethod
    def from_json(dict: Dict) -> 'ParametersProvider':
        if '__ParametersProvider__' in dict:
            del dict['__ParametersProvider__']
            params = ParametersProvider(dict['program'])
            del dict['program']
            params.parameters = dict
            return params
        else:
            return dict
