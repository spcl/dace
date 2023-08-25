from argparse import Namespace
import dace


class RunConfig:
    pattern: str
    use_dace_auto_opt: bool
    device: dace.DeviceType
    specialise_symbols: bool
    k_caching: bool
    change_stride: bool
    outside_loop_first: bool
    move_assignment_outside: bool

    def __init__(self, pattern: str = None, use_dace_auto_opt: bool = False,
                 device: dace.DeviceType = dace.DeviceType.GPU, specialise_symbols: bool = True,
                 k_caching: bool = False, change_stride: bool = False, outside_loop_first: bool = True,
                 move_assignment_outside: bool = True):
        self.pattern = pattern
        self.use_dace_auto_opt = use_dace_auto_opt
        self.device = device
        self.specialise_symbols = specialise_symbols
        self.k_caching = k_caching
        self.change_stride = change_stride
        self.outside_loop_first = outside_loop_first
        self.move_assignment_outside = move_assignment_outside

    def set_from_args(self, args: Namespace):
        keys = ['pattern', 'use_dace_auto_opt']
        args_dict = vars(args)
        for key in args_dict:
            if key in keys:
                setattr(self, key, args_dict[key])
        if 'specialise_symbols' in args_dict and args_dict['specialise_symbols']:
            self.specialise_symbols = True
        if 'not_specialise_symbols' in args_dict and args_dict['not_specialise_symbols']:
            self.specialise_symbols = False
        if 'k_caching' in args_dict and args_dict['k_caching']:
            self.k_caching = True
        if 'change_stride' in args_dict and args_dict['change_stride']:
            self.change_stride = True
        if 'no-outer-loop-first' in args_dict['no-outer-loop-first']:
            self.outside_loop_first = False

    def __len__(self):
        return len(self.pattern)

    def __str__(self):
        return f"RunConfig(pattern: {self.pattern}, use_dace_auto_opt: {self.use_dace_auto_opt}, " \
               f"device: {self.device}, specialise_symbols: {self.specialise_symbols}, " \
               f"k_caching: {self.k_caching}, change_stride: {self.change_stride})"
