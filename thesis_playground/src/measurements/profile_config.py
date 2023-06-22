from typing import List
import copy
from typing import Optional
import pandas as pd
import numpy as np
import dace

from utils.print import print_with_time
from utils.ncu import get_all_actions_filtered, get_frequencies, get_peak_performance, get_achieved_work, \
                      get_achieved_bytes, get_achieved_performance, get_runtime, get_cycles, get_all_actions
from utils.ncu_report import IAction
from utils.general import copy_to_device, get_programs_data, remove_build_folder, insert_heap_size_limit, get_inputs, \
                          get_outputs, use_cache, enable_debug_flags
from utils.execute_dace import RunConfig, compile_for_profile, gen_ncu_report, RNG_SEED
from execute.data import set_input_pattern
from execute.parameters import ParametersProvider
from execute.my_auto_opt import specialise_symbols
from measurements.flop_computation import get_number_of_bytes_2


class ProfileConfig:
    """
    Configuration how to profile one program
    """
    program: str
    sizes: List[ParametersProvider]
    set_heap_limit: bool
    heap_limit_str: str
    tot_time_repetitions: int
    ncu_repetitions: int
    ignore_action_re: Optional[str]
    size_identifiers: List[str]

    def __init__(self, program: str, sizes: List[ParametersProvider], size_identifiers: List[str],
                 set_heap_limit: bool = False, heap_limit_str: str = '',
                 tot_time_repetitions: int = 5, ncu_repetitions: int = 3,
                 ignore_action_re: Optional[str] = None):
        """
        Constructor

        :param program: The name of the program
        :type program: str
        :param sizes: The sizes to run
        :type sizes: List[ParametersProvider]
        :param size_identifiers: List of parameters which will change and are thus required to be recorded of reach
        measurement
        :type size_identifiers: List[str]
        :param set_heap_limit: If heap limit needs to be set in code, defaults to False
        :type set_heap_limit: bool, optional
        :param heap_limit_str: If heap limit is set, string describing it size, defaults to ''
        :type heap_limit_str: str, optional
        :param tot_time_repetitions: Number of repetitions for total runtime, defaults to 5
        :type tot_time_repetitions: int, optional
        :param ncu_repetitions: Number of repetitions for ncu measurements, defaults to 3
        :type ncu_repetitions: int, optional
        :param ignore_action_re: Regex for any ncu actions to ignore, defaults to None
        :type ignore_action_re: Optional[str], optional
        """
        self.program = program
        self.sizes = sizes
        self.size_identifiers = size_identifiers
        self.set_heap_limit = set_heap_limit
        self.heap_limit_str = heap_limit_str
        self.tot_time_repetitions = tot_time_repetitions
        self.ncu_repetitions = ncu_repetitions
        self.ignore_action_re = ignore_action_re

    def get_program_command_line_arguments(self, params: ParametersProvider, run_config: RunConfig) -> List[str]:
        """
        Provides the command line arguments to run the program once using run_program.py given the selected parameters
        and run_config

        :param params: The Parameters to use
        :type params: ParemtersProvider
        :param run_config: The RunConfig to use
        :type run_config: RunConfig
        :return: The command line arguments for run_program.py
        :rtype: List[str]
        """
        program_args = []
        if not run_config.specialise_symbols:
            program_args.append('--not-specialise-symbols')
            if self.set_heap_limit:
                program_args.append('--cache')

        for key, value in params.get_dict().items():
            if key in ['NBLOCKS', 'KLEV', 'KLON', 'KIDIA', 'KFDIA']:
                program_args.extend([f"--{key}", str(value)])
        if run_config.pattern is not None:
            program_args.extend(['--pattern', run_config.pattern])
        return program_args

    def get_action(self, ncu_report_path: str) -> IAction:
        """
        Get the relevant action from the ncu report. Filters out any actions where self.ignore_action_re matches the
        action name

        :param ncu_report_path: The path to the ncu report to read
        :type ncu_report_path: str
        :return: The relevant action
        :rtype: IAction
        """
        if self.ignore_action_re is None:
            actions = get_all_actions(ncu_report_path)
        else:
            actions = get_all_actions_filtered(ncu_report_path, self.ignore_action_re)
        action = actions[0]
        if len(actions) > 1:
            print(f"WARNING: Found more than one actions, there are {len(actions)}, taking the first ({action})")
        return action

    def compile(self, params: ParametersProvider, run_config: RunConfig,
                specialise_changing_sizes: bool = True) -> dace.SDFG:
        """
        Compiles the program for the given parameters and run config

        :param params: The Parameters to use
        :type params: ParametersProvider
        :param run_config: The run configuration to used
        :type run_config: RunConfig
        :param specialise_changing_sizes: Set to true if also the symbols in self.size_identifiers should be
        specialised, defaults to True
        :return: The compiled sdfg
        :rtype: dace.SDFG
        """
        remove_build_folder(self.program)
        enable_debug_flags()
        params_dict = params.get_dict()
        if not specialise_symbols:
            for symbol in self.size_identifiers:
                del params_dict[symbol]
        sdfg = compile_for_profile(self.program, params_dict, run_config)
        if self.set_heap_limit and not run_config.specialise_symbols:
            if self.heap_limit_str == "":
                print("WARNING: Should set heap string, but it is empty")
            else:
                programs = get_programs_data()['programs']
                insert_heap_size_limit(f"{programs[self.program]}_routine", self.heap_limit_str)
            use_cache(self.program)
        return sdfg

    def profile(self, run_config: RunConfig, ncu_report_path: Optional[str] = None,
                sdfg_path: Optional[str] = None) -> pd.DataFrame:
        """
        Profile this configuration

        :param run_config: The cun config to use
        :type run_config: RunConfig
        :param sdfg_path: Path to the SDFG to use. If None compiles the sdfg first, defaults to None
        :type sdfg_path: Optional[str], optional
        :param ncu_report_path: Path to store the ncu report in. If None stores in /tmp, defaults to None
        :type ncu_report_path: Optional[str], optional
        :return: The collected data in long format
        :rtype: pd.DataFrame
        """
        data = []
        programs = get_programs_data()['programs']
        routine_name = f"{programs[self.program]}_routine"
        ncu_report_path = '/tmp/profile.ncu-rep' if ncu_report_path is None else ncu_report_path

        # if sdfg_path is None:
        #     print_with_time("[ProfileConfig::profile] Compile unfixed SDFG for profiling")
        #     sdfg = self.compile(self.sizes[0], run_config)
        #     sdfg.save('/tmp/sdfg.sdfg')
        #     sdfg_path = '/tmp/sdfg.sdfg'
        # else:
        #     print_with_time(f"[ProfileConfig::profile] Read SDFG from {sdfg_path} for profiling")
        #     sdfg = SDFG.from_file(sdfg_path)

        for params in self.sizes:
            sdfg = self.compile(params, run_config)
            sdfg.save('/tmp/sdfg.sdfg')
            sdfg_path = '/tmp/sdfg.sdfg'
            specialise_symbols(sdfg, params.get_dict())

            rng = np.random.default_rng(RNG_SEED)
            inputs = get_inputs(self.program, rng, params)
            outputs = get_outputs(self.program, rng, params)
            if run_config.pattern is not None:
                set_input_pattern(inputs, outputs, params, self.program, run_config.pattern)

            sdfg.clear_instrumentation_reports()
            print_with_time(f"[ProfileConfig::profile] Run {self.program} {self.tot_time_repetitions} times to measure "
                            f"total runtime " f"{'with' if run_config.specialise_symbols else 'without'} symbols and "
                            f" KLON: {params['KLON']:,} KLEV: {params ['KLEV']:,} NBLOCKS: {params['NBLOCKS']:,}")
            inputs_device = copy_to_device(copy.deepcopy(inputs))
            outputs_device = copy_to_device(copy.deepcopy(outputs))
            for i in range(self.tot_time_repetitions):
                print_with_time(f"[ProfileConfig::profile] Starting run {i} for total time")
                sdfg(**inputs_device, **outputs_device)

            reports = sdfg.get_instrumentation_reports()
            size_data = {}
            for param in self.size_identifiers:
                size_data.update({param: params[param]})
            for index, report in enumerate(reports):
                keys = list(report.durations[(0, -1, -1)][f"SDFG {routine_name}"].keys())
                key = keys[0]
                if len(keys) > 1:
                    print(f"WARNING: Report has more than one key, taking only the first one. keys: {keys}")
                this_data = {
                    'program': self.program,
                    'run number': index,
                    'measurement': 'Total time',
                    'value': report.durations[(0, -1, -1)][f"SDFG {routine_name}"][key][0],
                    **size_data}
                data.append(this_data)
            print_with_time(f"[ProfileConfig::profile] Run {self.program} {self.ncu_repetitions} times with ncu "
                            f"{'with' if run_config.specialise_symbols else 'without'} symbols and "
                            f" KLON: {params['KLON']:,} KLEV: {params ['KLEV']:,} NBLOCKS: {params['NBLOCKS']:,}")
            for index in range(self.ncu_repetitions):
                program_args = self.get_program_command_line_arguments(params, run_config)
                program_args.extend(['--sdfg-file', sdfg_path])
                gen_ncu_report(self.program, ncu_report_path, run_config,
                               program_args=program_args,
                               ncu_args=['--set', 'full'])
                action = self.get_action(ncu_report_path)
                data.append({'program': self.program, 'run number': index, 'measurement': 'gpu frequency',
                             'value': get_frequencies(action)[0], **size_data})
                data.append({'program': self.program, 'run number': index, 'measurement': 'memory frequency',
                             'value': get_frequencies(action)[1], **size_data})
                data.append({'program': self.program, 'run number': index, 'measurement': 'peak performance',
                             'value': get_peak_performance(action)[0], **size_data})
                data.append({'program': self.program, 'run number': index, 'measurement': 'peak bandwidth',
                             'value': get_peak_performance(action)[1], **size_data})
                for key, value in get_achieved_work(action).items():
                    data.append({'program': self.program, 'run number': index, 'measurement': key, 'value': value,
                                 **size_data})
                data.append({'program': self.program, 'run number': index, 'measurement': 'measured bytes',
                             'value': get_achieved_bytes(action), **size_data})
                data.append({'program': self.program, 'run number': index, 'measurement': 'measured performance',
                             'value': get_achieved_performance(action)[0], **size_data})
                data.append({'program': self.program, 'run number': index, 'measurement': 'measured bandwidth',
                             'value': get_achieved_performance(action)[1], **size_data})
                data.append({'program': self.program, 'run number': index, 'measurement': 'runtime',
                             'value': get_runtime(action), **size_data})
                data.append({'program': self.program, 'run number': index, 'measurement': 'cycles',
                             'value': get_cycles(action), **size_data})
            theoretical_bytes = get_number_of_bytes_2(params, self.program)
            theoretical_bytes_temp = get_number_of_bytes_2(params, self.program, True)
            if theoretical_bytes is None:
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes total',
                             'value': -1, **size_data})
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes read',
                             'value': -1, **size_data})
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes written',
                             'value': -1, **size_data})
            else:
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes total',
                             'value': theoretical_bytes[0], **size_data})
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes read',
                             'value': theoretical_bytes[1], **size_data})
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes written',
                             'value': theoretical_bytes[2], **size_data})
            if theoretical_bytes_temp is None:
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp total',
                             'value': -1, **size_data})
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp read',
                             'value': -1, **size_data})
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp written',
                             'value': -1, **size_data})
            else:
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp total',
                             'value': theoretical_bytes_temp[0], **size_data})
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp read',
                             'value': theoretical_bytes_temp[1], **size_data})
                data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp written',
                             'value': theoretical_bytes_temp[2], **size_data})
        df = pd.DataFrame(data)
        index_cols = list(df.columns)
        index_cols.remove('value')
        return df.set_index(index_cols)
