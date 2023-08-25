from typing import List
import copy
from typing import Optional, Dict, Union
import pandas as pd
import numpy as np
import dace
import sympy
from dace.sdfg import SDFG
from numbers import Number
import logging

from utils.ncu import get_all_actions_filtered, get_frequencies, get_peak_performance, get_achieved_work, \
                      get_achieved_bytes, get_achieved_performance, get_runtime, get_cycles, get_all_actions, \
                      get_all_actions_matching_re, SummedIAction
from utils.ncu_report import IAction
from utils.general import get_programs_data, remove_build_folder, insert_heap_size_limit, get_inputs, \
                          get_outputs, use_cache, enable_debug_flags
from utils.gpu_general import copy_to_device
from utils.generate_sdfg import get_optimised_sdfg
from utils.execute_dace import compile_for_profile, gen_ncu_report, RNG_SEED
from utils.run_config import RunConfig
from execute.data import set_input_pattern
from execute.parameters import ParametersProvider
from execute.my_auto_opt import specialise_symbols
from measurements.flop_computation import get_number_of_bytes_2, get_number_of_flops

logger = logging.getLogger(__name__)


class ProfileConfig:
    """
    Configuration how to profile one program
    """
    program: str
    sizes: List[ParametersProvider]
    set_heap_limit: bool
    heap_limit_str: str
    heap_limit_expr: Optional[sympy.core.expr.Expr]
    tot_time_repetitions: int
    ncu_repetitions: int
    ignore_action_re: Optional[str]
    size_identifiers: List[str]
    ncu_kernels: Dict[str, str]
    use_basic_sdfg: bool

    def __init__(self, program: str, sizes: List[ParametersProvider], size_identifiers: List[str],
                 set_heap_limit: bool = False, heap_limit_str: str = '',
                 heap_limit_expr: Optional[sympy.core.expr.Expr] = None,
                 tot_time_repetitions: int = 5, ncu_repetitions: int = 3,
                 ignore_action_re: Optional[str] = None, ncu_kernels: Optional[Dict[str, str]] = None,
                 use_basic_sdfg: bool = False):
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
        :param heap_limit_expr: Sympy expression for the heap limit, can be given as an alternative to the string,
        defaults to None
        :type heap_limit_expr: Optional[sympy.core.expr.Expr]
        :param tot_time_repetitions: Number of repetitions for total runtime, defaults to 5
        :type tot_time_repetitions: int, optional
        :param ncu_repetitions: Number of repetitions for ncu measurements, defaults to 3
        :type ncu_repetitions: int, optional
        :param ignore_action_re: Regex for any ncu actions to ignore, defaults to None
        :type ignore_action_re: Optional[str], optional
        :param ncu_kernels: Dictionary of kernels to take from ncu. Key is kernel name to save, value is regex for
        kernel name as given by ncu, optional. If not given will take the kernel as specified by ignore_action_re. If
        there are multiple kernels per regex takes sum.
        :type ncu_kernels: Optional[Dict[str, str]]
        :param use_basic_sdfg: Whether to lazy load the basic sdfgs, defaults to False
        :type use_basic_sdfg: bool
        """
        self.program = program
        self.sizes = sizes
        self.size_identifiers = size_identifiers
        self.set_heap_limit = set_heap_limit
        self.heap_limit_str = heap_limit_str
        self.heap_limit_expr = heap_limit_expr
        self.tot_time_repetitions = tot_time_repetitions
        self.ncu_repetitions = ncu_repetitions
        self.ignore_action_re = ignore_action_re
        self.ncu_kernels = ncu_kernels
        self.use_basic_sdfg = use_basic_sdfg

    def __str__(self) -> str:
        return f"ProfileConfig for {self.program} profiling {self.tot_time_repetitions} for total time and " \
               f"{self.ncu_repetitions} using ncu"

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
            logger.warning(f"Found more than one actions, there are {len(actions)}, taking the first ({action})")
        return action

    @staticmethod
    def add_ncu_action_data(
            action: IAction,
            metadata: Dict[str, Union[str, int]]) -> List[Dict[str, Number]]:
        """
        Extracts different measurement points from the given ncu action

        :param action: The action to extract values from
        :type action: IAction
        :param metadata: The metadata to append to every data entry
        :type metadata: Dict[str, Union[str, int]]
        :return: List of data entries.
        :rtype: List[Dict[str, Number]]
        """
        data = []
        data.append({'measurement': 'gpu frequency', 'value': get_frequencies(action)[0], **metadata})
        data.append({'measurement': 'memory frequency', 'value': get_frequencies(action)[1], **metadata})
        data.append({'measurement': 'peak performance', 'value': get_peak_performance(action)[0], **metadata})
        data.append({'measurement': 'peak bandwidth', 'value': get_peak_performance(action)[1], **metadata})
        for key, value in get_achieved_work(action).items():
            data.append({'measurement': key, 'value': value, **metadata})
        data.append({'measurement': 'measured bytes', 'value': get_achieved_bytes(action), **metadata})
        data.append({'measurement': 'measured performance', 'value': get_achieved_performance(action)[0], **metadata})
        data.append({'measurement': 'measured bandwidth', 'value': get_achieved_performance(action)[1], **metadata})
        data.append({'measurement': 'runtime', 'value': get_runtime(action), **metadata})
        data.append({'measurement': 'cycles', 'value': get_cycles(action), **metadata})
        return data

    def compile(self, params: ParametersProvider, run_config: RunConfig,
                specialise_changing_sizes: bool = True, debug_mode: bool = False) -> dace.SDFG:
        """
        Compiles the program for the given parameters and run config. If heap_limit_str or heap_limit_expr is set and
        set_heap_limit is True, will insert the heap_limit_str into the code. If heap_limit_expr is given the
        heap_limit_str will be set to the evaluated expression given the parameters dict.

        :param params: The Parameters to use
        :type params: ParametersProvider
        :param run_config: The run configuration to used
        :type run_config: RunConfig
        :param specialise_changing_sizes: Set to true if also the symbols in self.size_identifiers should be
        specialised, defaults to True
        :param debug_mode: Set to true if compile for debugging, defaults to False
        :type debug_mode: bool
        :return: The compiled sdfg
        :rtype: dace.SDFG
        """
        remove_build_folder(self.program)
        if debug_mode:
            enable_debug_flags()
        params_dict = params.get_dict()
        if not specialise_symbols:
            for symbol in self.size_identifiers:
                del params_dict[symbol]
        if self.use_basic_sdfg:
            sdfg = get_optimised_sdfg(self.program, run_config, params, self.size_identifiers)
            sdfg.compile()
        else:
            sdfg = compile_for_profile(self.program, params_dict, run_config)
        if self.set_heap_limit and not run_config.specialise_symbols:
            if self.heap_limit_str == "" and self.heap_limit_expr is None:
                logger.warning("Should set heap string or expression, but both are empty")
            else:
                programs = get_programs_data()['programs']
                if self.heap_limit_expr is not None:
                    self.heap_limit_str = self.heap_limit_expr.evalf(subs=params.get_dict())
                insert_heap_size_limit(f"{programs[self.program]}_routine", self.heap_limit_str,
                                       debug_prints=debug_mode)
            use_cache(self.program)
        return sdfg

    def get_size_data_for_dataframe(self, params: ParametersProvider) -> Dict[str, Union[int, float]]:
        """
        Get dictionary of values of the provided parameters to be saved into the results dataframe.
        self.size_identifiers list the values to take.

        :param params: The parameters
        :type params: ParametersProvider
        :return: Dictionary, key is name of parameter, value its value
        :rtype: Dict[str, Union[int, float]]
        """
        size_data = {}
        for param in self.size_identifiers:
            size_data.update({param: params[param]})
        return size_data

    def profile_total_runtime(
            self,
            sdfg: SDFG,
            params: ParametersProvider,
            run_config: RunConfig) -> List[Dict[str, Union[str, float, int]]]:
        """
        Profile the total runtime of the given SDFG using the given parameters and run config

        :param sdfg: The SDFG to profile
        :type sdfg: SDFG
        :param params: The parameters to use, specifying input sizes
        :type params: ParametersProvider
        :param run_config: The run configuration to use
        :type run_config: RunConfig
        :return: List of dictionaries. Each dict is one total runtime measurement for one size.
        :rtype: List[Dict[str, Union[str, float, int]]]
        """
        logger.debug("foo1")
        programs = get_programs_data()['programs']
        routine_name = f"{programs[self.program]}_routine"
        rng = np.random.default_rng(RNG_SEED)
        logger.debug("foo2")
        inputs = get_inputs(self.program, rng, params)
        outputs = get_outputs(self.program, rng, params)
        logger.debug("foo3")

        if run_config.pattern is not None:
            set_input_pattern(inputs, outputs, params, self.program, run_config.pattern)

        sdfg.clear_instrumentation_reports()
        logger.info(f"Run {self.program} {self.tot_time_repetitions} times to measure "
                    f"total runtime " f"{'with' if run_config.specialise_symbols else 'without'} symbols and "
                    f" KLON: {params['KLON']:,} KLEV: {params ['KLEV']:,} NBLOCKS: {params['NBLOCKS']:,}")
        inputs_device = copy_to_device(copy.deepcopy(inputs))
        outputs_device = copy_to_device(copy.deepcopy(outputs))
        for i in range(self.tot_time_repetitions):
            logger.info(f"Starting run {i} for total time")
            sdfg(**inputs_device, **outputs_device)

        reports = sdfg.get_instrumentation_reports()
        size_data = self.get_size_data_for_dataframe(params)
        size_data['scope'] = "Total"
        size_data['num_kernels'] = -1
        data = []
        for index, report in enumerate(reports):
            keys = list(report.durations[(0, -1, -1)][f"SDFG {routine_name}"].keys())
            key = keys[0]
            if len(keys) > 1:
                logger.warning(f"Report has more than one key, taking only the first one. keys: {keys}")
            this_data = {
                'program': self.program,
                'run number': index,
                'measurement': 'Total time',
                'value': report.durations[(0, -1, -1)][f"SDFG {routine_name}"][key][0],
                **size_data}
            data.append(this_data)
        data.extend(self.get_theoretical_values(params, inputs, outputs))
        return data

    def profile_ncu(
            self,
            sdfg_path: str,
            ncu_report_path: str,
            params: ParametersProvider,
            run_config: RunConfig) -> List[Dict[str, Union[str, float, int]]]:
        """
        Profiles the kernels using ncu

        :param sdfg_path: Path to the sdfg to use
        :type sdfg_path: str
        :param ncu_report_path: The path where to store the ncu reprot
        :type ncu_report_path: str
        :param params: The parameters to use
        :type params: ParametersProvider
        :param run_config: The run configuration to use
        :type run_config: RunConfig
        :return: List of dictionaries. Each dit is one ncu measurement value for one size
        :rtype: List[Dict[str, Union[str, float, int]]]
        """

        logger.info(f"Run {self.program} {self.ncu_repetitions} times with ncu "
                    f"{'with' if run_config.specialise_symbols else 'without'} symbols and "
                    f" KLON: {params['KLON']:,} KLEV: {params ['KLEV']:,} NBLOCKS: {params['NBLOCKS']:,}")
        data = []
        for index in range(self.ncu_repetitions):
            program_args = self.get_program_command_line_arguments(params, run_config)
            program_args.extend(['--sdfg-file', sdfg_path])
            gen_ncu_report(self.program, ncu_report_path, run_config,
                           program_args=program_args,
                           ncu_args=['--set', 'full'])
            size_data = self.get_size_data_for_dataframe(params)
            metadata = {'program': self.program, 'run number': index, **size_data}
            if self.ncu_kernels is None:
                metadata['num_kernels'] = 0
                action = self.get_action(ncu_report_path)
                metadata['scope'] = str(action)
                logger.debug(f"No ncu kernels defined, taking action {action}")
                data.extend(ProfileConfig.add_ncu_action_data(action, metadata))
            else:
                for name, kernel_re in self.ncu_kernels.items():
                    actions = get_all_actions_matching_re(ncu_report_path, kernel_re)
                    action = SummedIAction(actions)
                    if len(action) == 0:
                        continue
                    metadata['scope'] = name
                    metadata['num_kernels'] = len(action)
                    data.extend(ProfileConfig.add_ncu_action_data(action, metadata))
        return data

    def get_theoretical_values(
            self,
            params: ParametersProvider,
            inputs: Dict[str, Union[Number, np.ndarray]],
            outputs: Dict[str, Union[Number, np.ndarray]]) -> List[Dict[str, Number]]:
        """
        Add theoretical flop and bytes information

        :param params: The parameters to use
        :type params: ParametersProvider
        :param inputs: The input data, used to compute flop count
        :type inputs: Dict[str, Union[Number, np.ndarray]]
        :param outputs: The input data, used to compute flop count
        :type outputs: Dict[str, Union[Number, np.ndarray]]
        :return: List of dictionaries. Each dict is one value
        :rtype: List[Dict[str, Number]]
        """
        data = []
        size_data = self.get_size_data_for_dataframe(params)
        size_data['scope'] = 'Total'
        size_data['num_kernels'] = -1
        theoretical_bytes = get_number_of_bytes_2(params, self.program)
        theoretical_bytes_temp = get_number_of_bytes_2(params, self.program, True)
        flop_count = get_number_of_flops(params, inputs, outputs, self.program)
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
            size_data['scope'] = 'temp'
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp total',
                         'value': -1, **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp read',
                         'value': -1, **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp written',
                         'value': -1, **size_data})
        else:
            size_data['scope'] = 'temp'
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp total',
                         'value': theoretical_bytes_temp[0], **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp read',
                         'value': theoretical_bytes_temp[1], **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical bytes temp written',
                         'value': theoretical_bytes_temp[2], **size_data})
        if flop_count is not None:
            size_data['scope'] = 'Total'
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical flops',
                         'value': flop_count.get_total_flops(), **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical adds',
                         'value': flop_count.adds, **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical muls',
                         'value': flop_count.muls, **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical divs',
                         'value': flop_count.divs, **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical minmax',
                         'value': flop_count.minmax, **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical abs',
                         'value': flop_count.abs, **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical powers',
                         'value': flop_count.powers, **size_data})
            data.append({'program': self.program, 'run number': 0, 'measurement': 'theoretical roots',
                         'value': flop_count.roots, **size_data})
        return data

    def profile(self, run_config: RunConfig, ncu_report_path: Optional[str] = None,
                sdfg_path: Optional[str] = None, debug_mode: bool = False) -> pd.DataFrame:
        """
        Profile this configuration

        :param run_config: The cun config to use
        :type run_config: RunConfig
        :param sdfg_path: Path to the SDFG to use. If None compiles the sdfg first, defaults to None
        :type sdfg_path: Optional[str], optional
        :param ncu_report_path: Path to store the ncu report in. If None stores in /tmp, defaults to None
        :type ncu_report_path: Optional[str], optional
        :param debug_mode: Set to true if compile for debugging, defaults to False
        :type debug_mode: bool
        :return: The collected data in long format
        :rtype: pd.DataFrame
        """
        data = []
        ncu_report_path = '/tmp/profile.ncu-rep' if ncu_report_path is None else ncu_report_path
        if sdfg_path is None:
            sdfg_path = '/tmp/sdfg.sdfg'
        else:
            logger.info(f"Save SDFG into {sdfg_path}")

        for params in self.sizes:
            logger.info(f"Profile with {', '.join([name +': ' + str(params[name]) for name in self.size_identifiers])}")
            sdfg = self.compile(params, run_config, debug_mode=debug_mode)
            logger.debug(f"Save SDFG into {sdfg_path}")
            sdfg.save(sdfg_path)

            if self.tot_time_repetitions > 0:
                data.extend(self.profile_total_runtime(sdfg, params, run_config))
            if self.ncu_repetitions > 0:
                data.extend(self.profile_ncu(sdfg_path, ncu_report_path, params, run_config))

        df = pd.DataFrame(data)
        index_cols = list(df.columns)
        index_cols.remove('value')
        return df.set_index(index_cols)
