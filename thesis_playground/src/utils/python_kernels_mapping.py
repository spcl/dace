from typing import Dict
import dace

from python_programs import vert_loop_7, vert_loop_7_1, vert_loop_7_1_no_klon, vert_loop_7_1_no_temp, \
                            vert_loop_wip, vert_loop_7_2, vert_loop_7_3, vert_loop_wip_scalar_offset, small_wip, \
                            small_wip_2, small_wip_3, small_wip_4, mwe_memlet_range, vert_loop_10


kernels = {
    'vert_loop_7': vert_loop_7,
    'vert_loop_7_1': vert_loop_7_1,
    'vert_loop_7_2': vert_loop_7_2,
    'vert_loop_7_3': vert_loop_7_3,
    'vert_loop_7_1_no_klon': vert_loop_7_1_no_klon,
    'vert_loop_7_1_no_temp': vert_loop_7_1_no_temp,
    'vert_loop_wip': vert_loop_wip,
    'vert_loop_wip_scalar_offset': vert_loop_wip_scalar_offset,
    'small_wip': small_wip,
    'small_wip_2': small_wip_2,
    'small_wip_3': small_wip_3,
    'small_wip_4': small_wip_4,
    'mwe_memlet_range': mwe_memlet_range,
    'vert_loop_10': vert_loop_10,
}


def get_function_by_name(name: str) -> Dict[str, dace.frontend.python.parser.DaceProgram]:
    """
    Get python kernel/program/function given its name

    :param name: The name of the kernel/program/function
    :type name: str
    :return: The DaCe kernel/function/program
    :rtype: Dict[str, dace.frontend.python.parser.DaceProgram]
    """
    return kernels[name]
