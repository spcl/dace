import copy
from pathlib import Path
import shutil
import numpy as np
import dace
import os
from typing import Callable, Dict, Any, List, Iterable, Tuple
from dace.soft_hier.utils.read_from_dump_file import get_address_and_read_from_file
from dace.soft_hier.utils.interleave_handler import InterleaveHandler
import subprocess
from dace.soft_hier.utils.generate_arch_config import generate_arg_cfg


class HardwareConfig:
    """Hardware configuration container"""

    def __init__(self,
                 hardware_thread_group_dims=(2, 2),
                 hbm_addr_base="0xc0000000",
                 hbm_addr_space="0x04000000",
                 tcdm_size="0x00100000",
                 cluster_zomem_size="0x00100000",
                 redmule_ce_height=64,
                 redmule_ce_width=64,
                 redmule_ce_pipe=1,
                 hbm_placement="2,2,2,2",
                 num_node_per_ctrl=1,
                 noc_link_width=4096,
                 num_hbm_channels=8,
                 dtype_input=np.uint16,
                 dtype_output=np.uint16,
                 dace_input_type=dace.uint16,
                 dace_output_type=dace.uint16):
        self.hardware_thread_group_dims = hardware_thread_group_dims
        self.hbm_addr_base = hbm_addr_base
        self.hbm_addr_space = hbm_addr_space
        self.tcdm_size = tcdm_size
        self.cluster_zomem_size = cluster_zomem_size
        self.redmule_ce_height = redmule_ce_height
        self.redmule_ce_width = redmule_ce_width
        self.redmule_ce_pipe = redmule_ce_pipe
        self.hbm_placement = hbm_placement
        self.num_node_per_ctrl = num_node_per_ctrl
        self.noc_link_width = noc_link_width
        self.num_hbm_channels = num_hbm_channels
        self.dtype_input = dtype_input
        self.dtype_output = dtype_output
        self.dace_input_type = dace_input_type
        self.dace_output_type = dace_output_type
        


def _get_gvsoc_path() -> str:
    """Get GVSOC path from environment or by locating gvsoc binary"""
    # First try environment variable
    if "GVSOC_PATH" in os.environ:
        return os.environ["GVSOC_PATH"]

    # Try to find gvsoc binary
    gvsoc_binary = shutil.which("gvsoc")
    if gvsoc_binary:
        # Get parent directory three times: bin -> install -> gvsoc_root
        return str(Path(gvsoc_binary).parent.parent.parent)

    # Fallback to default
    raise ValueError("GVSOC PATH COULD NOT BE FOUND, SET DACE_PATH OR ENSURE gvsoc IS IN PATH")


def _get_dace_path() -> str:
    """Get DACE path from environment or by locating sdfgcc binary"""
    # First try environment variable
    if "DACE_PATH" in os.environ:
        return os.environ["DACE_PATH"]

    # Try to find sdfgcc binary
    return Path(dace.__file__).parent.parent


def setup_environment():
    """Setup paths and environment"""
    GVSOC_PATH = _get_gvsoc_path()
    DACE_PATH = _get_dace_path()
    os.environ["GVSOC_INSTALL_PATH"] = GVSOC_PATH
    os.environ["GVSOC_DIR"] = GVSOC_PATH
    os.environ["GVSOC_PATH"] = GVSOC_PATH
    os.environ["PATH"] = f"{GVSOC_PATH}/third_party/toolchain/install/bin:{os.environ['PATH']}"
    os.environ["SHCC"] = f"{GVSOC_PATH}/third_party/toolchain/install/bin/riscv32-unknown-elf-gcc"
    os.environ[
        "CPLUS_INCLUDE_PATH"] = f"{DACE_PATH}/dace/runtime/include/dace/soft_hier/runtime/include:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"
    os.environ[
        "C_INCLUDE_PATH"] = f"{DACE_PATH}/dace/runtime/include/dace/soft_hier/runtime/include:{os.environ.get('C_INCLUDE_PATH','')}"
    os.environ["SOFTHIER_INSTALL_PATH"] = f"{GVSOC_PATH}/soft_hier/flex_cluster_sdk/runtime/"


def _parse_hbm_dump(filepath: str, num_channels: int, array_names_and_data: Iterable[Tuple[str, dace.data.Data]]) -> Dict[str, Dict[int, List[str]]]:
    """Parse HBM dump file into structured format"""
    sections: Dict[int, List[str]] = {}
    section_id = -1

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("HBM offset =="):
                section_id += 1
                sections[section_id] = []
            elif line.startswith("0x"):
                sections[section_id].append(line)

    array_names_and_data_sorted = sorted(array_names_and_data, key= lambda x:x[0])
    tiles_of_channel = dict()
    for arr_name, arr in array_names_and_data_sorted:
        tiles_of_channel[arr_name] = dict()
        for i in range(num_channels):
            tiles_of_channel[arr_name][i] = len([j for j in arr.hbm_placement_scheme if j == i])

    # Re-arrange into dictionary structure
    parsed = {}
    for i, (name, data) in enumerate(array_names_and_data_sorted):
        parsed[name] = {}
        for j in range(num_channels):
            parsed[name][j] = {}
            for k in range(tiles_of_channel[name][j]):
                offset = i * num_channels + j * tiles_of_channel[name][j] + k
                if offset in sections:
                    parsed[name][j][k] = sections[offset]

    return parsed


def _read_hbm_to_numpy(array_name: str, array: dace.data.Data, handler: InterleaveHandler, element_bytes: int, dtype: str,
                       parsed: Dict[str, Dict[int, Dict[int, List[str]]]], buffer: np.ndarray) -> None:
    """Read HBM data into NumPy array"""
    assert len(buffer.shape) == 2

    for i in range(buffer.shape[0]):
        for j in range(buffer.shape[1]):
            buffer[i, j] = get_address_and_read_from_file(i=i,
                                                          j=j,
                                                          interleave_handler=handler,
                                                          array_name=array_name,
                                                          array=array,
                                                          element_size_in_bytes=element_bytes,
                                                          dtype=dtype,
                                                          parsed_sections=parsed,
                                                          debug_print=False,
                                                          debug_i=None,
                                                          debug_j=None)


def compare(hardware_config: HardwareConfig,
            numpy_results: Dict[str, np.ndarray],
            sdfg_results: Dict[str, Any],
            interleave_handlers: Dict[str, InterleaveHandler],
            sdfg: dace.SDFG,
            tolerance: float = 1e-5) -> Dict[str, Any]:
    """
    Step 5: Compare NumPy reference with SDFG results

    Args:
        hw_config: Hardware configuration
        numpy_results: Results from NumPy computation
        sdfg_results: Results from SDFG execution
        data: Original data dictionary with handlers
        dump_path: Path to HBM dump file
        tolerance: Comparison tolerance for values

    Returns:
        Dictionary containing comparison results: {
            'all_match': bool,
            'details': Dict[str, Dict] (per-array comparison details),
            'execution_time_ns': int (if available)
        }
    """
    all_match = True
    print("=" * 80)
    print("STEP 5: Compare Results")
    print("=" * 80)

    dump_path = f"{_get_gvsoc_path()}/dump_0"

    # Parse HBM dump
    if not os.path.exists(dump_path):
        print(f"Warning: Dump file not found at {dump_path}")
        return {'all_match': False, 'details': {}}

    # SDFG data is needed for the access
    details = dict()
    numpy_result_names = numpy_results.keys()
    d = list()
    for name in numpy_result_names:
        d.append((name, sdfg.arrays[name]))
    parsed = _parse_hbm_dump(dump_path, hardware_config.num_hbm_channels, d)
    for name in numpy_results:
        numpy_array = numpy_results[name]
        sdfg_array = sdfg_results[name]
        handler = interleave_handlers[name]

        element_size = numpy_array.dtype.itemsize
        dtype_str = str(numpy_array.dtype).replace('numpy.', '')

        _read_hbm_to_numpy(name, sdfg.arrays[name], handler, element_size, dtype_str, parsed, sdfg_array)

        # Compare
        diff = np.abs(sdfg_array - numpy_array)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        matches = max_diff <= tolerance

        details[name] = {
            'matches': matches,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'numpy_array': numpy_array,
            'sdfg_array': sdfg_array
        }

        status = "✓ MATCH" if matches else "✗ MISMATCH"
        print(f"  {name}: {status} (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")

        if not matches:
            all_match = False

        diff = sdfg_array - numpy_array
        print(diff)

        # dump with 3 decimal places
        with open(f"array_dump_{name}_sdfg.txt", "w") as f:
            np.savetxt(f, sdfg_array, fmt="%04x")
        with open(f"array_dump_{name}_numpy.txt", "w") as f:
            np.savetxt(f, numpy_array, fmt="%04x")
        with open(f"array_dump_{name}_diff.txt", "w") as f:
            np.savetxt(f, diff, fmt="%04x")


    print()
    print("=" * 80)
    print("✓ ALL RESULTS MATCH" if all_match else "✗ SOME RESULTS DO NOT MATCH")
    print("=" * 80)
    print()

    return {'all_match': all_match, 'details': details}


def setup_architecture(hw_config: HardwareConfig):
    generate_arg_cfg(
        cluster_tcdm_size=hex(hw_config.tcdm_size),
        cluster_zomem_size=hex(hw_config.cluster_zomem_size),
        num_cluster_x=hw_config.hardware_thread_group_dims[0],
        num_cluster_y=hw_config.hardware_thread_group_dims[1],
        redmule_ce_height=hw_config.redmule_ce_height,
        redmule_ce_width=hw_config.redmule_ce_width,
        redmule_ce_pipe=hw_config.redmule_ce_pipe,
        hbm_start_base=hex(hw_config.hbm_addr_base),
        hbm_node_addr_space=hex(hw_config.hbm_addr_space),
        hbm_placement=hw_config.hbm_placement,
        num_node_per_ctrl=hw_config.num_node_per_ctrl,
        noc_link_width=hw_config.noc_link_width,
    )
    cwd = os.getcwd()
    subprocess.run(
        [
            "bash",
            "-c",
            f"cd {_get_gvsoc_path()} && source sourceme.sh "
            f"&& export cfg={cwd}/generated_arch.py "
            f"&& make hw",
        ],
        check=True,
    )


def setup_dace_config(hw_config: HardwareConfig):
    dace.config.Config.set("backend", "softhier", "HBM_ADDRESS_SPACE", value=str(hw_config.hbm_addr_space))
    dace.config.Config.set("backend", "softhier", "HBM_ADDRESS_BASE", value=str(hw_config.hbm_addr_base))
    dace.config.Config.set("backend", "softhier", "HBM_NUM_CHANNELS", value=int(hw_config.num_hbm_channels))


def setup_hw_env_dace(hw_config: HardwareConfig):
    setup_architecture(hw_config)
    setup_environment()
    setup_dace_config(hw_config)


def run_e2e_verification(hw_config: HardwareConfig,
                         data: Dict[str, Any],
                         interleave_handlers: Dict[str, Any],
                         numpy_fn: Callable,
                         sdfg_fn: Callable,
                         tolerance: float = 1e-3) -> Dict[str, Any]:
    # Step 1 Setup
    setup_hw_env_dace(hw_config)

    # Step 2 Copy Data
    numpy_data = data
    sdfg_data = copy.deepcopy(data)

    # Step 3 Run Numpy reference
    data["C"] = numpy_fn()

    # Step 4 Run SoftHier simulator
    ret_dict = sdfg_fn()
    sdfg = ret_dict["sdfg"]

    # Step 5 Compare Data
    comparison = compare(hw_config, numpy_data, sdfg_data, interleave_handlers, sdfg, tolerance)

    return comparison['all_match']
