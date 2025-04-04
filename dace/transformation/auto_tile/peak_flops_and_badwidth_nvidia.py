import cupy as cp

import pycuda.driver as cuda
import pycuda.autoinit  # Actually used
from dace.sdfg.analysis.cutout import SDFGCutout

# https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
def ConvertSMVer2Cores(major, minor):
    # Returns the number of CUDA cores per multiprocessor for a given
    # Compute Capability version. There is no way to retrieve that via
    # the API, so it needs to be hard-coded.
    return {
        # Tesla
        (1, 0):   8,
        (1, 1):   8,
        (1, 2):   8,
        (1, 3):   8,
        # Fermi
        (2, 0):  32,
        (2, 1):  48,
        # Kepler
        (3, 0): 192,
        (3, 2): 192,
        (3, 5): 192,
        (3, 7): 192,
        # Maxwell
        (5, 0): 128,
        (5, 2): 128,
        (5, 3): 128,
        # Pascal
        (6, 0):  64,
        (6, 1): 128,
        (6, 2): 128,
        # Volta
        (7, 0):  64,
        (7, 2):  64,
        # Turing
        (7, 5):  64,
        # Ampere
        (8, 0): 64,
        (8, 6): 128,
        (8, 7): 128,
        # Ada
        (8, 9): 128,
        # Hopper
        (9, 0): 128,
    }.get((major, minor), 64)   # unknown architecture, return a default value


def get_peak_flops_and_mem_bandwidth(device_id: int):
    device = cuda.Device(0)
    dev_prop = device.get_attributes()

    num_sms = dev_prop[cuda.device_attribute.MULTIPROCESSOR_COUNT]
    clock_speed_ghz = dev_prop[cuda.device_attribute.CLOCK_RATE] / 1e6
    cuda_cores_per_sm = ConvertSMVer2Cores(
        dev_prop[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR],
        dev_prop[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR])
    peak_flops = num_sms * cuda_cores_per_sm * 2 * clock_speed_ghz

    mem_bus_width_bits = dev_prop[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]
    mem_clock_speed_ghz = dev_prop[cuda.device_attribute.MEMORY_CLOCK_RATE] / 1e6
    mem_bandwidth_gb_s = (mem_bus_width_bits * mem_clock_speed_ghz * 2) / 8

    print(f"Peak FLOPs: {peak_flops} GFLOPs")
    print(f"Memory Bandwidth: {mem_bandwidth_gb_s} GB/s")
    return peak_flops, mem_bandwidth_gb_s


def get_arch(device_id: int):
    device = cuda.Device(device_id)
    dev_prop = device.get_attributes()
    major = dev_prop[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR],
    minor = dev_prop[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]
    return f"{major}{minor}"