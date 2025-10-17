import os
import numpy as np
from .interleave_handler import InterleaveHandler


def make_preload_elf(output_file_path,
                     np_arrays,
                     start_addresses=None,
                     hbm_node_addr_base=0xc0000000,
                     hbm_node_addr_space=0x04000000):
    """
    Generate an ELF file preloading numpy arrays.

    Parameters:
    - output_file_path (str): Path to save the output ELF file.
    - np_arrays (list of numpy.ndarray): List of numpy arrays to include in the ELF.
    - start_addresses (list of int or None): List of starting addresses for each array, or None.
      If None, addresses are auto-determined with 64-byte alignment.
    """
    NP_DTYPE_TO_C = {
        np.dtype('int8'): 'int8_t',
        np.dtype('uint8'): 'uint8_t',
        np.dtype('int16'): 'int16_t',
        np.dtype('uint16'): 'uint16_t',
        np.dtype('int32'): 'int32_t',
        np.dtype('uint32'): 'uint32_t',
        np.dtype('int64'): 'int64_t',
        np.dtype('uint64'): 'uint64_t',
        np.dtype('float16'): '_Float16',
        np.dtype('float32'): 'float',
        np.dtype('float64'): 'double',
    }

    ENV_PATH = os.environ.get("PATH")
    # Add RISC-V toolchain to PATH /scratch/dace4softhier/gvsoc/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin/
    # os.environ["PATH"] = f"{ENV_PATH}:/scratch/dace4softhier/gvsoc/third_party/toolchain/install/bin/"

    # Handle default for start_addresses
    if start_addresses is None:
        start_addresses = [None] * len(np_arrays)

    # Validate inputs
    if len(np_arrays) != len(start_addresses):
        raise ValueError("np_arrays and start_addresses must have the same length.")

    # 64-byte alignment
    alignment = 64
    current_address = hbm_node_addr_base  # Default starting address for auto-addressing

    # Step 1: Create "array.c"
    array_c_content = ['#include <stdint.h>']
    section_names = []

    for idx, (array, start_addr) in enumerate(zip(np_arrays, start_addresses)):
        # Determine C type from NumPy dtype
        c_type = NP_DTYPE_TO_C.get(array.dtype, None)
        if c_type is None:
            raise TypeError(f"Unsupported NumPy dtype: {array.dtype}")

        section_name = f".custom_section_{idx}"
        section_names.append(section_name)

        if start_addr is None:
            # Auto-determine the address with alignment
            start_addr = (current_address + alignment - 1) & ~(alignment - 1)
        else:
            # Ensure provided addresses are aligned
            if start_addr % alignment != 0:
                raise ValueError(f"Provided address {start_addr} is not {alignment}-byte aligned.")

        # Generate the array definition
        array_values = ", ".join(map(str, array.flatten()))
        array_c_content.append(
            f'{c_type} array_{idx}[] __attribute__((section("{section_name}"))) = {{{array_values}}};')

        current_address = start_addr + array.nbytes

    array_c_code = "\n".join(array_c_content)

    with open("array.c", "w") as f:
        f.write(array_c_code)

    # Step 2: Create "link.ld"
    link_ld_content = ["SECTIONS {"]
    current_address = hbm_node_addr_base  # Reset for linker script auto-addressing

    for idx, (array, start_addr) in enumerate(zip(np_arrays, start_addresses)):
        section_name = section_names[idx]

        if start_addr is None:
            # Auto-determine the address with alignment
            start_addr = (current_address + alignment - 1) & ~(alignment - 1)
        link_ld_content.append(f"    . = 0x{start_addr:X};\n    {section_name} : {{ *({section_name}) }}")
        current_address = start_addr + array.nbytes

    link_ld_content.append("}")
    link_ld_code = "\n".join(link_ld_content)

    with open("link.ld", "w") as f:
        f.write(link_ld_code)

    # Step 3: Compile the ELF file
    os.system("riscv64-unknown-elf-gcc -c array.c -o array.o")
    os.system(f"riscv64-unknown-elf-ld -T link.ld array.o -o {output_file_path}")
    os.system(
        f"riscv64-unknown-elf-strip --remove-section=.comment --remove-section=.Pulp_Chip.Info {output_file_path}")

    # Step 4: Cleanup
    os.remove("array.c")
    os.remove("link.ld")
    os.remove("array.o")


def make_preload_elf_hbm_interleaved_new(output_file_path,
                                         Handler_list: list[InterleaveHandler],
                                         KMN=None,
                                         hbm_node_addr_base=0xc0000000,
                                         hbm_node_addr_space=0x04000000,
                                         args_only=True):
    """
    Split np arrays into tiles and blocks and then use make_preload_elf to generate an ELF file preloading numpy arrays.

    """
    np_arrays = [handler.array for handler in Handler_list]
    split_schemes = [handler.split_scheme for handler in Handler_list]
    placement_schemes = [handler.placement_scheme for handler in Handler_list]
    hardware_block_sizes = [handler.block_shape for handler in Handler_list]
    total_channels = (Handler_list[0].cluster_dims[0] + Handler_list[0].cluster_dims[1]) * 2

    # 1) Combine relevant info with original indices
    arrays_info = [(idx, array, split_schemes[idx], placement_schemes[idx], hardware_block_sizes[idx])
                   for idx, array in enumerate(np_arrays)]

    # Helper to compute the number of channels from a placement scheme
    def get_num_channels(placement_scheme):
        return len(set(placement_scheme))

    # 2) Sort by number of channels used, descending
    arrays_info.sort(key=lambda x: get_num_channels(x[3]), reverse=True)
    #        index: x[0]
    #        array: x[1]
    # split_scheme: x[2]
    #placement_scheme: x[3]
    #hardware_block_size: x[4]

    # 3) Prepare channel start addresses
    current_start_address = 64
    start_addr_in_each_channel = [current_start_address] * total_channels

    # We'll store each array's start address in a list keyed by original index
    # so we can return them in the original order.
    start_addresses_in_original_order = [None] * len(np_arrays)

    # 4) Allocate arrays in sorted order
    for idx, array, split_scheme, placement_scheme, hardware_block_size in arrays_info:
        num_channels = get_num_channels(placement_scheme)

        # Basic tile check
        array_shape = array.shape
        tile_height = array_shape[0] // split_scheme[0]
        tile_length = array_shape[1] // split_scheme[1]
        if tile_height < hardware_block_size[0] or tile_length < hardware_block_size[1]:
            raise ValueError(f"Invalid tile size: {tile_height}x{tile_length}"
                             f" < {hardware_block_size[0]}x{hardware_block_size[1]}")

        # Compute how many bytes total and how many bytes per channel
        array_size = array.nbytes
        array_size_per_tile = array_size // len(placement_scheme)
        array_size_per_channel = array_size // num_channels

        # Grab the start address for the "first" channel in the group
        start_channel = placement_scheme[0]
        array_start_addr_in_channel = start_addr_in_each_channel[start_channel]
        print(f"array_start_addr_in_channel: {hex(array_start_addr_in_channel)}")

        # 5) Check if the placement scheme is valid
        valid_placement = True
        for channel_idx in set(placement_scheme):
            if start_addr_in_each_channel[channel_idx] != array_start_addr_in_channel:
                valid_placement = False
                Warning(f"Invalid placement scheme: {placement_scheme}, "
                        f"channel {channel_idx} start address mismatch")

        # 6) Store that start address in the array's original position
        if valid_placement:
            start_addresses_in_original_order[idx] = array_start_addr_in_channel
        else:
            # Choose the maximum start address among the channels used
            max_start_addr = max(start_addr_in_each_channel[channel_idx] for channel_idx in set(placement_scheme))
            start_addresses_in_original_order[idx] = max_start_addr
            # Update all channels used to this maximum start address
            for channel_idx in set(placement_scheme):
                start_addr_in_each_channel[channel_idx] = max_start_addr

        # 7) Advance addresses in all channels used by this array
        for i in range(len(placement_scheme)):
            channel_idx = placement_scheme[i]
            start_addr_in_each_channel[channel_idx] += array_size_per_tile

    args = start_addresses_in_original_order
    for arg in args:
        print(f"arg: {hex(arg)}")

    split_arrays = []
    split_arrays.append(args)
    split_arrays_start_addresses = []
    split_arrays_start_addresses.append(hbm_node_addr_base)  # for store args

    if not args_only:
        for array, split_scheme, placement_scheme, hardware_block_size, arg_start_addr in zip(
                np_arrays, split_schemes, placement_schemes, hardware_block_sizes, args):
            current_start_address = arg_start_addr
            print(f"current_start_address: {hex(current_start_address)}")
            block_height = hardware_block_size[0]
            block_length = hardware_block_size[1]
            block_size = block_height * block_length * np.dtype(array.dtype).itemsize
            print(f"block_size: {block_size}")
            # channel_start = placement_scheme[0]
            # print(f"channel_start: {channel_start}")
            # channel_end = placement_scheme[1]
            # print(f"channel_end: {channel_end}")
            # channel_stride = placement_scheme[2]
            # print(f"channel_stride: {channel_stride}")
            num_channels = get_num_channels(placement_scheme)
            array_shape = array.shape
            print(f"array_shape: {array_shape}")
            tile_height = array_shape[0] // split_scheme[0]
            print(f"tile_height: {tile_height}")
            tile_length = array_shape[1] // split_scheme[1]
            print(f"tile_length: {tile_length}")
            tile_size = tile_length * tile_height * np.dtype(array.dtype).itemsize
            print(f"tile_size: {tile_size}")
            for i in range(split_scheme[0]):
                for j in range(split_scheme[1]):
                    tile_idx = i * split_scheme[1] + j
                    print(f"tile_idx: {tile_idx}")
                    # channel_offset = tile_idx % num_channels
                    # print(f"channel_offset: {channel_offset}")
                    channel_idx = placement_scheme[tile_idx]
                    print(f"channel_idx: {channel_idx}")
                    # How many tiles have been assigned to this channel before this one?
                    tile_offset = sum(1 for t in range(tile_idx) if placement_scheme[t] == channel_idx)
                    print(f"tile_offset: {tile_offset}")
                    tile = array[i * tile_height:(i + 1) * tile_height, j * tile_length:(j + 1) * tile_length]
                    for bi in range(0, tile_height, block_height):
                        for bj in range(0, tile_length, block_length):
                            print(f"bi: {bi}, bj: {bj}")
                            block = tile[bi:bi + block_height, bj:bj + block_length]
                            split_arrays.append(block)
                            bi_index = bi // block_height
                            bj_index = bj // block_length
                            block_address = hbm_node_addr_base + current_start_address + channel_idx * hbm_node_addr_space + tile_offset * tile_size + (
                                bi_index * tile_length // block_length + bj_index) * block_size
                            print(f"block_address: {hex(block_address)}")
                            split_arrays_start_addresses.append(block_address)

    if KMN is not None:
        args.append(KMN[0])
        args.append(KMN[1])
        args.append(KMN[2])
    # args to np arrays
    args = np.array(args, dtype=np.uint32)

    # replace the args in split_arrays with new args
    split_arrays[0] = args

    make_preload_elf(output_file_path, split_arrays, split_arrays_start_addresses, hbm_node_addr_base,
                     hbm_node_addr_space)

    return args


def make_preload_elf_hbm_interleaved(output_file_path,
                                     np_arrays,
                                     split_schemes,
                                     placement_schemes,
                                     hardware_block_sizes,
                                     start_addresses=None,
                                     KMN=None,
                                     total_channels=16,
                                     hbm_node_addr_base=0xc0000000,
                                     hbm_node_addr_space=0x02000000,
                                     args_only=True):
    """
    Split np arrays into tiles and blocks and then use make_preload_elf to generate an ELF file preloading numpy arrays.

    """
    # 1) Combine relevant info with original indices
    arrays_info = [(idx, array, split_schemes[idx], placement_schemes[idx], hardware_block_sizes[idx])
                   for idx, array in enumerate(np_arrays)]

    # Helper to compute the number of channels from a placement scheme
    def get_num_channels(placement_scheme):
        start_channel, end_channel, stride = placement_scheme
        return (end_channel - start_channel + 1) // stride

    # 2) Sort by number of channels used, descending
    arrays_info.sort(key=lambda x: get_num_channels(x[3]), reverse=True)
    #        index: x[0]
    #        array: x[1]
    # split_scheme: x[2]
    #placement_scheme: x[3]
    #hardware_block_size: x[4]

    # 3) Prepare channel start addresses
    current_start_address = 64
    start_addr_in_each_channel = [current_start_address] * total_channels

    # We'll store each array's start address in a list keyed by original index
    # so we can return them in the original order.
    start_addresses_in_original_order = [None] * len(np_arrays)

    # 4) Allocate arrays in sorted order
    for idx, array, split_scheme, placement_scheme, hardware_block_size in arrays_info:
        start_channel, end_channel, stride = placement_scheme
        num_channels = get_num_channels(placement_scheme)

        # Basic tile check
        array_shape = array.shape
        tile_height = array_shape[0] // split_scheme[0]
        tile_length = array_shape[1] // split_scheme[1]
        if tile_height < hardware_block_size[0] or tile_length < hardware_block_size[1]:
            raise ValueError(f"Invalid tile size: {tile_height}x{tile_length}"
                             f" < {hardware_block_size[0]}x{hardware_block_size[1]}")

        # Compute how many bytes total and how many bytes per channel
        array_size = array.nbytes
        array_size_per_channel = array_size // num_channels

        # Grab the start address for the "first" channel in the group
        array_start_addr_in_channel = start_addr_in_each_channel[start_channel]

        # 5) Store that start address in the array's original position
        start_addresses_in_original_order[idx] = array_start_addr_in_channel

        # 6) Advance addresses in all channels used by this array
        for i in range(num_channels):
            channel_idx = (start_channel + i * stride) + total_channels
            channel_idx %= total_channels

            # Optional: Validate all channels in the group are at the same start
            if start_addr_in_each_channel[channel_idx] != array_start_addr_in_channel:
                raise ValueError(f"Invalid placement scheme: {placement_scheme}, "
                                 f"channel {channel_idx} start address mismatch")
            start_addr_in_each_channel[channel_idx] += array_size_per_channel

    args = start_addresses_in_original_order
    for arg in args:
        print(f"arg: {hex(arg)}")

    split_arrays = []
    split_arrays.append(args)
    split_arrays_start_addresses = []
    split_arrays_start_addresses.append(hbm_node_addr_base)  # for store args

    if not args_only:
        for array, split_scheme, placement_scheme, hardware_block_size, arg_start_addr in zip(
                np_arrays, split_schemes, placement_schemes, hardware_block_sizes, args):
            current_start_address = arg_start_addr
            print(f"current_start_address: {hex(current_start_address)}")
            block_height = hardware_block_size[0]
            block_length = hardware_block_size[1]
            block_size = block_height * block_length * np.dtype(array.dtype).itemsize
            print(f"block_size: {block_size}")
            channel_start = placement_scheme[0]
            print(f"channel_start: {channel_start}")
            channel_end = placement_scheme[1]
            print(f"channel_end: {channel_end}")
            channel_stride = placement_scheme[2]
            print(f"channel_stride: {channel_stride}")
            num_channels = (channel_end - channel_start + 1) // channel_stride
            array_shape = array.shape
            print(f"array_shape: {array_shape}")
            tile_height = array_shape[0] // split_scheme[0]
            print(f"tile_height: {tile_height}")
            tile_length = array_shape[1] // split_scheme[1]
            print(f"tile_length: {tile_length}")
            tile_size = tile_length * tile_height * np.dtype(array.dtype).itemsize
            print(f"tile_size: {tile_size}")
            for i in range(split_scheme[0]):
                for j in range(split_scheme[1]):
                    tile_idx = i * split_scheme[1] + j
                    print(f"tile_idx: {tile_idx}")
                    channel_offset = tile_idx % num_channels
                    print(f"channel_offset: {channel_offset}")
                    channel_idx = channel_start + channel_offset * channel_stride
                    channel_idx = (channel_idx + total_channels) % total_channels
                    print(f"channel_idx: {channel_idx}")
                    tile = array[i * tile_height:(i + 1) * tile_height, j * tile_length:(j + 1) * tile_length]
                    for bi in range(0, tile_height, block_height):
                        for bj in range(0, tile_length, block_length):
                            print(f"bi: {bi}, bj: {bj}")
                            block = tile[bi:bi + block_height, bj:bj + block_length]
                            split_arrays.append(block)
                            bi_index = bi // block_height
                            bj_index = bj // block_length
                            block_address = hbm_node_addr_base + current_start_address + channel_idx * hbm_node_addr_space + (
                                tile_idx // num_channels) * tile_size + (bi_index * tile_length // block_length +
                                                                         bj_index) * block_size
                            print(f"block_address: {hex(block_address)}")
                            split_arrays_start_addresses.append(block_address)

    if KMN is not None:
        args.append(KMN[0])
        args.append(KMN[1])
        args.append(KMN[2])
    # args to np arrays
    args = np.array(args, dtype=np.uint32)

    # replace the args in split_arrays with new args
    split_arrays[0] = args

    make_preload_elf(output_file_path, split_arrays, split_arrays_start_addresses, hbm_node_addr_base,
                     hbm_node_addr_space)

    return args
