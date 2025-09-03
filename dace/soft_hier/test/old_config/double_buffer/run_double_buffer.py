import dace
import typing
import numpy as np
import os
import argparse
from dace.transformation.dataflow import DoubleBuffering


M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")
def make_preload_elf(output_file_path, np_arrays, start_addresses=None):
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
        np.dtype('float16'): 'float16',
        np.dtype('float32'): 'float',
        np.dtype('float64'): 'double',
    }
    
    ENV_PATH = os.environ.get("PATH")
    # Add RISC-V toolchain to PATH /scratch/dace4softhier/gvsoc/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin/
    os.environ["PATH"] = f"{ENV_PATH}:/scratch/dace4softhier/gvsoc/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin/"
    
    # Handle default for start_addresses
    if start_addresses is None:
        start_addresses = [None] * len(np_arrays)

    # Validate inputs
    if len(np_arrays) != len(start_addresses):
        raise ValueError("np_arrays and start_addresses must have the same length.")

    # 64-byte alignment
    alignment = 64
    current_address = 0xc0000000  # Default starting address for auto-addressing

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
            f'{c_type} array_{idx}[] __attribute__((section("{section_name}"))) = {{{array_values}}};'
        )

        current_address = start_addr + array.nbytes

    array_c_code = "\n".join(array_c_content)

    with open("array.c", "w") as f:
        f.write(array_c_code)


    # Step 2: Create "link.ld"
    link_ld_content = ["SECTIONS {"]
    current_address = 0xc0000000  # Reset for linker script auto-addressing

    for idx, (array, start_addr) in enumerate(zip(np_arrays, start_addresses)):
        section_name = section_names[idx]

        if start_addr is None:
            # Auto-determine the address with alignment
            start_addr = (current_address + alignment - 1) & ~(alignment - 1)
        link_ld_content.append(
            f"    . = 0x{start_addr:X};\n    {section_name} : {{ *({section_name}) }}"
        )
        current_address = start_addr + array.nbytes

    link_ld_content.append("}")
    link_ld_code = "\n".join(link_ld_content)

    with open("link.ld", "w") as f:
        f.write(link_ld_code)

    # Step 3: Compile the ELF file
    os.system("riscv32-unknown-elf-gcc -c array.c -o array.o")
    os.system(f"riscv32-unknown-elf-ld -T link.ld array.o -o {output_file_path}")
    os.system(f"riscv32-unknown-elf-strip --remove-section=.comment --remove-section=.Pulp_Chip.Info {output_file_path}")

    # Step 4: Cleanup
    os.remove("array.c")
    os.remove("link.ld")
    os.remove("array.o")



M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")

def _my_gen_matmul_bd_sdfg_outerK(hardware_matmul_mnk: typing.Tuple,
                     global_storage: dace.dtypes.StorageType,
                     local_storage: dace.dtypes.StorageType,
                     device_schedule: dace.dtypes.ScheduleType,
                     thread_group_schedule: dace.dtypes.ScheduleType,
                     thread_group_dims: typing.Tuple,
                     input_float,
                     output_float,
                     coarsening_factor,
                     mmad_tasklet_str: str):
    sdfg = dace.SDFG("GEMM")
    tM, tN, tK = hardware_matmul_mnk
    tM *= coarsening_factor
    tN *= coarsening_factor
    tK *= coarsening_factor
    gM, gN = thread_group_dims

    main_state = sdfg.add_state("main")
    state = main_state

    arrs = dict()
    for arr_name, shape, ftype in [("A", (M, K), input_float), ("B", (K, N), input_float), ("C", (M, N), output_float)]:
        arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False)
        arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator", shape=(coarsening_factor*coarsening_factor, tM//coarsening_factor, tN//coarsening_factor), dtype=ftype, storage=local_storage, transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(
        name="gemm_entry",
        ndrange={"i" : dace.subsets.Range([(0, M-1, tM*gM)]),
                 "j" : dace.subsets.Range([(0, N-1, tN*gN)])},
        schedule=device_schedule
    )
    for name in ["A", "B", "C"]:
    # for name in ["A", "B"]:
        if name == "A" or name == "B":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            an = state.add_access(name)
            state.add_edge(an, None, dev_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            # an = state.add_access(name)
            dev_map_exit.add_out_connector(f"OUT_{name}")
            anc3 = state.add_access(name)
            state.add_edge(dev_map_exit, f"OUT_{name}", anc3, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(
        name="thread_group_mmad",
        ndrange={"gi" : dace.subsets.Range([(0, gM-1, 1)]),
                 "gj" : dace.subsets.Range([(0, gM-1, 1)])},
        schedule=thread_group_schedule
    )

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i:i + {tM} * {gM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j:j + {tN} * {gN}"])
            elif name == "C":
                access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")
    
    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK" : dace.subsets.Range([(0, K-1, tK//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
                if name == "A":
                    access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
                elif name == "B":
                    access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                elif name == "C":
                    access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                state.add_edge(thread_group_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
                thread_group_map_entry.add_out_connector(f"OUT_{name}")
                block_tiled_map_entry.add_in_connector(f"IN_{name}")
        # if name == "C":
        #     access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
        #     state.add_edge(block_tiled_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
        #     thread_group_map_exit.add_in_connector(f"IN_{name}")
        #     block_tiled_map_exit.add_out_connector(f"OUT_{name}")

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_group_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")


    local_coarsened_access_nodes = dict()
    for name, shape in [("A", (tM, tK//coarsening_factor)), ("B", (tK//coarsening_factor, tN))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_coarsened_{name}", shape=shape, dtype=input_float, storage=local_storage, transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_coarsened_{name}")
        local_coarsened_access_nodes[f"local_coarsened_{name}"] = an


    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={"ci" : dace.subsets.Range([(0, tM-1, tM//coarsening_factor)]),
                 "cj" : dace.subsets.Range([(0, tN-1, tN//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
                if name == "A":
                    access_str_1 = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", f"bK*{tK//coarsening_factor}:bK*{tK//coarsening_factor}+{tK//coarsening_factor}"])
                    access_str_2 = ", ".join([f"0:{tM}", f"0:{tK//coarsening_factor}"])
                elif name == "B":
                    access_str_1 = ", ".join([f"bK*{tK//coarsening_factor}:bK*{tK//coarsening_factor}+{tK//coarsening_factor}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                    access_str_2 = ", ".join([f"0:{tK//coarsening_factor}", f"0:{tN}"])
                elif name == "C":
                    access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                # state.add_edge(block_tiled_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
                state.add_edge(block_tiled_map_entry, f"OUT_{name}", local_coarsened_access_nodes[f"local_coarsened_{name}"], None, dace.memlet.Memlet(f"{name}[{access_str_1}]"))
                block_tiled_map_entry.add_out_connector(f"OUT_{name}")
                state.add_edge(local_coarsened_access_nodes[f"local_coarsened_{name}"], None, thread_coarsened_map_entry, f"IN_local_coarsened_{name}", dace.memlet.Memlet(f"local_coarsened_{name}[{access_str_2}]"))
                thread_coarsened_map_entry.add_in_connector(f"IN_local_coarsened_{name}")



    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", thread_coarsened_map_entry, f"IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_out_connector("OUT_accumulator")
    thread_coarsened_map_entry.add_in_connector("IN_accumulator")

    # local_access_nodes = dict()
    for name, shape in [("A", (tM//coarsening_factor, tK//coarsening_factor)), ("B", (tK//coarsening_factor, tN//coarsening_factor))]:
        thread_coarsened_map_entry.add_out_connector(f"OUT_local_coarsened_{name}")
    #     arrn, arr = sdfg.add_array(name=f"local_{name}", shape=shape, dtype=input_float, storage=local_storage, transient=True)
    #     arrs[arrn] = arr
    #     an = state.add_access(f"local_{name}")
    #     local_access_nodes[f"local_{name}"] = an
    #     if name == "A":
    #         access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
    #                                 f"bK:bK+{tK//coarsening_factor}"])
    #     elif name == "B":
    #         access_str = ", ".join([f"bK:bK+{tK//coarsening_factor}", 
    #                                 f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
    #     state.add_edge(thread_coarsened_map_entry, f"OUT_local_coarsened_{name}", an, None, dace.memlet.Memlet(f"local_coarsened_{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule", inputs={"_in_local_a", "_in_local_b", "_in_accumulator"}, outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str, language=dace.dtypes.Language.CPP)
    
    # state.add_edge(local_access_nodes["local_A"], None, matmul_tasklet, "_in_local_a", dace.memlet.Memlet(f"local_A[0:{tM//coarsening_factor}, 0:{tK//coarsening_factor}]"))
    # state.add_edge(local_access_nodes["local_B"], None, matmul_tasklet, "_in_local_b", dace.memlet.Memlet(f"local_B[0:{tK//coarsening_factor}, 0:{tN//coarsening_factor}]"))
    state.add_edge(thread_coarsened_map_entry, f"OUT_local_coarsened_A", matmul_tasklet, "_in_local_a", dace.memlet.Memlet(f"local_coarsened_A[0:{tK//coarsening_factor}, 0:{tN//coarsening_factor}]"))
    state.add_edge(thread_coarsened_map_entry, f"OUT_local_coarsened_B", matmul_tasklet, "_in_local_b", dace.memlet.Memlet(f"local_coarsened_B[0:{tK//coarsening_factor}, 0:{tN//coarsening_factor}]"))
    access_str = ", ".join([f"ci*{coarsening_factor} + cj", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(thread_coarsened_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    thread_coarsened_map_entry.add_out_connector("OUT_accumulator")

    # accumulator_an2 = state.add_access("accumulator")
    access_str = ", ".join([f"ci*{coarsening_factor} + cj", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    # state.add_edge(matmul_tasklet, "_out_accumulator", accumulator_an2, None, dace.memlet.Memlet(f"accumulator[{access_str}]")) 
    # state.add_edge(accumulator_an2, None, thread_coarsened_map_exit, f"IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    state.add_edge(matmul_tasklet, "_out_accumulator", thread_coarsened_map_exit, f"IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_accumulator")

    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(thread_coarsened_map_exit, f"OUT_accumulator", block_tiled_map_exit, f"IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    thread_coarsened_map_exit.add_out_connector("OUT_accumulator")
    block_tiled_map_exit.add_in_connector("IN_accumulator")

    accumulator_an3 = state.add_access("accumulator")
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet(f"accumulator[{access_str}]"))

    access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
    state.add_edge(accumulator_an3, None, thread_group_map_exit, f"IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    block_tiled_map_exit.add_out_connector("OUT_accumulator")
    thread_group_map_exit.add_in_connector("IN_C")

    DoubleBuffering.apply_to(sdfg, map_entry=block_tiled_map_entry, transient=local_coarsened_access_nodes["local_coarsened_A"])
    return sdfg

def _my_gen_matmul_bd_sdfg_innerK(hardware_matmul_mnk: typing.Tuple,
                     global_storage: dace.dtypes.StorageType,
                     local_storage: dace.dtypes.StorageType,
                     device_schedule: dace.dtypes.ScheduleType,
                     thread_group_schedule: dace.dtypes.ScheduleType,
                     thread_group_dims: typing.Tuple,
                     input_float,
                     output_float,
                     coarsening_factor,
                     mmad_tasklet_str: str):
    sdfg = dace.SDFG("GEMM")
    tM, tN, tK = hardware_matmul_mnk
    tM *= coarsening_factor
    tN *= coarsening_factor
    tK *= coarsening_factor
    gM, gN = thread_group_dims

    main_state = sdfg.add_state("main")
    state = main_state

    arrs = dict()
    for arr_name, shape, ftype in [("A", (M, K), input_float), ("B", (K, N), input_float), ("C", (M, N), output_float)]:
        arrn, arr = sdfg.add_array(name=arr_name, shape=shape, dtype=ftype, storage=global_storage, transient=False)
        arrs[arrn] = arr
    arrn, arr = sdfg.add_array(name="accumulator", shape=(coarsening_factor*coarsening_factor, tM//coarsening_factor, tN//coarsening_factor), dtype=ftype, storage=local_storage, transient=True)
    arrs[arrn] = arr

    dev_map_entry, dev_map_exit = main_state.add_map(
        name="gemm_entry",
        ndrange={"i" : dace.subsets.Range([(0, M-1, tM*gM)]),
                 "j" : dace.subsets.Range([(0, N-1, tN*gN)])},
        schedule=device_schedule
    )
    for name in ["A", "B", "C"]:
    # for name in ["A", "B"]:
        if name == "A" or name == "B":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            an = state.add_access(name)
            state.add_edge(an, None, dev_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"0:{n}" for n in arrs[name].shape])
            # an = state.add_access(name)
            dev_map_exit.add_out_connector(f"OUT_{name}")
            anc3 = state.add_access(name)
            state.add_edge(dev_map_exit, f"OUT_{name}", anc3, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    thread_group_map_entry, thread_group_map_exit = main_state.add_map(
        name="thread_group_mmad",
        ndrange={"gi" : dace.subsets.Range([(0, gM-1, 1)]),
                 "gj" : dace.subsets.Range([(0, gM-1, 1)])},
        schedule=thread_group_schedule
    )

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
            if name == "A":
                access_str = ", ".join([f"i:i + {tM} * {gM}", "0:K"])
            elif name == "B":
                access_str = ", ".join(["0:K", f"j:j + {tN} * {gN}"])
            elif name == "C":
                access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(dev_map_entry, f"OUT_{name}", thread_group_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_entry.add_out_connector(f"OUT_{name}")
            thread_group_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i:i + {gM} * {tM}", f"j:j + {gN} * {tN}"])
            state.add_edge(thread_group_map_exit, f"OUT_{name}", dev_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            dev_map_exit.add_in_connector(f"IN_{name}")
            thread_group_map_exit.add_out_connector(f"OUT_{name}")

    thread_coarsened_map_entry, thread_coarsened_map_exit = main_state.add_map(
        name="thread_coarsened",
        ndrange={"ci" : dace.subsets.Range([(0, tM-1, tM//coarsening_factor)]),
                 "cj" : dace.subsets.Range([(0, tN-1, tN//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    for name in ["A", "B", "C"]:
        if name == "A" or name == "B":
                if name == "A":
                    access_str = ", ".join([f"i + gi * {tM}:i + gi * {tM} + {tM}", "0:K"])
                elif name == "B":
                    access_str = ", ".join(["0:K", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                elif name == "C":
                    access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
                state.add_edge(thread_group_map_entry, f"OUT_{name}", thread_coarsened_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
                thread_group_map_entry.add_out_connector(f"OUT_{name}")
                thread_coarsened_map_entry.add_in_connector(f"IN_{name}")
        if name == "C":
            access_str = ", ".join([f"i + gj * {tM}:i + gj * {tM} + {tM}", f"j + gj * {tN}:j + gj * {tN} + {tN}"])
            state.add_edge(thread_coarsened_map_exit, f"OUT_{name}", thread_group_map_exit, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
            thread_group_map_exit.add_in_connector(f"IN_{name}")
            thread_coarsened_map_exit.add_out_connector(f"OUT_{name}")

    block_tiled_map_entry, block_tiled_map_exit = main_state.add_map(
        name="block_tiled",
        ndrange={"bK" : dace.subsets.Range([(0, K-1, tK//coarsening_factor)])},
        schedule=dace.dtypes.ScheduleType.SoftHier_Sequential
    )

    accumulator_an = state.add_access("accumulator")
    accumulator_an.setzero = True
    state.add_edge(thread_coarsened_map_entry, None, accumulator_an, None, dace.memlet.Memlet(None))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(accumulator_an, None, block_tiled_map_entry, f"IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    thread_group_map_entry


    for name in ["A", "B"]:
        if name == "A":
            access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", "0:K"])
        elif name == "B":
            access_str = ", ".join(["0:K", f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])

        state.add_edge(thread_coarsened_map_entry, f"OUT_{name}", block_tiled_map_entry, f"IN_{name}", dace.memlet.Memlet(f"{name}[{access_str}]"))
        block_tiled_map_entry.add_in_connector(f"IN_{name}")
        thread_coarsened_map_entry.add_out_connector(f"OUT_{name}")


    # Load
    local_access_nodes = dict()
    for name, shape in [("A", (tM//coarsening_factor, tK//coarsening_factor)), ("B", (tK//coarsening_factor, tN//coarsening_factor))]:
        block_tiled_map_entry.add_out_connector(f"OUT_{name}")
        arrn, arr = sdfg.add_array(name=f"local_{name}", shape=shape, dtype=input_float, storage=local_storage, transient=True)
        arrs[arrn] = arr
        an = state.add_access(f"local_{name}")
        local_access_nodes[f"local_{name}"] = an
        if name == "A":
            access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
                                    f"bK:bK+{tK//coarsening_factor}"])
        elif name == "B":
            access_str = ", ".join([f"bK:bK+{tK//coarsening_factor}", 
                                    f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
        state.add_edge(block_tiled_map_entry, f"OUT_{name}", an, None, dace.memlet.Memlet(f"{name}[{access_str}]"))

    # Connect local_A + local_B -> matmul -> accumulator
    matmul_tasklet = state.add_tasklet(name="mmad_redmule", inputs={"_in_local_a", "_in_local_b", "_in_accumulator"}, outputs={"_out_accumulator"},
                                       code=mmad_tasklet_str, language=dace.dtypes.Language.CPP)

    #for name in ["local_A", "local_B", "accumulate"]:
    #    state.add_edge()

    for name, an in local_access_nodes.items():
        state.add_edge(an, None, matmul_tasklet, "_in_" + name.lower(), dace.memlet.Memlet(name))
    state.add_edge(block_tiled_map_entry, f"OUT_accumulator", matmul_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # accumulator_an2 = state.add_access("accumulator")
    # state.add_edge(matmul_tasklet, f"_out_accumulator", accumulator_an2, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an2, None, block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet("accumulator"))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(matmul_tasklet, "_out_accumulator", block_tiled_map_exit, "IN_accumulator", dace.memlet.Memlet(f"accumulator[{access_str}]"))
    block_tiled_map_entry.add_in_connector("IN_accumulator")
    block_tiled_map_exit.add_in_connector("IN_accumulator")
    block_tiled_map_entry.add_out_connector("OUT_accumulator")
    block_tiled_map_exit.add_out_connector("OUT_accumulator")


    # assign_tasklet = state.add_tasklet(name="assign", inputs={"_in_accumulator"}, outputs={"_out_C"}, code="_out_C = _in_accumulator")
    # state.add_edge(block_tiled_map_exit, "OUT_C", assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator")) , "_in_C"

    # accumulator_an3 = state.add_access("accumulator")
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet("accumulator"))
    # state.add_edge(accumulator_an3, None, assign_tasklet, "_in_accumulator", dace.memlet.Memlet("accumulator"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", assign_tasklet, "_in_accumulator", dace.memlet.Memlet())

    # c_an2 = state.add_access("C")
    accumulator_an3 = state.add_access("accumulator")
    
    # state.add_edge(assign_tasklet, "_out_C", c_an2, None, dace.memlet.Memlet(f"C[{access_str}]"))
    # thread_coarsened_map_entry.add_out_connector(f"OUT_C")
    # state.add_edge(c_an2, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(assign_tasklet, "_out_C", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    # state.add_edge(block_tiled_map_exit, f"OUT_accumulator", thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    access_str = ", ".join([f"0:{coarsening_factor}*{coarsening_factor}", f"0:{tM//coarsening_factor}", f"0:{tN//coarsening_factor}"])
    state.add_edge(block_tiled_map_exit, f"OUT_accumulator", accumulator_an3, None, dace.memlet.Memlet(f"accumulator[{access_str}]"))
    access_str = ", ".join([f"i + gi * {tM} + ci * {tM//coarsening_factor}:i + gi * {tM} + ci * {tM//coarsening_factor} + {tM//coarsening_factor}", 
                            f"j + gj * {tN} + cj * {tN//coarsening_factor}:j + gj * {tN} + cj * {tN//coarsening_factor} + {tN//coarsening_factor}"])
    state.add_edge(accumulator_an3, None, thread_coarsened_map_exit, "IN_C", dace.memlet.Memlet(f"C[{access_str}]"))
    thread_coarsened_map_exit.add_in_connector("IN_C")

    DoubleBuffering.apply_to(sdfg, map_entry=block_tiled_map_entry, transient=local_access_nodes["local_A"])
    return sdfg




if __name__ == "__main__":
    # create input arguments
    parser = argparse.ArgumentParser(description="Run matrix multiplication with specified dimensions.")
    parser.add_argument("--K", type=int, default=8192, help="Dimension K")
    parser.add_argument("--M", type=int, default=8192, help="Dimension M")
    parser.add_argument("--N", type=int, default=8192, help="Dimension N")
    parser.add_argument("--hardware_matmul_mnk", type=int, nargs=3, default=(256, 256, 256), help="Hardware matmul dimensions (M, N, K)")
    parser.add_argument("--thread_group_dims", type=int, nargs=2, default=(4, 4), help="Thread group dimensions (gM, gN)")

    args = parser.parse_args()

    hardware_matmul_mnk = tuple(args.hardware_matmul_mnk)
    thread_group_dims = tuple(args.thread_group_dims)
    
    sdfg = _my_gen_matmul_bd_sdfg_innerK(hardware_matmul_mnk=hardware_matmul_mnk,
                            global_storage=dace.dtypes.StorageType.SoftHier_HBM,
                            local_storage=dace.dtypes.StorageType.SoftHier_TCDM,
                            device_schedule=dace.dtypes.ScheduleType.SoftHier_Device,
                            thread_group_schedule=dace.dtypes.ScheduleType.SoftHier_Cluster,
                            thread_group_dims=thread_group_dims,
                            input_float=dace.float16,
                            output_float=dace.float16,
                            coarsening_factor=1,
                            mmad_tasklet_str="flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_NONE_16);")

    sdfg.validate()
    K = args.K
    M = args.M
    N = args.N
    start_address = 0x00000000
    A_host = np.ones((M, K), dtype=np.float16)
    B_host = np.ones((K, N), dtype=np.float16)
    # random add some zeros to A and B
    # for i in range(M):
    #     for j in range(K):
    #         if np.random.rand() < 0.5:
    #             A_host[i, j] += 1
    # for i in range(K):
    #     for j in range(N):
    #         if np.random.rand() < 0.5:
    #             B_host[i, j] = 1

    # for i in range(M):
    #     for j in range(K):
    #         if np.random.rand() < 0.1:
    #             A_host[i, j] += 4
    # for i in range(K):
    #     for j in range(N):
    #         if np.random.rand() < 0.1:
    #             B_host[i, j] += 4
    C_host = np.zeros((M, N), dtype=np.float16)

    A_address = 64 + start_address
    B_address = 64 + A_host.nbytes + start_address
    C_address = 64 + A_host.nbytes + B_host.nbytes + start_address
    # create a uint32 np array to store the addresses
    args = np.array([A_address, B_address, C_address, K, M, N], dtype=np.uint32)
    # print args in hex
    print("args in hex:")
    for i in args:
        print(hex(i))
    # make_preload_elf("/usr/scratch/badile111/dace4softhier/gvsoc/output.elf", [args, A_host, B_host, C_host])
    make_preload_elf("/usr/scratch/badile111/dace4softhier/gvsoc/output.elf", [args])
    sdfg(A=A_host, B=B_host, C=C_host, M=M, N=N, K=K)
    # sdfg.compile()