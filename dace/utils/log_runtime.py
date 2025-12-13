import os
from pathlib import Path
import csv

import numpy as np


def write_runtime(name, variant, runtime_us, vlen=None, cpy=None, output_dir=".", filename=None):
    """
    Write runtime measurements to a CSV file.

    Parameters:
    -----------
    name : str
        Base name of the function/kernel being timed
    variant : str
        One of: "dace", "fortran", or "dace_vec"
    runtime_us : float
        Runtime in microseconds
    vlen : int, optional
        Vector length (required if variant is "dace_vec")
    cpy : bool, optional
        Whether copies were inserted (required if variant is "dace_vec")
    output_dir : str
        Directory to store runtime files (default: "runtimes")
    """

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate variant name
    if variant == "dace":
        variant_name = "dace"
    elif variant == "fortran":
        variant_name = "fortran"
    elif variant == "dace_vec":
        if vlen is None or cpy is None:
            raise ValueError("vlen and cpy must be provided for dace_vec variant")
        cpy_suffix = "w_cpy" if cpy else "no_cpy"
        variant_name = f"dace_vec_{vlen}_{cpy_suffix}"
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be 'dace', 'fortran', or 'dace_vec'")

    # Full name for this run
    full_name = f"{name}_{variant_name}"

    # Output file path
    if filename is None:
        output_file = os.path.join(output_dir, f"{name}_runtimes.csv")
    else:
        output_file = os.path.join(output_dir, f"{filename}.csv")


    # Check if file exists and has content to determine if we need to write header
    file_exists = os.path.isfile(output_file)
    file_has_content = False
    if file_exists:
        with open(output_file, 'r') as f:
            file_has_content = len(f.read().strip()) > 0

    # Write to CSV
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header only if file doesn't exist or is empty
        if not file_has_content:
            writer.writerow(['name', 'variant', 'runtime_us', 'vlen', 'cpy'])

        # Write data
        writer.writerow([
            name, variant_name,
            float(runtime_us), vlen if vlen is not None else 'None', cpy if cpy is not None else 'None'
        ])
    print(f"Logged runtime: {full_name} = {runtime_us} us")


def compare_row_col_dicts(data_C, data_F, rtol=1e-12, atol=1e-12):
    """
    Compare two dictionaries of NumPy arrays with detailed mismatch reporting.
    """

    all_ok = True

    for key in data_C.keys():
        vC = data_C[key]
        vF = data_F[key]

        # Only compare numpy arrays
        if not isinstance(vC, np.ndarray) or not isinstance(vF, np.ndarray):
            continue
        if key == "timer":
            continue

        arrC = np.asarray(vC).reshape(-1)
        arrF = np.asarray(vF).reshape(-1)

        # isclose mask
        is_close = np.isclose(arrC, arrF, rtol=rtol, atol=atol, equal_nan=True)
        all_close = np.all(is_close)
        if all_close:
            print(f"[Pass] {key}")
        else:
            # ------ MISMATCH CASE -------
            print(f"[Mismatch] {key}")
            all_ok = False

        mismatch_idx = np.where(~is_close)[0]
        num_mismatch = mismatch_idx.size
        total = arrC.size

        if not all_close:
            print(f"  mismatches: {num_mismatch} / {total} "
                  f"({100.0 * num_mismatch / total:.6f}%)")

        # Compute diffs
        abs_diff = np.abs(arrC - arrF)
        max_abs = np.nanmax(abs_diff)
        min_abs = np.nanmin(abs_diff)

        denom = np.maximum(np.abs(arrC), np.abs(arrF))
        rel_diff = np.zeros_like(abs_diff)
        mask = denom > 0
        rel_diff[mask] = abs_diff[mask] / denom[mask]
        max_rel = np.nanmax(rel_diff)

        print(f"  max abs diff: {max_abs:e}")
        print(f"  min abs diff: {min_abs:e}")
        print(f"  max rel diff: {max_rel:e}")

        if not all_close:
            print("  first mismatches:")
            for idx in mismatch_idx[:10]:
                print(f"    index {idx}: "
                      f"C={arrC[idx]}, F={arrF[idx]}, "
                      f"abs={abs_diff[idx]:e}, rel={rel_diff[idx]:e}")

    return all_ok
