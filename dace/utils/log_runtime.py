import os
from pathlib import Path
import csv

import numpy as np

def write_runtime(name, variant, runtime_us, vlen=None, cpy=None, output_dir="."):
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
    output_file = os.path.join(output_dir, f"{name}_runtimes.csv")
    
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
            name,
            variant_name,
            float(runtime_us),
            vlen if vlen is not None else 'None',
            cpy if cpy is not None else 'None'
        ])
    print(f"Logged runtime: {full_name} = {runtime_us} us")

def compare_row_col_dicts(data_C, data_F, rtol=1e-12, atol=1e-12):
    """
    Compare two dictionaries (row-major and column-major versions)
    using numpy.allclose on all ndarray entries.

    Scalars and non-array entries are skipped.

    Returns
    -------
    all_ok : bool
        True if all arrays match within tolerance.
    """
    all_ok = True

    for key in data_C.keys():
        vC = data_C[key]
        vF = data_F[key]

        # Only compare arrays
        if not isinstance(vC, np.ndarray) or not isinstance(vF, np.ndarray):
            continue
        if key == "timer":
            continue
        # Flatten both arrays so row/col layout differences disappear
        arrC = np.asarray(vC).reshape(-1)
        arrF = np.asarray(vF).reshape(-1)

        if not np.allclose(arrC, arrF, rtol=rtol, atol=atol, equal_nan=True):
            print(f"[Mismatch] {key}")
            all_ok = False
        else:
            print(f"[Pass] {key}")
    
    return all_ok
