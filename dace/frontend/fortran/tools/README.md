# Fortran Frontend Tools

This document describes the workflow for converting a large Fortran project into a simplified, self-contained Fortran file, and then generating an SDFG from it. This process is divided into two main steps, each using a dedicated script.

1.  **Preprocessing and Pruning**: Use `create_preprocessed_ast.py` to parse a Fortran project, apply a series of desugaring and simplification passes, and prune away any code not reachable from specified entry points. The output is a single, self-contained Fortran file.

2.  **SDFG Generation**: Use `create_singular_sdfg_from_ast.py` to take the preprocessed Fortran file and generate a single SDFG for a specified entry point.

This workflow is particularly useful for extracting specific computational kernels from large codebases like ICON.

## 1. How to generate a preprocessed AST?

The `create_preprocessed_ast.py` script is the first step. It takes a collection of Fortran source files, identifies all code relevant to one or more entry points, and simplifies it into a single Fortran file. This process involves resolving `USE` statements, inlining modules, removing unused code, and applying various desugaring transformations.

### Usage Example

Suppose you have a Fortran project (e.g., a checkout of ICON preprocessed with `cpp`) and you want to extract the `velocity_tendencies` subroutine from the `mo_velocity_advection` module.

-   **Input Source Directories**:
    -   `~/icon-project/src`
    -   `~/icon-project/externals/support`
-   **Entry Point**: `mo_velocity_advection.velocity_tendencies`
-   **Output File**: `~/preprocessed/velocity_tendencies.f90`
-   **Functions to make no-op**: You might want to nullify certain functions (e.g., for logging or timing) that are not relevant to the computation but cannot be pruned automatically. For example:
    -   `mo_exception.finish`
    -   `mo_real_timer.timer_start`
    -   `mo_real_timer.timer_stop`

You would run the script from the root of the DaCe repository as follows:

```shell
python -m dace.frontend.fortran.tools.create_preprocessed_ast \
       -i ~/icon-project/src \
       -i ~/icon-project/externals/support \
       -o ~/preprocessed/velocity_tendencies.f90 \
       -k mo_velocity_advection.velocity_tendencies \
       --noop mo_exception.finish \
       --noop mo_real_timer.timer_start \
       --noop mo_real_timer.timer_stop
```

This command will produce a single file, `~/preprocessed/velocity_tendencies.f90`, containing all the necessary modules, types, and subroutines for the `velocity_tendencies` function, in a simplified form.

### Key Arguments

-   `-i, --in_src`: Path to a source file or directory. Can be specified multiple times.
-   `-k, --entry_point`: The entry point to keep (e.g., `module.subroutine`). Can be specified multiple times.
-   `-o, --output_ast`: Path to the output Fortran file. If omitted, prints to standard output.
-   `--noop`: A function or subroutine to replace with an empty body.
-   `-d, --checkpoint_dir`: (Optional) A directory to store intermediate ASTs for debugging.

## 2. How to generate an SDFG from a preprocessed AST?

Once you have a self-contained, preprocessed Fortran file, you can use `create_singular_sdfg_from_ast.py` to generate an SDFG from it. This script can also perform constant propagation by injecting values from configuration files.

### Usage Example

Continuing the previous example:

-   **Input File**: `~/preprocessed/velocity_tendencies.f90`
-   **Entry Point**: `mo_velocity_advection.velocity_tendencies`
-   **Output SDFG**: `~/sdfgs/velocity_tendencies.sdfg`
-   **Config Injection Files**: (Optional) A directory `~/icon-configs/` containing `.ti` files for constant propagation.

You would run the script as follows:

```shell
python -m dace.frontend.fortran.tools.create_singular_sdfg_from_ast \
       -i ~/preprocessed/velocity_tendencies.f90 \
       -k mo_velocity_advection.velocity_tendencies \
       -o ~/sdfgs/velocity_tendencies.sdfg \
       -c ~/icon-configs/
```

This will generate, simplify, and save the final SDFG to `~/sdfgs/velocity_tendencies.sdfg`.

### Key Arguments

-   `-i, --in_src`: Path to the input Fortran file. Can be specified multiple times if the preprocessed AST is split across files.
-   `-k, --entry_point`: The single entry point to use for SDFG generation (e.g., `module.subroutine`).
-   `-o, --output_sdfg`: Path to the output SDFG file (e.g., `my_kernel.sdfg`).
-   `-c, --config_inject`: (Optional) Path to a file or directory containing configuration injection data.
-   `--keep_components`: (Optional) If specified, prevents the pruning of unused derived-type components.

## 3. Generating Multiple SDFGs

The workflow described above can be easily adapted to generate multiple, separate SDFGs from the same project.

1.  **Preprocess with all entry points**: Run `create_preprocessed_ast.py` once, but specify all desired entry points using multiple `-k` arguments. This creates a single, self-contained Fortran file that includes the code for all kernels.

2.  **Generate SDFGs individually**: Run `create_singular_sdfg_from_ast.py` multiple times, once for each entry point. Each run will use the same preprocessed Fortran file as input but will specify a different entry point with the `-k` flag and a different output file with the `-o` flag.

This approach is efficient as it avoids re-parsing and preprocessing the entire Fortran project for each kernel you want to extract.
