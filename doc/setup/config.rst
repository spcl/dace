.. _config:

Configuring DaCe
================

Various aspects of DaCe can be configured. When first run, the framework creates a file called ``.dace.conf``. The
file is written in YAML format and provides useful settings that can be modified either directly or overridden on a
case-by-case basis using the configuration API or environment variables.

.. note::
    Documentation for all configuration entries is available at the :ref:`config_schema`.



DaCe will first try to search for the configuration file in the ``DACE_CONFIG`` environment variable, if exists.
Otherwise, it will then look for a ``.dace.conf`` file in the current working directory. If not found,
it will look for it in the user's home directory. By default, if no file can be found a new one will be created in 
the home directory. If the home directory does not exist (e.g., in Docker containers), the file will be created in the
current working directory. If no configuration file can be created in any of the above paths, the default settings are used.


.. rubric::
    Changing configuration entries via environment variables


Environment variables that begin with ``DACE_`` and specify the entry, where categories are separated by underscores.
For example, 


.. rubric::
    Changing configuration entries via the API


TODO



.. rubric::
    Deciding the value of a configuration entry

The priority order for configuration files is as follows:

1. If a ``DACE_*`` environment variable is found, its value will always be used
2. If ``with dace.config.set_temporary(...)`` is used, for example:

.. code-block:: python

    # Temporarily enable profiling for one call
    with dace.config.set_temporary('profiling', value=True):
        dace_laplace(A, args.iterations)


3. A ``.dace.conf`` located in the current working directory
4. The ``.dace.conf`` located in the user's home directory or the path pointed to by the ``DACE_CONFIG`` environment variable



.. rubric::
    Useful configuration entries




General configuration:

 * ``DACE_debugprint`` (default: False): Print debugging information.
 * ``DACE_compiler_use_cache`` (default: False): Uses DaCe program cache instead of re-optimizing and compiling programs.
 * ``DACE_compiler_default_data_types`` (default: ``Python``): Chooses default types for integer and floating-point values. If ``Python`` is chosen, ``int`` and ``float`` are both 64-bit wide. If ``C`` is chosen, ``int`` and ``float`` are 32-bit wide.
 
Profiling:

 * ``DACE_profiling`` (default: False): Enables profiling measurement of the DaCe program runtime in milliseconds. Produces a log file and prints out median runtime.
 * ``DACE_treps`` (default: 100): Number of repetitions to run a DaCe program when profiling is enabled.
 
GPU programming and debugging:

 * ``DACE_compiler_cuda_backend`` (default: ``cuda``): Chooses the GPU backend to use (can be ``cuda`` for NVIDIA GPUs or ``hip`` for AMD GPUs).
 * ``DACE_compiler_cuda_syncdebug`` (default: False): If True, calls device-synchronization after every GPU kernel and checks for errors. Good for checking crashes or invalid memory accesses.
 
FPGA programming:

 * ``DACE_compiler_fpga_vendor``: (default: ``xilinx``): Can be ``xilinx`` for Xilinx FPGAs, or ``intel_fpga`` for Intel FPGAs.
 
SDFG interactive transformation:

 * ``DACE_optimizer_transform_on_call`` (default: False): Uses the transformation command line interface every time a ``@dace`` function is called.
 * ``DACE_optimizer_interface`` (default: ``dace.transformation.optimizer.SDFGOptimizer``): Controls the SDFG optimization process if ``transform_on_call`` is enabled. By default, uses the transformation command line interface.
 * ``DACE_optimizer_automatic_simplification`` (default: True): If False, skips automatic simplification in the Python frontend (see transformations tutorial for more information).
 
