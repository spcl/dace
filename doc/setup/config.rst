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
For example, TODO


.. rubric::
    Getting/setting configuration entries via the API


:class:`dace.config.Config` is a singleton class.

the hierarchy is used as separate arguments:

:func:`~dace.config.Config.get` for values, :func:`~dace.config.Config.get_bool` for boolean values

.. code-block:: python

    from dace.config import Config

    print('Synchronous debugging enabled:', Config.get_bool('compiler', 'cuda', 'syncdebug'))

    Config.set('frontend', 'unroll_threshold', value=11)


TODO

, we also provide an API to temporarily change the value of a configuration

.. code-block:: python

    # Temporarily enable profiling for one call
    with dace.config.set_temporary('profiling', value=True):
        dace_laplace(A, args.iterations)




.. rubric::
    Deciding the value of a configuration entry


If an entry is defined in multiple places, the priority order for determining the value is as follows:

1. If a ``DACE_*`` environment variable is found, its value will always be used
2. Otherwise, the API (:func:`~dace.config.Config.set`, :func:`~dace.config.set_temporary`) is used
3. Value located in a ``.dace.conf`` file in the current working directory
4. Lastly, the value will be searched in ``.dace.conf`` located in the user's home directory or the path pointed to by 
   the ``DACE_CONFIG`` environment variable


.. rubric::
    Useful configuration entries




General configuration:

 * ``debugprint`` (default: False): Print debugging information. If set to ``"verbose"``, prints more debugging information.
 * ``compiler.use_cache`` (default: False): Uses DaCe program cache instead of recompiling programs. Also useful for debugging
   code generation (see :ref:`debug_codegen`).
 * ``compiler.default_data_types`` (default: ``Python``): Chooses default types for integer and floating-point values. If 
   ``Python`` is chosen, ``int`` and ``float`` are both 64-bit wide. If ``C`` is chosen, ``int`` and ``float`` are 32-bit wide.
 * ``optimizer.automatic_simplification`` (default: True): If False, skips automatic simplification in the Python frontend 
   (see :ref:`simplify` for more information).
 
Profiling:

 * ``profiling`` (default: False): Enables profiling measurement of the DaCe program runtime in milliseconds. 
   Produces a log file and prints out median runtime. See :ref:`profiling` for more information.
 * ``treps`` (default: 100): Number of repetitions to run when profiling is enabled.
 
GPU programming and debugging:

 * ``compiler.cuda.backend`` (default: ``cuda``): Chooses the GPU backend to use (can be ``cuda`` for NVIDIA GPUs or 
   ``hip`` for AMD GPUs).
 * ``compiler.cuda.syncdebug`` (default: False): If True, calls device-synchronization after every GPU kernel and checks
   for errors. Good for checking crashes or invalid memory accesses.
 
FPGA programming:

 * ``compiler.fpga.vendor``: (default: ``xilinx``): Can be ``xilinx`` for Xilinx FPGAs, or ``intel_fpga`` for Intel FPGAs.
