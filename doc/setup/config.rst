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

An example configuration file, which changes two configuration entries, looks as follows:

.. code-block:: yaml

  compiler:
    cuda:
      default_block_size: 64,8,1  # Change GPU map block size

  debugprint: true  # Add more verbosity in printouts


When compiling programs, the configuration used to build it will also be saved along with the binary in the 
appropriate ``.dacecache`` folder. The configuration file in that folder contains *all* configuration entries, not
just the ones changed from default, for reproducibility purposes.

.. rubric::
    Changing configuration entries via environment variables


Any configuration entry can be overridden using environment variables. To do so, create a variable that starts with 
``DACE_`` followed by the configuration entry path. Dot (``.``) characters should be replaced with ``_``.

For example, setting the CPU compiler path (:envvar:`compiler.cpu.executable`) with an environment variable can be
done as follows:

.. code-block:: sh

    $ export DACE_compiler_cpu_executable=/path/to/clang++
    $ python my_program_with_clang.py


.. rubric::
    Getting/setting configuration entries via the API


Within DaCe, obtaining or modifying configuration entries is performed by accessing the :class:`dace.config.Config` 
singleton.

Get and set values with :func:`~dace.config.Config.get` and :func:`~dace.config.Config.set`.
For boolean values, use :func:`~dace.config.Config.get_bool` to convert more options (e.g., ``1``, ``True``, ``yes``) to
booleans. If the setting is in a hierarchy, pass it as separate arguments. Examples include:

.. code-block:: python

    from dace.config import Config

    print('Synchronous debugging enabled:', Config.get_bool('compiler', 'cuda', 'syncdebug'))

    Config.set('frontend', 'unroll_threshold', value=11)


We also provide a context manager API to temporarily change the value of a configuration (useful, for example, in 
unit tests, where configuration changes must not persist outside of a test):

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

 * :envvar:`debugprint`: Print debugging information. If set to ``"verbose"``, prints more debugging information.
 * :envvar:`compiler.use_cache`: Uses DaCe program cache instead of recompiling programs. Also useful for debugging
   code generation (see :ref:`debug_codegen`).
 * :envvar:`compiler.default_data_types`: Chooses default types for integer and floating-point values. If 
   ``Python`` is chosen, ``int`` and ``float`` are both 64-bit wide. If ``C`` is chosen, ``int`` and ``float`` are 32-bit wide.
 * :envvar:`optimizer.automatic_simplification`: If False, skips automatic simplification in the Python frontend 
   (see :ref:`simplify` for more information).
 
Profiling:

 * :envvar:`profiling`: Enables profiling measurement of the DaCe program runtime in milliseconds. 
   Produces a log file and prints out median runtime. See :ref:`profiling` for more information.
 * :envvar:`treps`: Number of repetitions to run when profiling is enabled.
 
GPU programming and debugging:

 * :envvar:`compiler.cuda.backend`: Chooses the GPU backend to use (can be ``cuda`` for NVIDIA GPUs or 
   ``hip`` for AMD GPUs).
 * :envvar:`compiler.cuda.syncdebug` (default: False): If True, calls device-synchronization after every GPU kernel and checks
   for errors. Good for checking crashes or invalid memory accesses.
 
FPGA programming:

 * :envvar:`compiler.fpga.vendor`: Can be ``xilinx`` for Xilinx FPGAs, or ``intel_fpga`` for Intel FPGAs.
