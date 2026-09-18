.. _config:

Configuring DaCe
================

Various aspects of DaCe can be configured through a YAML file called ``.dace.conf``, through ``DACE_*`` environment
variables, or through the configuration API. DaCe never creates or modifies a configuration file on its own:
:func:`~dace.config.Config.save` writes one explicitly.

.. note::
    Documentation for all configuration entries is available at the :ref:`config_schema`.



DaCe reads at most one configuration file, the first one found: a ``.dace.conf`` in the current working directory,
then the file pointed to by the ``DACE_CONFIG`` environment variable, then ``.dace.conf`` in the user's home
directory. If none exists, the schema defaults are used.

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

Environment variables are read once, when the configuration is loaded (at import time, or on an explicit
:func:`~dace.config.Config.load`); their values are coerced to the entry's declared type. Changing a ``DACE_*``
variable afterwards has no effect until the configuration is reloaded, and values set through the API always take
precedence over the environment.

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


If an entry is defined in multiple places, its value is decided when the configuration is loaded, with the following
sources in increasing priority:

1. The default value from the configuration schema
2. The configuration file (the first one found, see above)
3. A ``DACE_*`` environment variable

Values set through the API afterwards (:func:`~dace.config.Config.set`, :func:`~dace.config.set_temporary`,
:func:`~dace.config.temporary_config`) have the highest priority, since the environment is only consulted while
loading. Loading never writes the configuration file, so neither environment values nor API changes end up in
``.dace.conf`` unless :func:`~dace.config.Config.save` is called explicitly.


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
