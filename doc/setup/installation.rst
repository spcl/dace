.. _installation:

Installation
============

DaCe is routinely tested on and officially supports Python 3.7 or newer (Python 3.6 is also supported, but not actively tested).

Dependencies
------------

Most dependencies will be resolved when the package is installed with ``pip`` or ``setup.py``. Since DaCe compiles code,
however, it requires two more runtime dependencies to be installed and available in the ``PATH`` environment variable 
(if not, see :ref:`config` for how to configure different compiler paths):

 * A C++14-capable compiler (e.g., gcc 5.3+)
 * CMake 3.15 or newer

**GPU**: For NVIDIA GPUs, the ``cuda`` toolkit is also required, and AMD GPUs require HIP.

**FPGA**: Xilinx FPGAs require the Vitis suite and Intel FPGAs require the Intel FPGA SDK to be installed.


Installing with ``pip``
-----------------------

Optional dependencies (e.g., for testing)




Installing the latest (development) version:

.. code-block:: sh

  pip install git+https://github.com/spcl/dace.git



.. _fromsource:

Installing from source
----------------------

Clone **recursively**!!!

Sometimes CMake may not install properly (or scikit-build). Remove both entries and install separately.



.. _troubleshooting:

Troubleshooting
---------------

.. note::
  Can't find your issue? Look for similar issues or start a discussion on `GitHub Discussions <https://github.com/spcl/dace/discussions>`_.


Common issues with the DaCe Python module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  * If you are unable to install DaCe from ``pip`` due to a required dependency (most commonly CMake or ``scikit-build``
    on certain platforms), follow the instructions on :ref:`fromsource`.
  * If you are using DaCe from the git repository (installed from source) and getting missing dependencies or missing include files, make sure you cloned the repository recursively (with `git clone --recursive`) and that the submodules are up to date.
  * If you are running on Mac OS and getting compilation errors when calling DaCe programs, make sure you have OpenMP installed and configured with Apple Clang. Otherwise, you can use GCC to compile the code by following these steps:

      * Run ``brew install gcc``
      * Set your ``~/.dace.conf`` compiler configuration to use the installed GCC. For example, if you installed 
        version 9 (``brew install gcc@9``), run ``which g++-9`` and set the configuration entry called ``compiler.cpu.executable`` 
        (empty string by default) to the resulting path
      * Remove any ``.dacecache`` folders to clear the cache

  * (For Windows/Visual C++ users) If compilation fails in the linkage phase, try setting the following environment variable to force Visual C++ to use Multi-Threaded linkage:

    .. code-block:: text

      X:\path\to\dace> set _CL_=/MT


  * When using fast BLAS operators (for example, matrix multiplication with Intel MKL), sometimes CMake cannot find the
    required include files or libraries on its own. If a library is installed but not found, add the include folders to
    the ``CPATH`` environment variable, and the library folders to the ``LIBRARY_PATH`` and ``LD_LIBRARY_PATH`` environment
    variables.


Common issues with Visual Studio Code extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  * If the extension keeps notifying that DaCe is not installed when the SDFG transformation pane is used,
    it may be a sign that the Visual Studio Code terminal is misconfigured.

      * At the terminal pane (default: bottom right), choose the ``SDFG Optimizer`` pane

  * ERROR 500: look at log, it may offer more insights to the origin of the issue.

  * Transformed and transformation doesn't appear in the history pane? Bug in a transformation? Try using Undo (or ctrl-z)

  * If nothing else works and the editor seems stuck, close and reopen tab. If problem persists, the SDFG may be malformed,
    load it in Python (see :ref:`format`) and call ``sdfg.validate()`` to get more information about the issue and pinpoint
    the offending node/state/edge.

