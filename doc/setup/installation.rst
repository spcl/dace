.. _installation:

Installation
============

Dependencies
------------

Runtime dependencies:
 * A C++14-capable compiler (e.g., gcc 5.3+)
 * Python 3.7 or newer (Python 3.6 is supported but not actively tested)
 * CMake 3.15 or newer

Installing with ``pip``
-----------------------

Installing from source
----------------------



.. _troubleshooting:

Troubleshooting
---------------

  * If you are using DaCe from the git repository and getting missing dependencies or missing include files, make sure you cloned the repository recursively (with `git clone --recursive`) and that the submodules are up to date.
  * If you are running on Mac OS and getting compilation errors when calling DaCe programs, make sure you have OpenMP installed and configured with Apple Clang. Otherwise, you can use GCC to compile the code by following these steps:

      * Run ``brew install gcc``
      * Set your ``~/.dace.conf`` compiler configuration to use the installed GCC. For example, if you installed version 9 (`brew install gcc@9`), run `which g++-9` and set the config entry called `compiler.cpu.executable` (empty string by default) to the resulting path
      * Remove any ``.dacecache`` folders to clear the cache
  * (For Windows/Visual C++ users) If compilation fails in the linkage phase, try setting the following environment variable to force Visual C++ to use Multi-Threaded linkage:

    .. code-block:: text

      X:\path\to\dace> set _CL_=/MT


Common issues with Visual Studio Code extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other issues? Look for similar issues or start a discussion on our [GitHub Discussions](https://github.com/spcl/dace/discussions)!


