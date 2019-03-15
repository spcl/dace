![D](dace.svg)aCe - Data-Centric Parallel Programming
=====================================================

_Decoupling domain science from performance optimization._

DaCe compiles code in various programming languages and paradigms (Python/Numpy, MATLAB, TensorFlow) and maps it efficiently to **CPUs, GPUs, and FPGAs** with high utilization, on par with the state-of-the-art. The key feature driving DaCe is its Stateful DataFlow multiGraph (SDFG) *data-centric intermediate representation*: A transformable, interactive representation of code based on data movement.
With data-centric parallel programming, we enable **direct knowledge transfer** of performance optimization, regardless of the scientific application or the target processor.

DaCe can be written inline in Python and transformed in the command-line, or SDFGs can be interactively modified using the Data-centric Interactive Optimization Development Environment (DIODE).

For more information, see our [paper](http://www.arxiv.org/abs/1902.10345).

Tutorials
---------

* _Implicit Dataflow in Python (coming soon)_
* [Explicit Dataflow in Python](tutorials/explicit.ipynb)
* [SDFG API](tutorials/sdfg_api.ipynb)
* [Transformations](tutorials/transformations.ipynb)

Installation and Dependencies
-----------------------------

To install: `pip install dace`

Runtime dependencies:
 * A C++14-capable compiler (e.g., gcc 5.3+)
 * Python 3.5 or newer

Running DIODE may require additional dependencies:
 * `sudo apt-get install libgtksourceviewmm-3.0-dev libyaml-dev`
 * `sudo apt-get install python3-cairo python3-gi-cairo libgirepository1.0-dev xdot libwebkitgtk-dev libwebkitgtk-3.0-dev libwebkit2gtk-4.0-dev`
 * `pip install pygobject matplotlib`

To run DIODE on Windows, use MSYS2:
 * Download from http://www.msys2.org/
 * In the MSYS2 console, install all dependencies: `pacman -S mingw-w64-i686-gtk3 mingw-w64-i686-python2-gobject mingw-w64-i686-python3-gobject mingw-w64-i686-python3-cairo mingw-w64-i686-python3-pip mingw-w64-i686-gtksourceviewmm3 mingw-w64-i686-gcc mingw-w64-i686-boost mingw-w64-i686-python3-numpy mingw-w64-i686-python3-scipy mingw-w64-i686-python3-matplotlib`
 * Update MSYS2: `pacman -Syu`, close and restart MSYS2, then run `pacman -Su` to update the rest of the packages.

Publication
-----------

If you use DaCe, cite us:
```bibtex
@article{dace,
  author = {Ben-Nun, Tal and de Fine Licht, Johannes and Ziogas, Alexandros Nikolaos and Schneider, Timo and Hoefler, Torsten},
        title = {Stateful Dataflow Multigraphs: A Data-Centric Model for High-Performance Parallel Programs},
  journal   = {CoRR},
  volume    = {abs/1902.10345},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.10345},
  archivePrefix = {arXiv},
  eprint    = {1902.10345}
}
```

Configuration
-------------

DaCe creates a file called `.dace.conf` in the user's home directory. It provides useful settings that can be modified either directly in the file (YAML), within DIODE, or overriden on a case-by-case basis using environment variables that begin with `DACE_` and specify the setting (where categories are separated by underscores). The full configuration schema is located [here](dace/config_schema.yml).

Useful environment variable configurations include:

* `DACE_CONFIG` (default: `~/.dace.conf`): Override DaCe configuration file choice.

Context configuration:
 * `DACE_use_cache` (default: False): Uses DaCe program cache instead of re-optimizing and compiling programs.
 * `DACE_debugprint` (default: True): Print debugging information.
 
CPU target configuration:
 * `DACE_compiler_cpu_executable` (default: g++): Chooses the default C++ compiler for CPU code.
 * `DACE_compiler_cpu_additional_args` (default: None): Additional compiler flags (separated by spaces).
  
SDFG processing:
 * `DACE_optimizer_interface` (default: `dace.transformation.optimizer.SDFGOptimizer`): Controls the SDFG optimization process. If empty or class name is invalid, skips process. By default, uses the transformation command line interface.
 * `DACE_optimizer_visualize` (default: False): Visualizes optimization process by saving .dot (GraphViz) files after each pattern replacement.
 
Profiling:
 * `DACE_profiling` (default: False): Enables profiling measurement of the DaCe program runtime in milliseconds. Produces a log file and prints out median runtime.
 * `DACE_treps` (default: 100): Number of repetitions to run a DaCe program when profiling is enabled.
 

Contributing
------------
DaCe is an open-source project. We are happy to accept Pull Requests with your contributions!

License
-------
DaCe is published under the New BSD license, see LICENSE.

