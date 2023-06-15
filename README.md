[![Build Status](https://travis-ci.org/spcl/dace.svg?branch=master)](https://travis-ci.org/spcl/dace)
[![Documentation Status](https://readthedocs.org/projects/spcldace/badge/?version=latest)](https://spcldace.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/dace.svg)](https://badge.fury.io/py/dace)
[![codecov](https://codecov.io/gh/spcl/dace/branch/master/graph/badge.svg)](https://codecov.io/gh/spcl/dace)


![D](dace.svg)aCe - Data-Centric Parallel Programming 
=====================================================

_Decoupling domain science from performance optimization._

DaCe compiles code in various programming languages and paradigms (Python/Numpy, MATLAB, TensorFlow) and maps it efficiently to **CPUs, GPUs, and FPGAs** with high utilization, on par with the state-of-the-art. The key feature driving DaCe is its Stateful DataFlow multiGraph (SDFG) *data-centric intermediate representation*: A transformable, interactive representation of code based on data movement.
With data-centric parallel programming, we enable **direct knowledge transfer** of performance optimization, regardless of the scientific application or the target processor.

DaCe can be written inline in Python and transformed in the command-line/Jupyter Notebooks, or SDFGs can be interactively modified using the Data-centric Interactive Optimization Development Environment (DIODE, currently experimental).

For more information, see our [paper](http://www.arxiv.org/abs/1902.10345).

Tutorials
---------

* [Data-Centric Python Programs with NumPy](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/numpy_frontend.ipynb)
* [Explicit Dataflow in Python](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/explicit.ipynb)
* [SDFG API](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/sdfg_api.ipynb)
* [Transformations](https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/transformations.ipynb)

Installation and Dependencies
-----------------------------

To install: `pip install dace`

Runtime dependencies:
 * A C++14-capable compiler (e.g., gcc 5.3+)
 * Python 3.6 or newer
 * CMake 2.8.12 or newer (for Windows, CMake 3.15 is recommended)

Running
-------

**Python scripts:** Run DaCe programs (in implicit, explicit, or TensorFlow syntax) using Python directly.

**DIODE interactive development (experimental):**: Either run the installed script `diode`, or call `python3 -m diode.diode_server` from the shell. Then, follow the printed instructions to enter the web interface.

**Octave scripts (experimental):** `.m` files can be run using the installed script `dacelab`, which will create the appropriate SDFG file.

**Jupyter Notebooks:** DaCe is Jupyter-compatible. If a result is an SDFG or a state, it will show up directly in the notebook. See the [tutorials](tutorials) for examples.

**[SDFV (standalone SDFG viewer)](https://spcl.github.io/dace/sdfv.html):** To view SDFGs separately, run the `sdfv` installed script with the `.sdfg` file as an argument. Alternatively, you can use the link or open `diode/sdfv.html` directly and choose a file in the browser.

**Note for Windows/Visual C++ users:** If compilation fails in the linkage phase, try setting the following environment variable to force Visual C++ to use Multi-Threaded linkage:
```
X:\path\to\dace> set _CL_=/MT
```


Publication
-----------

If you use DaCe, cite us:
```bibtex
@inproceedings{dace,
  author    = {Ben-Nun, Tal and de~Fine~Licht, Johannes and Ziogas, Alexandros Nikolaos and Schneider, Timo and Hoefler, Torsten},
  title     = {Stateful Dataflow Multigraphs: A Data-Centric Model for Performance Portability on Heterogeneous Architectures},
  year      = {2019},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  series = {SC '19}
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
 
SDFG processing:
 * `DACE_optimizer_interface` (default: `dace.transformation.optimizer.SDFGOptimizer`): Controls the SDFG optimization process by choosing a Python handler. If empty or class name is invalid, skips process. By default, uses the transformation command line interface.
 
Profiling:
 * `DACE_profiling` (default: False): Enables profiling measurement of the DaCe program runtime in milliseconds. Produces a log file and prints out median runtime.
 * `DACE_treps` (default: 100): Number of repetitions to run a DaCe program when profiling is enabled.
 

Contributing
------------
DaCe is an open-source project. We are happy to accept Pull Requests with your contributions!

License
-------
DaCe is published under the New BSD license, see [LICENSE](LICENSE).

