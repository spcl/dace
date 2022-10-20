DaCe: Data-Centric Parallel Programming
=======================================

*Decoupling domain science from performance optimization.*

DaCe is a parallel programming framework that takes code in Python/NumPy and 
other programming languages, and maps it to high-performance **CPU, GPU, and FPGA** 
programs, which can be optimized to achieve state-of-the-art. Internally, DaCe 
uses the :ref:`sdfg` *data-centric intermediate 
representation*: A transformable, interactive representation of code based on 
data movement.
Since the input code and the SDFG are separate, it is possible to :ref:`optimize a 
program <optimization>` without changing its source, so that it stays 
readable. On the other hand, the used optimizations are customizable and user-extensible,
so they can be written once and reused in many applications.
With data-centric parallel programming, we enable **direct knowledge transfer** 
of performance optimization, regardless of the application or the target processor.

DaCe generates high-performance programs for:

 * Multi-core CPUs (tested on Intel, IBM POWER9, and ARM with SVE)
 * NVIDIA GPUs and AMD GPUs (see :ref:`how to use HIP in DaCe <amd>`)
 * Xilinx and Intel FPGAs



If you use DaCe, cite us:

.. code-block:: bibtex

    @inproceedings{dace,
      author    = {Ben-Nun, Tal and de~Fine~Licht, Johannes and Ziogas, Alexandros Nikolaos and Schneider, Timo and Hoefler, Torsten},
      title     = {Stateful Dataflow Multigraphs: A Data-Centric Model for Performance Portability on Heterogeneous Architectures},
      year      = {2019},
      booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
      series = {SC '19}
    }
  



.. toctree::
   :caption: User Guide
   :maxdepth: 2
   
   setup/quickstart
   setup/installation
   setup/config
   frontend/daceprograms
   sdfg/ir
   ide/vscode
   optimization/optimization
   setup/integration
   ide/cli
   general/glossary

.. general/faq

.. toctree::
   :caption: Developer Guide
   :maxdepth: 2

   general/structure
   general/debugging
   extensions/properties
   sdfg/transforming
   frontend/python
   codegen/codegen
   extensions/extensions
   general/errors

.. toctree::
   :caption: Module Reference
   :maxdepth: 2

   source/dace
   source/config_schema


Reference
=========

* :ref:`genindex`
* :ref:`modindex`
