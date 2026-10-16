The samples in this folder showcase the performance optimization APIs available in DaCe:

* `matmul.py`: Matrix-matrix multiplication, demonstrating how transformations and library nodes can be used to output
  high-performance code that competes with Intel MKL and NVIDIA CUBLAS.
* `tuning.py`: Sample that showcases the instrumentation interface for measuring internal data-centric application timers
  and using the power of the data-centric intermediate representation for auto-tuning data layouts.
* `ai_generated_matmul.py`: Matrix-matrix multiplication with a microkernel written by a language model (`dace.ai`),
  measured and revised with feedback. Needs a model provider, see the [documentation](https://spcldace.readthedocs.io/en/latest/optimization/ai.html).
