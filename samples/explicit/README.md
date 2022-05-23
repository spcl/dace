In this folder, there are multiple lower-level examples that use the DaCe explicit memory movement, tasklets, and 
parallel iterator syntax (Map and Consume scopes). These can be used in conjunction with high-level constructs such as
numpy for more control of the scheduling of the application. The samples are as follows:

* `histogram.py`: Explicit version of computing two-dimensional histograms, demonstrating the extended syntax of
  memlets, including ranges, volumes, and write-conflict resolution.
* `filter.py`: Predicate-based filtering, demonstrating the use of Streams (queue-semantics construct in DaCe).
* `fibonacci.py`: Fibonacci sequence, demonstrating Streams and the Consume scope for 
  handling dynamic tasks.
* `cc.py`: Shiloach-Viskin pointer-chasing connected components graph algorithm, showcasing concurrent parallel access,
  location constraints, and explicit data movement volume specification.
