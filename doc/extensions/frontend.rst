Frontend Design Guidelines
==========================

Trust the transformations: put statements in different states, :ref:`simplify` will take care of it.
don't try to perform premature optimization.

make things more data-centric
-----------------------------

preprocessing AST

helping expose memory access patterns to DaCe
if you know about symbols, use them

avoid pointers. if you can't: use arrays, then use views, then if all else fails use References

Encapsulation: callbacks and closures
-------------------------------------

Use a ``__state`` variable! create dummy dataflow so that operators are not reordered.
If you know more about the internal state (for example, one library does not interfere with others), use multiple
internal states.

Remember your closure!!!

Know your language
------------------
all AST nodes need to be accounted for (either by parsing, callbacks, or erroring out)

if you know of the behavior of certain calls or operators, use the right library nodes. For example:
the ``@`` operator and ``matmul`` library node.

assumptions
-----------

aliasing: if you have full program analysis - no problem.
if not, use the ``may_alias`` property of data containers.

by reference vs. by value: by reference-  views and references

limit the scope
---------------

limit your parser to work on a single scope (e.g., translation unit, function) - it is possible to create one 
object per scope (see Python frontend)

use nested sdfgs if you can (for each function call, each basic block)

