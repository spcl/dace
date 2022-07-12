.. _sdfg:

Stateful Dataflow multiGraphs (SDFG)
====================================

Philosophy
----------

data-centric vs. control-centric


The Language
------------

with pictures


Elements
~~~~~~~~

all the IR elements


Data Containers and Access Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data container types: array/scalar/stream.

Transient property, aliasing assumptions.

Views and references, see below.


Memlets
~~~~~~~

anatomy of a memlet


Symbols
~~~~~~~~
Scalars vs. symbols


Connectors
~~~~~~~~~~

Connectors have types.

Image with tasklet connectors (which are used in the tasklet), side by side with a map with two input connectors
and two input edges, and three output edges. Three of them marked in orange and the connector names are shown. 

Mention *memlet paths* and the general *memlet tree* that can go through arbitrary scopes


Parametric Parallelism
~~~~~~~~~~~~~~~~~~~~~~

Map consume
schedule types


Dynamic Map Ranges
~~~~~~~~~~~~~~~~~~~

Explain + example (image / embedded viewer)


Views and References
~~~~~~~~~~~~~~~~~~~~
view/reference

Use reference sparingly.


SDFG Builder API
----------------

``add_node`` etc.


What to Avoid
-------------

SDFGs are Turing complete. However, not everything can be represented concisely.

Parametric-depth recursion for example (could potentially make a stack, but will be slow)

References with different sizes (dynamic pointers etc.)

DaCe Frontends try to encapsulate those away


Frequently Asked Questions
--------------------------


when should I use a symbol and when should I use a scalar?

How do I perform dynamic memory allocation? array sizes are always associated with symbolic expressions.
 nested sdfg or symbolic size with symbolic assignment prior

