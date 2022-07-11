.. _sdfg:

Stateful Dataflow multiGraphs (SDFG)
====================================

Philosophy
----------

data-centric vs. control-centric

The Language
------------

with pictures

Basics
~~~~~~

all the IR elements

Data Containers and Access Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memlets
~~~~~~~

anatomy of a memlet


Symbols
~~~~~~~~
Scalars vs. symbols



Parametric Parallelism
~~~~~~~~~~~~~~~~~~~~~~

Map consume
schedule types


Dynamic Map Ranges
~~~~~~~~~~~~~~~~~~~

Explain + example (image / embedded viewer)

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

How do I perform dynamic memory allocation? 

