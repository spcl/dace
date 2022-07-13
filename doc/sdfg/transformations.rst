Pattern-Matching and Subgraph Transformations
=============================================

In DaCe, the easiest way to locally modify an SDFG is by data-centric transformations. Transformations are a powerful
tool to optimize applications in DaCe. You can go from naive code to state-of-the-art performance using only transformations.

All transformations extend the :class:`~dace.transformation.transformation.TransformationBase` class. There are three built-in types of transformations in DaCe:

  * **Pattern-matching Transformations** (extending :class:`~dace.transformation.transformation.PatternTransformation`): Transformations that require a certain 
    subgraph structure to match. Within this abstract class, there are two sub-classes:

      * :class:`~dace.transformation.transformation.SingleStateTransformation`: Patterns are limited to a single SDFG state.
      * :class:`~dace.transformation.transformation.MultiStateTransformation`: Patterns are given on a subgraph of an SDFG state machine.

    A pattern-matching must extend at least one of those two classes.
  * **Subgraph Transformations** (extending :class:`~dace.transformation.transformation.SubgraphTransformation`): Transformations that can operate on arbitrary
    subgraphs. 
  * Another form of (implicit) transformation is a Library node expansion (extending :class:`~dace.transformation.transformation.ExpandTransformation`). It is
    a class used for tracking when library nodes are expanded, and creating a library node implementation involves
    extending this class.

Transformations can have properties and those can be used when applying them: for example, tile sizes in :class:`~dace.transformation.dataflow.tiling.MapTiling`.

For more information on how to use and author data-centric transformations, see the `Using and Creating Transformations <https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/transformations.ipynb>`_
tutorial.


Pattern-Matching Transformations
--------------------------------

A pattern-matching transformation works on a specific subgraph pattern, and, using the API, can be used to find all
instances of that pattern and apply it anywhere.
Authoring such a transformation requires extending one of the two subclasses mentioned above 
(:class:`~dace.transformation.transformation.SingleStateTransformation` or :class:`~dace.transformation.transformation.MultiStateTransformation`), add static :class:`~dace.transformation.transformation.PatternNode` fields to the class to 
represent the pattern, and implement at least three methods:

  * ``expressions``: A method that returns a list of graph patterns that match this transformation.
  * ``can_be_applied``: A method that, given a subgraph candidate, checks for additional conditions whether it can be transformed.
  * ``apply``: A method that applies the transformation on the given SDFG.

An instance of the transformation class is associated with a specific match, so using the fields in the class relate
to a specific subgraph.

For example, the following sample transformation matches an access node connected to a map entry, and changes the map's
label to match the name of that access node:

.. code-block:: python

    from dace.sdfg import nodes, SDFG, SDFGState
    from dace.sdfg.utils import node_path_graph
    from dace.transformation import transformation as xf

    class MyTransformation(xf.SingleStateTransformation):
        # Pattern nodes are defined here and can be used in the class
        access = xf.PatternNode(nodes.AccessNode)
        map_node = xf.PatternNode(nodes.MapEntry)

        @classmethod
        def expressions(cls):
            # The pattern to match is ``access -> map_node``. Since this is a
            # class method, accessing ``cls.access`` gets the pattern node.
            return [node_path_graph(cls.access, cls.map_node)]
        
        # Because this is a Single-State Transformation, the first argument here
        # is ``state``
        def can_be_applied(self, state: SDFGState, expr_index: int, sdfg: SDFG,
                        permissive=False) -> bool:
            # We can now use ``self.access``, which refers to a specific subgraph
            # pattern match
            if self.access.data == 'mydata':
                return True
            
            # We only match patterns in which the access node is accessing 'mydata'
            return False


        def apply(self, state: SDFGState, sdfg: SDFG) -> nodes.MapEntry:
            # Here we apply the transformation, and can return any object. This
            # is sometimes used when transformations are composed together and
            # need to pass information to each other.
            self.map_node.label = 'mymap'
            return self.map_node



Subgraph Transformations
------------------------

Subgraph transformations can be applied to any subgraph that returns True for the ``can_be_applied`` method. It is used
when arbitrary local regions need to be modified, e.g., in :class:`~dace.transformation.subgraph.subgraph_fusion.SubgraphFusion`. The implementation is very similar to
pattern-matching transformations, but without the pattern. A simple example with a property would be:

.. code-block:: python

    from dace.sdfg import nodes, SDFG
    from dace.sdfg.utils import node_path_graph
    from dace.transformation import transformation as xf
    from dace.sdfg.graph import SubgraphView
    from dace.properties import make_properties, Property

    @make_properties
    class ExampleSubgraphXform(xf.SubgraphTransformation):
        """
        This string describes the transformation and will be shown in the Visual Studio Code plugin.
        """

        # Properties can be defined on Transformation classes as with other objects
        simplify = Property(desc="Simplify SDFG after applying transformation.", dtype=bool, default=False)

        def can_be_applied(self, sdfg: SDFG, subgraph: SubgraphView) -> bool:
            return True

        def apply(self, sdfg: SDFG) -> None:
            # First we obtain the subgraph view from the SDFG we matched in
            subgraph = self.subgraph_view(sdfg)

            # Then we can work on the graph normally
            for node in subgraph.nodes():
                # Do something complex...
                pass

            if self.simplify:
                sdfg.simplify()





