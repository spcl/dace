# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
API for SDFG analysis and manipulation Passes, as well as Pipelines that contain multiple dependent passes.
"""
from dace import properties, serialize
from dace.sdfg import SDFG, SDFGState, graph as gr, nodes, utils as sdutil, ScopeBlock

from enum import Flag, auto
from typing import Any, Dict, Iterator, List, Optional, Set, Type, Union
from dataclasses import dataclass


class Modifies(Flag):
    """
    Specifies which elements of an SDFG have been modified by a Pass/Transformation.
    This is used when deciding whether to rerun certain Passes for SDFG analysis.
    Note that this is a Python ``Flag``, which means values such as ``Memlets | Symbols`` are allowed.
    """
    Nothing = 0  #: Nothing was modified
    Descriptors = auto()  #: Data descriptors (e.g., arrays, streams) and their properties were modified
    Symbols = auto()  #: Symbols were modified
    States = auto()  #: The number of SDFG states and their connectivity (not their contents) were modified
    InterstateEdges = auto()  #: Contents (conditions/assignments) or existence of inter-state edges were modified
    AccessNodes = auto()  #: Access nodes' existence or properties were modified
    Scopes = auto()  #: Scopes (e.g., Map, Consume, Pipeline) or associated properties were created/removed/modified
    Tasklets = auto()  #: Tasklets were created/removed or their contents were modified
    NestedSDFGs = auto()  #: SDFG nesting structure or properties of NestedSDFG nodes were modified
    Memlets = auto()  #: Memlets' existence, contents, or properties were modified
    Nodes = AccessNodes | Scopes | Tasklets | NestedSDFGs  #: Modification of any dataflow node (contained in an SDFG state) was made
    Edges = InterstateEdges | Memlets  #: Any edge (memlet or inter-state) was modified
    Everything = Descriptors | Symbols | States | InterstateEdges | Nodes | Memlets  #: Modification to arbitrary parts of SDFGs (nodes, edges, or properties)


@properties.make_properties
class Pass:
    """
    An SDFG analysis or manipulation that registers as part of the SDFG history. Classes that extend ``Pass`` can be
    used for optimization purposes, to collect data on an entire SDFG, for cleanup, or other uses. Pattern-matching
    transformations, as well as the SDFG simplification process, extend Pass.
    Passes may depend on each other through a ``Pipeline`` object, which will ensure dependencies are met and that
    passes do not run redundantly.
    
    A Pass is defined by one main method: ``apply_pass``. This method receives the SDFG to manipulate/analyze, as well
    as the previous ``Pipeline`` results, if run in the context of a pipeline. The other three, pipeline-related
    methods are:
    * ``depends_on``: Which other passes this pass requires
    * ``modifies``: Which elements of the SDFG does this Pass modify (used to avoid re-applying when unnecessary)
    * ``should_reapply``: Given the modified elements of the SDFG, should this pass be rerun?

    :seealso: Pipeline
    """

    CATEGORY: str = 'Helper'

    def depends_on(self) -> Set[Union[Type['Pass'], 'Pass']]:
        """
        If in the context of a ``Pipeline``, which other Passes need to run first.

        :return: A set of Pass subclasses or objects that need to run prior to this Pass.
        """
        return set()

    def modifies(self) -> Modifies:
        """
        Which elements of the SDFG (e.g., memlets, state structure) are modified by this pass, if run successfully.

        :return: A ``Modifies`` set of flags of modified elements.
        """
        raise NotImplementedError

    def should_reapply(self, modified: Modifies) -> bool:
        """
        In the context of a ``Pipeline``, queries whether this Pass should be rerun after other passes have run
        and modified the SDFG.

        :param modified: Flags specifying which elements of the SDFG were modified.
        :return: True if this Pass should be rerun when the given elements are modified.
        """
        raise NotImplementedError

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        """
        Applies the pass to the given SDFG.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Some object if pass was applied, or None if nothing changed.
        """
        raise NotImplementedError

    def report(self, pass_retval: Any) -> Optional[str]:
        """
        Returns a user-readable string report based on the results of this pass.

        :param pass_retval: The return value from applying this pass.
        :return: A string with the user-readable report, or None if nothing to report.
        """
        return None

    def to_json(self, parent=None) -> Dict[str, Any]:
        props = serialize.all_properties_to_json(self)
        return {'type': 'Pass', 'transformation': type(self).__name__, 'CATEGORY': type(self).CATEGORY, **props}

    @staticmethod
    def from_json(json_obj: Dict[str, Any], context: Dict[str, Any] = None) -> 'Pass':
        pss = next(ext for ext in Pass.subclasses_recursive() if ext.__name__ == json_obj['transformation'])

        # Reconstruct the pass.
        ret = pss()
        context = context or {}
        context['transformation'] = ret
        serialize.set_properties_from_json(ret, json_obj, context=context, ignore_properties={'transformation', 'type'})
        return ret

    @classmethod
    def subclasses_recursive(cls) -> Set[Type['Pass']]:
        """
        Returns all subclasses of this class, including subclasses of subclasses.
        """
        subclasses = set(cls.__subclasses__())
        subsubclasses = set()
        for sc in subclasses:
            subsubclasses.update(sc.subclasses_recursive())

        # Ignore abstract classes.
        result = subclasses | subsubclasses
        result = set(sc for sc in result if not getattr(sc, '__abstractmethods__', False))

        return result

@properties.make_properties
class VisitorPass(Pass):
    """
    A simple type of Pass that provides a Python visitor object on an SDFG. Used for either analyzing an SDFG or
    modifying properties of existing elements (rather than their graph structure). Applying a visitor pass on an
    SDFG would call ``visit_<ElementType>`` methods on the SDFG elements with the element, its parent, and previous
    pipeline results, if in the context of a ``Pipeline``.

    For example:

    .. code-block:: python

        class HasWriteConflicts(VisitorPass):
            def __init__(self):
                self.found_wcr = False

            def visit_Memlet(self, memlet: dace.Memlet, parent: dace.SDFGState, pipeline_results: Dict[str, Any]):
                if memlet.wcr:
                    self.found_wcr = True

                    # If a value is returned, a dictionary key will be filled with the visited object and the value
                    return memlet.wcr

        wcr_checker = HasWriteConflicts()
        memlets_with_wcr = wcr_checker.apply_pass(sdfg, {})
        print('SDFG has write-conflicted memlets:', wcr_checker.found_wcr)
        print('Memlets:', memlets_with_wcr)
    """

    CATEGORY: str = 'Helper'

    def generic_visit(self, element: Any, parent: Any, pipeline_results: Dict[str, Any]) -> Any:
        """
        A default method that is called for elements that do not have a special visitor.

        :param element: The element to visit.
        :param parent: The parent of the visited element (e.g., SDFGState for dataflow elements, SDFG for SDFGStates).
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                            results as ``{Pass subclass name: returned object from pass}``. If not run in a
                            pipeline, an empty dictionary is expected.
        """
        return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[Any, Any]]:
        """
        Visits the given SDFG recursively, calling defined ``visit_*`` methods for each element.

        :param sdfg: The SDFG to recursively visit.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary of ``{element: return value}`` for visited elements with a non-None return value, or None
                 if nothing was returned.
        """
        results = {}
        for node, parent in sdfg.all_nodes_recursive():
            # Visit node (SDFGState, AccessNode, ...)
            f = getattr(self, f'visit_{type(node).__name__}', self.generic_visit)
            res = f(node, parent, pipeline_results)
            if res is not None:
                results[node] = res
        for edge, parent in sdfg.all_edges_recursive():
            # Visit edge (Edge, MultiConnectorEdge)
            f = getattr(self, f'visit_{type(edge).__name__}', self.generic_visit)
            res = f(edge, parent, pipeline_results)
            if res is not None:
                results[edge] = res

            # Visit edge data (Memlet, InterstateEdge)
            f = getattr(self, f'visit_{type(edge.data).__name__}', self.generic_visit)
            res = f(edge.data, parent, pipeline_results)
            if res is not None:
                results[edge.data] = res

        if not results:
            return None
        return results


@properties.make_properties
class StatePass(Pass):
    """
    A specialized Pass type that applies to each SDFG state separately. Such a pass is realized by implementing the
    ``apply`` method, which accepts a single state.
    
    :see: Pass
    """

    CATEGORY: str = 'Helper'

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[SDFGState, Optional[Any]]]:
        """
        Applies the pass to states of the given SDFG by calling ``apply`` on each state.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary of ``{state: return value}`` for visited states with a non-None return value, or None
                 if nothing was returned.
        """
        result = {}
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.nodes():
                retval = self.apply(state, pipeline_results)
                if retval is not None:
                    result[state] = retval

        if not result:
            return None
        return result

    def apply(self, state: SDFGState, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        """
        Applies this pass on the given state.

        :param state: The SDFG state to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Some object if pass was applied, or None if nothing changed.
        """
        raise NotImplementedError


@properties.make_properties
class ScopePass(Pass):
    """
    A specialized Pass type that applies to each scope (e.g., Map, Consume, Pipeline) separately. Such a pass is
    realized by implementing the ``apply`` method, which accepts a scope entry node and its parent SDFG state.
    
    :see: Pass
    """

    CATEGORY: str = 'Helper'

    def apply_pass(
        self,
        sdfg: SDFG,
        pipeline_results: Dict[str, Any],
    ) -> Optional[Dict[nodes.EntryNode, Optional[Any]]]:
        """
        Applies the pass to the scopes of the given SDFG by calling ``apply`` on each scope entry node.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary of ``{entry node: return value}`` for visited scopes with a non-None return value, or None
                 if nothing was returned.
        """
        result = {}
        for node, state in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.EntryNode):
                continue
            retval = self.apply(node, state, pipeline_results)
            if retval is not None:
                result[node] = retval

        if not result:
            return None
        return result

    def apply(self, scope: nodes.EntryNode, state: SDFGState, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        """
        Applies this pass on the given scope.

        :param scope: The entry node of the scope to apply the pass to.
        :param state: The parent SDFG state of the given scope.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Some object if pass was applied, or None if nothing changed.
        """
        raise NotImplementedError


@properties.make_properties
class ControlFlowScopePass(Pass):
    """
    A specialized Pass type that applies to each control flow scope (i.e., CFG) separately. Such a pass is
    realized by implementing the ``apply`` method, which accepts a CFG and the SDFG it belongs to.
    
    :see: Pass
    """

    CATEGORY: str = 'Helper'

    def apply_pass(
        self,
        sdfg: SDFG,
        pipeline_results: Dict[str, Any],
        **kwargs,
    ) -> Optional[Dict[nodes.EntryNode, Optional[Any]]]:
        """
        Applies the pass to the CFGs of the given SDFG by calling ``apply`` on each CFG.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary of ``{entry node: return value}`` for visited CFGs with a non-None return value, or None
                 if nothing was returned.
        """
        result = {}
        for scope_block in sdfg.all_state_scopes_recursive(recurse_into_sdfgs=False):
            retval = self.apply(scope_block, scope_block if isinstance(scope_block, SDFG) else scope_block.sdfg,
                                pipeline_results, **kwargs)
            if retval is not None:
                result[scope_block] = retval

        if not result:
            return None
        return result

    def apply(self, scope_block: ScopeBlock, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        """
        Applies this pass on the given scope.

        :param scope_block: The control flow scope block to apply the pass to.
        :param sdfg: The parent SDFG of the given scope.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Some object if pass was applied, or None if nothing changed.
        """
        raise NotImplementedError


@dataclass
@properties.make_properties
class Pipeline(Pass):
    """
    A pass pipeline contains multiple, potentially dependent Pass objects, and applies them in order. Each contained
    pass may depend on other (e.g., analysis) passes, which the Pipeline avoids rerunning depending on which elements
    were modified by applied passes. An example of a built-in pipeline is the SimplifyPass, which runs multiple
    complexity reduction passes and may reuse data across them. Prior results of applied passes are contained in
    the ``pipeline_results`` argument to ``apply_pass``, which can be used to access previous return values of Passes.

    A Pipeline in itself is a type of a Pass, so it can be arbitrarily nested in another Pipelines. Its dependencies
    and modified elements are unions of the contained Pass objects.

    Creating a Pipeline can be performed by instantiating the object with a list of Pass objects, or by extending the
    pipeline class (e.g., if pipeline order should be modified). The return value of applying a pipeline is a
    dictionary whose keys are the Pass subclass names and values are the return values of each pass. Example use:

    .. code-block:: python

        my_simplify = Pipeline([ScalarToSymbolPromotion(integers_only=False), ConstantPropagation()])
        results = my_simplify.apply_pass(sdfg, {})
        print('Promoted scalars:', results['ScalarToSymbolPromotion'])

    """

    CATEGORY: str = 'Helper'

    passes = properties.ListProperty(element_type=Pass,
                                     default=[],
                                     category='(Debug)',
                                     desc='List of passes that this pipeline contains')

    def __init__(self, passes: List[Pass]):
        self.passes = []
        self._pass_names = set(type(p).__name__ for p in passes)
        self.passes.extend(passes)

        # Add missing Pass dependencies
        self._add_dependencies(passes)

        self._depgraph: Optional[gr.OrderedDiGraph[Pass, None]] = None

        # Keep track of what is modified as the pipeline is executing
        self._modified: Modifies = Modifies.Nothing

    def _add_dependencies(self, passes: List[Pass]):
        """
        Verifies pass uniqueness in pipeline and adds missing dependencies from ``depends_on`` of each pass. 

        :param passes: The passes to add dependencies for.
        """
        unique_pass_types = set(type(p) for p in passes)
        check_if_unique: Set[Type[Pass]] = unique_pass_types

        if len(check_if_unique) != len(passes):
            pass_types = [type(p) for p in passes]
            dups = set([x for x in pass_types if pass_types.count(x) > 1])
            raise NameError('Duplicate pass types found in pipeline. Please use unique Pass type objects within one '
                            f'Pipeline. Duplicates: {dups}')

        # Traverse pass dependencies until there is nothing to visit
        passes_to_check = passes
        while len(passes_to_check) > 0:
            new_passes = []
            for p in passes_to_check:
                deps = p.depends_on()
                for dep in deps:
                    # If an object dependency is given, make sure it is unique so that dictionary works
                    if isinstance(dep, Pass):
                        if type(dep) in check_if_unique:
                            raise NameError(
                                f'Duplicate dependency passes given: "{type(dep).__name__}" is a Pass object dependency '
                                'that is already a dependency of a pass or used directly in the pipeline. Please use a '
                                'class instead of an object in the `depends_on` method.')

                        check_if_unique.add(type(dep))
                        self.passes.append(dep)
                        new_passes.append(dep)
                    elif isinstance(dep, type):
                        if dep not in check_if_unique:
                            check_if_unique.add(dep)
                            dep_obj = dep()  # Construct Pass object from type
                            self.passes.append(dep_obj)
                            new_passes.append(dep_obj)
                    else:
                        raise TypeError(f'Invalid pass type {type(dep).__name__} given to pipeline')
            passes_to_check = new_passes

    def modifies(self) -> Modifies:
        """
        Which elements of the SDFG (e.g., memlets, state structure) are modified by this pipeline, if run successfully.
        Computed as the union of all modified elements of each pass in the pipeline.
        
        :return: A ``Modifies`` set of flags of modified elements.
        """
        result = Modifies.Nothing
        for p in self.passes:
            result |= p.modifies()

        return result

    def should_reapply(self, modified: Modifies) -> bool:
        return any(p.should_reapply(modified) for p in self.passes)

    def depends_on(self) -> Set[Type[Pass]]:
        result = set()
        for p in self.passes:
            result.update(p.depends_on())
        return result

    def _make_dependency_graph(self) -> gr.OrderedDiGraph:
        """
        Makes an ordered dependency graph out of the passes in the pipeline to traverse when applying.
        """
        result = gr.OrderedDiGraph()
        ptype_to_pass = {type(p): p for p in self.passes}

        for p in self.passes:
            if p not in result._nodes:
                result.add_node(p)
            for dep in p.depends_on():
                # If a type, find it in self._passes
                if isinstance(dep, type):
                    dep = ptype_to_pass[dep]
                result.add_edge(dep, p)

        return result

    def iterate_over_passes(self, sdfg: SDFG) -> Iterator[Pass]:
        """
        Iterates over passes in the pipeline, potentially multiple times based on which elements were modified
        in the pass.
        Note that this method may be overridden by subclasses to modify pass order.

        :param sdfg: The SDFG on which the pipeline is currently being applied
        """
        # Lazily create dependency graph
        if self._depgraph is None:
            self._depgraph = self._make_dependency_graph()

        # Maintain a dictionary for each applied pass:
        # * Whenever a pass is applied, it is set to Nothing (as nothing was modified yet)
        # * As other passes apply, all existing passes union with what the current pass modified
        # This allows us to check, for each pass, whether it (and its dependencies) should reapply since it was last
        # applied.
        applied_passes: Dict[Pass, Modifies] = {}

        def reapply_recursive(p: Pass):
            """ Reapply pass dependencies in a recursive fashion. """
            # If pass should not reapply, skip
            if p in applied_passes and not p.should_reapply(applied_passes[p]):
                return

            # Check dependencies first
            for dep in self._depgraph.predecessors(p):
                yield from reapply_recursive(dep)

            yield p

        # Traverse dependency graph topologically and for every node, check if modified elements require
        # reapplying dependencies
        for p in sdutil.dfs_topological_sort(self._depgraph):
            p: Pass

            # If pass was visited (applied) and it (or any of its dependencies) needs reapplication
            for pass_to_apply in reapply_recursive(p):
                # Reset modified elements, yield pass, update the other applied passes with what changed
                self._modified = Modifies.Nothing

                yield pass_to_apply

                if self._modified != Modifies.Nothing:
                    for old_pass in applied_passes.keys():
                        applied_passes[old_pass] |= self._modified
                applied_passes[pass_to_apply] = Modifies.Nothing

    def apply_subpass(self, sdfg: SDFG, p: Pass, state: Dict[str, Any]) -> Optional[Any]:
        """
        Apply a pass from the pipeline. This method is meant to be overridden by subclasses.

        :param sdfg: The SDFG to apply the pass to.
        :param p: The pass to apply.
        :param state: The pipeline results state.
        :return: The pass return value.
        """
        return p.apply_pass(sdfg, state)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        state = pipeline_results
        retval = {}
        self._modified = Modifies.Nothing
        for p in self.iterate_over_passes(sdfg):
            r = self.apply_subpass(sdfg, p, state)
            if r is not None:
                state[type(p).__name__] = r
                retval[type(p).__name__] = r
                self._modified = p.modifies()

        if retval:
            return retval
        return None

    def to_json(self, parent=None) -> Dict[str, Any]:
        props = serialize.all_properties_to_json(self)
        return {
            'type': 'Pipeline',
            'transformation': type(self).__name__,
            'CATEGORY': type(self).CATEGORY,
            **props
        }


@properties.make_properties
class FixedPointPipeline(Pipeline):
    """
    A special type of Pipeline that applies its ``Pass`` objects in repeated succession until they all stop modifying
    the SDFG (i.e., by returning None).
    
    :see: Pipeline
    """

    CATEGORY: str = 'Helper'

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Applies the pipeline to the SDFG in repeated succession until the SDFG is no longer modified.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary of ``{Pass subclass name: return value}`` for applied passes, or None if no Passes
                 were applied in the context of this pipeline.
        """
        state = pipeline_results
        retval = {}
        while True:
            newret = super().apply_pass(sdfg, state)
            
            # Remove dependencies from pipeline
            if newret:
                newret = {k: v for k, v in newret.items() if k in self._pass_names}

            if not newret:
                if retval:
                    return retval
                return None
            state.update(newret)
            retval.update(newret)
