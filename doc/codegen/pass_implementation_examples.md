# Code Generation Pass Implementation Examples

This document provides concrete implementation examples for the proposed modular code generation passes described in the main design document.

## Example Pass Implementations

### 1. MetadataCollectionPass

```python
from typing import Any, Dict, Set, Optional
from dace import SDFG, dtypes, data, nodes
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class MetadataCollectionPass(ppl.Pass):
    """
    Collects metadata about the SDFG including free symbols, argument lists,
    constants, and shared transients.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Reapply if symbols, arrays, or structure changed
        return modified & (ppl.Modifies.Symbols | ppl.Modifies.AccessNodes | ppl.Modifies.CFG)

    def depends_on(self):
        return set()  # First analysis pass

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Collect SDFG metadata for use by downstream passes.

        :return: Metadata dictionary with symbols, arglist, constants, etc.
        """
        metadata = {
            'free_symbols': {},
            'symbols_and_constants': {},
            'arglist': {},
            'shared_transients': {},
            'constants': {},
        }

        # Collect free symbols for all SDFGs
        for nested_sdfg in sdfg.all_sdfgs_recursive():
            fsyms = nested_sdfg.free_symbols
            metadata['free_symbols'][nested_sdfg.cfg_id] = fsyms

            # Combine with constants
            constants = nested_sdfg.constants_prop.keys()
            metadata['symbols_and_constants'][nested_sdfg.cfg_id] = fsyms.union(constants)

            # Collect shared transients
            metadata['shared_transients'][nested_sdfg.cfg_id] = nested_sdfg.shared_transients(
                check_toplevel=False, include_nested_data=True
            )

        # Create argument list
        metadata['arglist'] = sdfg.arglist(
            scalars_only=False,
            free_symbols=metadata['free_symbols'][sdfg.cfg_id]
        )

        # Store in pipeline results
        return metadata
```

### 2. AllocationAnalysisPass

```python
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dace import SDFG, SDFGState, dtypes, data, nodes
from dace.transformation.passes.analysis import StateReachability
from dace.transformation import pass_pipeline as ppl, transformation
from dace.sdfg import scope as sdscope, utils
from dace.sdfg.analysis import cfg as cfg_analysis


@properties.make_properties
@transformation.explicit_cf_compatible
class AllocationAnalysisPass(ppl.Pass):
    """
    Determines allocation lifetimes and scopes for all data containers.
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.AccessNodes | ppl.Modifies.CFG | ppl.Modifies.Symbols)

    def depends_on(self):
        return {MetadataCollectionPass, StateReachability}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze allocation lifetimes and determine allocation scopes.

        :return: Allocation information dictionary
        """
        metadata = pipeline_results[MetadataCollectionPass.__name__]
        reachability = pipeline_results[StateReachability.__name__]

        allocation_info = {
            'to_allocate': defaultdict(list),
            'where_allocated': {},
            'allocation_scopes': {},
        }

        # Gather access instances for each SDFG
        access_instances = {}
        for nested_sdfg in sdfg.all_sdfgs_recursive():
            instances = defaultdict(list)
            array_names = set(nested_sdfg.arrays.keys())

            for state in cfg_analysis.blockorder_topological_sort(nested_sdfg, ignore_nonstate_blocks=True):
                for node in state.data_nodes():
                    if node.data in array_names:
                        instances[node.data].append((state, node))

                # Check interstate edges for symbol usage
                edge_fsyms = set()
                for e in state.parent_graph.all_edges(state):
                    edge_fsyms |= e.data.free_symbols
                for edge_array in edge_fsyms & array_names:
                    instances[edge_array].append((state, nodes.AccessNode(edge_array)))

            access_instances[nested_sdfg.cfg_id] = instances

        # Determine allocation scope for each array
        for nested_sdfg, name, desc in sdfg.arrays_recursive(include_nested_data=True):
            top_desc = nested_sdfg.arrays[name.split('.')[0]]

            if not top_desc.transient or name in nested_sdfg.constants_prop:
                continue

            alloc_scope, alloc_state = self._determine_allocation_scope(
                nested_sdfg, name, desc, access_instances[nested_sdfg.cfg_id],
                metadata['shared_transients'][nested_sdfg.cfg_id]
            )

            if alloc_scope is not None:
                first_state, first_node = access_instances[nested_sdfg.cfg_id].get(name, [(None, None)])[0]
                allocation_info['to_allocate'][alloc_scope].append(
                    (nested_sdfg, alloc_state or first_state, first_node, True, True, True)
                )
                allocation_info['where_allocated'][(nested_sdfg, name)] = alloc_scope
                allocation_info['allocation_scopes'][name] = alloc_scope

        return allocation_info

    def _determine_allocation_scope(self, sdfg, name, desc, access_instances, shared_transients):
        """Determine the appropriate allocation scope for a data container."""

        if desc.lifetime == dtypes.AllocationLifetime.Persistent:
            return sdfg.parent or sdfg, None
        elif desc.lifetime == dtypes.AllocationLifetime.Global:
            return sdfg.parent or sdfg, None
        elif desc.lifetime == dtypes.AllocationLifetime.SDFG:
            return sdfg, None
        elif desc.lifetime == dtypes.AllocationLifetime.State:
            return self._determine_state_scope(sdfg, name, access_instances)
        elif desc.lifetime == dtypes.AllocationLifetime.Scope:
            return self._determine_scope_scope(sdfg, name, access_instances, shared_transients)
        else:
            raise TypeError(f'Unrecognized allocation lifetime "{desc.lifetime}"')

    def _determine_state_scope(self, sdfg, name, access_instances):
        """Determine scope for state-lifetime arrays."""
        instances = access_instances.get(name, [])
        if not instances:
            return None, None

        states = {state for state, _ in instances}
        if len(states) == 1:
            return list(states)[0], list(states)[0]
        else:
            return sdfg, None  # Multi-state usage

    def _determine_scope_scope(self, sdfg, name, access_instances, shared_transients):
        """Determine scope for scope-lifetime arrays."""
        instances = access_instances.get(name, [])
        if not instances:
            return None, None

        # Check for multi-state usage
        states = {state for state, _ in instances}
        if len(states) > 1 or name in shared_transients:
            return sdfg, None

        # Find common scope within single state
        state = list(states)[0]
        sdict = state.scope_dict()
        common_scope = None

        for _, node in instances:
            if not isinstance(node, nodes.AccessNode):
                continue
            scope = sdict[node] or state
            if common_scope is None:
                common_scope = scope
            else:
                common_scope = sdscope.common_parent_scope(sdict, scope, common_scope)

        return common_scope, state
```

### 3. FrameCodeGenerationPass

```python
from typing import Any, Dict, List, Optional, Set
from dace import SDFG, dtypes, data
from dace.codegen.prettycode import CodeIOStream
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class FrameCodeGenerationPass(ppl.Pass):
    """
    Generates the main frame code including function signatures, state struct,
    and initialization/cleanup code.
    """

    CATEGORY: str = 'CodeGeneration'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing  # Generates code but doesn't modify SDFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified != ppl.Modifies.Nothing  # Any change requires regeneration

    def depends_on(self):
        return {MetadataCollectionPass, AllocationAnalysisPass, StateStructCreationPass, AllocationCodePass}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate frame code for the SDFG.

        :return: Frame code dictionary with global and local code
        """
        metadata = pipeline_results[MetadataCollectionPass.__name__]
        allocation_info = pipeline_results[AllocationAnalysisPass.__name__]
        state_struct = pipeline_results[StateStructCreationPass.__name__]
        allocation_code = pipeline_results[AllocationCodePass.__name__]

        global_stream = CodeIOStream()
        local_stream = CodeIOStream()

        # Generate file header
        self._generate_file_header(sdfg, global_stream)

        # Generate state struct definition
        if state_struct['struct_definition']:
            global_stream.write(state_struct['struct_definition'])

        # Generate function signature
        function_sig = self._generate_function_signature(sdfg, metadata['arglist'])
        local_stream.write(function_sig + ' {\n')

        # Generate local variable declarations
        self._generate_local_declarations(sdfg, metadata, local_stream)

        # Generate initialization code
        if state_struct['initialization_code']:
            local_stream.write(state_struct['initialization_code'])

        # Generate allocation code
        for scope, alloc_code in allocation_code['alloc_code_by_scope'].items():
            local_stream.write(alloc_code)

        # Generate main computation code (placeholder for now)
        local_stream.write('\n    // Generated state computation code goes here\n')

        # Generate deallocation code
        for scope, dealloc_code in allocation_code['dealloc_code_by_scope'].items():
            local_stream.write(dealloc_code)

        local_stream.write('}\n')

        return {
            'global_code': global_stream.getvalue(),
            'local_code': local_stream.getvalue(),
            'function_signature': function_sig,
            'header_includes': self._get_required_includes(sdfg, metadata),
        }

    def _generate_file_header(self, sdfg: SDFG, stream: CodeIOStream):
        """Generate file header with includes and defines."""
        stream.write('#include <dace/dace.h>\n')
        stream.write('#include <cmath>\n')
        stream.write('#include <cstring>\n')
        stream.write('#include <iostream>\n\n')

    def _generate_function_signature(self, sdfg: SDFG, arglist: Dict[str, data.Data]) -> str:
        """Generate the main function signature."""
        args = []
        for name, arg_type in arglist.items():
            if isinstance(arg_type, data.Scalar):
                args.append(f'{arg_type.dtype.ctype} {name}')
            elif isinstance(arg_type, data.Array):
                args.append(f'{arg_type.dtype.ctype}* __restrict__ {name}')
            elif isinstance(arg_type, data.Stream):
                args.append(f'{arg_type.as_arg(name)}')

        return f'void __program_{sdfg.name}({", ".join(args)})'

    def _generate_local_declarations(self, sdfg: SDFG, metadata: Dict, stream: CodeIOStream):
        """Generate local variable declarations."""
        # Declare symbols
        for sym in metadata['free_symbols'][sdfg.cfg_id]:
            if sym not in metadata['arglist']:
                stream.write(f'    int {sym};\n')

    def _get_required_includes(self, sdfg: SDFG, metadata: Dict) -> List[str]:
        """Get list of required header includes."""
        includes = ['<dace/dace.h>', '<cmath>', '<cstring>']

        # Add target-specific includes based on usage
        # This would be expanded based on target analysis

        return includes
```

### 4. Pipeline Configuration Examples

```python
class CodeGenerationPipeline(ppl.Pipeline):
    """Main code generation pipeline."""

    def __init__(self, validate: bool = True, target_specific: bool = True):
        passes = [
            # Phase 1: Analysis
            TypeInferencePass(),
            LibraryExpansionPass(),
            MetadataCollectionPass(),
            AllocationAnalysisPass(),
            ControlFlowAnalysisPass(),
            TargetAnalysisPass(),
        ]

        if target_specific:
            # Add conditional target-specific passes
            passes.extend([
                ConditionalPass(
                    condition=lambda r: 'cuda' in r.get('TargetAnalysisPass', {}).get('required_targets', []),
                    pass_instance=CUDAStreamAssignmentPass()
                ),
                ConditionalPass(
                    condition=lambda r: 'fpga' in r.get('TargetAnalysisPass', {}).get('required_targets', []),
                    pass_instance=FPGAPreprocessingPass()
                ),
            ])

        passes.extend([
            # Phase 2: Transformations
            CopyToMapPass(),
            TaskletLanguageLoweringPass(),
            MemletLoweringPass(),

            # Phase 3: Code Generation
            StateStructCreationPass(),
            AllocationCodePass(),
            FrameCodeGenerationPass(),
            TargetCodeGenerationPass(),
            HeaderGenerationPass(),

            # Phase 4: File Generation
            SDFGSplittingPass(),
            CodeObjectCreationPass(),
        ])

        super().__init__(passes)
        self.validate = validate


class ConditionalPass(ppl.Pass):
    """A pass that only runs if a condition is met."""

    def __init__(self, condition: callable, pass_instance: ppl.Pass):
        super().__init__()
        self.condition = condition
        self.pass_instance = pass_instance

    def modifies(self) -> ppl.Modifies:
        return self.pass_instance.modifies()

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return self.pass_instance.should_reapply(modified)

    def depends_on(self):
        return self.pass_instance.depends_on()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        if self.condition(pipeline_results):
            return self.pass_instance.apply_pass(sdfg, pipeline_results)
        return None


# Usage example
def generate_code_modular(sdfg: SDFG, validate: bool = True) -> List[CodeObject]:
    """
    New modular code generation entry point.
    """
    pipeline = CodeGenerationPipeline(validate=validate)
    results = pipeline.apply_pass(sdfg, {})

    # Extract final code objects
    return results[CodeObjectCreationPass.__name__]['code_objects']
```

## Integration with Existing System

### Backward Compatibility Wrapper

```python
def generate_code(sdfg: SDFG, validate: bool = True) -> List[CodeObject]:
    """
    Existing generate_code function, now using modular pipeline internally.
    Maintains full backward compatibility.
    """
    # Use new modular system internally
    return generate_code_modular(sdfg, validate)


def generate_code_legacy(sdfg: SDFG, validate: bool = True) -> List[CodeObject]:
    """
    Legacy implementation preserved for comparison and fallback.
    """
    # Original monolithic implementation
    # ... (existing code moved here)
```

### Incremental Migration Strategy

```python
class HybridCodeGenerationPipeline(ppl.Pipeline):
    """
    Hybrid pipeline that allows incremental migration from monolithic to modular.
    """

    def __init__(self, modular_passes: Set[str] = None):
        self.modular_passes = modular_passes or set()

        passes = []

        # Always use modular analysis passes
        passes.extend([
            MetadataCollectionPass(),
            AllocationAnalysisPass(),
        ])

        # Conditionally use modular code generation
        if 'frame_code' in self.modular_passes:
            passes.append(FrameCodeGenerationPass())
        else:
            passes.append(LegacyFrameCodePass())

        super().__init__(passes)
```

## Performance Considerations

### Caching Strategy

```python
@properties.make_properties
class CachedAnalysisPass(ppl.Pass):
    """Base class for analysis passes with caching support."""

    def __init__(self):
        super().__init__()
        self._cache = {}
        self._cache_key = None

    def _compute_cache_key(self, sdfg: SDFG) -> str:
        """Compute a cache key for the SDFG state."""
        # Simple hash of SDFG structure
        return str(hash((sdfg.cfg_id, len(sdfg.nodes()), len(sdfg.edges()))))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Any]:
        cache_key = self._compute_cache_key(sdfg)

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Compute result
        result = self._analyze(sdfg, pipeline_results)

        # Cache result
        self._cache[cache_key] = result
        return result

    def _analyze(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Any:
        """Override in subclasses."""
        raise NotImplementedError
```

## Testing Strategy

### Unit Tests for Individual Passes

```python
import unittest
from dace import SDFG, SDFGState, dtypes

class TestMetadataCollectionPass(unittest.TestCase):

    def test_simple_sdfg(self):
        """Test metadata collection on a simple SDFG."""
        sdfg = SDFG('test')
        state = sdfg.add_state('main')

        # Add some arrays and symbols
        sdfg.add_array('A', [10], dtypes.float64)
        sdfg.add_symbol('N', dtypes.int32)

        # Run pass
        pass_instance = MetadataCollectionPass()
        result = pass_instance.apply_pass(sdfg, {})

        # Verify results
        self.assertIn('free_symbols', result)
        self.assertIn('arglist', result)
        self.assertEqual(result['arglist']['A'].shape, [10])

    def test_nested_sdfg(self):
        """Test metadata collection with nested SDFGs."""
        # ... test implementation


class TestAllocationAnalysisPass(unittest.TestCase):

    def test_scope_allocation(self):
        """Test scope-based allocation analysis."""
        # ... test implementation

    def test_persistent_allocation(self):
        """Test persistent allocation analysis."""
        # ... test implementation


class TestCodeGenerationPipeline(unittest.TestCase):

    def test_full_pipeline(self):
        """Test complete code generation pipeline."""
        # ... test implementation
```

This implementation approach provides concrete examples of how the modular design would work in practice, with proper pass dependencies, caching strategies, and backward compatibility.
