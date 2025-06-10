# Extending the Modular Code Generation System

This document explains how to extend DaCe's modular code generation system to support new platforms and programming languages. The modular design makes it straightforward to add new backends without modifying the core pipeline.

## Table of Contents

1. [Extension Architecture Overview](#extension-architecture-overview)
2. [Creating a New Platform Backend](#creating-a-new-platform-backend)
3. [Adding Support for New Languages/IRs](#adding-support-for-new-languagesirs)
4. [Integration with the Pipeline](#integration-with-the-pipeline)
5. [Testing Extensions](#testing-extensions)
6. [Best Practices](#best-practices)

## Extension Architecture Overview

The modular code generation system is designed with extensibility as a core principle. Extensions can be added at several levels:

### Extension Points

1. **Target Code Generators**: New hardware platforms or programming models
2. **Language Backends**: New output languages or intermediate representations
3. **Transformation Passes**: Platform-specific optimizations or lowering
4. **Analysis Passes**: Platform-specific metadata collection

### Key Design Principles

- **Modularity**: Each extension is a self-contained component
- **Composability**: Extensions can be combined and reused
- **Inheritance**: Extensions can build upon existing components
- **Registration**: Extensions are discovered and registered automatically

## Creating a New Platform Backend

This section demonstrates how to add support for a new hardware platform (e.g., a custom accelerator, neuromorphic chip, or quantum processor).

### Example: Adding Support for a Neuromorphic Processor

#### Step 1: Define the Target Code Generator

```python
# File: dace/codegen/targets/neuromorphic.py

from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.codeobject import CodeObject
from dace import SDFG, nodes
from typing import List, Dict, Any

class NeuromorphicCodeGen(TargetCodeGenerator):
    """Code generator for neuromorphic processors."""

    target_name = "neuromorphic"
    title = "Neuromorphic Processor"
    language = "spike_train"  # Custom spike-based language

    def __init__(self):
        super().__init__()
        self._neuron_count = 0
        self._synapse_map = {}

    def can_handle(self, node) -> bool:
        """Check if this target can handle the given node."""
        # Handle specific node types for neuromorphic computation
        if isinstance(node, nodes.Tasklet):
            # Check if tasklet contains neuromorphic operations
            return self._is_neuromorphic_tasklet(node)
        elif isinstance(node, nodes.MapEntry):
            # Handle parallel spike processing
            return self._is_spike_parallel(node)
        return False

    def get_includes(self) -> List[str]:
        """Return required header includes."""
        return [
            "#include <neuromorphic_runtime.h>",
            "#include <spike_interface.h>",
        ]

    def get_dependencies(self) -> List[str]:
        """Return compilation dependencies."""
        return [
            "neuromorphic_runtime",
            "spike_processing_lib"
        ]

    def generate_code(self, sdfg: SDFG, pipeline_results: dict) -> List[CodeObject]:
        """Generate neuromorphic processor code."""
        code_objects = []

        # Extract neuromorphic-specific metadata
        neuron_config = self._analyze_neuron_requirements(sdfg, pipeline_results)

        # Generate spike train code
        spike_code = self._generate_spike_trains(sdfg, neuron_config)

        # Generate neuron configuration
        config_code = self._generate_neuron_config(neuron_config)

        # Create code objects
        code_objects.append(CodeObject(
            name=f"{sdfg.name}_spikes",
            code=spike_code,
            language=self.language,
            target=self.target_name,
            title="Spike Train Generation"
        ))

        code_objects.append(CodeObject(
            name=f"{sdfg.name}_config",
            code=config_code,
            language="yaml",  # Configuration in YAML
            target=self.target_name,
            title="Neuron Configuration"
        ))

        return code_objects

    def _is_neuromorphic_tasklet(self, tasklet) -> bool:
        """Check if tasklet contains neuromorphic operations."""
        # Look for spike-related operations in tasklet code
        neuromorphic_ops = ['spike', 'neuron', 'synapse', 'membrane_potential']
        return any(op in tasklet.code.code for op in neuromorphic_ops)

    def _analyze_neuron_requirements(self, sdfg: SDFG, pipeline_results: dict) -> dict:
        """Analyze SDFG to determine neuron and synapse requirements."""
        neuron_config = {
            'neuron_count': 0,
            'synapse_count': 0,
            'network_topology': {},
            'timing_requirements': {}
        }

        # Walk through SDFG and count required neurons/synapses
        for state in sdfg.nodes():
            for node in state.nodes():
                if self.can_handle(node):
                    self._count_neuromorphic_resources(node, neuron_config)

        return neuron_config

    def _generate_spike_trains(self, sdfg: SDFG, config: dict) -> str:
        """Generate spike train processing code."""
        code = []
        code.append("// Neuromorphic spike train processing")
        code.append(f"#define NEURON_COUNT {config['neuron_count']}")
        code.append(f"#define SYNAPSE_COUNT {config['synapse_count']}")
        code.append("")

        # Generate initialization code
        code.append("void initialize_neuromorphic_network() {")
        code.append("    setup_neuron_array(NEURON_COUNT);")
        code.append("    configure_synapses(SYNAPSE_COUNT);")
        code.append("}")
        code.append("")

        # Generate main processing loop
        code.append("void process_spike_trains() {")
        code.append("    for (int timestep = 0; timestep < simulation_time; timestep++) {")
        code.append("        update_membrane_potentials();")
        code.append("        process_spike_events();")
        code.append("        update_synaptic_weights();")
        code.append("    }")
        code.append("}")

        return "\n".join(code)
```

#### Step 2: Register the Target

```python
# File: dace/codegen/targets/__init__.py

# Add to existing imports
from .neuromorphic import NeuromorphicCodeGen

# Add to target registry
AVAILABLE_TARGETS = {
    'cpu': CPUCodeGen,
    'gpu': GPUCodeGen,
    'fpga': FPGACodeGen,
    'neuromorphic': NeuromorphicCodeGen,  # New target
}
```

#### Step 3: Add Target Detection Logic

```python
# File: dace/codegen/passes/target_analysis.py

class TargetAnalysisPass(Pass):
    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> None:
        targets = {}

        # Existing target detection logic...

        # Add neuromorphic detection
        if self._has_neuromorphic_nodes(sdfg):
            targets['neuromorphic'] = NeuromorphicCodeGen()

        pipeline_results['targets'] = targets

    def _has_neuromorphic_nodes(self, sdfg: SDFG) -> bool:
        """Check if SDFG contains neuromorphic computation nodes."""
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet):
                    # Check for neuromorphic operations in tasklet
                    neuromorphic_keywords = ['spike', 'neuron', 'synapse']
                    if any(kw in node.code.code for kw in neuromorphic_keywords):
                        return True
        return False
```

#### Step 4: Add Platform-Specific Transformation Pass

```python
# File: dace/codegen/passes/neuromorphic_lowering.py

from dace.transformation.pass_pipeline import Pass
from dace import SDFG, nodes

class NeuromorphicLoweringPass(Pass):
    """Lower high-level operations to neuromorphic primitives."""

    def modifies(self) -> bool:
        return True

    def should_reapply(self, modified: bool) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> None:
        """Convert mathematical operations to spike-based equivalents."""

        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.Tasklet):
                    self._lower_tasklet_to_spikes(node)

    def _lower_tasklet_to_spikes(self, tasklet):
        """Convert tasklet operations to spike-based operations."""
        code = tasklet.code.code

        # Replace mathematical operations with spike equivalents
        replacements = {
            'multiply': 'synapse_weight_modulation',
            'add': 'spike_accumulation',
            'threshold': 'membrane_potential_threshold',
            'sigmoid': 'neuron_activation_function'
        }

        for math_op, spike_op in replacements.items():
            code = code.replace(math_op, spike_op)

        tasklet.code.code = code
```

#### Step 5: Update Pipeline Configuration

```python
# File: dace/codegen/pipeline_config.py

def create_neuromorphic_pipeline():
    """Create pipeline configuration for neuromorphic targets."""
    return Pipeline([
        # Standard analysis passes
        TypeInferencePass(),
        MetadataCollectionPass(),
        AllocationAnalysisPass(),
        TargetAnalysisPass(),

        # Neuromorphic-specific passes
        ConditionalPass(
            condition=lambda results: 'neuromorphic' in results.get('targets', {}),
            pass=NeuromorphicLoweringPass()
        ),

        # Code generation
        FrameAndTargetCodeGenerationPass(),
    ])
```

## Adding Support for New Languages/IRs

This section demonstrates how to add support for generating code in a new programming language or intermediate representation.

### Example: Adding LLVM IR Support

#### Step 1: Create Language-Specific Code Generator

```python
# File: dace/codegen/targets/llvm_ir.py

from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.codeobject import CodeObject
from dace import SDFG, nodes, data
from typing import List, Dict, Any
import llvmlite.ir as ir

class LLVMIRCodeGen(TargetCodeGenerator):
    """Generate LLVM IR code from SDFG."""

    target_name = "llvm"
    title = "LLVM IR"
    language = "llvm_ir"

    def __init__(self):
        super().__init__()
        self.module = None
        self.builder = None
        self.function = None
        self.variables = {}

    def can_handle(self, node) -> bool:
        """LLVM IR can handle most computational nodes."""
        if isinstance(node, (nodes.Tasklet, nodes.MapEntry, nodes.AccessNode)):
            return True
        # Skip specialized nodes that need specific backends
        if isinstance(node, nodes.LibraryNode):
            return node.implementation in ['BLAS', 'pure']
        return False

    def get_includes(self) -> List[str]:
        """LLVM IR doesn't need traditional includes."""
        return []

    def get_dependencies(self) -> List[str]:
        """Return LLVM-related dependencies."""
        return ["llvm-runtime", "llvm-libs"]

    def generate_code(self, sdfg: SDFG, pipeline_results: dict) -> List[CodeObject]:
        """Generate LLVM IR code for the SDFG."""

        # Initialize LLVM module
        self.module = ir.Module(name=sdfg.name)

        # Generate function signature
        self._generate_function_signature(sdfg, pipeline_results)

        # Generate function body
        self._generate_function_body(sdfg, pipeline_results)

        # Create code object
        ir_code = str(self.module)

        return [CodeObject(
            name=f"{sdfg.name}_llvm",
            code=ir_code,
            language=self.language,
            target=self.target_name,
            title="LLVM IR Implementation"
        )]

    def _generate_function_signature(self, sdfg: SDFG, pipeline_results: dict):
        """Generate LLVM function signature from SDFG arguments."""

        # Analyze SDFG arguments
        args_info = pipeline_results.get('arguments', {})

        # Create LLVM function type
        arg_types = []
        for arg_name, arg_info in args_info.items():
            llvm_type = self._convert_dace_type_to_llvm(arg_info['type'])
            if arg_info.get('is_pointer', False):
                llvm_type = llvm_type.as_pointer()
            arg_types.append(llvm_type)

        # Create function type (void return for now)
        func_type = ir.FunctionType(ir.VoidType(), arg_types)

        # Create function
        self.function = ir.Function(self.module, func_type, name=f"dace_{sdfg.name}")

        # Create entry basic block
        block = self.function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        # Map arguments to variables
        for i, (arg_name, _) in enumerate(args_info.items()):
            self.variables[arg_name] = self.function.args[i]

    def _generate_function_body(self, sdfg: SDFG, pipeline_results: dict):
        """Generate LLVM IR for SDFG states."""

        for state in sdfg.nodes():
            self._generate_state_code(state)

        # Add return instruction
        self.builder.ret_void()

    def _generate_state_code(self, state):
        """Generate LLVM IR for a single state."""

        for node in state.nodes():
            if isinstance(node, nodes.Tasklet):
                self._generate_tasklet_code(node)
            elif isinstance(node, nodes.MapEntry):
                self._generate_map_code(node, state)

    def _generate_tasklet_code(self, tasklet):
        """Generate LLVM IR for a tasklet."""

        # Parse tasklet code and convert to LLVM operations
        code = tasklet.code.code.strip()

        # Simple example: convert basic arithmetic
        if '=' in code:
            lhs, rhs = code.split('=', 1)
            lhs = lhs.strip()
            rhs = rhs.strip()

            # Generate LLVM IR for the expression
            result = self._compile_expression(rhs)

            # Store result (simplified)
            if lhs in self.variables:
                self.builder.store(result, self.variables[lhs])

    def _compile_expression(self, expr: str):
        """Compile a simple expression to LLVM IR."""

        # Very simplified expression compiler
        # In practice, you'd want a proper parser

        if expr.isdigit():
            # Constant integer
            return ir.Constant(ir.IntType(32), int(expr))
        elif '+' in expr:
            # Addition
            operands = expr.split('+')
            left = self._compile_expression(operands[0].strip())
            right = self._compile_expression(operands[1].strip())
            return self.builder.add(left, right)
        elif expr in self.variables:
            # Variable reference
            return self.builder.load(self.variables[expr])

        # Default: return zero
        return ir.Constant(ir.IntType(32), 0)

    def _convert_dace_type_to_llvm(self, dace_type):
        """Convert DaCe data type to LLVM type."""

        if dace_type == data.int32:
            return ir.IntType(32)
        elif dace_type == data.int64:
            return ir.IntType(64)
        elif dace_type == data.float32:
            return ir.FloatType()
        elif dace_type == data.float64:
            return ir.DoubleType()
        else:
            # Default to i32
            return ir.IntType(32)
```

#### Step 2: Add Language Support to Code Object

```python
# File: dace/codegen/codeobject.py

# Add to LANGUAGE_MAPPINGS
LANGUAGE_MAPPINGS = {
    'cpp': 'C++',
    'cuda': 'CUDA',
    'llvm_ir': 'LLVM IR',  # New language
    'spike_train': 'Spike Train Language',  # From neuromorphic example
}

# Add to file extensions
FILE_EXTENSIONS = {
    'cpp': '.cpp',
    'cuda': '.cu',
    'llvm_ir': '.ll',  # LLVM IR file extension
    'spike_train': '.spike',
}
```

#### Step 3: Add LLVM-Specific Analysis Pass

```python
# File: dace/codegen/passes/llvm_analysis.py

from dace.transformation.pass_pipeline import Pass
from dace import SDFG, nodes

class LLVMAnalysisPass(Pass):
    """Analyze SDFG for LLVM IR generation requirements."""

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> None:
        """Collect LLVM-specific metadata."""

        llvm_info = {
            'requires_vectorization': False,
            'memory_access_patterns': {},
            'control_flow_complexity': 'simple',
            'parallelization_opportunities': []
        }

        # Analyze for vectorization opportunities
        for state in sdfg.nodes():
            if self._has_vectorizable_operations(state):
                llvm_info['requires_vectorization'] = True

        # Analyze memory access patterns
        llvm_info['memory_access_patterns'] = self._analyze_memory_accesses(sdfg)

        pipeline_results['llvm_analysis'] = llvm_info

    def _has_vectorizable_operations(self, state) -> bool:
        """Check if state contains vectorizable operations."""
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                # Maps are typically vectorizable
                return True
        return False

    def _analyze_memory_accesses(self, sdfg: SDFG) -> dict:
        """Analyze memory access patterns for optimization."""
        patterns = {
            'sequential': [],
            'strided': [],
            'random': []
        }

        # Simplified analysis - in practice would be more sophisticated
        for state in sdfg.nodes():
            for edge in state.edges():
                if edge.data.subset:
                    # Analyze access pattern
                    if self._is_sequential_access(edge.data.subset):
                        patterns['sequential'].append(edge)
                    elif self._is_strided_access(edge.data.subset):
                        patterns['strided'].append(edge)
                    else:
                        patterns['random'].append(edge)

        return patterns
```

#### Step 4: Integration with Pipeline

```python
# File: dace/codegen/pipeline_config.py

def create_llvm_pipeline():
    """Create pipeline configuration for LLVM IR generation."""
    return Pipeline([
        # Standard analysis passes
        TypeInferencePass(),
        MetadataCollectionPass(),
        AllocationAnalysisPass(),
        TargetAnalysisPass(),

        # LLVM-specific analysis
        ConditionalPass(
            condition=lambda results: 'llvm' in results.get('targets', {}),
            pass=LLVMAnalysisPass()
        ),

        # Code generation
        FrameAndTargetCodeGenerationPass(),
    ])

def create_multi_backend_pipeline():
    """Create pipeline that can generate multiple backends."""
    return Pipeline([
        # Common analysis passes
        TypeInferencePass(),
        MetadataCollectionPass(),
        AllocationAnalysisPass(),
        TargetAnalysisPass(),

        # Backend-specific analysis
        ConditionalPass(
            condition=lambda results: 'llvm' in results.get('targets', {}),
            pass=LLVMAnalysisPass()
        ),
        ConditionalPass(
            condition=lambda results: 'neuromorphic' in results.get('targets', {}),
            pass=NeuromorphicLoweringPass()
        ),

        # Code generation for all targets
        FrameAndTargetCodeGenerationPass(),
    ])
```

## Integration with the Pipeline

### Automatic Target Discovery

The modular system automatically discovers and registers new targets:

```python
# File: dace/codegen/passes/target_analysis.py

class TargetAnalysisPass(Pass):
    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> None:
        """Discover and configure appropriate targets for the SDFG."""

        targets = {}

        # Automatic target discovery based on node analysis
        for target_name, target_class in AVAILABLE_TARGETS.items():
            target_instance = target_class()

            # Check if any nodes in the SDFG can be handled by this target
            can_handle_any = False
            for state in sdfg.nodes():
                for node in state.nodes():
                    if target_instance.can_handle(node):
                        can_handle_any = True
                        break
                if can_handle_any:
                    break

            if can_handle_any:
                targets[target_name] = target_instance

        pipeline_results['targets'] = targets
        pipeline_results['target_priorities'] = self._compute_target_priorities(targets)

    def _compute_target_priorities(self, targets: dict) -> dict:
        """Compute priority order for targets when multiple can handle the same node."""
        priorities = {
            'neuromorphic': 100,  # Highest priority for neuromorphic nodes
            'fpga': 90,
            'gpu': 80,
            'llvm': 70,
            'cpu': 60,  # Lowest priority (fallback)
        }

        return {name: priorities.get(name, 50) for name in targets.keys()}
```

### Multi-Target Code Generation

The system can generate code for multiple targets simultaneously:

```python
# File: dace/codegen/passes/frame_and_target_generation.py

class FrameAndTargetCodeGenerationPass(Pass):
    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> None:
        """Generate code for all applicable targets."""

        targets = pipeline_results.get('targets', {})
        code_objects = []

        for target_name, target_generator in targets.items():
            try:
                target_code_objects = target_generator.generate_code(sdfg, pipeline_results)
                code_objects.extend(target_code_objects)
            except Exception as e:
                # Log error but continue with other targets
                print(f"Warning: Failed to generate code for target {target_name}: {e}")

        pipeline_results['code_objects'] = code_objects

        # Generate orchestration code if multiple targets are used
        if len(targets) > 1:
            orchestration_code = self._generate_orchestration_code(targets, pipeline_results)
            code_objects.append(orchestration_code)

    def _generate_orchestration_code(self, targets: dict, pipeline_results: dict) -> CodeObject:
        """Generate code to orchestrate multiple targets."""

        code = []
        code.append("// Multi-target orchestration code")
        code.append("#include <runtime/multi_target.h>")
        code.append("")

        # Initialize all targets
        for target_name in targets.keys():
            code.append(f"    initialize_{target_name}_runtime();")

        # Add synchronization points
        code.append("    synchronize_all_targets();")

        # Cleanup
        for target_name in targets.keys():
            code.append(f"    cleanup_{target_name}_runtime();")

        return CodeObject(
            name="multi_target_orchestration",
            code="\n".join(code),
            language="cpp",
            target="orchestration",
            title="Multi-Target Orchestration"
        )
```

## Testing Extensions

### Unit Testing New Targets

```python
# File: tests/codegen/test_neuromorphic_target.py

import unittest
from dace import SDFG, nodes
from dace.codegen.targets.neuromorphic import NeuromorphicCodeGen

class TestNeuromorphicTarget(unittest.TestCase):

    def setUp(self):
        self.target = NeuromorphicCodeGen()

    def test_can_handle_neuromorphic_tasklet(self):
        """Test that neuromorphic target can handle spike-related tasklets."""

        # Create a tasklet with neuromorphic operations
        tasklet = nodes.Tasklet(
            name="spike_processor",
            inputs={'input_spikes'},
            outputs={'output_spikes'},
            code="output_spikes = process_spike_train(input_spikes)"
        )

        self.assertTrue(self.target.can_handle(tasklet))

    def test_cannot_handle_regular_tasklet(self):
        """Test that neuromorphic target rejects non-neuromorphic tasklets."""

        tasklet = nodes.Tasklet(
            name="math_processor",
            inputs={'a', 'b'},
            outputs={'c'},
            code="c = a + b"
        )

        self.assertFalse(self.target.can_handle(tasklet))

    def test_code_generation(self):
        """Test code generation for neuromorphic SDFG."""

        # Create simple SDFG with neuromorphic tasklet
        sdfg = SDFG("test_neuromorphic")
        state = sdfg.add_state("main")

        tasklet = state.add_tasklet(
            "spike_gen",
            inputs={'input_data'},
            outputs={'spikes'},
            code="spikes = generate_spikes(input_data)"
        )

        # Mock pipeline results
        pipeline_results = {
            'arguments': {'input_data': {'type': 'float32', 'is_pointer': True}}
        }

        code_objects = self.target.generate_code(sdfg, pipeline_results)

        self.assertGreater(len(code_objects), 0)
        self.assertEqual(code_objects[0].target, "neuromorphic")
        self.assertIn("neuromorphic", code_objects[0].code)

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
# File: tests/codegen/test_multi_target_pipeline.py

import unittest
from dace import SDFG
from dace.codegen.pipeline_config import create_multi_backend_pipeline

class TestMultiTargetPipeline(unittest.TestCase):

    def test_neuromorphic_and_cpu_generation(self):
        """Test generating code for both neuromorphic and CPU targets."""

        # Create SDFG with mixed computation
        sdfg = SDFG("mixed_computation")
        state = sdfg.add_state("main")

        # Add CPU computation
        cpu_tasklet = state.add_tasklet(
            "cpu_math",
            inputs={'a', 'b'},
            outputs={'c'},
            code="c = a * b + 42"
        )

        # Add neuromorphic computation
        neuro_tasklet = state.add_tasklet(
            "spike_proc",
            inputs={'input_spikes'},
            outputs={'output_spikes'},
            code="output_spikes = neuron_activation(input_spikes)"
        )

        # Run multi-target pipeline
        pipeline = create_multi_backend_pipeline()
        pipeline_results = {}
        pipeline.apply_pass(sdfg, pipeline_results)

        # Check that code was generated for both targets
        code_objects = pipeline_results.get('code_objects', [])
        targets_found = {obj.target for obj in code_objects}

        self.assertIn('cpu', targets_found)
        self.assertIn('neuromorphic', targets_found)

    def test_llvm_ir_generation(self):
        """Test LLVM IR generation for computational SDFG."""

        # Create computational SDFG
        sdfg = SDFG("llvm_test")
        state = sdfg.add_state("compute")

        tasklet = state.add_tasklet(
            "vector_add",
            inputs={'a', 'b'},
            outputs={'c'},
            code="c = a + b"
        )

        # Force LLVM target
        pipeline_results = {
            'targets': {'llvm': LLVMIRCodeGen()}
        }

        pipeline = create_multi_backend_pipeline()
        pipeline.apply_pass(sdfg, pipeline_results)

        # Check LLVM IR was generated
        code_objects = pipeline_results.get('code_objects', [])
        llvm_objects = [obj for obj in code_objects if obj.language == 'llvm_ir']

        self.assertGreater(len(llvm_objects), 0)
        self.assertIn('define', llvm_objects[0].code)  # LLVM IR function definition

if __name__ == '__main__':
    unittest.main()
```

## Best Practices

### 1. Design Principles

- **Single Responsibility**: Each target should handle one type of hardware/language
- **Clear Interfaces**: Implement all required methods from `TargetCodeGenerator`
- **Error Handling**: Gracefully handle unsupported operations
- **Documentation**: Document what nodes your target can handle

### 2. Performance Considerations

- **Lazy Initialization**: Only initialize expensive resources when needed
- **Caching**: Cache expensive computations between passes
- **Incremental Generation**: Support incremental code generation when possible

### 3. Maintainability

- **Test Coverage**: Provide comprehensive unit and integration tests
- **Backward Compatibility**: Ensure new targets don't break existing functionality
- **Code Reuse**: Extend existing targets when possible rather than starting from scratch

### 4. Integration Guidelines

- **Registration**: Use the automatic registration system
- **Pipeline Integration**: Add conditional passes for target-specific optimizations
- **Documentation**: Update this document when adding new extension patterns

## Conclusion

The modular code generation system is designed to make adding new platforms and languages straightforward. The key is to:

1. Implement the `TargetCodeGenerator` interface for your new backend
2. Add any necessary analysis or transformation passes
3. Register your target with the system
4. Provide comprehensive tests

The examples above demonstrate the flexibility of the system - from specialized neuromorphic processors to general-purpose LLVM IR generation. The same patterns can be applied to add support for any new platform or programming language.
