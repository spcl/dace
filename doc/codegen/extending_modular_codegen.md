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

### Example: Adding Support for OpenCL Code Generation

This example demonstrates how to add OpenCL support, showcasing advanced features like custom allocation handling, memory copy management, and SDFG splitting for kernel generation.

#### Step 1: Define the OpenCL Target Code Generator

```python
# File: dace/codegen/targets/opencl.py

from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.codeobject import CodeObject
from dace import SDFG, nodes, data as dt
from typing import List, Dict, Any
import copy

class OpenCLCodeGen(TargetCodeGenerator):
    """Code generator for OpenCL devices (Intel FPGAs, CPUs, etc.)."""

    target_name = "opencl"
    title = "OpenCL"
    language = "opencl"

    def __init__(self):
        super().__init__()
        self._device_memories = {}
        self._kernel_sdfgs = {}

    def can_handle(self, node) -> bool:
        """Check if this target can handle the given node."""
        if isinstance(node, nodes.MapEntry):
            # Handle parallel maps that can be kernelized
            return self._is_kernelizable_map(node)
        elif isinstance(node, nodes.Tasklet):
            # Handle individual compute tasklets
            return self._is_opencl_compatible_tasklet(node)
        return False

    def get_includes(self) -> List[str]:
        """Return required header includes."""
        return [
            "#include <CL/cl.h>",
            "#include <opencl_runtime.h>",
        ]

    def get_dependencies(self) -> List[str]:
        """Return compilation dependencies."""
        return ["OpenCL", "opencl_runtime"]

    def generate_code(self, sdfg: SDFG, pipeline_results: dict) -> List[CodeObject]:
        """Generate OpenCL host and kernel code."""
        code_objects = []

        # Split SDFG into host and kernel portions
        host_sdfg, kernel_sdfgs = self._split_sdfg_for_opencl(sdfg, pipeline_results)

        # Generate kernel code objects (.cl files)
        for kernel_name, kernel_sdfg in kernel_sdfgs.items():
            kernel_code = self._generate_kernel_code(kernel_sdfg, pipeline_results)
            code_objects.append(CodeObject(
                name=f"{kernel_name}.cl",
                code=kernel_code,
                language="opencl",
                target=self.target_name,
                title=f"OpenCL Kernel: {kernel_name}"
            ))

        # Generate host code with OpenCL runtime calls
        host_code = self._generate_host_code(host_sdfg, kernel_sdfgs, pipeline_results)
        code_objects.append(CodeObject(
            name=f"{sdfg.name}_opencl_host.cpp",
            code=host_code,
            language="cpp",
            target=self.target_name,
            title="OpenCL Host Code"
        ))

        return code_objects

    def _split_sdfg_for_opencl(self, sdfg: SDFG, pipeline_results: dict) -> tuple:
        """Split SDFG into host and kernel portions."""
        host_sdfg = copy.deepcopy(sdfg)
        kernel_sdfgs = {}

        for state in sdfg.nodes():
            for node in state.nodes():
                if self.can_handle(node) and isinstance(node, nodes.MapEntry):
                    # Extract kernelizable map into separate SDFG
                    kernel_name = f"{sdfg.name}_{node.label}_kernel"
                    kernel_sdfg = self._extract_kernel_sdfg(node, state, sdfg)
                    kernel_sdfgs[kernel_name] = kernel_sdfg

        return host_sdfg, kernel_sdfgs

    def _generate_kernel_code(self, kernel_sdfg: SDFG, pipeline_results: dict) -> str:
        """Generate OpenCL kernel code from kernel SDFG."""
        code = []

        # Generate kernel signature
        kernel_params = self._analyze_kernel_parameters(kernel_sdfg, pipeline_results)
        code.append(f"__kernel void {kernel_sdfg.name}({', '.join(kernel_params)}) {{")

        # Generate kernel body from SDFG
        for state in kernel_sdfg.nodes():
            code.extend(self._generate_state_code(state, pipeline_results))

        code.append("}")
        return "\n".join(code)

    def _generate_host_code(self, host_sdfg: SDFG, kernel_sdfgs: dict, pipeline_results: dict) -> str:
        """Generate host code with OpenCL runtime management."""
        code = []

        # Include headers
        code.extend(self.get_includes())
        code.append("")

        # Generate OpenCL context initialization
        code.append("// OpenCL context and device setup")
        code.append("cl_context context;")
        code.append("cl_device_id device;")
        code.append("cl_command_queue queue;")
        code.append("")

        # Generate buffer allocations with custom OpenCL memory management
        allocation_info = pipeline_results.get('allocation_analysis', {})
        for array_name, allocation in allocation_info.items():
            code.extend(self._generate_buffer_allocation(array_name, allocation))

        # Generate kernel compilation and execution
        for kernel_name, kernel_sdfg in kernel_sdfgs.items():
            code.extend(self._generate_kernel_invocation(kernel_name, kernel_sdfg, pipeline_results))

        # Generate memory copy operations with async transfers
        copy_analysis = pipeline_results.get('copy_analysis', {})
        for copy_info in copy_analysis:
            code.extend(self._generate_async_copy(copy_info))

        return "\n".join(code)

    def _generate_buffer_allocation(self, array_name: str, allocation: dict) -> List[str]:
        """Generate OpenCL buffer allocation with custom memory flags."""
        code = []
        size = allocation.get('size', 0)
        access_pattern = allocation.get('access_pattern', 'read_write')

        # Map access pattern to OpenCL memory flags
        flag_map = {
            'read_only': 'CL_MEM_READ_ONLY',
            'write_only': 'CL_MEM_WRITE_ONLY',
            'read_write': 'CL_MEM_READ_WRITE'
        }
        flags = flag_map.get(access_pattern, 'CL_MEM_READ_WRITE')

        code.append(f"// Allocate OpenCL buffer for {array_name}")
        code.append(f"cl_mem {array_name}_buffer = clCreateBuffer(context, {flags}, {size}, NULL, &err);")
        code.append(f"check_opencl_error(err, \"Failed to allocate buffer for {array_name}\");")
        code.append("")

        return code

    def _generate_async_copy(self, copy_info: dict) -> List[str]:
        """Generate asynchronous memory copy operations."""
        code = []
        src = copy_info.get('source')
        dst = copy_info.get('destination')
        size = copy_info.get('size')

        if copy_info.get('direction') == 'host_to_device':
            code.append(f"// Async copy {src} -> {dst}")
            code.append(f"err = clEnqueueWriteBuffer(queue, {dst}_buffer, CL_FALSE, 0, {size}, {src}, 0, NULL, NULL);")
        elif copy_info.get('direction') == 'device_to_host':
            code.append(f"// Async copy {src} -> {dst}")
            code.append(f"err = clEnqueueReadBuffer(queue, {src}_buffer, CL_FALSE, 0, {size}, {dst}, 0, NULL, NULL);")

        code.append(f"check_opencl_error(err, \"Failed to copy {src} -> {dst}\");")
        code.append("")

        return code
```

#### Step 2: Add Custom Allocation Pass for OpenCL

```python
# File: dace/codegen/passes/opencl_allocation.py

from dace.transformation.pass_pipeline import Pass
from dace import SDFG, nodes, data as dt

class OpenCLAllocationPass(Pass):
    """Custom allocation analysis for OpenCL memory management."""

    def modifies(self) -> bool:
        return False  # Analysis pass doesn't modify SDFG

    def should_reapply(self, modified: bool) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> None:
        """Analyze memory allocations for OpenCL-specific optimizations."""

        allocation_analysis = {}
        copy_analysis = []

        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode):
                    # Analyze data access patterns for OpenCL buffer flags
                    array_info = self._analyze_array_access(node, state, sdfg)
                    allocation_analysis[node.data] = array_info

                elif isinstance(node, nodes.MapEntry):
                    # Analyze copy requirements for kernel boundaries
                    copy_info = self._analyze_kernel_copies(node, state, sdfg)
                    copy_analysis.extend(copy_info)

        pipeline_results['allocation_analysis'] = allocation_analysis
        pipeline_results['copy_analysis'] = copy_analysis

    def _analyze_array_access(self, access_node, state, sdfg) -> dict:
        """Determine OpenCL memory access patterns for optimal buffer allocation."""
        array_name = access_node.data
        array_desc = sdfg.arrays[array_name]

        # Analyze read/write patterns
        is_read = False
        is_written = False

        for edge in state.edges():
            if edge.src == access_node:
                is_read = True
            elif edge.dst == access_node:
                is_written = True

        # Determine access pattern
        if is_read and is_written:
            access_pattern = 'read_write'
        elif is_read:
            access_pattern = 'read_only'
        elif is_written:
            access_pattern = 'write_only'
        else:
            access_pattern = 'read_write'  # Default

        return {
            'size': array_desc.total_size * array_desc.dtype.bytes,
            'access_pattern': access_pattern,
            'dtype': str(array_desc.dtype),
            'shape': array_desc.shape
        }
```

#### Step 3: Add SDFG Splitting Pass

```python
# File: dace/codegen/passes/opencl_sdfg_splitting.py

from dace.transformation.pass_pipeline import Pass
from dace import SDFG, nodes, SDFGState
import copy

class OpenCLSDFGSplittingPass(Pass):
    """Split SDFG into host and kernel portions for OpenCL code generation."""

    def modifies(self) -> bool:
        return True  # Creates new SDFGs

    def should_reapply(self, modified: bool) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> None:
        """Split SDFG for OpenCL kernel extraction."""

        # Check if OpenCL target is present
        targets = pipeline_results.get('targets', {})
        if 'opencl' not in targets:
            return

        kernel_sdfgs = {}

        for state in sdfg.nodes():
            kernelizable_maps = self._find_kernelizable_maps(state)

            for map_node in kernelizable_maps:
                # Create kernel SDFG
                kernel_name = f"{sdfg.name}_{map_node.label}_kernel"
                kernel_sdfg = self._extract_kernel_sdfg(map_node, state, sdfg)
                kernel_sdfgs[kernel_name] = kernel_sdfg

                # Replace map with kernel call node in host SDFG
                self._replace_with_kernel_call(map_node, state, kernel_name)

        pipeline_results['kernel_sdfgs'] = kernel_sdfgs

    def _find_kernelizable_maps(self, state: SDFGState) -> list:
        """Find map nodes that can be converted to OpenCL kernels."""
        kernelizable = []

        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                # Check if map is suitable for kernelization
                if self._is_suitable_for_kernel(node, state):
                    kernelizable.append(node)

        return kernelizable

    def _extract_kernel_sdfg(self, map_node, state, parent_sdfg) -> SDFG:
        """Extract map scope into separate SDFG for kernel generation."""
        kernel_sdfg = SDFG(f"{parent_sdfg.name}_{map_node.label}_kernel")
        kernel_state = kernel_sdfg.add_state("kernel_state")

        # Copy map scope contents to kernel SDFG
        # This is a simplified version - real implementation would be more complex
        self._copy_map_scope_to_kernel(map_node, state, kernel_state, kernel_sdfg)

        return kernel_sdfg
```

#### Step 4: Update Pipeline Configuration

```python
# File: dace/codegen/pipeline_config.py

def create_opencl_pipeline():
    """Create pipeline configuration for OpenCL targets."""
    return Pipeline([
        # Standard analysis passes
        TypeInferencePass(),
        MetadataCollectionPass(),
        TargetAnalysisPass(),

        # OpenCL-specific analysis
        ConditionalPass(
            condition=lambda results: 'opencl' in results.get('targets', {}),
            pass=OpenCLAllocationPass()
        ),

        # Transformations
        ConditionalPass(
            condition=lambda results: 'opencl' in results.get('targets', {}),
            pass=OpenCLSDFGSplittingPass()
        ),

        # Code generation
        TargetCodeGenerationPass(),
    ])
```

#### Step 5: Target Registration and Integration

```python
# File: dace/codegen/targets/__init__.py

from .opencl import OpenCLCodeGen

AVAILABLE_TARGETS = {
    'cpu': CPUCodeGen,
    'gpu': GPUCodeGen,
    'fpga': FPGACodeGen,
    'opencl': OpenCLCodeGen,  # New OpenCL target
}
```

This example demonstrates several advanced features:

1. **Custom Allocation Management**: OpenCL-specific buffer allocation with memory access pattern analysis
2. **SDFG Splitting**: Automatic extraction of kernel portions into separate `.cl` files
3. **Asynchronous Memory Copies**: Host-device transfer optimization
4. **Target-Specific Passes**: Custom analysis and transformation passes for OpenCL
5. **Multi-File Generation**: Separate host (.cpp) and kernel (.cl) code objects

The modular design makes it straightforward to add sophisticated platform-specific optimizations while maintaining clean separation of concerns.

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

The OpenCL example above demonstrates the flexibility of the system, showing advanced features like custom allocation management, SDFG splitting, and multi-file code generation. The same patterns can be applied to add support for any new platform or programming language.
