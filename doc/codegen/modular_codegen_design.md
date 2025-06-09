# Modular Code Generator Design Document

This document outlines the design for refactoring DaCe's code generation system from a monolithic structure into a modular, pass-based pipeline architecture using DaCe's existing `Pass` and `Pipeline` infrastructure.

## Table of Contents

1. [Current System Analysis](#current-system-analysis)
2. [Proposed Pass Decomposition](#proposed-pass-decomposition)
3. [Abstract Pipeline Design](#abstract-pipeline-design)
4. [Information Flow and Reuse](#information-flow-and-reuse)
5. [Organizational Structure](#organizational-structure)
6. [Target Refactoring](#target-refactoring)
7. [Implementation Roadmap](#implementation-roadmap)

## Current System Analysis

### Current Code Generation Flow

The current code generation system in DaCe follows this monolithic structure in `dace.codegen.codegen.generate_code()`:

1. **Validation and Preprocessing**
   - SDFG validation (`sdfg.validate()`)
   - Type inference (`infer_types.infer_connector_types()`)
   - Default storage/schedule type setting (`infer_types.set_default_schedule_and_storage_types()`)
   - Library node expansion (`sdfg.expand_library_nodes()`)

2. **Frame Code Generator Setup**
   - Create `DaCeCodeGenerator` instance
   - Target instantiation (CPU, CUDA, FPGA, etc.)
   - Code generation target querying (`_get_codegen_targets()`)

3. **Target Preprocessing**
   - Each target runs its `preprocess()` method
   - Instrumentation provider instantiation

4. **Monolithic Code Generation** (in `DaCeCodeGenerator.generate_code()`)
   - Allocation lifetime determination (`determine_allocation_lifetime()`)
   - State struct creation
   - Frame code generation with recursive SDFG traversal

### Current Components Survey

#### Core Files:
- **`codegen.py`**: Main entry point and orchestration
- **`targets/framecode.py`**: Monolithic frame code generator
- **`dispatcher.py`**: Target routing and variable tracking
- **`control_flow.py`**: Structured control flow extraction
- **`compiler.py`**: CMake integration and compilation

#### Target System:
- **`targets/target.py`**: Base target interface
- **`targets/cpu.py`**: CPU/OpenMP code generation
- **`targets/cuda.py`**: CUDA/HIP GPU code generation
- **`targets/fpga.py`**: FPGA code generation
- **`targets/cpp.py`**: C++ utilities
- **`targets/mpi.py`**: MPI parallelization
- **`targets/rtl.py`**: RTL/SystemVerilog generation

#### Specialized Systems:
- **`targets/sve/`**: ARM SVE vectorization
- **`targets/mlir/`**: MLIR backend
- **`instrumentation/`**: Profiling and monitoring
- **`tools/`**: Type inference and runtime utilities

### Current Monolithic Behaviors in DaCeCodeGenerator

The `DaCeCodeGenerator` class currently handles numerous responsibilities that should be decomposed:

1. **Metadata Collection**
   - Free symbol analysis (`free_symbols()`)
   - Symbol and constant resolution
   - Argument list construction
   - Shared transient identification

2. **Allocation Analysis**
   - Allocation lifetime determination (`determine_allocation_lifetime()`)
   - Allocation scope determination
   - Memory layout analysis
   - Persistent/global memory handling

3. **Code Structure Creation**
   - State struct generation
   - Function signature creation
   - Header/footer generation
   - Constants definition

4. **Traversal and Generation**
   - Recursive SDFG traversal
   - State code generation dispatch
   - Target-specific code routing
   - Memory allocation/deallocation code

## Proposed Pass Decomposition

### Phase 1: Analysis Passes

#### 1. **TypeInferencePass**
- **Purpose**: Infer connector types and set default storage/schedule types
- **Input**: Raw SDFG
- **Output**: SDFG with inferred types, pipeline_results["type_info"]
- **Current Location**: `infer_types.py` functions

#### 2. **LibraryExpansionPass**
- **Purpose**: Expand all library nodes that haven't been expanded
- **Input**: Type-inferred SDFG
- **Output**: SDFG with expanded library nodes
- **Current Location**: `sdfg.expand_library_nodes()`

#### 3. **MetadataCollectionPass**
- **Purpose**: Collect free symbols, argument lists, constants, shared transients
- **Input**: Expanded SDFG
- **Output**: pipeline_results["metadata"] = {symbols, arglist, constants, shared_transients}
- **Current Location**: `DaCeCodeGenerator.__init__()`

#### 4. **AllocationAnalysisPass**
- **Purpose**: Determine allocation lifetimes and scopes for all data containers
- **Input**: SDFG with metadata
- **Output**: pipeline_results["allocation_info"] = {to_allocate, where_allocated}
- **Current Location**: `DaCeCodeGenerator.determine_allocation_lifetime()`

#### 5. **ControlFlowAnalysisPass**
- **Purpose**: Extract structured control flow from state machines
- **Input**: SDFG
- **Output**: pipeline_results["control_flow_tree"]
- **Current Location**: `control_flow.py` functions

#### 6. **TargetAnalysisPass**
- **Purpose**: Identify required code generation targets and dispatch routes
- **Input**: SDFG with metadata
- **Output**: pipeline_results["targets"] = {required_targets, dispatch_info}
- **Current Location**: `_get_codegen_targets()`

### Phase 2: Transformation Passes

#### 7. **CopyToMapPass**
- **Purpose**: Convert complex memory copies to Map nodes where needed
- **Input**: SDFG with targets identified
- **Output**: SDFG with transformed copies
- **Current Location**: `cuda.py` preprocessing, various target preprocessors
- **Applies To**: GPU strided copies, FPGA transfers

#### 8. **StreamAssignmentPass** (GPU-specific)
- **Purpose**: Assign CUDA streams for concurrent execution
- **Input**: GPU-targeted SDFG
- **Output**: SDFG with stream assignments, pipeline_results["stream_info"]
- **Current Location**: Embedded in CUDA code generation

#### 9. **TaskletLanguageLoweringPass**
- **Purpose**: Convert Python/generic tasklets to target language (C++/CUDA/etc.)
- **Input**: SDFG with tasklets
- **Output**: SDFG with lowered tasklets
- **Current Location**: Distributed across target generators

### Phase 3: Code Generation Passes

#### 10. **StateStructCreationPass**
- **Purpose**: Generate state struct definitions for persistent data
- **Input**: SDFG with allocation info
- **Output**: pipeline_results["state_struct"] = {struct_def, struct_init}
- **Current Location**: `DaCeCodeGenerator.generate_code()`

#### 11. **AllocationCodePass**
- **Purpose**: Generate allocation/deallocation code for each scope
- **Input**: SDFG with allocation analysis
- **Output**: pipeline_results["allocation_code"] = {alloc_code, dealloc_code}
- **Current Location**: `allocate_arrays_in_scope()`, `deallocate_arrays_in_scope()`

#### 12. **MemletLoweringPass**
- **Purpose**: Lower high-level memlets to explicit copy operations
- **Input**: SDFG with target analysis
- **Output**: SDFG with explicit copies as tasklets
- **Current Location**: Embedded in target-specific copy generation

#### 13. **FrameCodeGenerationPass**
- **Purpose**: Generate the main frame code with function signatures and dispatch
- **Input**: All previous pass results
- **Output**: pipeline_results["frame_code"] = {global_code, local_code}
- **Current Location**: `DaCeCodeGenerator.generate_code()`

#### 14. **TargetCodeGenerationPass**
- **Purpose**: Generate target-specific code files
- **Input**: SDFG with frame code
- **Output**: List of CodeObject instances
- **Current Location**: Target-specific `get_generated_codeobjects()`

#### 15. **HeaderGenerationPass**
- **Purpose**: Generate C/C++ header files for SDFG interface
- **Input**: SDFG with frame code
- **Output**: pipeline_results["headers"] = {call_header, sample_main}
- **Current Location**: `generate_headers()`, `generate_dummy()`

### Phase 4: File Generation Passes

#### 16. **SDFGSplittingPass**
- **Purpose**: Split complex SDFGs into multiple files if needed
- **Input**: Single SDFG with all code
- **Output**: List of SDFGs (one per target file)
- **Current Location**: Implicit in current system

#### 17. **CodeObjectCreationPass**
- **Purpose**: Create final CodeObject instances for each generated file
- **Input**: Split SDFGs and code
- **Output**: List[CodeObject] ready for compilation
- **Current Location**: End of `generate_code()`

## Abstract Pipeline Design

### Main Code Generation Pipeline

```python
class CodeGenerationPipeline(Pipeline):
    """Complete code generation pipeline for DaCe SDFGs."""

    def __init__(self):
        super().__init__([
            # Phase 1: Analysis
            TypeInferencePass(),
            LibraryExpansionPass(),
            MetadataCollectionPass(),
            AllocationAnalysisPass(),
            ControlFlowAnalysisPass(),
            TargetAnalysisPass(),

            # Phase 2: Transformations
            CopyToMapPass(),
            ConditionalPipeline([
                (lambda r: 'cuda' in r.get('targets', []), StreamAssignmentPass()),
                (lambda r: 'fpga' in r.get('targets', []), FPGAPreprocessingPass()),
            ]),
            TaskletLanguageLoweringPass(),

            # Phase 3: Code Generation
            StateStructCreationPass(),
            AllocationCodePass(),
            MemletLoweringPass(),
            FrameCodeGenerationPass(),
            TargetCodeGenerationPass(),
            HeaderGenerationPass(),

            # Phase 4: File Generation
            SDFGSplittingPass(),
            CodeObjectCreationPass(),
        ])
```

### Target-Specific Sub-Pipelines

```python
class CUDACodegenPipeline(Pipeline):
    """CUDA-specific code generation pipeline."""

    def __init__(self):
        super().__init__([
            CUDATargetValidationPass(),
            CUDAMemoryAnalysisPass(),
            CUDAKernelExtractionPass(),
            CUDAStreamAssignmentPass(),
            CUDAKernelGenerationPass(),
            CUDALaunchCodePass(),
        ])

class FPGACodegenPipeline(Pipeline):
    """FPGA-specific code generation pipeline."""

    def __init__(self):
        super().__init__([
            FPGAResourceAnalysisPass(),
            FPGAStreamingPass(),
            FPGAHLSGenerationPass(),
            FPGAHostCodePass(),
        ])
```

## Information Flow and Reuse

### Pipeline Results Schema

The `pipeline_results` dictionary will contain structured information to maximize reuse:

```python
pipeline_results = {
    # From TypeInferencePass
    "type_info": {
        "connector_types": Dict[Node, Dict[str, dtypes.typeclass]],
        "storage_types": Dict[str, dtypes.StorageType],
        "schedule_types": Dict[Node, dtypes.ScheduleType],
    },

    # From MetadataCollectionPass
    "metadata": {
        "free_symbols": Dict[int, Set[str]],
        "symbols_and_constants": Dict[int, Set[str]],
        "arglist": Dict[str, data.Data],
        "shared_transients": Dict[int, Set[str]],
    },

    # From AllocationAnalysisPass
    "allocation_info": {
        "to_allocate": DefaultDict[scope, List[allocation_tuple]],
        "where_allocated": Dict[Tuple[SDFG, str], scope],
        "allocation_scopes": Dict[str, scope],
    },

    # From TargetAnalysisPass
    "targets": {
        "required_targets": Set[TargetCodeGenerator],
        "dispatch_routes": Dict[Node, TargetCodeGenerator],
        "copy_dispatchers": Dict[edge, TargetCodeGenerator],
    },

    # From ControlFlowAnalysisPass
    "control_flow": {
        "control_flow_tree": Dict[SDFG, ControlFlow],
        "structured_regions": Dict[SDFG, List[ControlFlowRegion]],
    },

    # From StreamAssignmentPass (GPU)
    "stream_info": {
        "stream_assignments": Dict[Node, int],
        "stream_dependencies": nx.DiGraph,
        "synchronization_points": List[Tuple[Node, Node]],
    },

    # From StateStructCreationPass
    "state_struct": {
        "struct_definition": str,
        "struct_members": List[str],
        "initialization_code": str,
    },

    # From AllocationCodePass
    "allocation_code": {
        "alloc_code_by_scope": Dict[scope, str],
        "dealloc_code_by_scope": Dict[scope, str],
        "persistent_allocations": List[str],
    },

    # From FrameCodeGenerationPass
    "frame_code": {
        "global_code": str,
        "local_code": str,
        "function_signatures": Dict[str, str],
        "header_includes": List[str],
    },

    # From TargetCodeGenerationPass
    "target_code": {
        "code_objects": List[CodeObject],
        "target_specific_code": Dict[str, str],
        "compilation_flags": Dict[str, List[str]],
    },
}
```

### Information Reuse Strategies

1. **Caching Analysis Results**: Store expensive analysis results (allocation lifetime, control flow) to avoid recomputation
2. **Incremental Updates**: Allow passes to update only changed portions of the SDFG
3. **Shared Data Structures**: Use common data structures across passes for consistency
4. **Lazy Evaluation**: Compute expensive analyses only when needed by downstream passes

## Organizational Structure

### Current Structure
```
dace/codegen/
├── __init__.py
├── codegen.py              # Main entry point
├── targets/                # Target-specific generators
├── instrumentation/        # Profiling support
├── tools/                  # Utilities
├── dispatcher.py           # Target routing
├── control_flow.py         # Control flow analysis
├── compiler.py             # CMake integration
├── codeobject.py          # Code file representation
├── compiled_sdfg.py       # Runtime interface
└── ...
```

### Proposed Structure
```
dace/codegen/
├── __init__.py
├── codegen.py              # Simplified entry point
├── compiler/               # Compilation and build system
│   ├── __init__.py
│   ├── cmake.py            # CMake integration (from compiler.py)
│   ├── direct.py           # Direct compiler calls
│   ├── build_system.py     # Abstract build system interface
│   └── environments.py     # Environment-specific build logic
├── passes/                 # Code generation passes
│   ├── __init__.py
│   ├── analysis/           # Analysis passes
│   │   ├── __init__.py
│   │   ├── type_inference.py
│   │   ├── metadata_collection.py
│   │   ├── allocation_analysis.py
│   │   ├── control_flow_analysis.py
│   │   └── target_analysis.py
│   ├── transformation/     # Transformation passes
│   │   ├── __init__.py
│   │   ├── copy_to_map.py
│   │   ├── stream_assignment.py
│   │   ├── tasklet_lowering.py
│   │   └── memlet_lowering.py
│   ├── codegen/           # Code generation passes
│   │   ├── __init__.py
│   │   ├── state_struct.py
│   │   ├── allocation_code.py
│   │   ├── frame_code.py
│   │   ├── target_code.py
│   │   └── header_generation.py
│   ├── file_generation/   # File organization passes
│   │   ├── __init__.py
│   │   ├── sdfg_splitting.py
│   │   └── code_objects.py
│   └── pipelines.py       # Pre-built pipelines
├── targets/               # Target-specific generators (simplified)
│   ├── __init__.py
│   ├── base.py            # Base target interface (from target.py)
│   ├── openmp.py          # OpenMP backend (split from cpu.py)
│   ├── cpp.py             # Pure C++ backend
│   ├── gpu.py             # GPU backend (generalized from cuda.py)
│   ├── cuda.py            # CUDA-specific GPU
│   ├── hip.py             # HIP-specific GPU
│   ├── fpga/              # FPGA backends
│   └── specialized/       # Other specialized targets
├── runtime/               # Runtime interface (from compiled_sdfg.py)
└── utils/                 # Utilities (dispatcher, codeobject, etc.)
    ├── __init__.py
    ├── dispatcher.py
    ├── codeobject.py
    ├── prettycode.py
    └── common.py
```

## Target Refactoring

### Current Issues
- **CPU backend** actually does OpenMP generation
- **CUDA backend** is GPU-specific, not general GPU
- Poor separation between generic and specific backends

### Proposed Refactoring

#### 1. **C++ Backend** (`targets/cpp.py`)
- Pure C++ code generation without parallelization
- Base for other C++ based backends
- Sequential execution model
- Basic memory management

#### 2. **OpenMP Backend** (`targets/openmp.py`)
- Extends C++ backend with OpenMP directives
- CPU parallelization via OpenMP
- Current "CPU" backend functionality
- Shared memory parallelism

#### 3. **GPU Backend** (`targets/gpu.py`)
- Generic GPU programming model
- Common GPU concepts (kernels, memory hierarchy)
- Base for CUDA/HIP/OpenCL backends
- Device memory management

#### 4. **CUDA Backend** (`targets/cuda.py`)
- Extends GPU backend with CUDA specifics
- CUDA runtime API calls
- CUDA-specific optimizations
- PTX inline assembly support

#### 5. **HIP Backend** (`targets/hip.py`)
- Extends GPU backend with HIP specifics
- AMD GPU support
- ROCm integration
- Portable GPU code

### Target Hierarchy
```
TargetCodeGenerator (base)
├── CppCodeGen (sequential C++)
│   ├── OpenMPCodeGen (CPU parallelism)
│   └── MPICodeGen (distributed)
├── GPUCodeGen (generic GPU)
│   ├── CUDACodeGen (NVIDIA)
│   ├── HIPCodeGen (AMD)
│   └── OpenCLCodeGen (portable)
├── FPGACodeGen (FPGA base)
│   ├── XilinxCodeGen
│   └── IntelFPGACodeGen
└── SpecializedCodeGen
    ├── SVECodeGen (ARM vectors)
    ├── MLIRCodeGen
    └── RTLCodeGen
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. Create new directory structure
2. Move existing files to appropriate locations
3. Create base pass classes for code generation
4. Implement basic pipeline infrastructure

### Phase 2: Analysis Passes (Weeks 3-4)
1. Extract `TypeInferencePass` from existing code
2. Implement `MetadataCollectionPass`
3. Refactor `AllocationAnalysisPass` from `determine_allocation_lifetime()`
4. Create `TargetAnalysisPass` from `_get_codegen_targets()`

### Phase 3: Transformation Passes (Weeks 5-6)
1. Extract `CopyToMapPass` from CUDA preprocessing
2. Implement `StreamAssignmentPass` for GPU targets
3. Create `TaskletLanguageLoweringPass`
4. Develop `MemletLoweringPass`

### Phase 4: Code Generation Passes (Weeks 7-8)
1. Implement `StateStructCreationPass`
2. Create `AllocationCodePass`
3. Develop `FrameCodeGenerationPass`
4. Implement `TargetCodeGenerationPass`

### Phase 5: Target Refactoring (Weeks 9-10)
1. Split CPU backend into C++ and OpenMP
2. Generalize CUDA backend to GPU + CUDA specifics
3. Create HIP backend
4. Reorganize FPGA backends

### Phase 6: Integration and Testing (Weeks 11-12)
1. Create complete pipeline implementations
2. Comprehensive testing with existing test suite
3. Performance benchmarking
4. Documentation updates

### Phase 7: Compilation System (Weeks 13-14)
1. Implement direct compiler calls as alternative to CMake
2. Create abstract build system interface
3. Support for multiple output languages
4. Environment-specific build configurations

## Benefits of Modular Design

1. **Extensibility**: Easy to add new passes or modify existing ones
2. **Testability**: Each pass can be tested independently
3. **Maintainability**: Clear separation of concerns
4. **Reusability**: Passes can be composed into different pipelines
5. **Performance**: Incremental compilation and information reuse
6. **Verification**: Easier to verify correctness of individual passes
7. **Debugging**: Better error localization and debugging support

## Backward Compatibility

The new modular system will maintain full backward compatibility by:
1. Keeping the existing `generate_code()` API unchanged
2. Running the full default pipeline when called directly
3. Providing legacy wrappers for deprecated functionality
4. Gradual migration path for custom backends

## Conclusion

This modular design transforms DaCe's code generation from a monolithic system into a flexible, extensible pipeline architecture. By leveraging DaCe's existing pass infrastructure and carefully decomposing the current system, we can achieve better maintainability, extensibility, and verifiability while preserving all existing functionality.

The proposed design provides a clear roadmap for implementation that can be executed incrementally, allowing for thorough testing and validation at each step.
