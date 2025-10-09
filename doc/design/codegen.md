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
   - Type inference for connectors and storage/schedule types (`infer_types.infer_connector_types()`, `infer_types.set_default_schedule_and_storage_types()`)
   - Library node expansion (`sdfg.expand_library_nodes()`)
   - A second round of type inference following library node expansion

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

## Goals

1. **Modularity and Verifiability**: Each pass is small enough to be tested separately
2. **Target Disentanglement**: Clear separation between targets (CPU vs. OpenMP, CPP, CUDA from CPU)
3. **Simple Support for Other Target Languages**: If, e.g., SYCL is required as a target, or LLVM IR instead of C++,
   it should be easy to perform.


## Proposed Pass Decomposition

### Phase 1: Scheduling and Analysis Passes

#### 1. **ValidationPass**
- **Purpose**: Run SDFG validation prior to code generation
- **Input**: Input SDFG
- **Output**: None
- **Current Location**: `validate.py`

#### 2. **TypeInferencePass**
- **Purpose**: Infer connector types and set default storage/schedule types
- **Input**: Input SDFG
- **Output**: SDFG with inferred types, pipeline_results["type_info"]
- **Current Location**: `infer_types.py` functions

#### 3. **LibraryExpansionPass**
- **Purpose**: Expand all library nodes that haven't been expanded
- **Input**: Type-inferred SDFG
- **Output**: SDFG with expanded library nodes
- **Current Location**: `sdfg.expand_library_nodes()`

#### 4. **TypeInferencePass**
- **Purpose**: After expanding library nodes, run a second type inference pass if the SDFG changed
- **Input**: Library-expanded SDFG
- **Output**: SDFG with inferred types, updated pipeline_results["type_info"]
- **Current Location**: `infer_types.py` functions

#### 5. **MetadataCollectionPass**
- **Purpose**: Collect free symbols, argument lists, constants, shared transients
- **Input**: Expanded SDFG
- **Output**: pipeline_results["metadata"] = {symbols, arglist, constants, shared_transients}
- **Current Location**: `DaCeCodeGenerator.__init__()`

#### 6. **ControlFlowRaising**
- **Purpose**: Extract structured control flow from state machines, if Control Flow Regions were not already given
- **Input**: SDFG
- **Output**: SDFG with Control Flow Regions
- **Current Location**: Already exists

#### 7. **AllocationAnalysisPass**
- **Purpose**: Determine allocation lifetimes and scopes for all data containers
- **Input**: SDFG with metadata
- **Output**: SDFG with allocation/deallocation points stored in node metadata
- **Note**: Node metadata is used for allocation info in order to enable user-defined allocation scopes that supersede
            these decisions.
- **Current Location**: `DaCeCodeGenerator.determine_allocation_lifetime()`

#### 8. **StreamAssignmentPass** (mostly GPU-specific)
- **Purpose**: Assign streams for concurrent execution. Currently used for CUDA/HIP streams but can apply to other architectures
- **Input**: SDFG
- **Output**: SDFG with stream assignments stored in node metadata
- **Note**: Stream assignments are stored directly in `Node.metadata` fields rather than replicated in pipeline_results dictionary for efficiency
- **Current Location**: Embedded in CUDA code generation

### Phase 2: Lowering Passes

#### Target-Specific Preprocessing Passes
- **Purpose**: Perform preprocessing modifications on the SDFG based on the code generators that will be used next
- **Examples**: `FPGAPreprocessingPass` for FPGAs, `StreamAssignmentPass` for GPUs, `CopyToMap` for heterogeneous targets in general (see below)

#### 9. **LowerAllocations**
- **Purpose**: Add allocation/deallocation annotations (e.g., as tasklets) to the SDFG for each scope
- **Input**: SDFG with allocation analysis
- **Output**: SDFG with allocation/deallocation tasklets inserted
- **Current Location**: `allocate_arrays_in_scope()`, `deallocate_arrays_in_scope()`
- **Note**: This modifies the SDFG structure rather than generating code

#### 10. **CopyToMap**
- **Purpose**: Convert nontrivial memory copies to Map nodes where needed
- **Input**: SDFG with targets identified
- **Output**: SDFG with transformed copies
- **Current Location**: `cuda.py` preprocessing, various target preprocessors
- **Applies To**: GPU strided copies, FPGA transfers

#### 11. **LowerTaskletLanguage**
- **Purpose**: Convert Python/generic tasklets to tasklets in the target language (C++/CUDA/etc.)
- **Input**: SDFG with tasklets
- **Output**: SDFG with lowered tasklets
- **Current Location**: Distributed across target generators

#### 12. **LowerMemlets**
- **Purpose**: Lower high-level memlets to explicit copy operations
- **Input**: SDFG with target analysis
- **Output**: SDFG with explicit copies annotated (e.g., as tasklets)
- **Current Location**: Embedded in target-specific copy generation

#### 13. **SplitSDFGToTargets**
- **Purpose**: The final lowering step splits the single SDFG into an SDFG per target file.
               This means that, for example, a GPU kernel map will be converted to an ExternalSDFG call
               to another SDFG file that contains the kernel.
- **Input**: SDFG
- **Output**: List of SDFGs (one per target file)
- **Current Location**: Implicit in current system
- **Note**: This step makes explicit the separation of generated code into translation units, either due to necessity
            (e.g., different output language) or for performance (e.g., parallel compilation, avoiding recompilation).
            Using multiple SDFGs and adding user-defined knobs to tune this process could increase code generation
            and compilation scalability.

### Phase 3: Code Generation Passes

#### 14. **GenerateStateStruct**
- **Purpose**: Generate state struct definitions for persistent data
- **Input**: SDFG with allocation info
- **Output**: pipeline_results["state_struct"] = {struct_def, struct_init}
- **Current Location**: `DaCeCodeGenerator.generate_code()`

#### 15. **GenerateTargetCode**
- **Purpose**: Generate both frame code and target-specific code for each SDFG file by traversing the graph and emitting
               code for each element.
- **Input**: Split SDFGs with all previous analyses
- **Output**: pipeline_results["code_objects"] = List[CodeObject] with complete code
- **Current Location**: Combined from `DaCeCodeGenerator.generate_code()` and target-specific `get_generated_codeobjects()`
- **Note**: This pass may call individual target code generators (CppCodeGen, GPUCodeGen, FPGACodeGen, etc.) to
            generate platform-specific code

#### 14. **GenerateHeaders**
- **Purpose**: Generate C/C++ header files for SDFG interface
- **Input**: CodeObjects with complete code
- **Output**: pipeline_results["headers"] = {call_header, sample_main}
- **Current Location**: `generate_headers()`, `generate_dummy()`
- **Note**: This will also generate the code sample that DaCe provides within a cache folder

## Information Flow Design

### Main Code Generation Pipeline

```python
class CodeGenerationPipeline(Pipeline):
    """Complete code generation pipeline for DaCe SDFGs."""

    def __init__(self):
        super().__init__([
            # Phase 1: Scheduling
            ValidationPass(),
            TypeInferencePass(),
            LibraryExpansionPass(),
            TypeInferencePass(),
            MetadataCollectionPass(),
            ControlFlowRaising(),
            AllocationAnalysisPass(),
            StreamAssignmentPass(),

            # Phase 2: Lowering
            LowerAllocations(),
            ConditionalPipeline([
                (lambda r: 'cuda' in r.get('targets', []), CopyToMapPass()),
                (lambda r: 'fpga' in r.get('targets', []), FPGAPreprocessingPass()),
            ]),
            LowerTaskletLanguage(),
            LowerMemlets(),
            SplitSDFGToTargets(),

            # Phase 3: Code Generation
            GenerateStateStruct(),
            GenerateTargetCode(),
            GenerateHeaders(),
        ])
```

## Information Flow and Reuse

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

This is only a proposed structure and might change as implementation commences.

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
│   │   └── allocation_analysis.py
│   ├── transformation/     # Transformation passes
│   │   ├── __init__.py
│   │   ├── copy_to_map.py
│   │   ├── stream_assignment.py
│   │   ├── tasklet_lowering.py
│   │   ├── library_node_expansion.py
│   │   └── memlet_lowering.py
│   ├── generation/           # Code generation passes
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
│   ├── cpp.py             # Pure C++ backend (split from cpu.py and cpp.py)
│   ├── gpu.py             # GPU backend (generalized from cuda.py)
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
- Unified GPU programming model supporting both CUDA and HIP
- Common GPU concepts (kernels, memory hierarchy, device management)
- Runtime detection of CUDA/HIP capabilities
- Covers the superset of both NVIDIA and AMD GPU features
- Device memory management and kernel launch logic

### Target Hierarchy
```
TargetCodeGenerator (base)
├── CppCodeGen (sequential C++)
│   ├── OpenMPCodeGen (CPU parallelism)
│   ├── SVECodeGen (ARM vectors)
│   └── MPICodeGen (distributed)
├── GPUCodeGen (unified GPU backend)
│   ├── CUDACodeGen (NVIDIA specifics)
│   └── HIPCodeGen (AMD specifics)
├── FPGACodeGen (FPGA base)
│   ├── XilinxCodeGen
│   ├── IntelFPGACodeGen
|   └── RTLCodeGen
└── MLIRCodeGen
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
2. Refactor CUDA backend to GPU
3. Reorganize FPGA backends

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
