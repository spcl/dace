# DaCe Automatic Differentiation (Autodiff) Design Document

## Overview

The DaCe Automatic Differentiation system implements reverse-mode automatic differentiation (backpropagation) for SDFGs (Stateful DataFlow Graphs). It transforms forward computation graphs into backward passes that compute gradients efficiently. At the moment the AD module is tested on CPU with full GPU support to be added soon.

## Core Automatic Differentiation Algorithm

This section describes the automatic differentiation module in DaCe, which can be decomposed into the following main steps:

### 1. Pre-AD Transformations (if necessary)
This pass will call some SDFG transformations to prepare the input SDFG for AD. At the moment, the current passes included in this step are:
- Expanding library nodes with pure implementations from our repository
- Inline conditional regions into a state machine 

### 2. Critical Computation Subgraph (CCS) Extraction
The critical computation subgraph is the subset of the input SDFG that will require automatic differentiation. To extract this subgraph, we explore the input SDFG in reverse topological order, starting from the target output to differentiate up until the inputs to differentiate with respect to.


### 3. Reverse Each Component (State, Loop Region, CFR) Separately
Here, we define the reversal of each element in the SDFG from tasklets to states and LoopRegions.

### 4. Connect the Reversed Components
This consists of adding interstate edges between the reversed components according to how they were connected in the forward SDFG:
- Add the edge
- Add the necessary conditions on the interstate edges
- Store data-dependent control flow if necessary
- Add interstate edge assignments for the stored values if necessary


### 5. Forward Data from the Forward Pass to the Backward Pass
Data used in non-linear operations will need to be forwarded to the backward pass. This can be done through different strategies:
- Decide on the forwarding strategy (store-all, recompute-all, ILP-based, user-provided)
- Apply the chosen strategy to the forward and backward SDFGs


### 6. Post-AD Transformations (if necessary)
Post-AD cleanup transformations to simplify the SDFGs. For example, if we want to only compute the gradients correctly, we can apply DeadDataflowElimination to improve performance by removing some unnecessary parts of the forward pass.


## Implementation Architecture

### 1. Main Components

#### 1.1 Core Classes (`base_abc.py`)
- **`BackwardImplementation`**: Abstract base class for all backward pass implementations
  - Registry-based system using `@autoregister_params` decorator
  - Supports registration by node type or ONNX operation name
  - Provides `backward_can_be_applied()` and `backward()` methods

- **`BackwardContext`**: Context object passed to backward implementations
  ```python
  @dataclass
  class BackwardContext:
      forward_sdfg: SDFG
      forward_state: SDFGState  
      backward_sdfg: SDFG
      backward_state: SDFGState
      backward_generator: BackwardPassGenerator
  ```

- **`BackwardResult`**: Return type describing gradient computations
  - Maps input/output connectors to gradient names
  - Tracks which gradients need zero initialization

#### 1.2 Backward Pass Generator (`backward_pass_generator.py`)
The `BackwardPassGenerator` class is the core engine that orchestrates the entire backward pass construction:

**Key Responsibilities:**
- Traverses forward SDFG in reverse topological order
- Manages gradient flow and accumulation
- Handles memory management strategies (store_all, recompute)
- Coordinates with backward implementations
- Manages gradient initialization and cleanup

**Memory Management Strategies:**
- `store_all`: Store all intermediate values (memory-intensive but fast)
- `recompute`: Recompute values as needed (memory-efficient but slower)
- Mixed strategies with selective recomputation

#### 1.3 Analysis Module (`analysis.py`)
Provides dependency analysis utilities:
- **`dependency_analysis()`**: Analyzes read dependencies between arrays
- **`inverse_reachability()`**: Computes state reachability for gradient flow
- **`is_previously_written()`**: Determines if arrays need initialization

### 2. Implementation Registry

#### 2.1 DaCe Node Implementations (`implementations/dace_nodes.py`)
Implementations for core DaCe operations:

**ReverseReduce**: Handles reduction operations (Sum, Max, Min)
- Supports gradient scattering for Sum reductions
- Handles gradient propagation for Min/Max with appropriate masking
- Manages axes preservation during gradient computation

#### 2.2 ONNX Operation Implementations (`implementations/onnx_ops.py`) 
Backward implementations for ONNX operations:

**Key Patterns:**
- **Einsum operations**: Uses Einstein notation reversal
- **Element-wise operations**: Direct gradient propagation
- **Matrix operations**: Implements mathematical gradient rules
- **Activation functions**: Chain rule application

**Registration Pattern:**
```python
@autoregister_params(op="OperationName", name="implementation_name")
class ReverseOperation(BackwardImplementation):
    @staticmethod
    def backward(forward_node, context, given_gradients, required_gradients):
        # Implementation logic
        return backward_node, BackwardResult(...)
```

### 3. Integration Points

#### 3.1 PyTorch Integration (`torch.py`)
- **`make_backward_function()`**: Converts ONNX models to PyTorch-compatible backward functions
- Handles PyTorch tensor interoperability
- Manages gradient computation for neural network workflows

#### 3.2 Library Integration (`library/`)
- **`library.py`**: Core library functionality for autodiff
- **`python_frontend.py`**: Frontend integration with Python
- **`torch_integration.py`**: Deep PyTorch integration

### 4. Optimization Components

#### 4.1 Backward Pass Optimization (`optimize_backward_pass_generator.py`)
Advanced optimization strategies:

**Key Features:**
- **Auto-optimization**: Automatic selection of memory/computation trade-offs
- **SDFG preprocessing**: Optimizes forward graphs for efficient differentiation  
- **State fusion**: Combines related computation states
- **Memory planning**: Optimizes gradient storage and reuse

**Performance Analysis Integration:**
- Operational intensity analysis
- Work-depth analysis for parallel efficiency
- Memory footprint optimization using linear programming

#### 4.2 Optimization Functions
- **`autooptimize_sdfgs_for_ad()`**: Main auto-optimization entry point
- **`preprocess_fwd_sdfg()`**: Forward graph preprocessing
- **`fuse_states_cav()`**: State fusion optimization

### 5. Utility Functions (`utils.py`)

**Core Utilities:**
- **Descriptor Management**: 
  - `forward_in_desc_with_name()`: Get input data descriptors
  - `forward_out_desc_with_name()`: Get output data descriptors
  - `add_backward_desc_for_connector()`: Create gradient data descriptors

- **Type Management**:
  - `cast_consts_to_type()`: Type casting for gradient computations
  - Gradient shape and stride handling

### 6. Entry Points

#### 6.1 Main API (`autodiff.py`)
**`add_backward_pass()`**: Primary user-facing function
```python
def add_backward_pass(sdfg: SDFG,
                      outputs: List[Union[nodes.AccessNode, str]],
                      inputs: List[Union[nodes.AccessNode, str]], 
                      overwrite_strategy: str = "store_all",
                      data_to_recompute: List[str] = None,
                      autooptimize: bool = False,
                      separate_sdfgs: bool = False):
```

**Process Flow:**
1. SDFG validation and preprocessing
2. Control flow region inlining 
3. Backward pass generation via `BackwardPassGenerator`
4. Optional auto-optimization
5. Return backward SDFG (separate or integrated)

## Design Patterns

### 1. Registry Pattern
The system uses a registry pattern for backward implementations:
- Automatic registration via decorators
- Runtime lookup by node type or operation name
- Extensible for custom operations

### 2. Context Pattern
Backward implementations receive rich context:
- Access to both forward and backward SDFGs
- State information for both passes
- Generator reference for coordination

### 3. Strategy Pattern
Memory management uses configurable strategies:
- Store-all for performance
- Recompute for memory efficiency  
- Mixed strategies for balanced trade-offs

### 4. Builder Pattern
The `BackwardPassGenerator` builds backward graphs incrementally:
- Node-by-node processing
- Gradient flow management
- Dependency resolution

## Key Algorithms

### 1. Reverse-Mode AD Algorithm
1. **Forward Analysis**: Analyze dependencies and memory requirements
2. **Topological Traversal**: Process nodes in reverse topological order
3. **Gradient Accumulation**: Accumulate gradients at fan-in points
4. **Memory Management**: Apply chosen storage/recomputation strategy

### 2. Gradient Flow Resolution
1. **Output Gradients**: Start from provided output gradients (loss derivatives)
2. **Chain Rule Application**: Apply chain rule through backward implementations
3. **Gradient Routing**: Route gradients through SDFG structure
4. **Input Gradients**: Collect required input gradients

### 3. Memory Optimization
1. **Liveness Analysis**: Determine when intermediate values are needed
2. **Recomputation Planning**: Decide what to store vs. recompute
3. **Linear Programming**: Optimize memory/computation trade-offs
4. **State Fusion**: Merge compatible computation states

## Current Limitations

### 1. Supported Operations
- Limited to specific node types and ONNX operations
- No support for control flow differentiation
- Inplace operations not supported

### 2. Memory Management
- Store-all strategy can be memory-intensive
- Recomputation strategies need refinement
- Limited automatic optimization heuristics

### 3. Hardware Support
- Primary focus on CPU/GPU targets
- FPGA support needs validation
- Memory coalescing optimizations needed

## Extension Points

### 1. Adding New Operations
1. Implement `BackwardImplementation` subclass
2. Define mathematical gradient computation
3. Register with appropriate decorators
4. Add unit tests

### 2. Custom Memory Strategies
1. Extend memory management in `BackwardPassGenerator`
2. Implement custom recomputation logic
3. Add performance analysis integration

### 3. Hardware-Specific Optimizations
1. Add target-specific backward implementations
2. Implement hardware-aware memory management
3. Add performance tuning for specific architectures

## Future Enhancements

### 1. Control Flow Differentiation
- Support for loops and conditionals in backward pass
- Dynamic graph unrolling strategies
- Gradient checkpointing for long sequences

### 2. Advanced Optimizations
- Automatic mixed-precision training
- Gradient compression techniques
- Cross-node optimization passes

### 3. Debugging and Profiling
- Gradient flow visualization
- Performance profiling tools
- Numerical gradient verification

## Testing Strategy

### 1. Unit Tests
- Individual backward implementation testing
- Gradient correctness verification
- Performance regression testing

### 2. Integration Tests
- End-to-end model differentiation
- Hardware target validation
- Memory management verification

### 3. Performance Tests
- Benchmarking against reference implementations
- Memory usage profiling
- Scalability testing

---

## Proposed Modular Architecture

### Current Architecture Issues Analysis

The existing DaCe autodiff system suffers from several architectural problems that limit its maintainability, extensibility, and testability:

#### 1. Monolithic Design
- **`backward_pass_generator.py`**: Contains 5,553 lines - far exceeding maintainable file size limits
- **Single Responsibility Violation**: Core algorithms, memory management, optimization, and utilities are intermingled
- **Cognitive Overload**: Developers must understand entire system to make small changes

#### 2. Poor Separation of Concerns
- Memory management logic scattered throughout core generator
- Analysis functions mixed with implementation details
- Integration code (PyTorch, library) embedded in core modules
- No clear boundaries between different functional areas

#### 3. Limited Extensibility
- Adding new memory strategies requires modifying core generator
- New operation implementations must navigate complex registration system
- Optimization passes tightly coupled to main generation logic
- Framework integrations hard-coded into existing modules

#### 4. Testing and Maintenance Challenges
- Monolithic files difficult to unit test in isolation
- Complex interdependencies make refactoring risky
- No clear module boundaries for parallel development
- Hard to identify performance bottlenecks

#### 5. Current File Size Distribution
```
backward_pass_generator.py:           5,553 lines (too large)
implementations/onnx_ops.py:          2,138 lines (too large)
optimize_backward_pass_generator.py:  1,252 lines (large)
implementations/dace_nodes.py:          534 lines (acceptable)
utils.py:                               323 lines (acceptable)
```

### Proposed Modular Structure

#### Directory Organization
```
dace/autodiff/
├── __init__.py                              # Clean public API exports
├── api/                                     # High-level user-facing APIs
│   ├── __init__.py
│   ├── main.py                             # Core add_backward_pass API
│   └── torch_api.py                        # PyTorch-specific API functions
├── core/                                    # Core autodiff algorithms and logic
│   ├── __init__.py
│   ├── abc.py                              # Abstract base classes (from base_abc.py)
│   ├── context.py                          # BackwardContext management
│   ├── generator.py                        # Main BackwardPassGenerator (refactored)
│   ├── traversal.py                        # Graph traversal algorithms
│   └── gradient_flow.py                    # Gradient flow and accumulation logic
├── memory/                                  # Memory management strategies
│   ├── __init__.py
│   ├── strategies.py                       # Memory strategy interface and base classes
│   ├── store_all.py                        # Store-all strategy implementation
│   ├── recompute.py                        # Recomputation strategy implementation
│   └── optimizer.py                        # ILP-based memory optimization
├── analysis/                                # Analysis and dependency tracking
│   ├── __init__.py
│   ├── dependencies.py                     # Dependency analysis (from analysis.py)
│   ├── liveness.py                         # Liveness analysis for memory optimization
│   └── reachability.py                     # State reachability analysis
├── implementations/                         # Backward pass implementations
│   ├── __init__.py
│   ├── registry.py                         # Implementation registry management
│   ├── base.py                             # Base implementation utilities
│   ├── nodes/                              # DaCe node implementations
│   │   ├── __init__.py
│   │   ├── reduction.py                    # Reduction operations (from dace_nodes.py)
│   │   ├── map.py                          # Map and nested SDFG operations
│   │   ├── tasklet.py                      # Tasklet differentiation
│   │   └── access.py                       # AccessNode handling
│   └── onnx/                               # ONNX operation implementations
│       ├── __init__.py
│       ├── math_ops.py                     # Basic math operations (Add, Mul, etc.)
│       ├── linalg_ops.py                   # Linear algebra (MatMul, Einsum, etc.)
│       ├── activation_ops.py               # Activation functions (ReLU, Sigmoid, etc.)
│       ├── conv_ops.py                     # Convolution and pooling operations
│       └── shape_ops.py                    # Reshape, Transpose, etc.
├── optimization/                            # Optimization and performance enhancements
│   ├── __init__.py
│   ├── auto_optimizer.py                   # Auto-optimization logic
│   ├── preprocessing.py                    # Forward SDFG preprocessing
│   ├── state_fusion.py                     # State fusion optimizations
│   └── performance_analysis.py             # Performance analysis integration
├── integration/                             # External framework integration
│   ├── __init__.py
│   ├── pytorch/                            # PyTorch integration
│   │   ├── __init__.py
│   │   ├── converter.py                    # ONNX model conversion
│   │   ├── function.py                     # Backward function creation
│   │   └── tensor_utils.py                 # Tensor handling utilities
│   └── library/                            # DaCe library integration
│       ├── __init__.py
│       ├── nodes.py                        # Library node integration (from library/)
│       └── frontend.py                     # Frontend integration
├── utils/                                   # Utility functions and helpers
│   ├── __init__.py
│   ├── descriptors.py                      # Data descriptor utilities (from utils.py)
│   ├── connectors.py                       # Connector management utilities
│   ├── type_casting.py                     # Type casting and conversion
│   └── validation.py                       # SDFG validation helpers
└── exceptions.py                            # Custom exception classes
```

### Architectural Design Patterns

#### 1. Strategy Pattern for Memory Management
```python
class MemoryStrategy(ABC):
    @abstractmethod
    def should_store(self, array_name: str, state: SDFGState) -> bool:
        pass
    
    @abstractmethod 
    def should_recompute(self, array_name: str, state: SDFGState) -> bool:
        pass

class StoreAllStrategy(MemoryStrategy):
    # Implementation for storing all intermediate values
    
class RecomputeStrategy(MemoryStrategy):  
    # Implementation for recomputing values as needed
    
class OptimalStrategy(MemoryStrategy):
    # ILP-based optimal strategy
```

#### 2. Registry Pattern for Implementations
```python
class BackwardImplementationRegistry:
    @staticmethod
    def register(node_type=None, op=None, name=None):
        # Enhanced registration with better organization
        
    @staticmethod  
    def get_implementation(node, state, sdfg):
        # Improved lookup with fallback mechanisms
```

#### 3. Factory Pattern for Context Creation
```python
class BackwardContextFactory:
    @staticmethod
    def create_context(forward_sdfg, forward_state, backward_sdfg, backward_state, generator):
        # Centralized context creation with validation
```

#### 4. Observer Pattern for Performance Monitoring
```python
class PerformanceObserver:
    def on_node_processed(self, node, timing_info):
        # Performance monitoring hooks
    
    def on_memory_decision(self, array_name, decision, rationale):
        # Memory decision tracking
```

### Key Architectural Improvements

#### 1. Modular Core Generator (core/generator.py)
Split the monolithic `BackwardPassGenerator` into focused components:
- **Main orchestration**: High-level backward pass coordination
- **Node processing**: Delegate to specialized processors
- **Memory coordination**: Delegate to memory strategies  
- **Gradient management**: Delegate to gradient flow manager

Target: Reduce from 5,553 lines to ~500 lines through delegation

#### 2. Pluggable Memory Strategies (memory/)
- **Interface-based design**: All strategies implement common interface
- **Configurable selection**: Choose strategy based on requirements
- **Performance optimization**: ILP-based optimal strategy for advanced users
- **Easy extension**: Add new strategies without core changes

#### 3. Organized Implementation Registry (implementations/)
- **Type-based organization**: Group related implementations together
- **Hierarchical structure**: ONNX ops organized by functionality
- **Simplified registration**: Cleaner decorator-based system
- **Better discoverability**: Clear organization makes finding implementations easier

#### 4. Comprehensive Analysis Suite (analysis/)
- **Dependency tracking**: Enhanced dependency analysis
- **Liveness analysis**: Support memory optimization decisions
- **Reachability analysis**: Support control flow handling
- **Performance analysis**: Integration with optimization components

#### 5. Clean Integration Layer (integration/)
- **Framework isolation**: Framework-specific code contained
- **Extensible design**: Easy to add new framework integrations
- **Standard interfaces**: Common patterns across integrations
- **Minimal coupling**: Integration code doesn't affect core algorithms

### Migration Strategy

#### Phase 1: Foundation (Week 1-2)
1. Create new directory structure
2. Move and refactor abstract base classes to `core/abc.py`
3. Extract utility functions to dedicated modules
4. Set up new public API structure

#### Phase 2: Core Refactoring (Week 3-4)  
1. Split `backward_pass_generator.py` into focused modules
2. Extract memory management to `memory/` package
3. Move analysis functions to `analysis/` package
4. Update internal imports and dependencies

#### Phase 3: Implementation Organization (Week 5-6)
1. Split `onnx_ops.py` by operation type
2. Reorganize `dace_nodes.py` into focused modules  
3. Update registration system
4. Add implementation discovery utilities

#### Phase 4: Integration and Testing (Week 7-8)
1. Move integration code to `integration/` package
2. Add comprehensive unit tests for all modules
3. Performance regression testing
4. Documentation updates

### Benefits of Modular Architecture

#### 1. Maintainability
- **Focused modules**: Each file has single, clear responsibility
- **Reduced complexity**: Easier to understand and modify individual components
- **Better organization**: Logical grouping makes navigation intuitive
- **Clearer dependencies**: Module boundaries make dependencies explicit

#### 2. Extensibility  
- **Plugin architecture**: New strategies and implementations easy to add
- **Framework integration**: New frameworks can be added without core changes
- **Custom optimizations**: Optimization passes can be composed modularly
- **Hardware targets**: Target-specific optimizations can be added cleanly

#### 3. Testability
- **Unit testing**: Individual modules can be tested in isolation
- **Mock dependencies**: Clean interfaces enable effective mocking
- **Performance testing**: Individual components can be profiled separately
- **Regression testing**: Focused tests prevent breaking changes

#### 4. Performance
- **Optimization opportunities**: Modular design enables targeted optimizations
- **Memory efficiency**: Better memory strategy selection and tuning
- **Parallel development**: Multiple developers can work on different modules
- **Profiling**: Easier to identify and fix performance bottlenecks

#### 5. Code Quality
- **SOLID principles**: Single responsibility, open/closed, dependency inversion
- **Design patterns**: Consistent use of proven architectural patterns
- **Documentation**: Easier to document focused, cohesive modules
- **Code review**: Smaller, focused changes easier to review

### Target Metrics

#### File Size Distribution (Post-Refactoring)
```
Core modules:           < 500 lines each
Memory strategies:      < 300 lines each  
Implementation files:   < 300 lines each
Analysis modules:       < 400 lines each
Integration modules:    < 200 lines each
Utility modules:        < 200 lines each
```

#### Quality Metrics
- **Cyclomatic complexity**: < 10 per function
- **Test coverage**: > 85% for all modules
- **Documentation coverage**: 100% for public APIs
- **Performance regression**: < 5% overhead from modularization

### Backward Compatibility

#### Public API Preservation
```python
# Existing public API remains unchanged
from dace.autodiff import add_backward_pass, make_backward_function
from dace.autodiff import BackwardImplementation, BackwardContext, BackwardResult
```

#### Internal API Migration
- **Deprecation warnings**: For internal APIs that will change
- **Migration guide**: Documentation for updating custom implementations
- **Compatibility layer**: Temporary forwarding for deprecated imports
- **Version planning**: Clear timeline for breaking changes

This modular architecture transforms the DaCe autodiff system from a monolithic design into a well-organized, extensible framework that follows software engineering best practices while preserving all existing functionality and maintaining backward compatibility.

---

This design document provides a comprehensive overview of the DaCe automatic differentiation system architecture and can serve as a foundation for further development and optimization work.