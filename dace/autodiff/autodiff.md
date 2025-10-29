Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
# DaCe Automatic Differentiation (AD) System - Design Document

## Table of Contents

1. [Introduction](#1-introduction)
2. [Directory Structure](#2-directory-structure)
3. [Core Components](#3-core-components)
4. [Data Forwarding System](#4-data-forwarding-system)
5. [Backward Implementations](#5-backward-implementations)
6. [Library Integration](#6-library-integration)
7. [PyTorch Integration](#7-pytorch-integration)
8. [Gradient Accumulation and Clearing](#8-gradient-accumulation-and-clearing)

---

## 1. Introduction

### 1.1 Purpose

The DaCe Automatic Differentiation (AD) module provides **reverse-mode automatic differentiation** for Stateful DataFlow Graphs (SDFGs). It enables automatic computation of gradients for optimized numerical kernels, making it possible to differentiate DaCe programs for machine learning, optimization, and scientific computing applications.

### 1.2 Reverse-Mode AD Fundamentals

Reverse-mode automatic differentiation (also known as backpropagation) computes gradients by:

1. **Forward Pass**: Execute the original computation and record intermediate values
2. **Backward Pass**: Traverse the computation graph in reverse, accumulating gradients using the chain rule

For a function `f: ℝⁿ → ℝᵐ`, reverse-mode AD efficiently computes the gradient when `m << n` (common in ML where loss is scalar).

**Example**: For a composite function `y = f(g(h(x)))`:
- **Forward pass**: Compute and store intermediate values:
  - `a = h(x)`
  - `b = g(a)`
  - `y = f(b)`
- **Backward pass**: Apply chain rule in reverse order. Given seed gradient `∂L/∂y`:
  - Compute `∂L/∂b = ∂L/∂y · (∂f/∂b)` where `(∂f/∂b)` is evaluated at the stored value of `b`
  - Compute `∂L/∂a = ∂L/∂b · (∂g/∂a)` where `(∂g/∂a)` is evaluated at the stored value of `a`
  - Compute `∂L/∂x = ∂L/∂a · (∂h/∂x)` where `(∂h/∂x)` is evaluated at the input value `x`

### 1.3 Key Features

- **Control Flow Support**: Handles loops (`LoopRegion`) and conditionals
- **Data Forwarding Strategies**: Flexible tradeoff between memory (store intermediates) and computation (recompute on demand)
- **Extensible Backward Implementations**: Registry-based system for adding backward rules for new operations
- **ONNX Integration**: Differentiate ONNX neural network models imported into DaCe
- **PyTorch Compatibility**: Integration with PyTorch's autograd system via `torch.autograd.Function`
- **Library Node Support**: Backward implementations for DaCe standard library (BLAS, reductions, etc.)
- **Nested SDFG Differentiation**: Recursive backward pass generation for nested SDFGs

### 1.4 Use Cases

1. **Machine Learning Training**: Compute gradients for neural network parameters
2. **Sensitivity Analysis**: Determine how outputs change with respect to inputs
3. **Optimization**: Gradient-based optimization of physical simulations
4. **Inverse Problems**: Solve inverse problems by differentiating forward models
5. **Scientific Computing**: Adjoint methods for PDEs and large-scale simulations

### 1.5 Component Interaction Flow

```
Input: Forward SDFG + Output Arrays + Input Arrays
       ▼
1. add_backward_pass() - Entry point
   • Validate SDFG
   • Simplify (optional)
   • Inline control flow (conditionals)
       ▼
2. BackwardPassGenerator.__init__()
   • Convert AccessNodes/strings to internal format
   • Initialize mappings (reverse_map, array_grad_map, etc.)
   • Set data forwarding strategy
       ▼
3. BackwardPassGenerator.backward()
   • Reverse states in topological order
   • For each state:
     a. Reverse nodes (AccessNode, Tasklet, Map, etc.)
     b. Find backward implementation via registry
     c. Call implementation.backward()
     d. Connect gradients with WCR
       ▼
4. DataForwardingManager.forward_data_to_backward_pass()
   • Identify intermediates needed in backward pass
   • Check if overwritten
   • Apply strategy (store or recompute)
       ▼
5. Simplify and validate (optional)
       ▼
Output: Backward SDFG with gradients computed
```

---

## 2. Directory Structure

### 2.1 File Organization

```
dace/autodiff/
├── __init__.py                      # Main API exports
│   └── Exports: add_backward_pass, BackwardPassGenerator,
│                BackwardImplementation, AutoDiffException, etc.
│
├── autodiff.py                      # Entry point
│   └── add_backward_pass() - High-level API
│
├── base_abc.py                      # Abstract base classes
│   ├── BackwardImplementation (ABC)
│   ├── BackwardContext (dataclass)
│   ├── BackwardResult (dataclass)
│   ├── AutoDiffException
│   └── find_backward_implementation()
|
├── backward_pass_generator.py      # Core AD engine
│   └── BackwardPassGenerator class - Main differentiation algorithm
│
├── analysis.py                      # SDFG analysis
│   ├── dependency_analysis()
│   ├── inverse_reachability()
│   └── is_previously_written()
│
├── utils.py                         # Utility functions
│   ├── Descriptor management
│   ├── Symbolic differentiation
│   ├── Graph traversal
│   └── Loop analysis
│
├── torch.py                         # PyTorch integration
│   └── make_backward_function() - Convert ONNX to PyTorch differentiable
│
├── data_forwarding/                 # Store or recompute strategies
│   ├── __init__.py                  # Package exports
│   ├── manager.py                   # Strategy coordinator
│   │   └── DataForwardingManager
│   ├── store.py                     # Store strategy
│   │   └── resolve_overwrite_with_store()
│   └── recompute.py                 # Recompute strategy
│       └── resolve_overwrite_with_recomputation()
│           get_recomputation_nsdfg()
│
├── implementations/                 # Backward rules for node types
│   ├── __init__.py                  # Package exports
│   ├── dace_nodes.py                # Pure SDFG elements
│   │   └── DaceNodeBackwardImplementations
│   │       ├── _reverse_AccessNode()
│   │       ├── _reverse_Tasklet()
│   │       ├── _reverse_MapEntry()
│   │       ├── _reverse_MapExit()
│   │       └── _reverse_NestedSDFG()
│   ├── dace_library_nodes.py        # Library operations
│   │   ├── ReverseReduce
│   │   ├── ReverseTranspose
│   │   └── ... (other library nodes)
│   └── onnx_ops.py                  # ONNX operations
│       ├── ONNXConvBackward
│       ├── ONNXMatMulBackward
│       └── ... (50+ ONNX ops)
│
└── library/                         # Library integrations
    ├── __init__.py                  # Package exports
    ├── library.py                   # BackwardPass node
    │   ├── ParameterArray (data descriptor)
    │   ├── BackwardPass (LibraryNode)
    │   └── ExpandBackwardPass (expansion)
    ├── torch_integration.py         # PyTorch hooks
    └── python_frontend.py           # Frontend hooks
```


## 3. Core Components

### 3.1 Entry Point: `autodiff.py`

**Location**: [autodiff.py](autodiff.py)

The main entry point for users to add backward passes to SDFGs.


#### 3.1.1 Workflow

```
┌─────────────────────┐
│ 1. Validate SDFG    │
└──────────┬──────────┘
           ▼
    ┌───────────────┐
    │ 2. Simplify   │
    └──────┬────────┘
           ▼
┌─────────────────────────────────┐
│ 3. Inline Control Flow          │
│    (conditionals, not loops)    │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────────┐
│ 4. Create Backward SDFG             │
│    (if separate_sdfgs flag is True) │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────┐
│ 5. Initialize BackwardPass-     │
│    Generator                    │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│ 6. generator.backward()         │
│    (main differentiation)       │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────┐
│ 7. Validate SDFG    │
└──────────┬──────────┘
           ▼
    ┌───────────────┐
    │ 8. Simplify   │
    └──────┬────────┘
           ▼
┌─────────────────────┐
│ 9. Return SDFG      │
└─────────────────────┘
```

#### 3.1.2 Key Constraints

- **Supported Nodes**:
  - Maps, AccessNodes, Tasklets, LoopRegions, ControlFlowRegions (inlined into state machine)
  - Reductions (Sum, Min, Max)
  - ONNXOps (with registered backward implementations)
  - NestedSDFGs

---

### 3.2 BackwardPassGenerator: The Core AD Engine

**Location**: [backward_pass_generator.py](backward_pass_generator.py)

The `BackwardPassGenerator` class is the core of the AD system. It orchestrates the entire backward pass generation process.

#### 3.2.1 Key Data Structures

The generator maintains several mappings and data structures:

- **Configuration**:
  - `sdfg`: Forward SDFG
  - `backward_sdfg`: Backward SDFG (can be same or separate)
  - `given_gradients_data`: Output arrays (seed gradients provided)
  - `required_gradients_data`: Input arrays (gradients to compute)
  - `data_forwarding_strategy`: "store_all", "recompute_all", "user_defined"

- **Generated Mappings**:
  - `reverse_map: Dict[Node, Node]`: Forward node → backward node
  - `reversed_states_map: Dict[SDFGState, SDFGState]`: Forward state → backward state
  - `array_grad_map: Dict[str, str]`: Array name → gradient array name
  - `result_map: Dict[Node, BackwardResult]`: Forward node → BackwardResult

- **Analysis Results**:
  - `read_only_arrays`: Arrays never written to
  - `backward_grad_arrays`: Gradient array descriptors
  - `backward_input_arrays`: Forward values needed in backward pass
  - `data_to_forward`: List of data to forward from forward to backward

#### 3.2.2 Main Algorithm: `backward()`

**Steps**:

1. **Initialize gradient arrays** for all required outputs
2. **Compute state order** (topological sort of SDFG states)
3. **Extract the Critical Computation Subgraph (CCS) of each state**
4. **Reverse the CCS of states** in reverse topological order:
   - Create backward state
   - Reverse nodes within CCS of the state
   - Connect gradients between reversed nodes
5. **Reverse loop regions** by generating loop regions in the backward pass
6. **Handle data forwarding** (store or recompute intermediates)
7. **Create interstate edges** to reverse control flow and connect all reversed components
8. **Return** backward result with gradient mappings

#### 3.2.3 State Reversal

For each forward state, the generator:

1. Creates a corresponding backward state
2. For each node in the CCS of the state:
   - Finds appropriate backward implementation from registry
   - Determines given/required gradients
   - Calls `implementation.backward()`
   - Stores mapping and result
3. Connects gradients between reversed nodes

---

### 3.3 Abstract Base Classes: `base_abc.py`

**Location**: [base_abc.py](base_abc.py)

#### 3.3.1 BackwardImplementation (ABC)

The abstract base class for all backward implementations.

```python
@dace.registry.make_registry
class BackwardImplementation(abc.ABC):

    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: SDFGState,
                                sdfg: SDFG) -> bool:
        """Check if this implementation can be applied to the node."""
        return True

    @staticmethod
    @abc.abstractmethod
    def backward(
        forward_node: nd.Node,
        context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:
        """Generate the backward pass for this node."""
        ...
```

**Registration Example**:

```python

# For ONNX operations
@dace.registry.autoregister_params(op="MatMul", name="pure")
class MatMulBackward(BackwardImplementation):
    ...
```

#### 3.3.2 BackwardContext (Dataclass)

Contains all context information needed by backward implementations:

```python
@dataclasses.dataclass
class BackwardContext:
    forward_sdfg: SDFG              # The forward SDFG
    backward_sdfg: SDFG             # The backward SDFG
    backward_generator: BackwardPassGenerator  # The generator (for utilities)
```

#### 3.3.3 BackwardResult (Dataclass)

Returns information about the generated backward node:

```python
@dataclasses.dataclass
class BackwardResult:
    """Result of differentiating a node."""

    # Mapping from forward input connector → gradient connector name
    required_grad_names: Dict[Optional[str], Optional[str]]

    # Mapping from forward output connector → gradient connector name
    given_grad_names: Dict[Optional[str], Optional[str]]

    # Which gradients should be zero-initialized
    zero_init: Dict[Optional[str], Optional[bool]]
```

#### 3.3.4 find_backward_implementation()

Looks up the registered backward implementation for a node by:

1. Querying `BackwardImplementation.extensions()` registry
2. Filtering by `node_type` (for DaCe nodes) or `op` (for ONNX)
3. Checking `backward_can_be_applied()` for each candidate
4. Returning first valid implementation

---

### 3.4 Analysis Utilities: `analysis.py`

**Location**: [analysis.py](analysis.py)

Provides SDFG analysis functions used by the AD engine:

#### 3.4.1 dependency_analysis()

Computes transitive read dependencies for each array. For example, if `C = A + B`, then `dependencies["C"] = {"A", "B"}`. Uses graph traversal and transitive closure to build a complete dependency map.

#### 3.4.2 inverse_reachability()

For each state, computes the set of predecessor states that can reach it. Uses DaCe's `StateReachability` analysis pass.

#### 3.4.3 is_previously_written()

Determines if an array was written before a given node in a state. Used by data forwarding to determine if an intermediate value needs to be stored (because it will be overwritten).

Checks both:
1. Current state (concurrent subgraphs)
2. Predecessor states

---

### 3.5 Utility Functions: `utils.py`

**Location**: [utils.py](utils.py)

The `utils.py` module contains many helper functions organized into categories:

#### 3.5.1 Descriptor Management

- `add_backward_desc()`: Add gradient array descriptor to backward SDFG
- `add_backward_desc_for_connector()`: Add backward descriptor for specific connector
- Helper functions for managing array descriptors and data types

#### 3.5.2 Symbolic Differentiation

- `differentiate_tasklet()`: Symbolically differentiates tasklet code using AST parsing and SymPy
- Converts tasklet expressions to symbolic form, computes derivatives, and generates backward code

#### 3.5.3 Graph Traversal

- `get_all_path_edges()`: Gets all edges on paths from source to target
- `concurrent_subgraphs()`: Finds concurrent execution regions in a state
- Helper functions for navigating SDFG structures

#### 3.5.4 Loop Analysis

- `state_within_loop()`: Checks if a state is inside a loop region
- `get_loop_carried_dependencies()`: Finds arrays with loop-carried dependencies
- Loop-specific helper functions

---

## 4. Data Forwarding System

### 4.1 The Core Problem

During backward pass generation, we often need access to intermediate values from the forward pass:

**Example**:
```python
# Forward
y = sigmoid(x)
z = y * y

# Backward (to compute dL/dx)
dL/dy = dL/dz * 2y    # Need y from forward pass!
dL/dx = dL/dy * y * (1 - y)  # Need y again!
```

**Two strategies**:
1. **Store**: Save `y` during forward pass, load it during backward
   - **Pro**: Fast backward pass (no recomputation)
   - **Con**: High memory usage
2. **Recompute**: Recompute `y = sigmoid(x)` during the backward pass
   - **Pro**: Low memory usage (no storage required)
   - **Con**: Slower backward pass due to recomputation cost

---

### 4.2 DataForwardingManager: `manager.py`

**Location**: [data_forwarding/manager.py](data_forwarding/manager.py)

Coordinates the data forwarding strategy.

#### 4.2.1 Strategy Selection

The manager provides three strategies:

1. **`store_all`** (default): Store all intermediate values
   - Fastest backward pass
   - Highest memory usage
   - Best for memory-rich environments

2. **`recompute_all`**: Recompute all intermediate values
   - Experimental feature to test recomputation capabilities

3. **`user_defined`**: User specifies which arrays to recompute
   - Balanced approach
   - Requires domain knowledge
   - Allows fine-grained control

#### 4.2.2 Main Algorithm

For each data item that needs to be forwarded:

1. Determine if the data is overwritten before backward pass needs it
2. If overwritten, choose resolution strategy (store or recompute)
3. Apply strategy:
   - **Store**: Create copy before overwrite, load in backward
   - **Recompute**: Extract computation subgraph, inline in backward

#### 4.2.3 Overwrite Detection Algorithm

**Problem**: Determine if an intermediate value is overwritten before the backward pass needs it

**Algorithm**:
```
is_overwritten(array, state, node):
  1. Check if array is written in concurrent subgraphs
  2. Check if array is written in successor states
  3. If either is true, the array is overwritten
  4. Apply data forwarding strategy (store or recompute)
```

**Uses**: `is_previously_written()` from `analysis.py`

---

### 4.3 Store Strategy: `store.py`

**Location**: [data_forwarding/store.py](data_forwarding/store.py)

**Key Function**: `resolve_overwrite_with_store()`

**Approach**:

```
Forward Pass State       Backward Pass State
┌──────────────┐        ┌──────────────┐
│ Compute x    │        │              │
│ Store x_copy │        │ Load x_copy  │
│ Overwrite x  │        │ Use in grad  │
└──────────────┘        └──────────────┘
```

**Steps**:
1. Create a storage descriptor for the intermediate value
2. Add a copy operation in the forward state (before overwrite)
3. Add a load operation in the backward state (when needed)
4. Update memlets to use the stored copy

---

### 4.4 Recompute Strategy: `recompute.py` (Experimental!)

**Location**: [data_forwarding/recompute.py](data_forwarding/recompute.py)

**Key Function**: `resolve_overwrite_with_recomputation()`

**Approach**:

```
Forward Pass State       Backward Pass State
┌──────────────┐        ┌──────────────┐
│ Compute x    │        │ Recompute x  │
│              │   →    │ Use in grad  │
│ Overwrite x  │        │              │
└──────────────┘        └──────────────┘
```

**Steps**:
1. Extract the computation subgraph that produces the value
2. Create a nested SDFG containing the recomputation logic
3. Inline the nested SDFG in the backward state
4. Connect inputs and outputs appropriately

**Subgraph Extraction** (`get_recomputation_nsdfg()`):
- Performs backward breadth-first search (BFS) from the data node to find all dependencies
- Copies nodes and edges into a new nested SDFG
- Handles map scopes and connectors
- Ensures all dependencies are included

---

## 5. Backward Implementations

### 5.1 DaCe Core Nodes: `dace_nodes.py`

**Location**: [implementations/dace_nodes.py](implementations/dace_nodes.py)

Implements backward passes for core SDFG elements.

#### 5.1.1 AccessNode

**Purpose**: Create gradient AccessNode

**Approach**:
- Forward: `AccessNode("x")`
- Backward: `AccessNode("grad_x")` with `setzero=True`

Also handles view connectors for arrays with views or subsets.

#### 5.1.2 Tasklet

**Purpose**: Symbolically differentiate tasklet code

**Approach**:
1. Parse tasklet code to AST
2. Extract output expressions
3. Use SymPy to compute symbolic derivatives
4. Generate backward code: `grad_input = grad_output * derivative`

**Example**:
- Forward: `y = x * x`
- Backward: `grad_x = grad_y * 2 * x`

#### 5.1.3 Maps

**Purpose**: Reverse map structure

Maps are special: `MapEntry` and `MapExit` nodes are swapped in the backward pass.

**Forward**:
```
AccessNode → MapEntry → [Tasklet in scope] → MapExit → AccessNode
```

**Backward**:
```
AccessNode → MapEntry (reversed) → [Tasklet_grad in scope] → MapExit (reversed) → AccessNode
```

**Approach**:
- `MapEntry` → `MapExit` in backward pass
- `MapExit` → `MapEntry` in backward pass
- Connectors inverted: `IN_X` ↔ `OUT_X`
- Same map object used for both

#### 5.1.4 NestedSDFG

**Purpose**: Recursively differentiate nested SDFGs

**Approach**:
1. Recursively call `add_backward_pass()` on nested SDFG
2. Map forward connectors to backward connectors
3. Handle symbols and interstate edges
4. Ensure proper gradient flow through nested boundaries

#### 5.1.5 LoopRegions

**Purpose**: Reverse loops in the forward SDFG

**Approach**:
Loops are reversed by creating a backward loop that iterates in the reverse direction to process gradients.


```
# Forward loop:
for i in range(N):
    y[i+1] = f(x[i])

# Backward loop:
for i in reversed(range(N)):
    grad_x[i] = grad_f(x[i]) * grad_y[i+1]
```

---

### 5.2 DaCe Library Nodes: `dace_library_nodes.py`

**Location**: [implementations/dace_library_nodes.py](implementations/dace_library_nodes.py)

Implements backward passes for DaCe library nodes.

#### 5.2.1 Key Implementations

| Operation | Backward Implementation | Notes |
|-----------|------------------------|-------|
| **Reduce (Sum)** | Broadcast gradient to match input shape | Handles axis reduction |
| **Reduce (Max/Min)** | Gradient flows only to max/min elements | Requires forward values |
| **Transpose** | Transpose gradient with inverted axes | Simple permutation |
| **MatMul** | Matrix multiplications with transposed matrices | BLAS operations |
| **Conv2D** | Convolution with rotated/transposed kernels | Uses backward_data/backward_filter |
| **Pooling** | Unpool gradient to input locations | Requires pooling indices |
| **BatchNorm** | Compute mean/variance gradients | Requires saved statistics |

---

### 5.3 ONNX Operations: `onnx_ops.py`

**Location**: [implementations/onnx_ops.py](implementations/onnx_ops.py)

Implements backward passes for 50+ ONNX operations. Each implementation follows the ONNX operator specification for gradient computation.

**Categories**:

- **Element-wise**: Add, Sub, Mul, Div, Sqrt, Exp, Log, Pow, etc.
- **Activation**: Relu, Sigmoid, Tanh, Softmax, etc.
- **Matrix**: MatMul, Gemm, BatchMatMul
- **Convolution**: Conv, ConvTranspose
- **Pooling**: MaxPool, AveragePool, GlobalAveragePool
- **Normalization**: BatchNormalization, LayerNormalization
- **Reduction**: ReduceSum, ReduceMean, ReduceMax, etc.
- **Shape**: Reshape, Transpose, Concat, Split, Squeeze, Unsqueeze
- **Advanced**: Gather, Scatter, Einsum, etc.

Each ONNX backward implementation is registered with `@dace.registry.autoregister_params(op="OpName")`.

---

## 6. Library Integration

### 6.1 BackwardPass Library Node: `library.py`

**Location**: [library/library.py](library/library.py)

Provides a library node for encapsulating backward passes as reusable components.

#### 6.1.1 ParameterArray

A special data descriptor for gradient accumulation buffers that mimics PyTorch Parameters.

#### 6.1.2 BackwardPass

A library node that wraps a backward pass SDFG, allowing backward passes to be composed and reused like other library operations.

#### 6.1.3 ExpandBackwardPass

Expands the `BackwardPass` library node into the full SDFG. Handles:
- Gradient initialization (zero or provided seed)
- Parameter gradient accumulation

---

## 7. PyTorch Integration

### 7.1 Overview: `torch.py`

**Location**: [torch.py](torch.py)

Enables the integration between DaCe AD and PyTorch's autograd system.

### 7.2 make_backward_function()

**Purpose**: Convert ONNX model to PyTorch-differentiable function

**Signature**:
```python
def make_backward_function(
    forward_sdfg: SDFG,
    inputs: List[str],
    outputs: List[str],
    parameters: Optional[List[str]] = None
) -> Type[torch.autograd.Function]:
```

**Returns**: PyTorch `autograd.Function` subclass with:
- `forward()`: Compiles and runs forward SDFG
- `backward()`: Compiles and runs backward SDFG
- Handles PyTorch tensor ↔ DaCe array conversion
- Supports scalar inputs/outputs
- Manages parameter gradients

### 7.3 Integration Flow

```
PyTorch Model
       ↓
DaCe ONNX Import
       ↓
Forward SDFG
       ↓
add_backward_pass()
       ↓
Backward SDFG
       ↓
make_backward_function()
       ↓
torch.autograd.Function
       ↓
Use in PyTorch training loop
```

---

## 8. Gradient Accumulation and Clearing

### 8.1 Gradient Accumulation

**Problem**: Multiple paths can contribute to same gradient

**Example**:
```
       ┌─→ y1 ─┐
   x ──┤       ├─→ z
       └─→ y2 ─┘
```

Both `y1` and `y2` contribute to `grad_x`.

**Solution**: Write-Conflict Resolution (WCR)

When connecting gradients, use WCR on memlets:
```python
memlet.wcr = "lambda a, b: a + b"
```

This ensures multiple gradient contributions are summed correctly.

### 8.2 Gradient Clearing

**Problem**: Overwritten arrays in the forward pass require clearing the gradients of the corresponding gradient arrays to allow the always-accumulate solution presented above.

**When to Clear Gradients**:
- In the backward pass, at the corresponding point where arrays in the forward pass where overwritten.

**Implementation Strategies**:

1. **Zero Initialization for all intermediate arrays**: Set all gradient arrays to zero before backward pass
   ```python
   # In DaCe, gradient arrays can be initialized with setzero=True
   grad_array = AccessNode("grad_x", setzero=True)
   ```

2. **Manual Clearing**: Explicitly zero out gradient arrays if necessary
   ```python
   # Reset gradients if an overwrite is detected in dace/autodiff/backward_pass_generator.py
   self._zero_out_gradient(forward_state=forward_state,
                            forward_node=node,
                            memlet=edge.data)
   ```
