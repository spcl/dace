Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
# DaCe ONNX Integration Library - Design Document

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Directory Structure](#3-directory-structure)
4. [Core Components](#4-core-components)
5. [Import Pipeline](#5-import-pipeline)
6. [Shape Inference System](#6-shape-inference-system)
7. [Implementation Strategies](#7-implementation-strategies)
8. [Key Algorithms](#8-key-algorithms)
9. [Extension Points](#9-extension-points)

---

## 1. Introduction

### 1.1 Purpose

The DaCe ONNX Integration Library enables **the importing and executing of ONNX (Open Neural Network Exchange) models** within the DaCe framework. It provides a pipeline for converting ONNX neural network models into optimized DaCe SDFGs (Stateful DataFlow Graphs) that can run efficiently on CPUs, GPUs, and other accelerators.

### 1.2 Current Capabilities

- **Model Import**: Load ONNX models from files or protobuf objects
- **Shape Inference**: Automatic computation of tensor shapes (symbolic and concrete)
- **Multi-Strategy Implementations**: Pure (correctness), optimized (performance), hardware-specific (GPU/FPGA)
- **Type Safety**: Schema-based validation and type checking
- **Framework Integration**: Interoperability with PyTorch and NumPy

### 1.3 Use Cases

1. **ML Inference Optimization**: Optimize pre-trained models for production deployment
2. **Hardware Acceleration**: Leverage DaCe's code generation for GPU/FPGA execution
3. **Cross-Framework Compatibility**: Run PyTorch/TensorFlow models in DaCe ecosystem
4. **Research and Experimentation**: Analyze and optimize neural network architectures
5. **Custom Optimization**: Apply DaCe transformations to ML workloads
6. **Benchmarking**: Compare performance across different implementations

### 1.4 ONNX Background

ONNX is an open standard for representing machine learning models, supported by major frameworks:
- **Export**: PyTorch, TensorFlow, Keras, scikit-learn
- **Operators**: 150+ standard operations (Conv, MatMul, Attention, etc.)
- **Opsets**: Versioned operator specifications (current: opset 18)
- **Use**: Model exchange, optimization, deployment

---

## 2. Architecture Overview

### 2.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  ONNXModel   │  │ ONNX Backend │  │ Direct ONNX Op  │   │
│  │ (main API)   │  │   (testing)  │  │   calls         │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘   │
└─────────┼──────────────────┼───────────────────┼────────────┘
          │                  │                   │
          └──────────────────┼───────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   IMPORT PIPELINE                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │    ONNXImporter (onnx_importer.py)                   │   │
│  │  1. Load Model    → 4. Graph Construction            │   │
│  │  2. Simplify      → 5. Weight Management             │   │
│  │  3. Shape Infer   → 6. Compilation                   │   │
│  └──────────────────┬───────────────────────────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
                      │
         ┌────────────┼───────────┐
         ▼            ▼           ▼
┌──────────────┐ ┌─────────┐ ┌──────────────────┐
│   REGISTRY   │ │ SCHEMA  │ │ SHAPE INFERENCE  │
├──────────────┤ ├─────────┤ ├──────────────────┤
│ Dynamic Node │ │ Type    │ │ Symbolic Shape   │
│ Generation   │ │ System  │ │ Inference        │
│              │ │         │ │ (Microsoft impl) │
│ • 100+ ops   │ │ • Valid-│ │                  │
│ • Versioning │ │   ation │ │ • Dynamic dims   │
│ • Properties │ │ • Constr-│ │ • Auto-merge     │
│ • Connectors │ │   aints │ │ • Concrete eval  │
└──────────────┘ └─────────┘ └──────────────────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
┌────────────────────────────────────────────────────────────┐
│              IMPLEMENTATION LAYER                          │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐     │
│  │   Pure      │ │  Optimized   │ │  Hardware        │     │
│  │   (SDFGs)   │ │  (img ops)   │ │  (cuDNN, etc)    │     │
│  ├─────────────┤ ├──────────────┤ ├──────────────────┤     │
│  │ Reference   │ │ Performance  │ │ GPU/FPGA         │     │
│  │ impl for    │ │ focused      │ │ specialized      │     │
│  │ correctness │ │ operations   │ │ libraries        │     │
│  └─────────────┘ └──────────────┘ └──────────────────┘     │
└────────────────────────────────────────────────────────────┘
                      ▼
              DaCe SDFG with ONNX Nodes
                      ▼
           Expansion → Optimization → Code Generation
```

### 2.2 Component Interaction Flow

```
ONNX Model File
       ↓
ONNXModel.__init__()
       ↓
1. onnx.checker.check_model() → Validate
       ↓
2. shape_inference.infer_shapes() → Compute shapes
       ↓
3. onnxsim.simplify() (optional) → Optimize ONNX graph
       ↓
4. Create SDFG structure
       ↓
5. For each ONNX node:
   ├─→ get_onnx_node(op_type, version) → Retrieve node class
   ├─→ Create instance with attributes
   ├─→ Add connectors from schema
   └─→ Create edges with memlets
       ↓
6. Load weights (initializers)
       ↓
7. Handle outputs (scalar promotion, return arrays)
       ↓
8. Apply GPU transformations (if cuda=True)
       ↓
SDFG with ONNX Library Nodes
       ↓
Compilation (triggered by first call or explicit compile()):
   ├─→ Expand ONNX nodes (select implementation)
   ├─→ Apply DaCe optimizations
   ├─→ Generate C++/CUDA code
   └─→ Compile to binary
       ↓
Execution:
   ├─→ Infer runtime symbols from input shapes
   ├─→ Call compiled function with inputs + weights
   └─→ Return outputs (NumPy or PyTorch tensors)
```

---

## 3. Directory Structure

### 3.1 File Organization

```
dace/libraries/onnx/
├── __init__.py                          # Library registration
│   └── Exports: ONNXModel, ONNXOp, schemas, backend
│
├── onnx_importer.py                     # Main entry point
│   └── ONNXModel class - Import pipeline orchestrator
│
├── schema.py                            # Type system
│   ├── @onnx_representation decorator
│   ├── ONNXSchema
│   ├── ONNXAttribute
│   ├── ONNXParameter
│   └── ONNXTypeConstraint
│
├── converters.py                        # Type conversions
│   ├── convert_onnx_proto()
│   ├── onnx_tensor_type_to_typeclass()
│   ├── clean_onnx_name()
│   └── convert_attribute_proto()
│
├── forward_implementation_abc.py       # Implementation interface
│   └── ONNXForward (ABC + registry)
│
├── nodes/                               # ONNX operation nodes
│   ├── onnx_op.py                       # Base class
│   │   └── ONNXOp - Abstract superclass for all ONNX ops
│   ├── onnx_op_registry.py              # Dynamic generation
│   │   ├── _get_all_schemas()
│   │   ├── _create_node_class()
│   │   └── get_onnx_node() / has_onnx_node()
│   └── node_utils.py                    # Utilities
│       └── parse_variadic_param()
│
├── op_implementations/                  # Implementation strategies
│   ├── pure_implementations.py          # Reference impl
│   │   └── 40+ operations (Add, Conv, MatMul, Softmax, etc.)
│   ├── img_op_implementations.py        # Image ops
│   │   └── Optimized Conv, Pool, BatchNorm
│   ├── criteria_implementations.py      # Conditional selection
│   └── utils.py                         # Helpers
│       ├── @op_implementation decorator
│       ├── @python_pure_op_implementation
│       ├── program_for_node()
│       └── empty_sdfg_for_node()
│
└── shape_inference/                     # Dynamic shape support
    ├── symbolic_shape_infer.py          # Symbolic inference
    │   └── SymbolicShapeInference class
    └── shape_inference.py               # Wrapper
        └── infer_shapes()

```

### 3.2 File Size Distribution

| File | Lines | Purpose |
|------|-------|---------|
| `pure_implementations.py` | 3052 | Reference implementations for correctness |
| `symbolic_shape_infer.py` | 1976 | Symbolic shape inference (Microsoft) |
| `onnx_importer.py` | 711 | Main import pipeline orchestrator |
| `img_op_implementations.py` | 586 | Optimized image operations |
| `schema.py` | 390 | Type system and validation |
| `onnx_op_registry.py` | 325 | Dynamic node class generation |
| `onnx_op.py` | 294 | Base class for ONNX operations |
| `converters.py` | 257 | Type conversion utilities |
| `utils.py` | 228 | Implementation helpers |

---

## 4. Core Components

### 4.1 ONNXModel: The Main Entry Point

**Location**: [onnx_importer.py](onnx_importer.py)

The `ONNXModel` class is the primary interface for importing and executing ONNX models.

#### Key Features

- **Model Loading**: Loads models from files or ONNX protobuf objects
- **Automatic Optimization**: Provides optional ONNX-level simplification
- **Shape Inference**: Handles dynamic and symbolic shapes automatically
- **Weight Management**: Loads and manages model parameters efficiently
- **Compilation**: Supports lazy or explicit compilation to optimized code
- **Execution**: Provides direct `__call__` interface with NumPy/PyTorch tensors
- **GPU Support**: Automatic GPU transformation when `cuda=True`

#### Constructor Signature

```python
class ONNXModel:
    def __init__(
        self,
        name: str,
        model: Union[str, onnx.ModelProto],
        cuda: bool = False,
        apply_strict: bool = False,
        auto_optimize: bool = True,
        onnx_simplify: bool = True,
        infer_shapes: bool = True,
        auto_merge: bool = False
    ):
        """
        Import an ONNX model into DaCe.

        Args:
            name: Name for the generated SDFG
            model: Path to .onnx file or onnx.ModelProto object
            cuda: Enable GPU execution
            apply_strict: Strict ONNX validation
            auto_optimize: Apply DaCe optimizations on first run
            onnx_simplify: Apply onnx-simplifier before import
            infer_shapes: Run shape inference
            auto_merge: Auto-merge conflicting symbolic shapes
        """
```

#### Main Methods

- **`__call__()`**: Execute the model with inputs
- **`compile()`**: Explicitly compile the SDFG
- **`save()`**: Save compiled model to disk
- **`infer_symbols()`**: Infer symbolic dimension values from input shapes

---

### 4.2 Registry System: Dynamic Node Generation

**Location**: [nodes/onnx_op_registry.py](nodes/onnx_op_registry.py)

The registry system **dynamically generates Python classes** for all ONNX operations at import time, eliminating the need to manually write 100+ node classes.

#### How It Works

**Process**:
```
1. Query ONNX for all supported operations
   ↓
2. For each operation (e.g., "Conv"):
   ├─ Get all versions (e.g., Conv_1, Conv_11, Conv_13)
   ├─ Convert ONNX OpSchema to ONNXSchema
   └─ For each version:
       ├─ Create Python properties from attributes
       ├─ Generate __init__ constructor
       ├─ Add input/output connectors
       ├─ Generate documentation
       ├─ Create class with type()
       └─ Register with DaCe library system
   ↓
3. Store in global registry:
   _ONNX_OPS["Conv"][11] = ONNXConv_11
   _ONNX_OPS["Conv"][13] = ONNXConv_13
   ↓
4. Export latest version to module:
   ONNXConv = ONNXConv_13
```

#### Generated Class Structure

For each ONNX operation, the registry generates:

- **Class Name**: `ONNX{OpName}_{Version}` (e.g., `ONNXConv_11`)
- **Properties**: One DaCe property per ONNX attribute
- **Constructor**: Validates required attributes, sets defaults
- **Connectors**: Input/output connectors from schema
- **Schema**: Embedded `ONNXSchema` for validation
- **Implementations**: Linked expansion transformations
- **Documentation**: Auto-generated from ONNX docs

#### API Functions

```python
def has_onnx_node(name: str) -> bool:
    """Check if ONNX operation is supported."""

def get_onnx_node(name: str, opset_version: int = None) -> Type[ONNXOp]:
    """Get ONNX node class by name and version."""
```

---

### 4.3 Schema System: Type Safety

**Location**: [schema.py](schema.py)

The schema system provides a Python representation layer for ONNX protobuf schemas, enabling type-safe interactions.

#### Key Components

**ONNXSchema** - Complete operation specification:
```python
@dataclass
class ONNXSchema:
    name: str                          # Operation name (e.g., "Conv")
    since_version: int                 # First opset supporting this
    doc: str                           # Documentation
    inputs: List[ONNXParameter]        # Input specifications
    outputs: List[ONNXParameter]       # Output specifications
    attributes: Dict[str, ONNXAttribute]  # Attribute specs
    type_constraints: Dict[str, ONNXTypeConstraint]  # Type constraints
```

**ONNXParameter** - Input/output parameter:
```python
@dataclass
class ONNXParameter:
    name: str                          # Parameter name
    type_str: str                      # Type constraint reference
    param_type: ONNXParameterType      # Single/Optional/Variadic
    description: str                   # Documentation
    homogeneous: bool                  # For variadic params
```

**ONNXAttribute** - Operation configuration:
```python
@dataclass
class ONNXAttribute:
    name: str                          # Attribute name
    type: ONNXAttributeType           # Int/Float/String/Tensor/etc.
    required: bool                     # Must be provided?
    default_value: Any                 # Default if not provided
    description: str                   # Documentation
```

**ONNXTypeConstraint** - Allowed types:
```python
@dataclass
class ONNXTypeConstraint:
    type_param_str: str                # Type parameter (e.g., "T")
    allowed_types: List[typeclass]     # Allowed DaCe types
    description: str                   # Documentation
```

#### The @onnx_representation Decorator

Enables creating Python classes from ONNX protobufs:

```python
@onnx_representation(onnx.TensorProto)
class ONNXTensor:
    dims: List[int]
    data_type: int
    # ... other fields
```

Automatically generates:
- `__init__()` constructor
- `from_onnx_proto()` class method
- `from_json()` / `to_json()` serialization
- Registration in the global protobuf registry

---

### 4.4 ONNXOp Base Class

**Location**: [nodes/onnx_op.py](nodes/onnx_op.py)

`ONNXOp` is the abstract base class for all ONNX operation nodes in DaCe SDFGs.

#### Key Methods

- **`iter_inputs_in_onnx_order()`**: Get input edges in schema order
- **`iter_outputs_in_onnx_order()`**: Get output edges in schema order
- **`iter_edges()`**: Iterate all edges with input/output flag
- **Validation**: Automatic schema-based validation during SDFG construction

#### Properties

- `schema`: The operation's ONNXSchema
- `backward_implementation`: Which backward impl to use (for autodiff)
- `implementations`: Available forward implementations
- `default_implementation`: Default expansion strategy

---

### 4.5 Type Converters

**Location**: [converters.py](converters.py)

Provides bidirectional conversion between ONNX, DaCe, NumPy, and PyTorch type systems.

#### Key Functions

**Type Conversion**:
- `onnx_tensor_type_to_typeclass()`: ONNX type enum → DaCe typeclass
- `typeclass_to_onnx_tensor_type_int()`: DaCe typeclass → ONNX type enum
- `convert_onnx_proto()`: Generic protobuf → Python conversion
- `convert_attribute_proto()`: ONNX AttributeProto → Python value

**Name Sanitization**:
- `clean_onnx_name()`: Makes ONNX names valid DaCe identifiers
  - Prefixes digit-starting names: `123` → `ONNX_123`
  - Replaces special characters: `.` → `DOT`, `:` → `COLON`, `/` → `SLASH`

**Helper Functions**:
- `get_proto_attr()`: Provides safe protobuf attribute access with encoding checks

---

## 5. Import Pipeline

### 5.1 Complete Workflow

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Model Loading and Validation                   │
├─────────────────────────────────────────────────────────┤
│ 1. Load ONNX model (from file or protobuf)              │
│ 2. Run onnx.checker.check_model()                       │
│ 3. Validate model conforms to ONNX spec                 │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Shape Inference                                 │
├─────────────────────────────────────────────────────────┤
│ 1. Run symbolic shape inference                          │
│ 2. Compute concrete shapes where possible               │
│ 3. Create symbolic dimensions for dynamic shapes        │
│ 4. Auto-merge conflicting symbols (optional)            │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: ONNX-Level Optimization (optional)              │
├─────────────────────────────────────────────────────────┤
│ 1. Apply onnxsim.simplify()                             │
│    - Constant folding                                    │
│    - Dead code elimination                               │
│    - Operator fusion                                     │
│ 2. Validate optimization preserves semantics            │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: SDFG Construction                               │
├─────────────────────────────────────────────────────────┤
│ 1. Create empty SDFG with initial state                 │
│ 2. Register inputs/outputs as data descriptors          │
│ 3. For each ONNX node:                                  │
│    a. Get node class from registry                      │
│    b. Extract and convert attributes                    │
│    c. Create node instance                              │
│    d. Add input/output connectors                       │
│    e. Create AccessNodes for data                       │
│    f. Add edges with memlets                            │
│ 4. Handle special cases (Constants, Identities)         │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 5: Weight Management                               │
├─────────────────────────────────────────────────────────┤
│ 1. Load initializers (weights/biases) from ONNX         │
│ 2. Convert to PyTorch tensors                           │
│ 3. Store in self.weights dictionary                     │
│ 4. Create corresponding DaCe arrays (non-transient)     │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 6: Output Handling                                 │
├─────────────────────────────────────────────────────────┤
│ 1. Promote scalars to arrays (CPU only)                 │
│ 2. Create return arrays (__return, __return_0, etc.)    │
│ 3. Add copy-out state for outputs                       │
│ 4. Fuse states for efficiency                           │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 7: GPU Transformation (if cuda=True)               │
├─────────────────────────────────────────────────────────┤
│ 1. Apply sdfg.apply_gpu_transformations()               │
│ 2. Convert memory to GPU_Global storage                 │
│ 3. Add GPU kernel launch infrastructure                 │
└──────────────────────┬──────────────────────────────────┘
                       ▼
            SDFG with ONNX Library Nodes
```

### 5.2 Node Construction Details

For each ONNX operation in the graph:

**Step 1: Operation Lookup**
```python
if not has_onnx_node(node.op_type):
    raise ValueError(f"Unsupported operation: {node.op_type}")
```

**Step 2: Attribute Extraction**
```python
attributes = {attr.name: convert_attribute_proto(attr)
              for attr in node.attribute}
```

**Step 3: Node Class Retrieval**
```python
node_class = get_onnx_node(node.op_type, model_opset_version)
```

**Step 4: Instance Creation**
```python
dace_node = node_class(name=node.name, **attributes)
```

**Step 5: Connector and Edge Creation**
```python
for input_param in node_class.schema.inputs:
    # Validate parameter type (Single/Optional/Variadic)
    # Create or reuse AccessNode
    # Add connector to operation node
    # Create Memlet edge with full array semantics
```

### 5.3 Special Handling

- **Constants**: Directly added to weights, no node created
- **Identities**: Can be elided during optimization
- **Variadic Parameters**: Use naming convention `param_name__index`
- **Optional Parameters**: Checked for presence, skipped if absent

---

## 6. Shape Inference System

### 6.1 Purpose and Motivation

ONNX models often have **dynamic shapes** where tensor dimensions depend on runtime inputs:
- Batch size: Variable number of samples
- Sequence length: Variable-length sequences (NLP)
- Image dimensions: Variable-size images

Shape inference computes tensor shapes either symbolically or concretely for all intermediate tensors in the model.

### 6.2 Integration

**Location**: [shape_inference/symbolic_shape_infer.py](shape_inference/symbolic_shape_infer.py)

Called during model import:
```python
model = shape_inference.infer_shapes(model, auto_merge=auto_merge)
```

### 6.3 Capabilities

**Symbolic Dimensions**:
```python
# Input shape: [batch_size, 3, 224, 224]
# After Conv: [batch_size, 64, 112, 112]
# After Pool: [batch_size, 64, 56, 56]
```

**Concrete Evaluation**:
```python
# Known: kernel_size=3, stride=2, padding=1, input_size=224
# Computed: output_size = (224 + 2*1 - 3) / 2 + 1 = 112
```

**Broadcasting**:
```python
# Shape A: [batch, 256, 1, 1]
# Shape B: [batch, 256, 7, 7]
# Result:  [batch, 256, 7, 7]
```

**Auto-Merge** (optional):
```python
# Before: tensor_0: [batch_0, seq_len_0]
#         tensor_1: [batch_1, seq_len_1]
# After:  tensor_0: [batch, seq_len]
#         tensor_1: [batch, seq_len]
```

### 6.4 Implementation Details

The symbolic shape inference is based on a **Microsoft-sourced implementation** that includes:

- Helper functions for dimension extraction and axis handling
- `SymbolicShapeInference` class with per-operation rules
- Sympy-based symbolic computation
- Integration with ONNX's native shape inference
- Special handling for complex operations (Reshape, Transpose, Concat)

### 6.5 DaCe Integration

Symbolic dimensions are added to the SDFG symbol table:
```python
for dim_name in symbolic_dimensions:
    sdfg.add_symbol(dim_name, dace.int64)
```

At runtime, DaCe infers symbol values from input shapes:
```python
symbols = {}
if 'batch_size' in sdfg.symbols:
    symbols['batch_size'] = input_tensor.shape[0]
```

---

## 7. Implementation Strategies

### 7.1 The ONNXForward Interface

**Location**: [forward_implementation_abc.py](forward_implementation_abc.py)

```python
@make_registry
class ONNXForward(abc.ABC):
    """Abstract base for ONNX operation implementations."""

    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        """Check if implementation is applicable."""
        return True

    @staticmethod
    @abc.abstractmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> Union[Node, SDFG]:
        """Expand node to DaCe constructs."""
        ...
```

### 7.2 Implementation Types

#### 1. Pure Implementations

**Location**: [op_implementations/pure_implementations.py](op_implementations/pure_implementations.py)

**Purpose**: Provides reference implementations focused on correctness

**Characteristics**:
- Written in Python/NumPy style
- Automatically parsed via the DaCe Python frontend
- Semantically correct according to ONNX specifications
- May not be optimally performant until further transformations are applied

**Example Operations**:
- Mathematical: Log, Exp, Sqrt, Pow, Abs
- Reductions: ReduceMean, ReduceSum, ReduceMax
- Shape manipulation: Reshape, Transpose, Squeeze, Unsqueeze
- Array operations: Concat, Split, Gather, Scatter
- Element-wise: Add, Sub, Mul, Div

**Implementation Pattern**:
```python
@python_pure_op_implementation
def Relu(X: dace.float32[H, W]):
    """Pure implementation of ReLU activation."""
    return np.maximum(X, 0)
```

**Process**:
1. Decorator creates an `ONNXForward` subclass
2. Function is parsed via the DaCe Python frontend
3. Converted to SDFG with maps and tasklets
4. Result: Efficient parallel code generation

#### 2. Optimized Implementations

**Location**: [op_implementations/img_op_implementations.py](op_implementations/img_op_implementations.py)

**Purpose**: Provides performance-optimized implementations for specific operations

**Examples**:
- `Conv`: Optimized convolution with im2col or Winograd
- `MaxPool/AveragePool`: Efficient pooling operations
- `BatchNormalization`: Fused batch normalization

**Characteristics**:
- Hand-crafted SDFG construction
- May use library calls (BLAS, cuDNN)
- Optimized for specific hardware/configurations

#### 3. Hardware-Specific Implementations

**Concept**: Implementations optimized for specific hardware

**Examples** (potential):
- `cuDNN` implementations for GPU (Conv, Pool, BatchNorm)
- `MKL-DNN` implementations for CPU
- `FPGA` implementations for reconfigurable hardware

**Selection via Applicability**:
```python
@op_implementation(op="Conv", name="cudnn")
class CuDNNConv(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node, state, sdfg):
        return sdfg.gpu and has_cudnn()
```

### 7.3 Implementation Selection

**Process**:

1. Query the registry for the operation's implementations
2. Filter by applicability: `forward_can_be_applied()`
3. Prefer user-specified implementation (if set)
4. Fall back to the default implementation
5. Expand the node using the selected implementation

**Priority Order**:
1. User-specified implementation (node property)
2. First applicable implementation (by registration order)
3. Default implementation (usually "pure")

### 7.4 Common Implementation Patterns

#### Pattern A: Pure Python with Decorator

```python
@python_pure_op_implementation
def Softmax(X: dace.float32[N, M], axis: int = -1):
    """Softmax activation function."""
    exp_x = np.exp(X - np.max(X, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

#### Pattern B: Manual SDFG Construction

```python
@op_implementation(op="MatMul", name="blas")
class BLASMatMul(ONNXForward):
    @staticmethod
    def forward(node, state, sdfg):
        # Create nested SDFG
        nsdfg = dace.SDFG(f"{node.label}_matmul")
        nstate = nsdfg.add_state()

        # Use BLAS library node
        from dace.libraries.blas import MatMul
        matmul_node = MatMul("matmul")

        # Connect inputs/outputs
        # ...

        return nsdfg
```

#### Pattern C: Library Call Integration

```python
@op_implementation(op="Conv", name="optimized")
class OptimizedConv(ONNXForward):
    @staticmethod
    def forward(node, state, sdfg):
        # Leverage existing DaCe library nodes
        from dace.libraries.standard import Conv2D

        # Convert ONNX semantics to library call
        conv_node = Conv2D(...)

        # Return library node (further expanded by DaCe)
        return conv_node
```

### 7.5 Implementation Utilities

**Location**: [op_implementations/utils.py](op_implementations/utils.py)

**Key Functions**:

- `@op_implementation(op, name)`: Register implementation with registry
- `@python_pure_op_implementation`: Create implementation from Python function
- `program_for_node()`: Convert Python function to nested SDFG
- `empty_sdfg_for_node()`: Create empty nested SDFG template

---

## 8. Key Algorithms

### 8.1 Dynamic Node Class Generation

**Algorithm**: Creates Python classes at import time

```
For each ONNX operation in onnx.defs.get_all_schemas():
    1. Extract OpSchema from ONNX
    2. Convert to ONNXSchema (DaCe representation)
    3. For each version of the operation:
        a. Generate class name: ONNX{OpName}_{Version}
        b. Create properties from attributes:
           - Map ONNX types to DaCe property types
           - Set defaults and required flags
        c. Generate __init__ constructor:
           - Validate required attributes provided
           - Convert types (e.g., StringLiteral → str)
           - Set up connectors for parameters
        d. Generate documentation from schema
        e. Create class with type():
           cls = type(cls_name, (ONNXOp,), attrs)
        f. Register as DaCe library node:
           cls = dace.library.node(cls)
        g. Link implementations:
           - Query ONNXForward.extensions()
           - Create ExpandTransformation wrappers
           - Register with node class
        h. Store in registry:
           _ONNX_OPS[op_name][version] = cls
    4. Export latest version to module:
       globals()[f"ONNX{OpName}"] = latest_version
```

**Result**: 100+ operation classes generated automatically, ready for use

### 8.2 Schema-Based Validation

**Algorithm**: Validates node construction

```
When creating ONNX node instance:
    1. Check required attributes provided:
       missing = required_attrs - provided_attrs
       if missing: raise ValueError(...)

    2. Validate connector usage:
       For each edge connected to node:
           a. Determine parameter (input/output)
           b. Check parameter type (Single/Optional/Variadic)
           c. Validate connector naming:
              - Single/Optional: exact name
              - Variadic: name__index format
           d. Verify edge data type matches constraints

    3. Type constraint checking:
       For each connector with type constraint:
           a. Get connector data type
           b. Look up constraint allowed types
           c. Verify type in allowed set
           d. If not: raise validation error
```

### 8.3 Runtime Symbol Inference

**Algorithm**: Infers symbolic dimension values from inputs

```
When executing ONNXModel:
    1. Collect all symbols in SDFG:
       symbols = sdfg.free_symbols

    2. For each input tensor:
       For each dimension in tensor.shape:
           if dimension_name in symbols:
               inferred_symbols[dimension_name] = dimension_value

    3. Verify all required symbols inferred:
       missing = symbols - inferred_symbols.keys()
       if missing: raise ValueError(...)

    4. Pass symbols to compiled SDFG:
       result = compiled_sdfg(inputs..., **inferred_symbols)
```

### 8.4 Type Conversion Pipeline

**Algorithm**: Converts between type systems

```
ONNX Type → DaCe Type:
    1. Extract ONNX type enum (e.g., TensorProto.FLOAT)
    2. Look up in cached mapping:
       dace_type = onnx_to_dace_type_map[onnx_type]
    3. Return DaCe typeclass (e.g., dace.float32)

DaCe Type → NumPy Type:
    1. Get DaCe typeclass
    2. Extract numpy_dtype property
    3. Return numpy dtype (e.g., np.float32)

NumPy Type → PyTorch Type:
    1. Look up in numpy_to_torch_dtype_dict
    2. Return torch dtype (e.g., torch.float32)
```

---

## 9. Extension Points

### 9.1 Adding New ONNX Operations

If an ONNX operation is not yet supported, you can add it by creating an implementation:

**Step 1: Create Implementation Class**

```python
from dace.libraries.onnx.forward_implementation_abc import ONNXForward
from dace.libraries.onnx.op_implementations.utils import op_implementation

@op_implementation(op="CustomOp", name="pure")
class CustomOpImplementation(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node, state, sdfg):
        # Check if this implementation is applicable
        return True

    @staticmethod
    def forward(node, state, sdfg):
        # Create nested SDFG for operation
        # ...
        return nested_sdfg
```

**Step 2: Register Implementation**

The `@op_implementation` decorator automatically registers the implementation with the ONNXForward registry.

**Step 3: Use in Models**

The operation will now be available when importing ONNX models that use it.

### 9.2 Custom Implementations for Existing Operations

Override the default implementation with a custom one:

```python
@op_implementation(op="Conv", name="my_optimized_conv")
class MyOptimizedConv(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node, state, sdfg):
        # Only apply for specific configurations
        return (node.kernel_shape == [3, 3] and
                node.stride == [1, 1])

    @staticmethod
    def forward(node, state, sdfg):
        # Custom optimized implementation
        # ...
```

**Selection**: Set `node.default_implementation = "my_optimized_conv"` or allow DaCe to select automatically based on applicability.
