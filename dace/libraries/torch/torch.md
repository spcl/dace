# DaCe PyTorch Integration Library - Design Document

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Directory Structure](#3-directory-structure)
4. [Core Components](#4-core-components)
5. [Dispatcher Strategies](#5-dispatcher-strategies)
6. [Integration Pipeline](#6-integration-pipeline)
7. [Zero-Copy Tensor Sharing](#7-zero-copy-tensor-sharing)
8. [Autograd Integration](#8-autograd-integration)

---

## 1. Introduction

### 1.1 Purpose

The DaCe PyTorch Integration Library provides **bidirectional integration** between PyTorch's neural network framework and DaCe's high-performance SDFG execution engine. It enables:

- **Optimizing PyTorch models** using DaCe's dataflow transformations
- **Accelerating training and inference** with optimized compiled code

### 1.2 Current Capabilities

- **Model Optimization**: Convert PyTorch `nn.Module` to optimized DaCe SDFGs
- **Automatic Differentiation**: Integration with PyTorch's autograd system
- **Dual Dispatch**: C++ extension (performance) or CTypes (flexibility)
- **Training Support**: Backward pass generation and gradient computation

### 1.3 Integration Directions

The library supports bidirectional data flow:

**1. PyTorch → DaCe (Primary Direction)**:
```python
# Wrap PyTorch model for DaCe optimization
dace_module = DaceModule(pytorch_model, dummy_inputs, backward=True)

# Use as drop-in replacement
output = dace_module(input_tensor)
loss.backward()  # Autograd works!
```

**Workflow**: PyTorch Model → ONNX Export → DaCe SDFG → Compiled Code → PyTorch Operator

**2. DaCe → PyTorch (Zero-Copy Access)**:
```python
# DaCe arrays accessible as PyTorch tensors (no copy)
torch_tensor = array_to_torch_tensor(ptr, dace_descriptor)
```

**Mechanism**: DLPack protocol for memory sharing

### 1.4 Use Cases

1. **Neural Network Optimization**: Speed up inference for production deployment
2. **Training Acceleration**: Optimize forward and backward passes for faster training
3. **Custom Operators**: Implement custom PyTorch operations with DaCe
4. **Research**: Experiment with dataflow-level optimizations on ML models
5. **Mixed Workflows**: Combine PyTorch layers with DaCe-optimized modules

---

## 2. Architecture Overview

### 2.1 High-Level System Diagram

```
┌───────────────────────────────────────────────────────────┐
│                    USER INTERFACE                         │
│  ┌────────────────────────────────────────────────────┐   │
│  │   DaceModule (pytorch_model, dummy_inputs, ...)    │   │
│  │   • Wraps PyTorch nn.Module                        │   │
│  │   • Provides PyTorch-compatible interface          │   │
│  │   • Supports forward + backward passes             │   │
│  └──────────────────┬─────────────────────────────────┘   │
└─────────────────────┼─────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────┐
│              ONNX EXPORT PIPELINE                         │
│  ┌────────────────────────────────────────────────────┐   │
│  │ torch.onnx.export()                                │   │
│  │    PyTorch Model → ONNX ModelProto                 │   │
│  └──────────────────┬─────────────────────────────────┘   │
└─────────────────────┼─────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              SDFG CONSTRUCTION                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ ONNXModel (onnx_proto)                             │ │
│  │    ONNX → DaCe SDFG (Forward)                      │ │
│  └────────────────────┬───────────────────────────────┘ │
│                       │                                 │
│                       ▼ (if backward=True)              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ BackwardPassGenerator                              │ │
│  │    Forward SDFG → Backward SDFG                    │ │
│  └──────────────────┬─────────────────────────────────┘ │
└─────────────────────┼───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            DISPATCHER SELECTION                             │
│  ┌──────────────────┐           ┌──────────────────┐        │
│  │ C++ Extension    │    OR     │ CTypes Module    │        │
│  ├──────────────────┤           ├──────────────────┤        │
│  │ • Performance    │           │ • No param limit │        │
│  │ • Native PyTorch │           │ • Faster compile │        │
│  │ • 64 param limit │           │ • Pure Python    │        │
│  └────────┬─────────┘           └────────┬─────────┘        │
└───────────┼──────────────────────────────┼──────────────────┘
            │                              │
            └──────────┬───────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│            CODE GENERATION & COMPILATION                │
│  ┌────────────────────────────────────────────────────┐ │
│  │ SDFG.compile() → Shared Library (.so)              │ │
│  │ C++ Codegen → PyTorch Operator Registration        │ │
│  │ State Initialization → Handle Creation             │ │
│  └──────────────────┬─────────────────────────────────┘ │
└─────────────────────┼───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            CALLABLE PYTORCH OPERATOR                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ dace_module(inputs) → outputs                        │   │
│  │ • Zero-copy tensor access via DLPack                 │   │
│  │ • Stateful execution via handles                     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Flow

```
User Code: dace_module = DaceModule(model, dummy_inputs, backward=True)
       ↓
1. Store model and configuration
       ↓
User Code: output = dace_module(actual_input)  # First call
       ↓
2. Detect function is None → Initialize SDFG
       ↓
3. Export to ONNX
   a. torch.onnx.export(model, dummy_inputs)
   b. Save parameters and model structure
       ↓
4. Import ONNX to DaCe
   a. ONNXModel(onnx_proto)
   b. Create forward SDFG
       ↓
5. Generate Backward (if backward=True)
   a. Determine required gradients
   b. BackwardPassGenerator.backward()
   c. Create backward SDFG
   d. Identify forwarded transients
       ↓
6. Compile SDFGs
   a. forward_sdfg.compile()
   b. backward_sdfg.compile() (if applicable)
   c. Initialize with dummy inputs
       ↓
7. Select and Initialize Dispatcher
   If compile_torch_extension:
   a. Generate C++ code with autograd
   b. Compile as PyTorch extension
   c. Register as torch.ops.dace_{name}.{name}
   Else:
   a. Create Python autograd.Function
   b. Wrap with CTypes calls
       ↓
8. Create Wrapper Function
   a. Accept user inputs
   b. Pass state handles as first args
   c. Call compiled operator
   d. Return outputs
       ↓
9. Execute and Return
   a. Zero-copy tensor access via DLPack
   b. Call native SDFG code
   c. Return PyTorch tensors
       ↓
User Code: loss.backward()  # Backward pass
       ↓
10. PyTorch calls backward function
    a. Recover saved tensors from context
    b. Allocate gradient buffers
    c. Call backward SDFG
    d. Return input gradients
```

---

## 3. Directory Structure

### 3.1 File Organization

```
dace/libraries/torch/
├── __init__.py                          # Library exports
│   └── Exports: PyTorch, PyTorchCUDA environment classes
│
├── dlpack.py                            # Zero-copy tensor sharing
│   ├── DLPack structure definitions
│   ├── Type conversion mappings
│   └── array_to_torch_tensor() - Main conversion function
│
├── dispatchers/                         # Dispatcher implementations
│   ├── __init__.py                      # Package exports
│   │   └── Exports: get_ctypes_dispatcher, register_and_compile_torch_extension
│   │
│   ├── common.py                        # Shared utilities
│   │   ├── DaCeMLTorchFunction dataclass
│   │   ├── get_arglist()
│   │   └── compile_and_init_sdfgs()
│   │
│   ├── cpp_torch_extension.py           # C++ extension generator
│   │   ├── Type conversion utilities
│   │   ├── C++ code generation for forward/backward
│   │   ├── Autograd function generation
│   │   ├── Tensor initialization
│   │   └── register_and_compile_torch_extension()
│   │
│   └── ctypes_module.py                 # CTypes dispatcher
│       ├── init_remaining_parameters()
│       ├── callable_for_fwd_module()
│       ├── callable_for_bwd_module()
│       └── get_ctypes_dispatcher()
│
└── environments/                        # Build configuration
    ├── __init__.py                      # Package exports (1 line)
    └── pytorch_env.py                   # PyTorch environments
        ├── PyTorch (CPU environment)
        └── PyTorchCUDA (GPU environment)

```

### 3.2 Component Responsibilities

| Component | Lines | Purpose |
|-----------|-------|---------|
| `cpp_torch_extension.py` | 717 | C++ code generation for PyTorch operators |
| `ctypes_module.py` | 230 | CTypes-based dispatcher for large models |
| `dlpack.py` | 199 | Zero-copy tensor sharing via DLPack |
| `common.py` | 122 | Shared dispatcher utilities |
| `pytorch_env.py` | 94 | CMake build configuration |
| `__init__.py` (dispatchers) | 25 | Dispatcher exports |
| `__init__.py` (main) | 17 | Library exports |

---

## 4. Core Components

### 4.1 DaceModule: The Main Entry Point

**Location**: [dace/frontend/python/module.py](../../frontend/python/module.py)

#### Constructor Signature

```python
class DaceModule:
    def __init__(
        self,
        module: torch.nn.Module,
        dummy_inputs: Tuple[torch.Tensor, ...],
        cuda: bool = False,
        backward: bool = False,
        compile_torch_extension: bool = True,
        auto_optimize: bool = True,
        **onnx_kwargs
    ):
        """
        Wrap a PyTorch module for DaCe optimization.

        Args:
            module: PyTorch module to optimize
            dummy_inputs: Sample inputs for shape inference and tracing
            cuda: Enable GPU execution
            backward: Generate backward pass for training
            compile_torch_extension: Use C++ extension (True) or CTypes (False)
            auto_optimize: Apply DaCe optimizations
            **onnx_kwargs: Additional arguments for torch.onnx.export()
        """
```

#### Key Methods

- **`__call__(*inputs)`**: Execute the optimized module
- **`_initialize_sdfg(inputs)`**: Lazy compilation on first call
- **`_call_params()`**: Get model parameters as tensors

#### Workflow Summary

1. **Initialization**: Store model and configuration
2. **First Call**: Export to ONNX → Import to SDFG → Compile → Create dispatcher
3. **Subsequent Calls**: Direct execution via the compiled operator
4. **Backward**: Automatic execution via PyTorch autograd integration

---

### 4.2 DLPack Bridge: Zero-Copy Tensor Sharing

**Location**: [dlpack.py](dlpack.py)

The DLPack bridge enables **zero-copy conversion** between DaCe arrays and PyTorch tensors.

#### DLPack Structure Definitions

**Type System**:
```python
class DLDeviceType(ctypes.c_int):
    kDLCPU = 1
    kDLGPU = 2
    # ... other devices

class DLDataTypeCode(ctypes.c_uint8):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4

class DLDataType(ctypes.Structure):
    _fields_ = [('type_code', DLDataTypeCode),
                ('bits', ctypes.c_uint8),
                ('lanes', ctypes.c_uint16)]
```

**Tensor Representation**:
```python
class DLTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.c_void_p),         # Raw pointer
        ('ctx', DLContext),                # Device info
        ('ndim', ctypes.c_int),            # Number of dimensions
        ('dtype', DLDataType),             # Data type
        ('shape', ctypes.POINTER(ctypes.c_int64)),   # Shape array
        ('strides', ctypes.POINTER(ctypes.c_int64)), # Strides array
        ('byte_offset', ctypes.c_uint64)   # Byte offset
    ]
```

#### Zero-Copy Conversion

**Function**: `array_to_torch_tensor(ptr, desc)`

**Process**:
1. Map the DaCe storage type to DLDeviceType
2. Convert the DaCe dtype to DLDataType
3. Create shape and strides arrays
4. Build the DLTensor structure
5. Wrap in DLManagedTensor with a no-op deleter
6. Create a PyCapsule with name "dltensor"
7. Call `torch.utils.dlpack.from_dlpack(capsule)` to create the PyTorch tensor
8. Store the DLPack structure as a tensor attribute (prevents garbage collection)

**Memory Ownership**:
- Data is owned by the DaCe SDFG state struct
- No-op deleter ensures that DaCe manages deallocation
- PyTorch tensor is a **view** into DaCe memory (zero-copy)

**Type Mapping**:
```python
dace_to_dldtype_dict = {
    dace.float32: DLDataType(kDLFloat, 32, 1),
    dace.float64: DLDataType(kDLFloat, 64, 1),
    dace.int32: DLDataType(kDLInt, 32, 1),
    # ... complete mapping
}
```

---

### 4.3 Common Dispatcher Utilities

**Location**: [dispatchers/common.py](dispatchers/common.py)

#### DaCeMLTorchFunction Dataclass

```python
@dataclasses.dataclass
class DaCeMLTorchFunction:
    """Encapsulates a compiled DaCe module with PyTorch interface."""
    function: Callable              # The callable (torch op or Python function)
    compiled_sdfgs: List[CompiledSDFG]  # [forward, backward] (or just [forward])
    ptr: List[torch.Tensor]         # State handle pointers [fwd_handle, bwd_handle]
```

**Purpose**: Provides a uniform interface regardless of dispatcher choice (C++ or CTypes)

#### compile_and_init_sdfgs()

**Function Signature**:
```python
def compile_and_init_sdfgs(
    module: DaceModule,
    dummy_inputs: Tuple[torch.Tensor, ...]
) -> Union[
    Tuple[CompiledSDFG, torch.Tensor],  # No backward
    Tuple[CompiledSDFG, torch.Tensor,   # With backward
          CompiledSDFG, torch.Tensor]
]:
```

**Process**:
1. Compile the forward SDFG
2. Construct arguments from dummy inputs and parameters
3. Infer symbols from input shapes
4. Allocate forwarded transients (for backward pass)
5. Initialize the forward SDFG state
6. Extract the state handle as `torch.tensor([libhandle])`
7. If backward is enabled:
   - Compile the backward SDFG
   - Allocate gradient buffers
   - Initialize the backward SDFG state
   - Extract the backward handle
8. Return the compiled SDFGs and handles

#### get_arglist()

**Function**:
```python
def get_arglist(module: DaceModule) -> Tuple[List[str], List[str]]:
    """Extracts input and output names with ONNX name cleaning."""
    inputs = [clean_onnx_name(name) for name in module.dace_model.inputs]
    outputs = [clean_onnx_name(name) for name in module.dace_model.outputs]
    return inputs, outputs
```

---

### 4.4 PyTorch Environment Configuration

**Location**: [environments/pytorch_env.py](environments/pytorch_env.py)

Defines the CMake build configuration for linking against PyTorch libraries.

#### PyTorch Environment (CPU)

```python
@dace.library.environment
class PyTorch:
    """Environment for building PyTorch C++ operators (CPU)."""

    cmake_includes = torch.utils.cpp_extension.include_paths()

    @staticmethod
    def cmake_libraries():
        """Locate and return PyTorch library paths."""
        library_names = ["c10", "torch", "torch_cpu", "torch_python"]
        # Search in torch.utils.cpp_extension.library_paths()
        return library_paths

    cmake_compile_flags = ["-D_GLIBCXX_USE_CXX11_ABI=0"]  # ABI compatibility
```

#### PyTorchCUDA Environment (GPU)

```python
@dace.library.environment
class PyTorchCUDA:
    """Environment for building PyTorch C++ operators (CUDA)."""

    cmake_includes = torch.utils.cpp_extension.include_paths()

    @staticmethod
    def cmake_libraries():
        """Locate and return PyTorch CUDA library paths."""
        library_names = ["c10", "torch", "torch_cpu", "torch_cuda",
                         "torch_python", "c10_cuda"]
        return library_paths + ["cudart"]
```

**Integration with DaCe**:
- Registered via the `@dace.library.environment` decorator
- DaCe's CMake generator uses these settings for linker configuration
- Ensures that compiled code can call the PyTorch C++ API

---

## 5. Dispatcher Strategies

### 5.1 Why Two Dispatchers?

The library provides two dispatcher implementations to handle different use cases:

| Feature | C++ Extension | CTypes Module |
|---------|--------------|---------------|
| **Performance** | High (native call) | Good (small overhead) |
| **Parameter Limit** | 64 parameters | Unlimited |
| **Compilation Time** | Slower (C++ compile) | Faster (no codegen) |
| **Registration** | `torch.ops.dace_name` | Python function |

### 5.2 C++ PyTorch Extension

**Location**: [dispatchers/cpp_torch_extension.py](dispatchers/cpp_torch_extension.py)

#### Overview

Generates C++ code that registers a custom PyTorch operator with native autograd support.

#### Type Conversion Utilities

**DaCe → PyTorch C++ Types**:
```python
_REPLACED_CTYPES = {
    dace.int64: "int64_t",
    dace.uint64: "uint64_t",
    dace.float16: "at::Half"
}

def torch_ctype(dtype: dace.typeclass) -> str:
    """Convert DaCe type to PyTorch C++ type string."""
    if isinstance(dtype, dace.pointer):
        return "int64_t"
    elif dtype in _REPLACED_CTYPES:
        return _REPLACED_CTYPES[dtype]
    else:
        return dtype.ctype  # e.g., "float", "double"
```

**DaCe → PyTorch Tensor Dtype**:
```python
_TYPECLASS_TO_TORCH_DTYPE_STR = {
    dt.bool: "kBool",
    dt.int8: "kInt8",
    dt.float32: "kFloat32",
    dt.float64: "kFloat64",
    # ... complete mapping
}
```

#### Tensor Initialization Code Generation

**Function**: `tensor_init_for_desc()`

**Purpose**: Generates C++ code to allocate PyTorch tensors

**Approach**:
- Checks if tensor is a constant (from weights)
- If constant: embeds values as a C++ initializer list
- If output: allocates with `torch::zeros()` or `torch::empty()`
- Sets proper dtype, device (CPU/CUDA), and layout

**Example Output**:
```cpp
Tensor output = torch::zeros(
    {10, 256},
    torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU)
        .layout(torch::kStrided)
);
```

#### Forward Function Code Generation

**Generated Structure**:
```cpp
Tensor forward_function(
    int64_t fwd_handle_ptr,
    int64_t bwd_handle_ptr,  // if backward
    const Tensor& input_0_,
    const Tensor& input_1_,
    // ... more inputs
) {
    // 1. Initialize outputs
    Tensor output = torch::zeros({...}, torch::TensorOptions()...);

    // 2. Ensure inputs are contiguous
    Tensor input_0 = input_0_.contiguous();

    // 3. Extract pointers
    float *input_0_ptr = reinterpret_cast<float*>(input_0.data_ptr<float>());
    float *output_ptr = reinterpret_cast<float*>(output.data_ptr<float>());

    // 4. Call SDFG
    MySDFGHandle_t handle = reinterpret_cast<MySDFGHandle_t>(fwd_handle_ptr);
    __program_my_sdfg(handle, input_0_ptr, output_ptr);

    // 5. Return outputs
    return output;  // or std::make_tuple(...) for multiple
}
```

#### Autograd Function Code Generation

**Generated Structure**:
```cpp
class MySDFGFunction : public torch::autograd::Function<MySDFGFunction> {
public:
    static Tensor forward(
        AutogradContext *ctx,
        int64_t fwd_handle_ptr,
        int64_t bwd_handle_ptr,
        const Tensor& input_
    ) {
        // Run forward pass
        Tensor output = forward_function(fwd_handle_ptr, bwd_handle_ptr, input_);

        // Save for backward
        ctx->save_for_backward({input_, output});

        // Save non-I/O transients
        ctx->saved_data["intermediate"] = intermediate_value;

        // Save backward handle
        ctx->saved_data["bwd_handle"] = bwd_handle_ptr;

        return output;
    }

    static tensor_list backward(
        AutogradContext *ctx,
        tensor_list grad_outputs
    ) {
        // 1. Recover saved tensors
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto intermediate = ctx->saved_data["intermediate"].toTensor();

        // 2. Get backward handle
        int64_t bwd_handle_ptr = ctx->saved_data["bwd_handle"].toInt();
        MySDFGBackwardHandle_t bwd_handle =
            reinterpret_cast<MySDFGBackwardHandle_t>(bwd_handle_ptr);

        // 3. Allocate gradient buffers
        Tensor grad_input = torch::zeros({...});  // or empty if zero_init=False

        // 4. Get output gradients
        Tensor grad_output = grad_outputs[0].contiguous();

        // 5. Call backward SDFG
        __program_my_sdfg_backward(
            bwd_handle,
            grad_output.data_ptr<float>(),
            input.data_ptr<float>(),
            intermediate.data_ptr<float>(),
            grad_input.data_ptr<float>()
        );

        // 6. Return gradients (None for non-differentiable)
        return {Tensor(), Tensor(), grad_input};  // None for handles, grad for input
    }
};
```

#### Operator Registration

**Generated Code**:
```cpp
// Register operator
TORCH_LIBRARY(dace_my_sdfg, m) {
    m.def("my_sdfg", forward_function);
}

// Register autograd if backward enabled
TORCH_LIBRARY_IMPL(dace_my_sdfg, Autograd, m) {
    m.impl("my_sdfg", MySDFGFunction::apply);
}
```

#### Compilation Process

**Function**: `register_and_compile_torch_extension()`

**Steps**:
1. Generate the complete C++ source code
2. Write to a temporary file
3. Use `torch.utils.cpp_extension.load()` for JIT compilation
4. Link against:
   - PyTorch libraries (from environment)
   - Compiled SDFG shared library
5. Return the operator accessible via `torch.ops.dace_name.name`

**Limitations**:
- PyTorch dispatcher has **64 parameter limit**
- Longer compilation time (~seconds)
- Requires C++ compiler

---

### 5.3 CTypes Module

**Location**: [dispatchers/ctypes_module.py](dispatchers/ctypes_module.py)

#### Overview

A pure Python dispatcher that calls compiled SDFGs via ctypes, avoiding C++ code generation.

#### When to Use

- Models with >64 parameters
- Rapid development/iteration
- Environments where C++ compilation is problematic
- Prototyping and debugging

#### Forward-Only Callable

**Function**: `callable_for_fwd_module()`

**Generated Function**:
```python
def forward(*inputs):
    kwargs = {}

    # Set inputs
    for i, input_name in enumerate(input_names):
        kwargs[input_name] = inputs[i].contiguous()

    # Initialize outputs
    for name in output_names:
        kwargs[name] = create_output_array(
            {},
            forward_compiled.sdfg.arrays[name],
            use_torch=True,
            zeros=False
        )

    # Add constants
    kwargs.update(constants)

    # Call SDFG (ctypes handles conversion)
    return forward_compiled(**kwargs)
```

#### Forward+Backward Callable

**Function**: `callable_for_bwd_module()`

**Generated Autograd Function**:
```python
class DifferentiableFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        kwargs = {}

        # Set inputs
        for i, input_name in enumerate(input_names):
            kwargs[input_name] = inputs[i].contiguous()

        # Initialize outputs + forwarded transients
        for name in outputs_and_forwarded:
            kwargs[name] = create_output_array(...)

        # Call forward SDFG
        outputs = forward_compiled(**kwargs, **constants)

        # Save I/O tensors for backward
        ctx.save_for_backward(*(kwargs[name] for name in forwarded_io_names))

        # Save non-I/O transients as attributes
        for name in forwarded_non_io_names:
            setattr(ctx, f"dace_saved_{name}", kwargs[name])

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        kwargs = {}

        # Recover saved I/O tensors
        saved = ctx.saved_tensors
        for name, val in zip(forwarded_io_names, saved):
            kwargs[name] = val

        # Recover non-I/O transients
        for name in forwarded_non_io_names:
            kwargs[name] = getattr(ctx, f"dace_saved_{name}")

        # Allocate gradient buffers
        for grad_name, zero_init, desc in gradient_descriptors:
            kwargs[grad_name] = create_output_array(..., zeros=zero_init)

        # Set output gradients from PyTorch
        for grad_name, grad_val in zip(output_gradient_names, grad_outputs):
            kwargs[grad_name] = grad_val.contiguous()

        # Call backward SDFG
        backward_compiled(**kwargs)

        # Return input gradients (None for non-differentiable)
        return tuple(kwargs.get(grad_name) for grad_name in input_gradient_names)

return DifferentiableFunction.apply
```

#### Parameter Handling

**Function**: `init_remaining_parameters()`

**Purpose**: Extracts constant parameters (model weights) that are neither inputs nor outputs

**Process**:
1. Identify parameters not in the input/output lists
2. Verify they exist in `module.dace_model.clean_weights`
3. Transfer to CUDA if needed
4. Return as a constants dictionary

---

## 6. Integration Pipeline

### 6.1 Complete Workflow

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Initialization                                 │
├─────────────────────────────────────────────────────────┤
│ dace_module = DaceModule(model, dummy_inputs, ...)      │
│                                                         │
│ 1. Store PyTorch model reference                        │
│ 2. Store configuration (cuda, backward, dispatcher)     │
│ 3. Set function = None (lazy compilation)               │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: First Forward Call                             │
├─────────────────────────────────────────────────────────┤
│ output = dace_module(actual_input)                      │
│                                                         │
│ Detect function is None → Trigger _initialize_sdfg()    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: ONNX Export                                    │
├─────────────────────────────────────────────────────────┤
│ 1. Call torch.onnx.export(model, dummy_inputs, ...)     │
│ 2. Save exported ONNX ModelProto                        │
│ 3. Extract and save model parameters                    │
│ 4. Remove initializers that overlap with inputs         │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: ONNX → DaCe SDFG                               │
├─────────────────────────────────────────────────────────┤
│ 1. Create ONNXModel(onnx_proto)                         │
│    - Import ONNX graph to SDFG                          │
│    - Run shape inference                                │
│    - Apply simplifications                              │
│ 2. Store forward SDFG as module.sdfg                    │
│ 3. Apply post_onnx_hooks (if any)                       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 5: Backward SDFG Generation (if backward=True)    │
├─────────────────────────────────────────────────────────┤
│ 1. Determine required gradients:                        │
│    - Model inputs (if not in clean_weights)             │
│    - Parameters with requires_grad=True                 │
│                                                         │
│ 2. Call make_backward_function():                       │
│    a. Create backward SDFG                              │
│    b. Initialize BackwardPassGenerator                  │
│    c. Generate reverse operations                       │
│    d. Identify forwarded transients                     │
│                                                         │
│ 3. Modify forward SDFG:                                 │
│    - Make forwarded arrays non-transient (outputs)      │
│    - Convert scalars to size-1 arrays                   │
│                                                         │
│ 4. Store:                                               │
│    - module.forward_sdfg                                │
│    - module.backward_sdfg                               │
│    - module._ad_result (BackwardResult)                 │
│    - module._ad_inp_arrs (forwarded arrays)             │
│                                                         │
│ 5. Apply post_autodiff_hooks (if any)                   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 6: SDFG Compilation                               │
├─────────────────────────────────────────────────────────┤
│ Call compile_and_init_sdfgs(module, dummy_inputs):      │
│                                                         │
│ 1. Compile forward SDFG → forward_compiled              │
│ 2. Construct arguments from dummy inputs + parameters   │
│ 3. Call _call_args() to infer symbols                   │
│ 4. Allocate forwarded transients (if backward)          │
│ 5. Initialize forward SDFG state                        │
│ 6. Extract state handle: fwd_handle=compiled._libhandle │
│                                                         │
│ 7. If backward:                                         │
│    a. Compile backward SDFG → backward_compiled         │
│    b. Allocate gradient buffers                         │
│    c. Initialize backward SDFG state                    │
│    d. Extract backward handle                           │
│                                                         │
│ 8. Apply post_compile_hooks (if any)                    │
│                                                         │
│ 9. Return compiled SDFGs and handles                    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 7: Dispatcher Generation                          │
├─────────────────────────────────────────────────────────┤
│ If compile_torch_extension:                             │
│   ├─→ register_and_compile_torch_extension()            │
│   │   1. Generate C++ code with autograd                │
│   │   2. Compile as PyTorch extension                   │
│   │   3. Register operator                              │
│   │   4. Return torch.ops.dace_name.name                │
│   │                                                     │
│ Else:                                                   │
│   └─→ get_ctypes_dispatcher()                           │
│       1. Create Python autograd.Function                │
│       2. Wrap compiled SDFGs with ctypes calls          │
│       3. Return callable                                │
│                                                         │
│ Return DaCeMLTorchFunction(function,compiled_sdfgs,ptrs)│
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 8: Wrapper Function Creation                      │
├─────────────────────────────────────────────────────────┤
│ Create forward() wrapper:                               │
│                                                         │
│ def forward(*args):                                     │
│     return compiled_function.function(                  │
│         *compiled_function.ptr,     # State handles     │
│         *args,                      # User inputs       │
│         *parameters_to_pass)        # Model params      │
│                                                         │
│ Store as module.function                                │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 9: Execution                                      │
├─────────────────────────────────────────────────────────┤
│ Forward Pass:                                           │
│ 1. User calls dace_module(input)                        │
│ 2. Wrapper extracts .contiguous() tensors               │
│ 3. Zero-copy access via DLPack (if needed)              │
│ 4. Call compiled SDFG with pointers                     │
│ 5. Return PyTorch tensors                               │
│                                                         │
│ Backward Pass (if backward=True):                       │
│ 1. User calls loss.backward()                           │
│ 2. PyTorch autograd calls backward function             │
│ 3. Recover saved tensors from context                   │
│ 4. Allocate gradient buffers                            │
│ 5. Call backward SDFG                                   │
│ 6. Return input gradients to PyTorch                    │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Data Transformations

**Input Transformation** (PyTorch → DaCe):
```
torch.Tensor (user input)
    ↓ .contiguous()
torch.Tensor (contiguous memory)
    ↓ .data_ptr<T>() (C++ extension) or direct pass (CTypes)
Raw pointer / PyTorch tensor
    ↓ Passed to SDFG
SDFG operates on memory
```

**Output Transformation** (DaCe → PyTorch):
```
Allocate torch.Tensor (zeros or empty)
    ↓ Extract .data_ptr<T>()
Raw pointer
    ↓ Pass to SDFG as output parameter
SDFG fills memory
    ↓ No copy needed
Return torch.Tensor (already owns memory)
```

**Constant Transformation**:
```
PyTorch model parameters
    ↓ Extract in ONNX export
ONNX initializers
    ↓ Save as clean_weights
Embed in C++ (C++ extension) or pass as kwargs (CTypes)
```

---

## 7. Zero-Copy Tensor Sharing

### 7.1 The DLPack Protocol

**Purpose**: Industry-standard protocol for zero-copy tensor exchange between frameworks

**Key Concept**: Shares memory pointers and metadata between frameworks without copying data

### 7.2 DaCe → PyTorch Conversion

**Function**: `array_to_torch_tensor(ptr, desc)`

**Complete Process**:

**Step 1: Device Mapping**
```python
if desc.storage == dtypes.StorageType.GPU_Global:
    device_type = DLDeviceType.kDLGPU
elif desc.storage in [StorageType.CPU_Heap, StorageType.Default]:
    device_type = DLDeviceType.kDLCPU
```

**Step 2: Type Conversion**
```python
dtype = dace_to_dldtype_dict[desc.dtype]
# e.g., dace.float32 → DLDataType(kDLFloat, 32, 1)
```

**Step 3: Shape and Strides**
```python
shape = (ctypes.c_int64 * len(desc.shape))(*desc.shape)
strides = (ctypes.c_int64 * len(desc.shape))(*desc.strides)
```

**Step 4: DLTensor Construction**
```python
dltensor = DLTensor(
    data=ptr,           # Raw pointer from DaCe
    ctx=DLContext(device_type, device_id=0),
    ndim=len(desc.shape),
    dtype=dtype,
    shape=shape,
    strides=strides,
    byte_offset=0
)
```

**Step 5: Managed Tensor Wrapper**
```python
managed = DLManagedTensor(
    dl_tensor=dltensor,
    manager_ctx=None,
    deleter=no_op_deleter  # DaCe owns memory
)
```

**Step 6: PyCapsule Creation**
```python
capsule = PyCapsule.New(
    ctypes.byref(managed),
    b"dltensor",
    None
)
```

**Step 7: PyTorch Conversion**
```python
tensor = torch.utils.dlpack.from_dlpack(capsule)
tensor._dace_dlpack = managed  # Prevent GC
```

### 7.3 Memory Lifecycle

**Ownership**:
- The DaCe SDFG state struct owns the memory
- PyTorch tensor is a **view** that shares the memory
- No-op deleter ensures that DaCe handles deallocation

**Safety**:
- Keep the SDFG state alive as long as tensors exist
- State handles are stored as `torch.Tensor` objects (ref-counted)
- PyTorch's garbage collector won't free memory prematurely

**Use Cases**:
- Return DaCe outputs as PyTorch tensors
- Access intermediate SDFG arrays from PyTorch
- Enable PyTorch operations on DaCe memory

---

## 8. Autograd Integration

### 8.1 Backward Pass Generation

**Entry Point**: `make_backward_function()` (in `dace/autodiff/torch.py`)

**Workflow**:

**Step 1: Determine Required Gradients**
```python
required_grads = []
for param_name in model.parameters():
    if param_name.requires_grad and param_name not in inputs:
        required_grads.append(param_name)
```

**Step 2: Create Backward SDFG**
```python
generator = BackwardPassGenerator(
    forward_sdfg,
    backward_sdfg,
    given_gradients=model_outputs,
    required_gradients=model_inputs + required_params
)
backward_result = generator.backward()
```

**Step 3: Identify Forwarded Transients**
- Identifies values needed for gradient computation
- Example: For `y = x * w`, the backward pass needs both `x` and `w`
- These are marked as non-transient (outputs) in the forward SDFG

**Step 4: Modify Forward SDFG**
- Makes forwarded arrays non-transient
- Converts scalar outputs to size-1 arrays
- Ensures proper storage types

### 8.2 C++ Extension Autograd

**Forward Method**:
```cpp
static Tensor forward(AutogradContext *ctx, int64_t fwd_handle,
                     int64_t bwd_handle, Tensor input) {
    // Execute forward SDFG
    Tensor output = forward_function(fwd_handle, bwd_handle, input);

    // Save I/O tensors
    ctx->save_for_backward({input, output});

    // Save non-I/O transients (not saved by PyTorch)
    ctx->saved_data["intermediate"] = intermediate_value;

    // Save backward handle
    ctx->saved_data["bwd_handle"] = bwd_handle;

    return output;
}
```

**Backward Method**:
```cpp
static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    // 1. Recover saved tensors
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto intermediate = ctx->saved_data["intermediate"].toTensor();

    // 2. Get handles
    int64_t bwd_handle = ctx->saved_data["bwd_handle"].toInt();

    // 3. Allocate gradient buffers
    Tensor grad_input = torch::zeros({...});  // If zero_init=True
    // OR
    Tensor grad_input = torch::empty({...});  // If zero_init=False

    // 4. Get output gradients
    Tensor grad_output = grad_outputs[0].contiguous();

    // 5. Call backward SDFG
    __program_backward(
        bwd_handle,
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        grad_input.data_ptr<float>()
    );

    // 6. Return gradients (None for handles, grads for inputs)
    return {Tensor(), Tensor(), grad_input};
}
```

### 8.3 CTypes Autograd

**Forward Method**:
```python
@staticmethod
def forward(ctx, *inputs):
    kwargs = {}

    # Set inputs
    for i, name in enumerate(input_names):
        kwargs[name] = inputs[i].contiguous()

    # Allocate outputs + forwarded transients
    for name in all_output_names:
        kwargs[name] = create_output_array(...)

    # Call forward SDFG
    forward_compiled(**kwargs, **constants)

    # Save I/O for backward
    ctx.save_for_backward(*(kwargs[n] for n in forwarded_io_names))

    # Save non-I/O transients as attributes
    for name in forwarded_non_io_names:
        setattr(ctx, f"dace_saved_{name}", kwargs[name])

    return tuple(kwargs[n] for n in model_output_names)
```

**Backward Method**:
```python
@staticmethod
def backward(ctx, *grad_outputs):
    kwargs = {}

    # Recover I/O tensors
    saved = ctx.saved_tensors
    for name, val in zip(forwarded_io_names, saved):
        kwargs[name] = val

    # Recover non-I/O transients
    for name in forwarded_non_io_names:
        kwargs[name] = getattr(ctx, f"dace_saved_{name}")

    # Allocate gradient buffers
    for grad_name, zero_init in gradient_specs:
        kwargs[grad_name] = create_output_array(..., zeros=zero_init)

    # Set output gradients
    for grad_name, grad_val in zip(out_grad_names, grad_outputs):
        kwargs[grad_name] = grad_val.contiguous()

    # Call backward SDFG
    backward_compiled(**kwargs)

    # Return input gradients
    return tuple(kwargs.get(g) for g in input_grad_names)
```

### 8.4 Gradient Accumulation

**BackwardResult Structure**:
```python
required_grad_names = {
    "input_0": "grad_input_0",
    "param_weight": "grad_param_weight"
}

given_grad_names = {
    "output": "grad_output"
}

zero_init = {
    "grad_input_0": True,    # Initialize to zero
    "grad_param_weight": False  # Don't initialize (accumulate)
}
```

**Usage**:
- `zero_init=True`: First gradient computation (allocate and initialize to zeros)
- `zero_init=False`: Accumulate into existing buffer (for gradient accumulation)

---
