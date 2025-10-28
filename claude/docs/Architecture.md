# Modular Platform Architecture

**A comprehensive guide to understanding the Modular/Mojo stack**

This document provides a detailed, pedagogical walkthrough of the Modular Platform architecture, including the MAX inference framework and the Mojo programming language. It traces data flows, compilation pipelines, and component interactions with annotated code references.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [MAX Component Architecture](#2-max-component-architecture)
3. [Mojo Compiler Pipeline](#3-mojo-compiler-pipeline)
4. [Mojo-MLIR Interoperability](#4-mojo-mlir-interoperability)
5. [Mojo-Python Interoperability](#5-mojo-python-interoperability)
6. [Data Flow Examples](#6-data-flow-examples)
7. [Build System](#7-build-system)

---

## 1. System Overview

The Modular Platform consists of two main components:

- **MAX**: A high-performance AI inference framework with OpenAI-compatible APIs
- **Mojo**: A systems programming language that bridges Python ergonomics with C/C++ performance

### High-Level Stack Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐ │
│  │  max serve     │  │  Python Scripts │  │  Mojo Programs   │ │
│  │  (HTTP API)    │  │  (max.pipelines)│  │  (.mojo files)   │ │
│  └────────┬───────┘  └────────┬────────┘  └─────────┬────────┘ │
└───────────┼──────────────────┼───────────────────────┼──────────┘
            │                  │                       │
            ▼                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Python High-Level APIs                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ max.nn   │◄─┤max.graph │◄─┤max.engine│◄─┤ max.driver   │   │
│  │(NN Ops)  │  │(Graph IR)│  │(Compile) │  │(Device Mgmt) │   │
│  └──────────┘  └──────────┘  └────┬─────┘  └──────────────┘   │
└────────────────────────────────────┼──────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Python-Native Bridge (C++/Mojo)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    max._core                              │  │
│  │  (MLIR Types, Operations, Device/Tensor ABIs)            │  │
│  └───────────────────────────┬──────────────────────────────┘  │
└────────────────────────────────┼──────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│            Compiled Mojo/C++ Layer (Native Code)                │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ max.kernels     │  │max._core_mojo│  │ max.compiler     │  │
│  │ (Mojo Kernels)  │  │(Mojo-Py FFI)│  │ (Compiler Utils) │  │
│  └────────┬────────┘  └──────────────┘  └──────────────────┘  │
└───────────┼─────────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │   Hardware    │
    │ CPU/GPU/TPU   │
    └───────────────┘
```

### Repository Structure

```
modular/
├── mojo/                      # Mojo language implementation
│   ├── stdlib/                # Standard library (OPEN SOURCE)
│   │   ├── stdlib/            # Library source code
│   │   ├── test/              # Unit tests
│   │   └── benchmarks/        # Performance benchmarks
│   ├── docs/                  # User documentation
│   ├── proposals/             # Language RFCs
│   └── integration-test/      # Integration tests
├── max/                       # MAX inference framework
│   ├── _core/                 # Core C++/Mojo bridge (type stubs)
│   ├── _core_mojo/            # Mojo-Python bindings
│   ├── kernels/               # High-performance Mojo kernels
│   ├── graph/                 # Graph construction API (Python)
│   ├── engine/                # Compilation & execution (Python)
│   ├── driver/                # Device management (Python)
│   ├── serve/                 # HTTP inference server (Python)
│   ├── pipelines/             # Model architectures (Python)
│   ├── nn/                    # Neural network operators (Python)
│   └── compiler/              # Compiler utilities (Mojo)
├── bazel/                     # Build system configuration
└── examples/                  # Usage examples
```

---

## 2. MAX Component Architecture

MAX is organized into distinct layers, from high-level Python APIs down to hardware-optimized Mojo kernels.

### 2.1 max._core - Core Runtime Bridge

**Location:** [max/_core/](../max/_core/)

**Purpose:** Provides Python type stubs (`.pyi` files) for core functionality implemented in compiled C++/Mojo. This is the lowest-level interface between Python and native code.

#### Key Files

| File | Purpose | Key Types/Functions |
|------|---------|-------------------|
| [__init__.pyi](../max/_core/__init__.pyi) | Core MLIR types | `Attribute`, `Type`, `Value`, `Operation`, `Block`, `Region` |
| [driver.pyi](../max/_core/driver.pyi) | Device & tensor ops | `Device`, `Accelerator`, `CPU`, `Tensor`, `DeviceStream` |
| [engine.pyi](../max/_core/engine.pyi) | Model execution | `Model`, `InferenceSession`, `TensorSpec`, `MojoValue` |
| [graph.pyi](../max/_core/graph.pyi) | Graph utilities | `Analysis`, `load_modular_dialects()`, dtype conversions |
| [dtype.pyi](../max/_core/dtype.pyi) | Data type definitions | `DType` and related types |

#### MLIR Dialect Definitions

[max/_core/dialects/](../max/_core/dialects/) contains type stubs for MLIR dialects:

- `mo.pyi` - Modular Operations dialect
- `builtin.pyi` - MLIR builtin dialect
- `kgen.pyi` - Code generation dialect
- `rmo.pyi` - Runtime Modular Operations
- `m.pyi`, `mosh.pyi` - Additional internal dialects

**Dependencies:** None (this is the foundation layer)

---

### 2.2 max._core_mojo - Mojo-Python FFI

**Location:** [max/_core_mojo/](../max/_core_mojo/)

**Purpose:** Provides specific Mojo functions exposed as Python modules for performance-critical operations.

#### Key Implementation

[mojo_module.mojo](../max/_core_mojo/mojo_module.mojo) - Block hasher implementation:

```mojo
from python import PythonObject
from python.bindings import PythonModuleBuilder

@export
fn PyInit__core_mojo() -> PythonObject:
    var m = PythonModuleBuilder("_core_mojo")
    m.def_py_function[mojo_block_hasher]("mojo_block_hasher")
    return m.finalize()

fn mojo_block_hasher(
    py_array: PythonObject,
    ...
) raises -> PythonObject:
    # High-performance token hashing for KV cache
    # Direct memory access to numpy arrays (zero-copy)
    ...
```

**Purpose:** Optimized Mojo implementations for operations like block hashing used in KV cache management.

**Dependencies:**
- [mojo/stdlib/stdlib/python/](../mojo/stdlib/stdlib/python/) - Python interop
- CPython C API

---

### 2.3 max.kernels - High-Performance Compute Kernels

**Location:** [max/kernels/](../max/kernels/)

**Purpose:** Low-level, hardware-optimized compute kernels written in Mojo. These are the building blocks for all numerical operations in MAX.

#### Directory Structure

```
max/kernels/
├── src/                       # Kernel implementations
│   ├── linalg/                # Linear algebra (GEMM, GEMV, matmul)
│   │   ├── matmul/            # Matrix multiplication kernels
│   │   ├── gemm/              # General matrix multiply
│   │   └── gemv/              # Matrix-vector multiply
│   ├── nn/                    # Neural network operations
│   │   ├── attention/         # Attention mechanisms
│   │   │   ├── gpu/           # GPU implementations
│   │   │   └── amd/           # AMD-specific optimizations
│   │   ├── conv/              # Convolutions
│   │   ├── pooling/           # Pooling operations
│   │   └── activations/       # Activation functions
│   ├── quantization/          # Quantized operations
│   ├── kv_cache/              # Key-value cache implementations
│   ├── layout/                # Memory layout utilities
│   ├── register/              # Register-level operations
│   ├── comm/                  # Communication primitives (NCCL, etc.)
│   ├── shmem/                 # Shared memory utilities
│   ├── weights_registry/      # Weight management
│   ├── _cublas/               # NVIDIA cuBLAS bindings
│   ├── _cudnn/                # NVIDIA cuDNN bindings
│   ├── _rocblas/              # AMD ROCm BLAS bindings
│   └── extensibility/         # Custom kernel API
├── test/                      # Unit tests (mirrors src/)
└── benchmarks/                # Performance benchmarks
    ├── gpu/                   # GPU benchmarks
    ├── linalg/                # Linear algebra benchmarks
    └── autotune/              # Auto-tuning tools
```

#### Example: Matrix Multiplication Kernel

[max/kernels/src/linalg/matmul/](../max/kernels/src/linalg/matmul/) contains multiple implementations:

- Generic CPU implementation with SIMD
- GPU implementations (CUDA/ROCm)
- Vendor library wrappers (cuBLAS, ROCm BLAS)
- Platform-specific dispatch tables

**Key Concept: Dispatch Tables**

Kernels use dispatch tables for architecture-specific optimizations:

```mojo
# Platform-specific implementations selected at runtime
from .dispatch_table_a100_gpu import matmul as matmul_a100
from .dispatch_table_amd import matmul as matmul_amd

fn matmul(A, B, C, device):
    if device.is_nvidia_a100():
        matmul_a100(A, B, C)
    elif device.is_amd_mi300():
        matmul_amd(A, B, C)
    else:
        matmul_generic(A, B, C)
```

**Build Commands:**

```bash
# Build all kernels
./bazelw build //max/kernels/...

# Build specific kernel
./bazelw build //max/kernels/src/linalg:linalg

# Run benchmarks
./bazelw run //max/kernels/benchmarks/gpu:bench_matmul -- \
    env_get_int[M]=1024 env_get_int[N]=1024 env_get_int[K]=1024

# Run tests
./bazelw test //max/kernels/test/linalg:test_matmul
```

**Dependencies:**
- Mojo standard library
- Vendor libraries (cuBLAS, cuDNN, ROCm)
- CUDA/ROCm toolchains

---

### 2.4 max.graph - Graph Construction API

**Location:** [max/graph/](../max/graph/)

**Purpose:** Python API for building dataflow computation graphs. This is the high-level programming model for defining neural network models.

#### Key Files (73 Python files)

| File | Purpose | Key Classes |
|------|---------|-------------|
| [graph.py](../max/graph/graph.py) | Core graph builder | `Graph`, `KernelLibrary` |
| [value.py](../max/graph/value.py) | Value types | `TensorValue`, `BufferValue`, `_ChainValue` |
| [type.py](../max/graph/type.py) | Type system | `TensorType`, `BufferType`, `Shape`, `DeviceKind` |
| [weight.py](../max/graph/weight.py) | Weight management | `Weight`, sharding strategies |
| [shape.py](../max/graph/shape.py) | Shape inference | Symbolic dimensions, broadcasting |
| [dim.py](../max/graph/dim.py) | Dimension types | `StaticDim`, `SymbolicDim`, `AlgebraicDim` |

#### Operations Directory

[max/graph/ops/](../max/graph/ops/) - 50+ operation files:

- `elementwise.py` - Element-wise operations (add, mul, relu, etc.)
- `matmul.py` - Matrix multiplication
- `conv.py` - Convolution operations
- `attention.py` - Attention mechanisms
- `quantize.py` - Quantization/dequantization
- `custom.py` - Custom operation integration
- ... many more

#### Usage Pattern

```python
from max.graph import Graph, TensorType, ops
from max.graph.weight import Weight

# Define input types
input_type = TensorType(shape=[1, 512], dtype=DType.float32)
weight_type = TensorType(shape=[512, 768], dtype=DType.float32)

# Build graph
with Graph("linear_layer", input_types=[input_type]) as graph:
    # Get input value
    x = graph.inputs[0]

    # Load weight
    weight = Weight(weight_type, name="weight")
    W = graph.constant(weight)

    # Matrix multiplication
    result = ops.matmul(x, W)

    # Activation
    output = ops.relu(result)

    # Mark output
    graph.output(output)
```

**How it works:**

1. Operations return `TensorValue` objects (symbolic, not concrete values)
2. Each operation adds nodes to the graph's internal MLIR representation
3. The graph captures the computation structure, not the data
4. Compilation (via `max.engine`) converts this to executable code

#### Custom Kernels via KernelLibrary

```python
from max.graph import Graph, KernelLibrary

# Load custom Mojo kernels
kernel_lib = KernelLibrary.from_path("custom_ops.mojopkg")

with Graph("custom_graph", kernel_library=kernel_lib) as graph:
    # Custom operations are now available
    result = ops.custom.my_custom_op(input_tensor)
    graph.output(result)
```

**Dependencies:**
- `max._core` - MLIR types and operations
- `max.driver` - Device references

---

### 2.5 max.driver - Device & Tensor Management

**Location:** [max/driver/](../max/driver/)

**Purpose:** High-level Python API for device management and tensor operations. Wraps `max._core.driver`.

#### Key Files

| File | Purpose |
|------|---------|
| [driver.py](../max/driver/driver.py) | Device abstractions |
| [tensor.py](../max/driver/tensor.py) | Tensor utilities |

#### Key Classes

**Device Management:**

```python
from max.driver import Device, CPU, Accelerator, DeviceSpec

# Query available devices
num_gpus = accelerator_count()

# Create device specifications
devices = [
    CPU(),
    Accelerator(0),  # GPU 0
    Accelerator(1),  # GPU 1
]

# Initialize devices
device_handles = load_devices(devices)
```

**Tensor Operations:**

```python
from max.driver import Tensor
import numpy as np

# Create tensor from numpy (host memory)
np_array = np.array([1, 2, 3], dtype=np.float32)
host_tensor = Tensor.from_numpy(np_array)

# Transfer to device
device_tensor = host_tensor.to_device(Accelerator(0))

# Transfer back to host
result = device_tensor.to_host()
result_np = result.to_numpy()
```

**Device Streams (Async Operations):**

```python
from max.driver import DeviceStream

# Create stream for async execution
stream = DeviceStream(Accelerator(0))

# Launch operations asynchronously
stream.copy_async(src_tensor, dst_tensor)
stream.sync()  # Wait for completion
```

**Dependencies:**
- `max._core.driver` - Core tensor and device implementations
- `max._core_types.driver.DLPackArray` - DLPack protocol for zero-copy numpy interop

---

### 2.6 max.engine - Model Compilation & Execution

**Location:** [max/engine/](../max/engine/)

**Purpose:** Compiles graphs into executable models and manages inference execution.

#### Key Files

[api.py](../max/engine/api.py) - Main compilation and execution API

#### Key Classes

**InferenceSession - The Compilation Context:**

```python
from max.engine import InferenceSession
from max.driver import CPU, Accelerator

# Create session with devices
session = InferenceSession()

# Configure compilation
session.set_mojo_define("M", 1024)  # Set compile-time parameters
session.set_mojo_define("USE_TENSOR_CORES", True)

# Compile from graph object
model = session.compile_from_object(
    graph,
    devices=[CPU(), Accelerator(0)]
)

# Or compile from saved graph
model = session.compile_from_path("model.mogg")
```

**Model - The Executable:**

```python
from max.driver import Tensor

# Get input/output specifications
print(model.input_metadata)   # List[TensorSpec]
print(model.output_metadata)  # List[TensorSpec]
print(model.devices)          # Device placement info

# Execute model
input_tensors = [Tensor.from_numpy(input_data)]
outputs = model.execute(*input_tensors)

# Or use __call__ syntax
outputs = model(input_tensors[0])
```

**TensorSpec - Tensor Metadata:**

```python
class TensorSpec:
    name: str
    dtype: DType
    shape: List[int]
    device: Device
```

#### Compilation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Input: max.graph.Graph (Python object)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Lower to MLIR (mo/kgen dialects)                         │
│    - Graph ops → MLIR operations                            │
│    - Shape inference                                         │
│    - Type checking                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Apply Optimization Passes                                │
│    - Constant folding                                        │
│    - Operator fusion                                         │
│    - Layout optimization                                     │
│    - Memory planning                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Lower to Mojo Kernels                                    │
│    - Graph ops → Kernel calls                               │
│    - Insert synchronization                                  │
│    - Device placement                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Link Custom Extensions (.mojopkg)                        │
│    - Load custom kernel libraries                           │
│    - Resolve symbols                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Code Generation                                          │
│    - Generate runtime code                                   │
│    - Create execution plan                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Output: max.engine.Model (executable)                    │
└─────────────────────────────────────────────────────────────┘
```

**Dependencies:**
- `max._core.engine` - Core compilation engine
- `max.driver` - Device and tensor management
- `mojo.paths` - Mojo package handling

---

### 2.7 max.serve - Inference Server

**Location:** [max/serve/](../max/serve/) (57 Python files)

**Purpose:** Production-ready HTTP inference server with OpenAI-compatible API. Handles request scheduling, batching, and model serving.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HTTP Layer (FastAPI)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ OpenAI API   │  │ KServe API   │  │ SageMaker API      │    │
│  │ /v1/chat/... │  │ /v2/models/..│  │ /invocations       │    │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘    │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Scheduler Layer (ZMQ)                         │
│  ┌────────────────────┐  ┌──────────────────┐                  │
│  │ Prefill Scheduler  │  │ Decode Scheduler │                  │
│  │ (First tokens)     │  │ (Continuation)   │                  │
│  └─────────┬──────────┘  └────────┬─────────┘                  │
└────────────┼─────────────────────┼────────────────────────────────┘
             │                     │
             ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Worker Pool                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Worker 0    │  │ Worker 1    │  │ Worker N    │            │
│  │ (GPU 0)     │  │ (GPU 1)     │  │ (GPU N)     │            │
│  │ Model+Cache │  │ Model+Cache │  │ Model+Cache │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Components

##### 1. API Server

[api_server.py](../max/serve/api_server.py) - FastAPI application entry point

```python
from fastapi import FastAPI
from max.serve.router.openai_routes import router as openai_router

app = FastAPI()
app.include_router(openai_router)

# OpenAI-compatible endpoints:
# POST /v1/chat/completions
# POST /v1/completions
# POST /v1/embeddings
# GET  /v1/models
```

##### 2. Request Schedulers

[scheduler/](../max/serve/scheduler/) - 13 files implementing various scheduling strategies:

- [text_generation_scheduler.py](../max/serve/scheduler/text_generation_scheduler.py) - Main text generation scheduler
- [prefill_scheduler.py](../max/serve/scheduler/prefill_scheduler.py) - Initial prompt processing
- [decode_scheduler.py](../max/serve/scheduler/decode_scheduler.py) - Token-by-token generation
- [embeddings_scheduler.py](../max/serve/scheduler/embeddings_scheduler.py) - Embedding generation
- [queues.py](../max/serve/scheduler/queues.py) - ZMQ-based inter-process queues

**Continuous Batching:**

The scheduler dynamically batches requests to maximize GPU utilization:

```python
# Conceptual flow
while True:
    # Collect waiting requests
    batch = []
    while len(batch) < max_batch_size:
        if request_queue.has_pending():
            batch.append(request_queue.pop())
        else:
            break

    # Execute batch
    if batch:
        outputs = model.execute(batch)

        # Some sequences complete, others continue
        for req, output in zip(batch, outputs):
            if output.is_complete:
                respond_to_client(req, output)
            else:
                decode_queue.push(req)  # Continue in next iteration
```

##### 3. Pipeline Workers

[pipelines/](../max/serve/pipelines/) - 7 files:

- [llm.py](../max/serve/pipelines/llm.py) - `TokenGeneratorPipeline`, `AudioGeneratorPipeline`
- [model_worker.py](../max/serve/pipelines/model_worker.py) - Worker process management
- [kvcache_worker.py](../max/serve/pipelines/kvcache_worker.py) - KV cache management
- [telemetry_worker.py](../max/serve/pipelines/telemetry_worker.py) - Metrics collection

**Worker Process:**

```python
class ModelWorker:
    def __init__(self, model_path, device_id):
        # Load compiled model
        session = InferenceSession()
        self.model = session.compile_from_path(model_path)
        self.device = Accelerator(device_id)
        self.kv_cache = KVCache(max_tokens=2048)

    def process_batch(self, requests):
        # Prepare inputs
        input_ids = [r.token_ids for r in requests]

        # Execute model
        logits = self.model(
            input_ids,
            cache=self.kv_cache,
            positions=[r.position for r in requests]
        )

        # Sample next tokens
        next_tokens = self.sampler.sample(logits, temperature=0.7)
        return next_tokens
```

##### 4. HTTP Routers

[router/](../max/serve/router/) - API endpoint implementations:

- [openai_routes.py](../max/serve/router/openai_routes.py) - OpenAI-compatible API
- [kserve_routes.py](../max/serve/router/kserve_routes.py) - KServe protocol
- [sagemaker_routes.py](../max/serve/router/sagemaker_routes.py) - AWS SageMaker

#### Request Lifecycle

```
1. HTTP Request arrives
   ↓ [openai_routes.py::OpenAIResponseGenerator]

2. Parse & validate request
   ↓ [request.py::TextGenerationRequest]

3. Queue for scheduling
   ↓ [scheduler/queues.py::ZMQQueue.push()]

4. Prefill phase (first tokens)
   ↓ [scheduler/prefill_scheduler.py]

5. Send to worker
   ↓ [pipelines/model_worker.py::ModelWorker.process()]

6. Execute model
   ↓ [max.engine.Model.execute()]

7. Decode phase (continuation)
   ↓ [scheduler/decode_scheduler.py]

8. Stream tokens back
   ↓ [router/openai_routes.py::stream_response()]

9. Client receives SSE stream
```

#### Configuration

[config.py](../max/serve/config.py) - Server configuration:

```python
class ServerConfig:
    model_path: str           # Path to compiled model
    devices: List[Device]     # GPU devices to use
    max_batch_size: int       # Max concurrent requests
    max_sequence_length: int  # Max tokens per request
    continuous_batching: bool # Enable dynamic batching
    tensor_parallel: int      # Tensor parallelism degree
    pipeline_parallel: int    # Pipeline parallelism degree
    lora_adapters: List[str]  # LoRA adapter paths
```

**Starting the Server:**

```bash
# Using MAX CLI
max serve --model modularai/Llama-3.1-8B-Instruct-GGUF

# With custom config
max serve \
    --model ./my_model.mogg \
    --devices 0,1,2,3 \
    --max-batch-size 128 \
    --tensor-parallel 4
```

**Dependencies:**
- FastAPI, Uvicorn - Web framework
- ZMQ - Inter-process communication
- `max.interfaces` - Pipeline interfaces
- `max.pipelines` - Model pipeline definitions
- `max.nn.kv_cache` - KV cache management

---

### 2.8 max.pipelines - Model Architectures

**Location:** [max/pipelines/](../max/pipelines/)

**Purpose:** Complete model pipeline implementations for various architectures. Connects models, tokenizers, and generation logic.

#### Directory Structure

```
max/pipelines/
├── core/                      # Core pipeline abstractions
├── lib/                       # Pipeline library
│   ├── config.py              # PipelineConfig
│   ├── pipeline.py            # TextGenerationPipeline
│   ├── tokenizer.py           # Tokenizer wrappers
│   ├── sampling/              # Sampling strategies
│   ├── kv_cache_config.py     # KV cache configuration
│   └── registry.py            # PIPELINE_REGISTRY
├── architectures/             # Model implementations (26+ models)
│   ├── llama/                 # Llama family
│   ├── mistral/               # Mistral family
│   ├── phi/                   # Phi models
│   ├── qwen/                  # Qwen models
│   ├── deepseek/              # DeepSeek models
│   ├── gemma/                 # Gemma models
│   └── ...                    # Many more
└── dataprocessing/            # Data processing utilities
```

#### Key Abstractions

**PipelineConfig:**

```python
from max.pipelines.lib.config import PipelineConfig

config = PipelineConfig(
    model_path="Llama-3.1-8B-Instruct",
    tokenizer_path="tokenizer.json",
    max_sequence_length=2048,
    max_batch_size=32,
    devices=[Accelerator(0)],
    kv_cache_config=KVCacheConfig(
        strategy="paged",  # or "ragged"
        block_size=16,
    ),
)
```

**TextGenerationPipeline:**

```python
from max.pipelines.lib.pipeline import TextGenerationPipeline

# Create pipeline
pipeline = TextGenerationPipeline(config)

# Generate text
output = pipeline.generate(
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,
)
print(output.text)
```

#### Example: Llama Architecture

[architectures/llama/](../max/pipelines/architectures/llama/) - Llama implementation:

**Model Definition:**

```python
from max.graph import Graph, ops
from max.nn import Linear, RMSNorm, Embedding
from max.nn.attention import AttentionWithRope

class LlamaModel:
    def __init__(self, config):
        self.embed = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            LlamaLayer(config) for _ in range(config.num_layers)
        ]
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = Linear(config.hidden_size, config.vocab_size)

    def __call__(self, input_ids, kv_cache, positions):
        # Embedding
        x = self.embed(input_ids)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, kv_cache[i], positions)

        # Output projection
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

class LlamaLayer:
    def __init__(self, config):
        self.attn = AttentionWithRope(
            num_heads=config.num_heads,
            hidden_size=config.hidden_size,
            rope_dim=config.rope_dim,
        )
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_norm = RMSNorm(config.hidden_size)
        self.post_attn_norm = RMSNorm(config.hidden_size)

    def __call__(self, x, kv_cache, positions):
        # Pre-norm attention with residual
        attn_out = self.attn(
            self.input_norm(x),
            kv_cache=kv_cache,
            positions=positions,
        )
        x = x + attn_out

        # Pre-norm MLP with residual
        mlp_out = self.mlp(self.post_attn_norm(x))
        x = x + mlp_out

        return x
```

**Registration:**

```python
# In architectures/llama/__init__.py
from max.pipelines.lib.registry import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register(
    "llama",
    model_class=LlamaModel,
    tokenizer_class=LlamaTokenizer,
    config_class=LlamaConfig,
)
```

#### Supported Models

Current implementations (26+ architectures):

- **Llama family:** Llama 2, Llama 3, Llama 3.1, Code Llama
- **Mistral family:** Mistral, Mixtral (MoE)
- **Other:** Phi, Qwen, DeepSeek, Gemma, Falcon, Yi, InternLM, Baichuan, ...

**Dependencies:**
- `max.nn` - Neural network layers
- `max.graph` - Graph construction
- `max.engine` - Model compilation
- HuggingFace tokenizers library

---

### 2.9 max.nn - Neural Network Operators

**Location:** [max/nn/](../max/nn/)

**Purpose:** High-level neural network building blocks implemented in Python. These are graph-level operators that lower to kernel implementations.

#### Key Modules

| Module | Purpose | Key Components |
|--------|---------|----------------|
| [linear.py](../max/nn/linear.py) | Dense layers | `Linear`, `MLP` |
| [attention/](../max/nn/attention/) | Attention mechanisms | `MultiHeadAttention`, `AttentionWithRope` |
| [conv.py](../max/nn/conv.py) | Convolutions | `Conv1D`, `Conv2D`, `Conv3D` |
| [embedding.py](../max/nn/embedding.py) | Embeddings | `Embedding`, `PositionalEmbedding` |
| [norm/](../max/nn/norm/) | Normalization | `LayerNorm`, `RMSNorm`, `GroupNorm` |
| [rotary_embedding.py](../max/nn/rotary_embedding.py) | RoPE | `RotaryEmbedding` |
| [kv_cache/](../max/nn/kv_cache/) | KV cache | `PagedKVCache`, `RaggedKVCache` |
| [transformer/](../max/nn/transformer/) | Transformer blocks | `Transformer`, `TransformerBlock` |
| [lora/](../max/nn/lora/) | LoRA adapters | `LoRALinear`, `LoRAAttention` |
| [moe/](../max/nn/moe/) | Mixture of Experts | `MoE`, `TopKRouter` |
| [parallel/](../max/nn/parallel/) | Tensor parallelism | `ColumnParallel`, `RowParallel` |
| [comm/](../max/nn/comm/) | Communication | `allreduce`, `allgather` |
| [sampling/](../max/nn/sampling/) | Token sampling | `greedy`, `top_k`, `top_p` |

#### Example: Linear Layer

[linear.py](../max/nn/linear.py) implementation:

```python
from max.graph import ops, TensorValue

class Linear:
    """Linear transformation: y = xW^T + b"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features

        # Weight: [out_features, in_features]
        self.weight = Weight(
            TensorType(shape=[out_features, in_features])
        )

        if bias:
            self.bias = Weight(
                TensorType(shape=[out_features])
            )
        else:
            self.bias = None

    def __call__(self, x: TensorValue) -> TensorValue:
        # Matrix multiplication: [batch, in] @ [in, out]^T
        #   = [batch, in] @ [out, in]
        #   = [batch, out]
        output = ops.matmul(x, self.weight, transpose_b=True)

        if self.bias is not None:
            output = ops.add(output, self.bias)

        return output
```

#### Example: Multi-Head Attention

[attention/multihead.py](../max/nn/attention/multihead.py) implementation:

```python
class MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_cache: Optional[KVCache] = None,
    ):
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads

        # Projections
        self.q_proj = Linear(hidden_size, hidden_size)
        self.k_proj = Linear(hidden_size, hidden_size)
        self.v_proj = Linear(hidden_size, hidden_size)
        self.o_proj = Linear(hidden_size, hidden_size)

        self.num_heads = num_heads
        self.kv_cache = kv_cache

    def __call__(
        self,
        x: TensorValue,
        mask: Optional[TensorValue] = None,
        positions: Optional[TensorValue] = None,
    ) -> TensorValue:
        batch, seq_len, hidden = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # [batch, seq, hidden]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to [batch, num_heads, seq, head_dim]
        Q = ops.reshape(Q, [batch, seq_len, self.num_heads, self.head_dim])
        K = ops.reshape(K, [batch, seq_len, self.num_heads, self.head_dim])
        V = ops.reshape(V, [batch, seq_len, self.num_heads, self.head_dim])

        Q = ops.transpose(Q, [0, 2, 1, 3])  # [batch, heads, seq, dim]
        K = ops.transpose(K, [0, 2, 1, 3])
        V = ops.transpose(V, [0, 2, 1, 3])

        # Use KV cache if provided
        if self.kv_cache is not None:
            K, V = self.kv_cache.update(K, V, positions)

        # Scaled dot-product attention
        scores = ops.matmul(Q, K, transpose_b=True)  # [batch, heads, seq, seq]
        scores = ops.div(scores, ops.sqrt(float(self.head_dim)))

        if mask is not None:
            scores = ops.add(scores, mask)  # Masked positions = -inf

        attn_weights = ops.softmax(scores, dim=-1)
        attn_output = ops.matmul(attn_weights, V)  # [batch, heads, seq, dim]

        # Reshape back
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [batch, seq_len, hidden])

        # Output projection
        output = self.o_proj(attn_output)
        return output
```

**Note:** These operations don't execute immediately. They build a graph of operations that gets compiled by `max.engine`.

**Dependencies:**
- `max.graph` - Graph construction
- `max.kernels` (indirectly) - Lowered to Mojo kernels during compilation

---

### 2.10 max.compiler - Compiler Utilities

**Location:** [max/compiler/](../max/compiler/)

**Purpose:** Compiler-internal utilities for kernel registration and graph lowering.

#### Key File

[src/__init__.mojo](../max/compiler/src/__init__.mojo):

```mojo
from compiler_internal import (
    StaticTensorSpec,
    register,
    view_kernel,
)
```

**Components:**

- `StaticTensorSpec` - Static tensor specifications for compile-time
- `register` - Register custom kernels with the compiler
- `view_kernel` - Utilities for inspecting kernel implementations

**Usage (Internal):**

```mojo
# Register a custom kernel
@register("custom.my_op")
fn my_kernel_impl(
    input: StaticTensorSpec,
    output: StaticTensorSpec
):
    # Kernel implementation
    ...
```

**Dependencies:**
- `compiler_internal` (Mojo builtin - closed source)

---

### 2.11 Component Integration Summary

```
┌───────────────────────────────────────────────────────────────┐
│                    Data Flow Through MAX                       │
└───────────────────────────────────────────────────────────────┘

1. USER CODE (Python)
   │
   ├─ max.serve
   │   └─> HTTP request → Scheduler → Worker pool
   │
   ├─ max.pipelines
   │   └─> Model architecture definition
   │       └─> uses max.nn layers
   │
   └─ max.nn
       └─> Neural network operators
           └─> uses max.graph.ops

2. GRAPH CONSTRUCTION (Python)
   │
   └─ max.graph
       ├─> Build computation graph (symbolic)
       ├─> Manage weights
       └─> Produce MLIR representation

3. COMPILATION (Python → Native)
   │
   └─ max.engine
       ├─> Lower graph to MLIR dialects (mo, kgen)
       ├─> Apply optimization passes
       ├─> Lower to kernel calls
       ├─> Link custom extensions (.mojopkg)
       └─> Generate executable Model

4. DEVICE MANAGEMENT (Python wrapper → Native)
   │
   └─ max.driver
       ├─> Device initialization
       ├─> Tensor allocation
       ├─> Data transfer (host ↔ device)
       └─> uses max._core.driver (native)

5. NATIVE BRIDGE (Python stubs → C++/Mojo)
   │
   └─ max._core
       ├─> MLIR types and operations
       ├─> Device and tensor ABIs
       └─> Dialect definitions

6. EXECUTION (Native Mojo)
   │
   ├─ max.kernels
   │   ├─> Execute compute operations
   │   ├─> Hardware-specific optimizations
   │   └─> Vendor library calls (cuBLAS, etc.)
   │
   └─ max._core_mojo
       └─> Specialized Mojo-Python FFI

7. HARDWARE
   └─> CPU / GPU / TPU
```

---

## 3. Mojo Compiler Pipeline

**IMPORTANT:** The Mojo compiler core is **closed source**. The actual compiler implementation (parser, type checker, MLIR dialect definitions, passes, and code generation) is **not** present in this repository.

What IS available:
- ✅ Standard library source code (written in Mojo)
- ✅ MLIR operation usage (consuming compiler features)
- ✅ Build system configuration
- ✅ Documentation and examples

What is NOT available:
- ❌ Compiler source code
- ❌ MLIR dialect TableGen definitions (.td files)
- ❌ Compilation passes
- ❌ Lowering transformations

### 3.1 Compiler Binary Location

The compiler is downloaded from Modular's servers:

**Configuration:** [bazel/mojo.MODULE.bazel](../bazel/mojo.MODULE.bazel)

```python
# Nightly version reference
VERSION = "25.7.0.dev2025102805"

# Download URL
URL = "https://dl.modular.com/public/nightly/python"
```

### 3.2 Inferred Compilation Pipeline

Based on standard library artifacts and MLIR operation usage, the compilation flow is:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Source Code (.mojo files)                                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Parsing & AST Generation [CLOSED SOURCE]                     │
│    - Lexical analysis                                            │
│    - Syntax parsing                                              │
│    - AST construction                                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Type Checking & Elaboration [CLOSED SOURCE]                  │
│    - Type inference                                              │
│    - Trait conformance checking                                  │
│    - Parametric resolution (@parameter evaluation)               │
│    - Lifetime analysis                                           │
│    - Ownership checking                                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. KGEN Dialect Generation [CLOSED SOURCE]                      │
│    - Parametric functions → kgen.generator                       │
│    - Closures → kgen.capture_list.*                             │
│    - Compile-time constants → kgen.param.constant               │
│    - Deferred evaluation → !kgen.deferred                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. POP Dialect Lowering [CLOSED SOURCE]                         │
│    - Mojo types → POP types (!pop.scalar, !pop.simd)           │
│    - Mojo ops → POP ops (pop.add, pop.mul, pop.cast_*)         │
│    - SIMD operations → pop.simd ops                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. LIT Dialect Integration [CLOSED SOURCE]                      │
│    - Insert ownership tracking:                                  │
│      · lit.ownership.mark_initialized                            │
│      · lit.ownership.mark_destroyed                              │
│    - Lifetime validation                                         │
│    - Reference management (lit.ref.to_pointer)                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Standard MLIR Lowering [CLOSED SOURCE]                       │
│    - POP → LLVM dialect                                         │
│    - Index operations (index.add, index.cmp, etc.)              │
│    - LLVM intrinsics (via pop.call_llvm_intrinsic)             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. LLVM IR Generation [USES OPEN SOURCE LLVM]                  │
│    - LLVM IR emission                                            │
│    - Optimization passes                                         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. Target Code Generation [USES OPEN SOURCE LLVM]              │
│    - Assembly generation                                         │
│    - Object code generation                                      │
│    - Executable linking                                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. Output Artifacts                                            │
│    ├─> Executable binary                                        │
│    ├─> .mojopkg package (for imports)                          │
│    └─> .so shared library (for Python bindings)                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Build Commands

**Building Standard Library:**

```bash
# From repository root
./bazelw build //mojo/stdlib/stdlib

# Output location
# bazel-bin/mojo/stdlib/stdlib/stdlib.mojopkg
```

**Building Python Extension:**

```bash
# Build .so for Python import
mojo build mojo_module.mojo --emit shared-lib -o mojo_module.so

# Can then be imported in Python
# import mojo_module
```

**Build Rules:**

[bazel/internal/mojo_library.bzl](../bazel/internal/mojo_library.bzl):

```python
load("@rules_mojo//mojo:mojo_library.bzl", _upstream_mojo_library = "mojo_library")

# Wraps external mojo_library rule from @rules_mojo
# (which is also closed source)
```

---

## 4. Mojo-MLIR Interoperability

Mojo's integration with MLIR is through three custom dialects plus standard MLIR dialects.

### 4.1 The Three Core Dialects

#### KGEN - Code Generation Dialect

**Purpose:** Manages compilation, code generation, and parametric programming

**Documentation:** References in standard library code

**Key Types:**

```
!kgen.pointer<T>          // Pointer types
!kgen.string              // Compile-time strings
!kgen.dtype               // Data type representation
!kgen.type                // Type representation
!kgen.none                // None type
!kgen.deferred            // Deferred evaluation
```

**Key Operations (found in stdlib):**

[mojo/stdlib/stdlib/compile/compile.mojo:265](../mojo/stdlib/stdlib/compile/compile.mojo#L265):
```mojo
__mlir_op.`kgen.compile_offload`  // Compile for different targets
```

[mojo/stdlib/stdlib/builtin/_closure.mojo:25,29,39](../mojo/stdlib/stdlib/builtin/_closure.mojo#L25):
```mojo
__mlir_op.`kgen.capture_list.create`  // Create closure capture list
__mlir_op.`kgen.capture_list.copy`    // Copy captures
__mlir_op.`kgen.capture_list.expand`  // Expand captures
```

[mojo/stdlib/stdlib/builtin/_location.mojo](../mojo/stdlib/stdlib/builtin/_location.mojo):
```mojo
__mlir_op.`kgen.source_loc`  // Get source location for debugging
```

[mojo/stdlib/stdlib/builtin/rebind.mojo](../mojo/stdlib/stdlib/builtin/rebind.mojo):
```mojo
__mlir_op.`kgen.rebind`  // Rebind references
```

[mojo/stdlib/stdlib/builtin/variadics.mojo](../mojo/stdlib/stdlib/builtin/variadics.mojo):
```mojo
__mlir_op.`kgen.pack.load`  // Load from variadic pack
__mlir_op.`kgen.pack.gep`   // Get element pointer in pack
```

[mojo/stdlib/stdlib/builtin/constrained.mojo](../mojo/stdlib/stdlib/builtin/constrained.mojo):
```mojo
__mlir_op.`kgen.param.assert`  // Compile-time assertions
```

**Key Attributes:**

```mojo
#kgen.compile_offload_closure<target, func>
#kgen.get_linkage_name<target, func>
#kgen.get_type_name<type, qualified_builtins>
```

---

#### POP - Parametric Operations Dialect

**Purpose:** High-level parametric operations on top of LLVM

**Documentation:** [mojo/stdlib/docs/internal/pop_dialect.md](../mojo/stdlib/docs/internal/pop_dialect.md)

**Key Types:**

```
!pop.scalar<dtype>         // Single value with specific dtype
!pop.simd<size, dtype>     // SIMD vector
!pop.array<size, type>     // Static array
!pop.union<types...>       // Union types
!pop.int_literal           // Compile-time integer
!pop.float_literal         // Compile-time float
```

**Key Operations (found in stdlib):**

[mojo/stdlib/stdlib/builtin/int.mojo:315](../mojo/stdlib/stdlib/builtin/int.mojo#L315):
```mojo
__mlir_op.`pop.cast_to_builtin`    // Mojo type → LLVM type
__mlir_op.`pop.cast_from_builtin`  // LLVM type → Mojo type
```

[mojo/stdlib/stdlib/sys/intrinsics.mojo:77,85](../mojo/stdlib/stdlib/sys/intrinsics.mojo#L77):
```mojo
__mlir_op.`pop.call_llvm_intrinsic`  // Call LLVM intrinsics
```

[mojo/stdlib/stdlib/builtin/_closure.mojo:35](../mojo/stdlib/stdlib/builtin/_closure.mojo#L35):
```mojo
__mlir_op.`pop.aligned_free`  // Free aligned memory
```

**SIMD Operations:**

```
pop.add, pop.sub, pop.mul, pop.div        // Arithmetic
pop.and, pop.or, pop.xor                  // Bitwise
pop.cmp                                   // Comparison
pop.select                                // Conditional select
pop.load, pop.store                       // Memory access
pop.broadcast                             // Broadcast scalar to SIMD
pop.shuffle                               // Shuffle SIMD elements
```

---

#### LIT - Lifetime & Ownership Dialect

**Purpose:** Manages value lifetimes, ownership tracking, and memory safety

**Key Types:**

```
!lit.origin.set  // Origin tracking for references
```

**Key Operations (found in stdlib):**

[mojo/stdlib/stdlib/builtin/coroutine.mojo:151,153,244,252](../mojo/stdlib/stdlib/builtin/coroutine.mojo#L151):
```mojo
__mlir_op.`lit.ref.to_pointer`              // Reference → pointer
__mlir_op.`lit.ownership.mark_initialized`  // Mark value initialized
__mlir_op.`lit.ownership.mark_destroyed`    // Mark value destroyed
__mlir_op.`lit.raise`                       // Raise exception
```

**Purpose in Mojo:**

The LIT dialect enables Rust-like ownership semantics:

```mojo
fn example(owned value: String):
    # lit.ownership.mark_initialized inserted here

    use(value)  # Value is owned by this function

    # lit.ownership.mark_destroyed inserted here
    # Destructor called automatically
```

---

### 4.2 Standard MLIR Dialects

#### Index Dialect

Used for `Int` operations in the standard library.

[mojo/stdlib/stdlib/builtin/int.mojo:428-742](../mojo/stdlib/stdlib/builtin/int.mojo#L428):

```mojo
# Integer arithmetic
__mlir_op.`index.add`     # Addition
__mlir_op.`index.sub`     # Subtraction
__mlir_op.`index.mul`     # Multiplication
__mlir_op.`index.divs`    # Signed division
__mlir_op.`index.divu`    # Unsigned division
__mlir_op.`index.rems`    # Signed remainder
__mlir_op.`index.remu`    # Unsigned remainder

# Bitwise operations
__mlir_op.`index.shl`     # Shift left
__mlir_op.`index.shrs`    # Shift right signed
__mlir_op.`index.shru`    # Shift right unsigned
__mlir_op.`index.and`     # Bitwise AND
__mlir_op.`index.or`      # Bitwise OR
__mlir_op.`index.xor`     # Bitwise XOR

# Comparisons
__mlir_op.`index.cmp`     # Generic comparison
```

**Example Usage:**

```mojo
@always_inline("builtin")
fn __add__(self, rhs: Int) -> Int:
    return __mlir_op.`index.add`(
        self._value,
        rhs._value
    )
```

#### LLVM Dialect

Accessed via POP's `call_llvm_intrinsic`:

[mojo/stdlib/stdlib/sys/intrinsics.mojo:177](../mojo/stdlib/stdlib/sys/intrinsics.mojo#L177):

```mojo
# Masked memory operations
llvm_intrinsic["llvm.masked.gather", ...]
llvm_intrinsic["llvm.masked.scatter", ...]

# Math intrinsics
llvm_intrinsic["llvm.sqrt", ...]
llvm_intrinsic["llvm.sin", ...]
llvm_intrinsic["llvm.cos", ...]
```

---

### 4.3 Compile-Time Evaluation

Mojo heavily uses compile-time evaluation via the **MLIR Interpreter**.

**Documentation:** [mojo/stdlib/docs/internal/compiler.md:14](../mojo/stdlib/docs/internal/compiler.md#L14)

> "The MLIR Interpreter is the mechanism by which Mojo evaluates code at compile time. It runs before elaboration, meaning some information is not available."

#### @parameter Decorator

```mojo
@parameter
fn compute_factorial(n: Int) -> Int:
    if n <= 1:
        return 1
    return n * compute_factorial(n - 1)

# This runs at compile time!
alias FACT_5 = compute_factorial(5)  # = 120
```

#### @always_inline("builtin") Decorator

**Documentation:** [mojo/stdlib/docs/internal/mlir.md:16](../mojo/stdlib/docs/internal/mlir.md#L16)

> "always_inline('builtin') enables symbolic inlining for parameter expressions. Example: `T[a+b]` → `T[Int{index.add(a.value, b.value)}]`. This avoids interpreter overhead for dependent types."

```mojo
@always_inline("builtin")
fn add_indices(a: Int, b: Int) -> Int:
    return __mlir_op.`index.add`(a, b)

# Type-level computation uses symbolic inlining
alias Sum = add_indices(10, 20)  # Computed at compile time
```

---

### 4.4 Decorator System

[mojo/stdlib/stdlib/builtin/int.mojo](../mojo/stdlib/stdlib/builtin/int.mojo) examples:

```mojo
@register_passable("trivial")  // Line 206 - Pass by value, trivially copyable
struct Int:
    var _value: __mlir_type.index

@always_inline("builtin")     // Line 292 - Inline with symbolic evaluation
fn __add__(self, rhs: Int) -> Int:
    return __mlir_op.`index.add`(self._value, rhs._value)

@always_inline("nodebug")      // Line 65 - Inline without debug info
fn _some_internal_fn():
    ...

@parameter                     // Compile-time evaluation
fn compute_at_compile_time():
    ...
```

---

### 4.5 Introspection & Metaprogramming

#### Compile-Time Information

[mojo/stdlib/stdlib/compile/compile.mojo:208](../mojo/stdlib/stdlib/compile/compile.mojo#L208):

```mojo
fn compile_info[
    func_type: AnyTrivialRegType,
    func: func_type,
    emission_kind: StaticString = "asm",  # "asm", "llvm", "llvm-opt", "object"
    target: _TargetType = _current_target(),
]() -> CompiledFunctionInfo[...]:
    """
    Get compile-time information about a function:
    - Assembly/IR code
    - Function linkage name
    - Module name
    - Number of captures
    - Capture sizes
    """
```

**Usage Example:**

```mojo
fn my_function(x: Int) -> Int:
    return x * 2

# Get assembly at compile time
alias info = compile_info[my_function.type, my_function, "asm"]()
print(info.code)  # Prints assembly code
```

---

### 4.6 Type System Integration

#### MLIR Type Wrappers

```mojo
# Mojo types wrap MLIR types
struct Int:
    var _value: __mlir_type.index  # MLIR index type

struct Float32:
    var _value: __mlir_type.f32    # MLIR f32 type

struct Bool:
    var _value: __mlir_type.i1     # MLIR i1 type
```

#### Attribute System

```mojo
# MLIR attributes for compile-time values
alias MY_CONST = __mlir_attr.`42 : index`
alias MY_FLOAT = __mlir_attr.`3.14 : f32`
```

---

## 5. Mojo-Python Interoperability

Mojo provides deep, bidirectional interoperability with Python through CPython's C API.

### 5.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python World                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Python Code                                              │  │
│  │  >>> import mojo_module                                   │  │
│  │  >>> result = mojo_module.factorial(10)                   │  │
│  └──────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────────┘
                              │ (CPython C API)
┌─────────────────────────────┼──────────────────────────────────┐
│                    FFI Bridge                                    │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │  PythonModuleBuilder                                      │  │
│  │  - def_function[mojo_fn]("factorial")                     │  │
│  │  - Creates PyMethodDef table                              │  │
│  │  - Wraps Mojo function with C ABI                         │  │
│  └──────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────────┘
                              │
┌─────────────────────────────┼──────────────────────────────────┐
│                    Mojo World                                    │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │  Mojo Implementation                                      │  │
│  │  fn factorial(n: Int) -> Int: ...                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

                         ║ BIDIRECTIONAL ║

┌─────────────────────────────────────────────────────────────────┐
│                    Mojo World                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Mojo Code                                                │  │
│  │  var np = Python.import_module("numpy")                   │  │
│  │  var arr = np.array([1, 2, 3])                            │  │
│  └──────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────────┘
                              │ (via PythonObject)
┌─────────────────────────────┼──────────────────────────────────┐
│                    FFI Bridge                                    │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │  PythonObject                                             │  │
│  │  - Wraps PyObject*                                        │  │
│  │  - Manages reference counting                             │  │
│  │  - Operator overloading                                   │  │
│  └──────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────────┘
                              │ (CPython C API)
┌─────────────────────────────┼──────────────────────────────────┐
│                    Python World                                  │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │  Python Runtime (libpython)                               │  │
│  │  - NumPy arrays, Python objects, etc.                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 5.2 Python Calling Mojo

#### Step 1: Define Mojo Function

**Example:** [mojo/docs/code/manual/python/mojo-from-python/mojo_module.mojo](../mojo/docs/code/manual/python/mojo-from-python/mojo_module.mojo)

```mojo
fn factorial(py_n: PythonObject) raises -> PythonObject:
    """Compute factorial of n"""
    # Convert from Python to Mojo
    var n = Int(py_n)

    # Mojo computation
    var result = 1
    for i in range(1, n + 1):
        result *= i

    # Convert back to Python
    return PythonObject(result)
```

#### Step 2: Build Python Extension Module

```mojo
from python import PythonObject
from python.bindings import PythonModuleBuilder

@export
fn PyInit_mojo_module() -> PythonObject:
    """Module initialization (called by Python import)"""
    var m = PythonModuleBuilder("mojo_module")

    # Register functions
    m.def_function[factorial](
        "factorial",
        docstring="Compute n!"
    )

    return m.finalize()
```

**Key File:** [mojo/stdlib/stdlib/python/bindings.mojo:266-529](../mojo/stdlib/stdlib/python/bindings.mojo#L266)

#### Step 3: Build Shared Library

```bash
mojo build mojo_module.mojo --emit shared-lib -o mojo_module.so
```

#### Step 4: Use from Python

```python
import mojo_module

result = mojo_module.factorial(10)
print(result)  # 3628800
```

---

### 5.3 Mojo Calling Python

#### Import Python Modules

[mojo/stdlib/stdlib/python/python.mojo:208-241](../mojo/stdlib/stdlib/python/python.mojo#L208):

```mojo
from python import Python

fn use_numpy() raises:
    # Import module
    var np = Python.import_module("numpy")

    # Call Python functions
    var arr = np.array([1, 2, 3, 4, 5])

    # Access attributes
    print(arr.shape)  # (5,)

    # Call methods
    var mean = arr.mean()
    print(mean)  # 3.0
```

#### PythonObject - The Bridge Type

**File:** [mojo/stdlib/stdlib/python/python_object.mojo:89-1628](../mojo/stdlib/stdlib/python/python_object.mojo#L89)

```mojo
@register_passable
struct PythonObject:
    var _obj_ptr: PyObjectPtr  # Pointer to CPython PyObject
```

**Creating PythonObjects:**

```mojo
# From Mojo primitives (lines 209-296)
var py_bool = PythonObject(True)
var py_int = PythonObject(123)
var py_float = PythonObject(3.14)
var py_str = PythonObject("hello")

# Collections (lines 308-379)
var py_list: PythonObject = [1, 2, 3]
var py_dict: PythonObject = {"key": "value"}
var py_set: PythonObject = {1, 2, 3}
```

**Using PythonObjects:**

```mojo
var obj = Python.import_module("math")

# Attribute access (line 421)
var pi = obj.pi

# Method calls
var result = obj.sqrt(16.0)

# Operators (lines 600-850)
var a = PythonObject(10)
var b = PythonObject(20)
var c = a + b  # __add__ calls Python's + operator
```

---

### 5.4 Low-Level Implementation

#### CPython C API Bindings

**File:** [mojo/stdlib/stdlib/python/_cpython.mojo:728-1157](../mojo/stdlib/stdlib/python/_cpython.mojo#L728)

```mojo
struct CPython:
    var lib: DLHandle  # Dynamic library handle to libpython
    var version: PythonVersion

    # Function pointers to CPython API
    var _Py_IncRef: Py_IncRef.type
    var _Py_DecRef: Py_DecRef.type
    var _PyObject_GetAttrString: PyObject_GetAttrString.type
    var _PyObject_Call: PyObject_Call.type
    # ... hundreds more
```

**Loading CPython:**

[_cpython.mojo:1392-1549](../mojo/stdlib/stdlib/python/_cpython.mojo#L1392):

```mojo
fn __init__(out self):
    # Load libpython dynamically
    self.lib = DLHandle("libpython3.so")

    # Resolve function pointers
    self._Py_IncRef = self.lib.get_function["Py_IncRef"]()
    self._Py_DecRef = self.lib.get_function["Py_DecRef"]()
    # ... resolve all API functions
```

---

#### Memory Layout: PyMojoObject

**File:** [mojo/stdlib/stdlib/python/bindings.mojo:125-159](../mojo/stdlib/stdlib/python/bindings.mojo#L125)

```mojo
struct PyMojoObject[T: AnyType]:
    """Bridge between Mojo values and Python objects"""

    var ob_base: PyObject  # MUST BE FIRST (C ABI compatibility)
    var mojo_value: T      # The actual Mojo value
    var is_initialized: Bool
```

**Why `ob_base` must be first:**

Python expects every `PyObject*` to start with reference count and type pointer. This struct is **binary-compatible** with CPython's object layout.

---

#### Reference Counting

**File:** [mojo/stdlib/stdlib/python/python_object.mojo:381-400](../mojo/stdlib/stdlib/python/python_object.mojo#L381)

```mojo
fn __copyinit__(out self, existing: Self):
    """Copying increments refcount"""
    # from_borrowed calls Py_IncRef
    self = Self(from_borrowed=existing._obj_ptr)

fn __del__(deinit self):
    """Destruction decrements refcount"""
    ref cpy = Python().cpython()
    with GILAcquired(Python(cpy)):  # MUST hold GIL!
        cpy.Py_DecRef(self._obj_ptr)
```

---

#### GIL Management

**File:** [mojo/stdlib/stdlib/python/_cpython.mojo:1186-1267](../mojo/stdlib/stdlib/python/_cpython.mojo#L1186)

```mojo
struct GILAcquired:
    """RAII wrapper for GIL acquisition"""
    var python: Python
    var gil_state: PyGILState_STATE

    fn __enter__(mut self):
        self.gil_state = self.python.cpython().PyGILState_Ensure()

    fn __exit__(mut self):
        self.python.cpython().PyGILState_Release(self.gil_state)

# Usage
with GILAcquired(Python()):
    # Safe to manipulate Python objects here
    var obj = create_python_object()
    modify(obj)
# GIL released automatically
```

---

#### Zero-Copy NumPy Interop

**Example:** [max/_core_mojo/mojo_module.mojo:42-127](../max/_core_mojo/mojo_module.mojo#L42)

```mojo
struct PyArrayObject[dtype: DType]:
    """Direct access to NumPy array internals (zero-copy)"""
    var data: UnsafePointer[Scalar[dtype]]
    var nd: Int
    var dimensions: UnsafePointer[Int]
    var strides: UnsafePointer[Int]

fn process_array(py_array: PythonObject) raises -> PythonObject:
    # Cast PythonObject to PyArrayObject (no copy!)
    var arr_ptr = UnsafePointer[PyArrayObject[DType.int32]](
        unchecked_downcast_value=py_array
    )

    # Direct memory access
    var data_ptr = arr_ptr[].data
    var length = arr_ptr[].dimensions[0]

    # Process data in place
    for i in range(length):
        data_ptr[i] *= 2

    return py_array  # Return modified array (same object)
```

---

### 5.5 Type Conversion Traits

#### ConvertibleToPython

**File:** [mojo/stdlib/stdlib/python/conversions.mojo:23-36](../mojo/stdlib/stdlib/python/conversions.mojo#L23)

```mojo
trait ConvertibleToPython:
    fn to_python_object(var self) raises -> PythonObject
```

**Implemented by:**

- `Int`, `Float64`, `Float32`
- `Bool`, `String`
- `List[T]`, `Dict[K, V]`, `Set[T]`

**Usage:**

```mojo
var mojo_int = 42
var py_int = mojo_int.to_python_object()  # Explicit

# Or implicit via __init__
var py_int2 = PythonObject(mojo_int)
```

---

#### ConvertibleFromPython

**File:** [mojo/stdlib/stdlib/python/conversions.mojo:39-54](../mojo/stdlib/stdlib/python/conversions.mojo#L39)

```mojo
trait ConvertibleFromPython:
    fn __init__(out self, obj: PythonObject) raises
```

**Usage:**

```mojo
var py_obj = PythonObject(5)
var mojo_int = Int(py_obj)  # Calls Int.__init__(PythonObject)

var py_str = PythonObject("hello")
var mojo_str = String(py_str)
```

---

### 5.6 Advanced: Exposing Mojo Types to Python

**File:** [mojo/stdlib/stdlib/python/bindings.mojo:531-1043](../mojo/stdlib/stdlib/python/bindings.mojo#L531)

```mojo
struct PythonTypeBuilder:
    """Build Python type objects for Mojo types"""

    fn bind[T: Representable](type_name: StaticString) -> Self
    fn def_method[method](...)
    fn def_py_init[init_func](...)
    fn def_repr[repr_func](...)
```

**Example:**

```mojo
@value
struct Point:
    var x: Float64
    var y: Float64

fn point_init(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    var x = Float64(args[0])
    var y = Float64(args[1])

    # Store in PyMojoObject
    var point = Point(x, y)
    store_mojo_value(py_self, point)
    return PythonObject.none()

fn point_repr(py_self: PythonObject) raises -> PythonObject:
    var point = get_mojo_value[Point](py_self)
    return PythonObject(f"Point({point.x}, {point.y})")

@export
fn PyInit_point_module() -> PythonObject:
    var m = PythonModuleBuilder("point_module")

    # Create type
    var point_type = PythonTypeBuilder.bind[Point]("Point")
    point_type.def_py_init[point_init]()
    point_type.def_repr[point_repr]()

    # Add to module
    m.add_type(point_type)

    return m.finalize()
```

**Usage in Python:**

```python
import point_module

p = point_module.Point(3.0, 4.0)
print(p)  # Point(3.0, 4.0)
```

---

### 5.7 Key Implementation Files

| File | Purpose | Key Components |
|------|---------|----------------|
| [python/python.mojo](../mojo/stdlib/stdlib/python/python.mojo) | High-level API | `Python.import_module()`, `Python.evaluate()` |
| [python/python_object.mojo](../mojo/stdlib/stdlib/python/python_object.mojo) | Core bridge type | `PythonObject`, operators, refcounting |
| [python/_cpython.mojo](../mojo/stdlib/stdlib/python/_cpython.mojo) | C API bindings | `CPython`, `PyObject`, GIL management |
| [python/bindings.mojo](../mojo/stdlib/stdlib/python/bindings.mojo) | Extension building | `PythonModuleBuilder`, `PythonTypeBuilder` |
| [python/conversions.mojo](../mojo/stdlib/stdlib/python/conversions.mojo) | Type conversion | `ConvertibleToPython`, `ConvertibleFromPython` |

---

## 6. Data Flow Examples

### 6.1 Example: Text Generation Request

This traces a complete request through the MAX stack.

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. HTTP Request                                                  │
│    POST /v1/chat/completions                                     │
│    {"messages": [{"role": "user", "content": "Hello!"}]}        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. FastAPI Router                                                │
│    File: max/serve/router/openai_routes.py                      │
│    - Parse OpenAI format request                                 │
│    - Validate parameters                                         │
│    - Create TextGenerationRequest                                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Tokenization                                                  │
│    File: max/pipelines/lib/tokenizer.py                         │
│    - Load tokenizer (HuggingFace)                                │
│    - Encode: "Hello!" → [128000, 9906, 0]                       │
│    - Add special tokens (BOS, EOS)                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Prefill Scheduler                                             │
│    File: max/serve/scheduler/prefill_scheduler.py               │
│    - Queue tokens for processing                                 │
│    - Wait for available worker                                   │
│    - Batch with other prefill requests if possible               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Model Worker (Prefill Phase)                                 │
│    File: max/serve/pipelines/model_worker.py                    │
│    - Allocate KV cache slots                                     │
│    - Prepare input tensor: shape=[1, 3] (batch=1, seq=3)        │
│    - Set position IDs: [0, 1, 2]                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Model Execution (Prefill)                                    │
│    File: max/engine/api.py → Model.execute()                    │
│                                                                  │
│    a) Embedding Lookup                                           │
│       File: max/nn/embedding.py                                  │
│       [128000, 9906, 0] → [1, 3, 4096] tensor                   │
│                                                                  │
│    b) Transformer Layers (32 layers)                             │
│       File: max/pipelines/architectures/llama/                   │
│       For each layer:                                            │
│         - Self-attention with RoPE                               │
│           File: max/nn/attention/                                │
│           Lowers to: max/kernels/src/nn/attention/               │
│         - Update KV cache                                        │
│           File: max/nn/kv_cache/                                 │
│           Lowers to: max/kernels/src/kv_cache/                   │
│         - MLP (SwiGLU)                                           │
│           File: max/nn/linear.py                                 │
│           Lowers to: max/kernels/src/linalg/matmul/              │
│                                                                  │
│    c) Final Layer Norm                                           │
│       File: max/nn/norm/                                         │
│                                                                  │
│    d) LM Head Projection                                         │
│       File: max/nn/linear.py                                     │
│       [1, 3, 4096] → [1, 3, 128256] logits                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Token Sampling                                                │
│    File: max/nn/sampling/                                        │
│    - Take logits for last position: [128256] shape              │
│    - Apply temperature scaling: logits /= 0.7                    │
│    - Top-p filtering (nucleus sampling)                          │
│    - Sample token: 34521 (hypothetical)                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. Decode Scheduler                                              │
│    File: max/serve/scheduler/decode_scheduler.py                │
│    - Move request to decode queue                                │
│    - Store KV cache pointers                                     │
│    - Batch with other decode requests                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. Model Execution (Decode) - LOOP                              │
│    - Process single new token: [34521]                           │
│    - Use cached K, V from previous tokens                        │
│    - Append new K, V to cache                                    │
│    - Output: [128256] logits                                     │
│    - Sample next token: 12345                                    │
│    - Repeat until EOS or max_tokens                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. Detokenization                                               │
│     File: max/pipelines/lib/tokenizer.py                        │
│     - Decode token IDs to text                                   │
│     - [34521, 12345, ...] → "Hi there! How can I help?"         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 11. HTTP Response (Streaming)                                   │
│     File: max/serve/router/openai_routes.py                     │
│     - SSE stream: data: {"choices": [{"delta": {...}}]}         │
│     - Each token sent as generated                               │
│     - Final message with stop reason                             │
└─────────────────────────────────────────────────────────────────┘
```

**Key Kernel Calls During Execution:**

1. **Embedding:** [max/kernels/src/layout/](../max/kernels/src/layout/) - Gather operation
2. **MatMul (QKV projection):** [max/kernels/src/linalg/matmul/](../max/kernels/src/linalg/matmul/) - cuBLAS/ROCm BLAS
3. **RoPE:** [max/kernels/src/nn/attention/](../max/kernels/src/nn/attention/) - Custom GPU kernel
4. **Attention (QK^T):** [max/kernels/src/linalg/matmul/](../max/kernels/src/linalg/matmul/) - Batch matmul
5. **Softmax:** [max/kernels/src/nn/activations/](../max/kernels/src/nn/) - GPU reduce kernel
6. **Attention (V):** [max/kernels/src/linalg/matmul/](../max/kernels/src/linalg/matmul/)
7. **MLP MatMul:** [max/kernels/src/linalg/matmul/](../max/kernels/src/linalg/matmul/)
8. **SwiGLU:** [max/kernels/src/nn/activations/](../max/kernels/src/nn/) - Fused activation

---

### 6.2 Example: Custom Kernel Integration

This shows how to write and integrate a custom Mojo kernel.

#### Step 1: Write Mojo Kernel

**File:** `my_kernels/custom_matmul.mojo`

```mojo
from max.graph import TensorType, ops
from max.driver import Tensor

fn custom_matmul[
    M: Int, N: Int, K: Int,
    dtype: DType
](
    A: Tensor[dtype],
    B: Tensor[dtype],
    out C: Tensor[dtype]
):
    """Custom matrix multiplication kernel

    A: [M, K]
    B: [K, N]
    C: [M, N] = A @ B
    """

    # Tile sizes for cache optimization
    alias TILE_M = 64
    alias TILE_N = 64
    alias TILE_K = 16

    # Iterate over tiles
    for m_tile in range(0, M, TILE_M):
        for n_tile in range(0, N, TILE_N):
            # Accumulator tile
            var acc_tile = Tensor[dtype].zeros([TILE_M, TILE_N])

            for k_tile in range(0, K, TILE_K):
                # Load tiles from A and B
                var a_tile = A.slice(
                    [m_tile, k_tile],
                    [TILE_M, TILE_K]
                )
                var b_tile = B.slice(
                    [k_tile, n_tile],
                    [TILE_K, TILE_N]
                )

                # Compute tile matmul
                acc_tile += matmul_tile(a_tile, b_tile)

            # Store result
            C.store_slice([m_tile, n_tile], acc_tile)

fn matmul_tile[dtype: DType](
    A: Tensor[dtype],  # [TILE_M, TILE_K]
    B: Tensor[dtype],  # [TILE_K, TILE_N]
) -> Tensor[dtype]:   # [TILE_M, TILE_N]
    # Inner tile computation with SIMD
    # ... implementation details ...
    pass
```

#### Step 2: Build Kernel Package

```bash
# Build .mojopkg
./bazelw build //my_kernels:custom_matmul

# Or using mojo directly
mojo package my_kernels/ -o custom_matmul.mojopkg
```

#### Step 3: Use in Graph

```python
from max.graph import Graph, KernelLibrary, TensorType, ops
from max.engine import InferenceSession

# Load custom kernels
kernel_lib = KernelLibrary.from_path("custom_matmul.mojopkg")

# Build graph with custom kernels
input_type = TensorType(shape=[128, 256], dtype=DType.float32)
weight_type = TensorType(shape=[256, 512], dtype=DType.float32)

with Graph(
    "custom_graph",
    input_types=[input_type],
    kernel_library=kernel_lib
) as graph:
    x = graph.inputs[0]
    W = graph.constant(weight_type, name="weight")

    # Use custom matmul
    result = ops.custom.custom_matmul(x, W)

    graph.output(result)

# Compile
session = InferenceSession()
model = session.compile_from_object(graph)

# Execute
output = model(input_tensor)
```

---

## 7. Build System

### 7.1 Bazel Overview

The Modular repository uses Bazel for all builds.

**Wrapper Script:** [bazelw](../bazelw)

```bash
#!/bin/bash
# Wrapper that ensures correct Bazel version
exec bazel "$@"
```

**Usage:**

```bash
# Always use ./bazelw from repo root
./bazelw build //...
./bazelw test //...
./bazelw run //mojo/stdlib:example
```

---

### 7.2 Key Configuration Files

#### MODULE.bazel

[MODULE.bazel](../MODULE.bazel) - Main Bazel module configuration:

```python
module(
    name = "modular",
    version = "25.7.0",
)

# External dependencies
bazel_dep(name = "rules_mojo", version = "...")
bazel_dep(name = "rules_python", version = "...")
bazel_dep(name = "llvm", version = "...")
```

#### mojo.MODULE.bazel

[bazel/mojo.MODULE.bazel](../bazel/mojo.MODULE.bazel) - Mojo toolchain:

```python
# Mojo compiler version
MOJO_VERSION = "25.7.0.dev2025102805"

# Download location
MOJO_URL = "https://dl.modular.com/public/nightly/python"
```

---

### 7.3 Build Targets

#### Mojo Standard Library

```bash
# Build stdlib
./bazelw build //mojo/stdlib/stdlib

# Output: bazel-bin/mojo/stdlib/stdlib/stdlib.mojopkg
```

#### MAX Kernels

```bash
# Build all kernels
./bazelw build //max/kernels/...

# Build specific kernel
./bazelw build //max/kernels/src/linalg:matmul

# Build and run benchmark
./bazelw run //max/kernels/benchmarks/gpu:bench_matmul
```

#### Tests

```bash
# Run all tests
./bazelw test //...

# Run specific test suite
./bazelw test //mojo/stdlib/test/collections/...
./bazelw test //max/kernels/test/linalg:test_matmul

# With specific configs
./bazelw test --config=asan //...           # AddressSanitizer
./bazelw test --config=remote-a10 //...     # Run on A10 GPU
./bazelw test --runs_per_test=10 //...      # Multiple runs
```

---

### 7.4 Mojo Library Build Rule

[bazel/internal/mojo_library.bzl](../bazel/internal/mojo_library.bzl):

```python
load("@rules_mojo//mojo:mojo_library.bzl", _upstream_mojo_library = "mojo_library")

def mojo_library(**kwargs):
    """Wrapper around upstream mojo_library rule"""
    _upstream_mojo_library(**kwargs)
```

**Example BUILD file:**

```python
load("//bazel/internal:mojo_library.bzl", "mojo_library")

mojo_library(
    name = "my_lib",
    srcs = ["my_lib.mojo"],
    deps = [
        "//mojo/stdlib/stdlib",
    ],
)
```

---

## Summary

This document has provided a comprehensive overview of the Modular Platform architecture, covering:

1. **MAX Components** - From high-level Python APIs (serve, pipelines, nn, graph) down to low-level Mojo kernels
2. **Mojo Compiler Pipeline** - The (mostly closed-source) compilation flow from .mojo to binaries via MLIR
3. **MLIR Integration** - Three custom dialects (KGEN, POP, LIT) plus standard MLIR dialects
4. **Python Interop** - Bidirectional FFI via CPython C API with zero-copy capabilities
5. **Data Flow** - Complete request traces through the stack
6. **Build System** - Bazel-based build with specific commands for different components

The architecture demonstrates a clear layering:
- Python APIs for productivity
- Graph IR for portability
- Mojo kernels for performance
- Hardware backends for execution

Each layer has well-defined interfaces, enabling modularity while maintaining end-to-end performance optimization.
