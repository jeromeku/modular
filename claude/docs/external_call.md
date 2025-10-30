# How `external_call` Works in Mojo's MLIR-Based Compilation

## Overview

`external_call` is a **compiler intrinsic** in Mojo that enables seamless interoperation with C ABI functions. This document explains how it works conceptually in Mojo's MLIR-based compilation pipeline, from high-level Mojo source code down to machine code execution.

## Table of Contents

1. [High-Level Compilation Flow](#high-level-compilation-flow)
2. [Detailed Compilation Steps](#detailed-compilation-steps)
3. [MLIR Dialects Involved](#mlir-dialects-involved)
4. [Symbol Resolution and Linking](#symbol-resolution-and-linking)
5. [Optimization Attributes](#optimization-attributes)
6. [Complete Example: DeviceContext](#complete-example-devicecontext)
7. [Memory Layout and ABI Compatibility](#memory-layout-and-abi-compatibility)
8. [Why This Design?](#why-this-design)

---

## High-Level Compilation Flow

```
Mojo Source (external_call)
    ↓
Parser/Elaboration
    ↓
POP Dialect (pop.external_call) - Platform-independent IR
    ↓
Type Lowering & Conversion
    ↓
LLVM Dialect (llvm.call)
    ↓
LLVM IR (call @symbol)
    ↓
Link Time Symbol Resolution
    ↓
Machine Code (actual call instruction)
```

---

## Detailed Compilation Steps

### Step 1: Mojo Source → MLIR POP Dialect

When you write in Mojo:
```mojo
external_call["AsyncRT_DeviceContext_HtoD_async", _ConstCharPtr](
    ctx_handle,
    dst_buffer,
    src_pointer,
)
```

The compiler performs three transformations:

#### A. Symbol String → Compile-Time Constant

**Implementation**: [ffi.mojo:751](../../mojo/stdlib/stdlib/sys/ffi.mojo#L751)

```mojo
alias callee_kgen_string = _get_kgen_string["AsyncRT_DeviceContext_HtoD_async"]()
```

This creates an MLIR attribute `!kgen.string`:
```mlir
"AsyncRT_DeviceContext_HtoD_async" : !kgen.string
```

The symbol name is **baked into the IR** as a compile-time constant.

#### B. Argument Pack Loading

**Implementation**: [ffi.mojo:752](../../mojo/stdlib/stdlib/sys/ffi.mojo#L752)

```mojo
var loaded_pack = args.get_loaded_kgen_pack()
```

This is crucial! Mojo passes arguments as **references** internally, but C functions expect **values**. The `kgen.pack.load` operation dereferences all arguments:

**From**: [variadics.mojo:622-629](../../mojo/stdlib/stdlib/builtin/variadics.mojo#L622-L629)
```mojo
fn get_loaded_kgen_pack(self) -> Self._loaded_kgen_pack_type:
    """This returns the stored KGEN pack after loading all of the elements."""
    return __mlir_op.`kgen.pack.load`(self.get_as_kgen_pack())
```

MLIR representation:
```mlir
%loaded = kgen.pack.load(%pack) : (!kgen.pack<ref<i64>, ref<ptr>, ref<ptr>>)
                                 -> !kgen.pack<i64, ptr, ptr>
```

#### C. Emit POP External Call

**Implementation**: [ffi.mojo:754-758](../../mojo/stdlib/stdlib/sys/ffi.mojo#L754-L758)

```mojo
return __mlir_op.`pop.external_call`[
    func=callee_kgen_string,
    _type=_ConstCharPtr,
](loaded_pack)
```

This emits a high-level MLIR operation:
```mlir
%result = pop.external_call[
    func = "AsyncRT_DeviceContext_HtoD_async" : !kgen.string,
    _type = !pop.pointer<scalar<i8>>
](<%ctx : !pop.pointer<opaque>,
  %dst : !pop.pointer<opaque>,
  %src : !pop.pointer<scalar<f32>>>)
  -> !pop.pointer<scalar<i8>>
```

---

### Step 2: POP Dialect → LLVM Dialect

The **POP (Parametric Operations) dialect** is Mojo's platform-independent IR. It's similar to LLVM IR but supports:
- Parametric types (generics)
- Compile-time evaluation
- Platform-independent semantics

**Documentation**: [pop_dialect.md](../../mojo/stdlib/docs/internal/pop_dialect.md)

During lowering, `pop.external_call` is transformed:

```mlir
// POP dialect (high-level)
%result = pop.external_call["AsyncRT_DeviceContext_HtoD_async"](...)
    -> !pop.pointer<scalar<i8>>

// ↓ Lowering ↓

// LLVM dialect (low-level)
%result = llvm.call @AsyncRT_DeviceContext_HtoD_async(%ctx, %dst, %src) {
    fastmath = #llvm.fastmath<none>,
} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
```

**Key transformations:**
- `!pop.pointer<T>` → `!llvm.ptr` (LLVM opaque pointers)
- `!pop.scalar<f32>` → `f32`
- Symbol name becomes a function reference `@AsyncRT_...`
- Calling convention is set to C ABI

---

### Step 3: LLVM Dialect → LLVM IR

The LLVM dialect is directly converted to LLVM IR text format:

```llvm
; LLVM IR
declare ptr @AsyncRT_DeviceContext_HtoD_async(ptr, ptr, ptr)

define ... {
  ...
  %result = call ptr @AsyncRT_DeviceContext_HtoD_async(
      ptr %ctx_handle,
      ptr %dst_buffer,
      ptr %src_pointer
  )
  ...
}
```

At this point:
- The function is **declared** but not defined
- It's marked as an external symbol
- The symbol name is exactly as specified in the Mojo code

---

### Step 4: Symbol Resolution and Linking

#### Static Linking (most common for compiler runtime)

```bash
# Conceptually, the linker does:
mojo build myprogram.mojo

# Internally:
# 1. Compile Mojo to object file
mojo-compiler myprogram.mojo -o myprogram.o

# 2. Link against compiler runtime
ld myprogram.o \
   -lMojoCompilerRT \    # Contains KGEN_CompilerRT_* symbols
   -lAsyncRT \            # Contains AsyncRT_* symbols
   -lc \                  # Standard C library
   -o myprogram
```

The linker resolves `@AsyncRT_DeviceContext_HtoD_async` by:
1. Searching the provided libraries for the symbol
2. Finding it in `libAsyncRT.so` (or `.dylib`/`.dll`)
3. Patching the call site with the actual address (or creating a PLT entry)

#### Dynamic Linking (for `DLHandle`)

```mojo
var handle = DLHandle("libcustom.so")
var func = handle.get_function[fn(Int) -> Int]("my_function")
```

This uses POSIX `dlopen`/`dlsym`:
```c
void* handle = dlopen("libcustom.so", RTLD_LAZY);
void* func_ptr = dlsym(handle, "my_function");
```

Symbol resolution happens **at runtime**, not compile time.

---

## MLIR Dialects Involved

### 1. KGEN Dialect (Mojo-specific)

**Key types and operations:**
- `!kgen.string` - Compile-time string constants
- `!kgen.pack<T...>` - Heterogeneous argument packs
- `kgen.pack.load` - Dereference packed references
- Handles parametric operations and compile-time code generation

**Purpose**: Compile-time code generation, parameter handling, and metaprogramming

### 2. POP Dialect (Parametric Operations)

**Documentation**: [pop_dialect.md](../../mojo/stdlib/docs/internal/pop_dialect.md)

> POP dialect serves two purposes: pre-elaboration (for parametric programming) and post-elaboration (platform-independent distribution format)

**Key operations:**
- `pop.external_call` - High-level external function call
- `!pop.pointer<T>` - Typed pointers
- `!pop.scalar<type>` - Scalar types
- Platform-independent, serializable IR

**Purpose**: Platform-independent intermediate representation suitable for distribution and tooling

### 3. LLVM Dialect (Standard MLIR)

**Key operations:**
- `llvm.call` - Standard function calls
- `!llvm.ptr` - LLVM opaque pointers
- Standard LLVM types and operations

**Purpose**: Direct mapping to LLVM IR for optimization and code generation

---

## Symbol Resolution and Linking

### Two Categories of External Symbols

#### A. Compiler Runtime Functions (KGEN_CompilerRT_*)

**From**: [faq.md](../../mojo/stdlib/docs/faq.md)

> Mojo depends on certain features that are still written in C++, collectively called "the compiler runtime." This may manifest in the standard library code through references like `KGEN_CompilerRT_AsyncRT_CreateRuntime`.

Examples from the codebase:
- `KGEN_CompilerRT_GetOrCreateGlobal` - Global variable management
- `KGEN_CompilerRT_AsyncRT_CreateRuntime` - Async runtime initialization
- `KGEN_CompilerRT_GetStackTrace` - Stack trace capture
- `KGEN_CompilerRT_NumPhysicalCores` - System info queries

These are **resolved at link time** by the Mojo linker, which links against the compiler runtime library.

#### B. Standard C Library Functions

Examples:
- `free` - Memory deallocation
- `malloc` - Memory allocation
- `fprintf` - Formatted I/O

These follow standard C calling conventions and are resolved by the system linker.

---

## Optimization Attributes

The compiler can attach LLVM attributes for optimization:

**Implementation**: [ffi.mojo:780-799](../../mojo/stdlib/stdlib/sys/ffi.mojo#L780-L799)

```mojo
_external_call_const["malloc", UnsafePointer[Byte]](size)
```

Generates:
```mlir
%result = pop.external_call[
    func = "malloc",
    resAttrs = [{llvm.noundef}],           // Result is never undef
    funcAttrs = ["willreturn"],            // Always returns
    memory = #llvm.memory_effects<         // Memory behavior
        other = none,
        argMem = none,
        inaccessibleMem = write
    >
](<%size : i64>)
```

### Attribute Categories

**Function attributes:**
- `willreturn` - Function will eventually return
- `allockind` - Allocation/deallocation kind
- `alloc-family` - Allocator family (e.g., "malloc")

**Argument attributes:**
- `llvm.allocptr` - Marks pointer as allocation pointer

**Result attributes:**
- `llvm.noundef` - Result is never undefined

**Memory effects:**
```
#llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
```

### Enabled Optimizations

These attributes enable aggressive optimizations:
- **CSE (Common Subexpression Elimination)**: Pure functions can be deduplicated
- **Dead Code Elimination**: Calls with no side effects and unused results can be removed
- **Inlining**: Compiler can inline through external calls with known semantics
- **Alias Analysis**: Memory effects guide pointer aliasing analysis

---

## Complete Example: DeviceContext

Let's trace a real-world example from the MAX kernel library through the entire pipeline.

### Mojo Code

```mojo
ctx.enqueue_copy(device_buffer, host_ptr)
```

### Step 1: High-Level Mojo

**Source**: [device_context.mojo:5500-5534](../../mojo/stdlib/stdlib/gpu/host/device_context.mojo#L5500-L5534)

```mojo
fn enqueue_copy[dtype: DType](
    self,
    dst_buf: DeviceBuffer[dtype],
    src_ptr: UnsafePointer[Scalar[dtype]],
) raises:
    _checked(
        external_call[
            "AsyncRT_DeviceContext_HtoD_async",
            _ConstCharPtr,
            _DeviceContextPtr,
            _DeviceBufferPtr,
            UnsafePointer[Scalar[dtype]],
        ](
            self._handle,      # DeviceContext pointer
            dst_buf._handle,   # DeviceBuffer pointer
            src_ptr,           # Host memory pointer
        )
    )
```

### Step 2: MLIR POP Dialect (after lowering)

```mlir
%error = pop.external_call[
    func = "AsyncRT_DeviceContext_HtoD_async" : !kgen.string,
    _type = !pop.pointer<scalar<i8>>
](<%ctx_handle : !pop.pointer<opaque>,
  %buf_handle : !pop.pointer<opaque>,
  %host_ptr : !pop.pointer<scalar<f32>>>)
-> !pop.pointer<scalar<i8>>

// Error checking
%is_null = pop.cmp eq %error, null
pop.if %is_null {
    pop.return
} else {
    // Raise exception with error string
}
```

### Step 3: LLVM Dialect

```mlir
%error = llvm.call @AsyncRT_DeviceContext_HtoD_async(
    %ctx_handle, %buf_handle, %host_ptr
) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr

%is_null = llvm.icmp "eq" %error, %null : !llvm.ptr
llvm.cond_br %is_null, ^success, ^error

^error:
  llvm.call @raise_exception(%error)
```

### Step 4: LLVM IR

```llvm
declare ptr @AsyncRT_DeviceContext_HtoD_async(ptr, ptr, ptr)

%error = call ptr @AsyncRT_DeviceContext_HtoD_async(
    ptr %ctx_handle,
    ptr %buf_handle,
    ptr %host_ptr
)

%is_null = icmp eq ptr %error, null
br i1 %is_null, label %success, label %error

error:
  call void @raise_exception(ptr %error)
```

### Step 5: x86-64 Assembly (conceptual)

```asm
; Arguments already in registers: %rdi, %rsi, %rdx (System V ABI)
call AsyncRT_DeviceContext_HtoD_async@PLT  ; PLT for dynamic linking
test %rax, %rax                             ; Check if null
je .Lsuccess
mov %rdi, %rax                              ; Error string to arg1
call raise_exception@PLT
.Lsuccess:
```

### Step 6: Link Time

```bash
# Linker resolves the symbol
ld ... -lAsyncRT

# libAsyncRT.so exports:
# const char* AsyncRT_DeviceContext_HtoD_async(void*, void*, void*)

# Linker patches the PLT entry to point to the actual function
```

### Step 7: Runtime Execution

The actual CUDA call happens inside `AsyncRT_DeviceContext_HtoD_async`:
```cpp
// Inside AsyncRT runtime (C++)
const char* AsyncRT_DeviceContext_HtoD_async(
    const DeviceContext* ctx,
    const DeviceBuffer* dst,
    const void* src
) {
    // Extract CUDA stream from context
    cudaStream_t stream = ctx->cuda_stream();

    // Get device pointer and size from buffer
    void* dst_ptr = dst->device_ptr();
    size_t size = dst->size_bytes();

    // Asynchronous host-to-device copy
    cudaError_t err = cudaMemcpyAsync(
        dst_ptr,
        src,
        size,
        cudaMemcpyHostToDevice,
        stream
    );

    // Return error string if failed, null if success
    return (err != cudaSuccess) ? cudaGetErrorString(err) : nullptr;
}
```

---

## Memory Layout and ABI Compatibility

### Mojo Reference Semantics vs C Value Semantics

**Mojo Reference Semantics:**
```mojo
fn my_function(ref x: Int):  # x is a reference (!lit.ref<i64>)
```

**C Function Semantics:**
```c
void my_function(int64_t x);  // x is passed by value
```

### The Solution: `kgen.pack.load`

The `kgen.pack.load` operation ensures ABI compatibility:

```mlir
// Before load: !kgen.pack<ref<i64>, ref<ptr>>
%ref_pack = ...

// After load: !kgen.pack<i64, ptr>
%value_pack = kgen.pack.load(%ref_pack)

// Now safe to pass to C function
%result = pop.external_call["my_c_function"](%value_pack)
```

### Type Conversions

| Mojo Type | KGEN Type | POP Type | LLVM Type | C Type |
|-----------|-----------|----------|-----------|--------|
| `Int` | `!kgen.int<si, 64>` | `!pop.scalar<si64>` | `i64` | `int64_t` |
| `Float32` | `!kgen.float<32>` | `!pop.scalar<f32>` | `f32` | `float` |
| `UnsafePointer[T]` | `!kgen.pointer<T>` | `!pop.pointer<T>` | `!llvm.ptr` | `T*` |
| `Bool` | `!kgen.bool` | `!pop.scalar<i1>` | `i1` | `bool` |

---

## Why This Design?

### 1. Type Safety
Compile-time checking of external function signatures ensures:
- Argument types match expectations
- Return type is correctly specified
- No silent ABI mismatches

### 2. Zero Overhead
`@always_inline("nodebug")` ensures:
- No function call overhead
- Direct emission of MLIR operations
- No runtime cost compared to hand-written LLVM IR

### 3. Platform Independence
POP dialect allows:
- Cross-platform IR distribution
- Portable binary format for libraries
- Hardware-agnostic intermediate representation

### 4. Optimization
LLVM attributes enable:
- Aggressive optimization (CSE, DCE, inlining)
- Better alias analysis
- Function attribute propagation

### 5. Flexibility
Supports both:
- Static linking (compile-time symbol resolution)
- Dynamic linking (runtime symbol resolution via `DLHandle`)

### 6. Debugging
Symbol names are:
- Preserved through compilation
- Visible in backtraces
- Accessible in debuggers (gdb, lldb)

---

## Key Implementation Files

1. **[ffi.mojo](../../mojo/stdlib/stdlib/sys/ffi.mojo)** - Main `external_call` implementation
2. **[variadics.mojo](../../mojo/stdlib/stdlib/builtin/variadics.mojo)** - Argument pack handling
3. **[device_context.mojo](../../mojo/stdlib/stdlib/gpu/host/device_context.mojo)** - Real-world usage example
4. **[pop_dialect.md](../../mojo/stdlib/docs/internal/pop_dialect.md)** - POP dialect documentation
5. **[faq.md](../../mojo/stdlib/docs/faq.md)** - Compiler runtime information

---

## Summary

`external_call` is a sophisticated compiler intrinsic that bridges Mojo's high-level abstractions to C ABI functions through a multi-stage compilation pipeline:

1. **Compile Time**: Creates `!kgen.string` symbol constants
2. **Elaboration**: Loads argument references into values via `kgen.pack.load`
3. **High-Level IR**: Emits `pop.external_call` in platform-independent POP dialect
4. **Lowering**: Converts to `llvm.call` with proper C ABI
5. **LLVM Backend**: Generates standard LLVM IR with external symbol reference
6. **Link Time**: System/dynamic linker resolves symbol to actual function address
7. **Runtime**: Direct machine code call with zero Mojo overhead

This design allows Mojo to seamlessly interoperate with C libraries while maintaining its own high-level abstractions for safety, ownership, and parametric programming!

---

## Related Topics

- **DeviceContext to CUDA Translation**: See how high-level GPU operations map to CUDA Runtime/Driver API calls
- **MLIR Dialects**: Understanding KGEN, POP, and LLVM dialects in Mojo
- **ABI Compatibility**: How Mojo ensures correct calling conventions with C libraries
- **Compiler Runtime**: The role of `KGEN_CompilerRT_*` functions in Mojo programs
