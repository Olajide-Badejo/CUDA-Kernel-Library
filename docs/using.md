# Using the library

How to get CKL into your own build, what the first program looks like, and the
contracts you have to honor once it compiles. Everything here describes what is
in release 1.1.0; where a thing is declared but not implemented yet, it says so.

- [Consuming an install prefix with `find_package`](#consuming-an-install-prefix-with-find_package)
- [Consuming the source tree with `add_subdirectory`](#consuming-the-source-tree-with-add_subdirectory)
- [Consuming with pkg-config](#consuming-with-pkg-config)
- [A complete first program](#a-complete-first-program)
- [The C ABI](#the-c-abi)
- [Contracts](#contracts)

Building the library itself is [`building.md`](building.md). Calling it from
Fortran is [`fortran.md`](fortran.md). The four programs in `examples/` are the
same material as runnable code.

## Consuming an install prefix with `find_package`

Install the library once:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14
cmake --build build
cmake --install build --prefix /opt/ckl
```

Then, in your own project, four lines:

```cmake
find_package(CKL REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE ckl::ckl)
```

Configure with `-DCMAKE_PREFIX_PATH=/opt/ckl` if the prefix is not already on
the search path. `ckl::ckl` is an interface target that pulls in the headers and
all eight libraries; the package also finds `CUDAToolkit` and `Threads` for you,
because the vendor libraries are `PRIVATE` on the individual targets and on a
static build they reach the export as `$<LINK_ONLY:...>`.

The individual targets are exported too, if you want a narrower link line:

| Target | Contents |
|---|---|
| `ckl::headers` | Interface only: the include directories and `CUDA::cudart`. |
| `ckl::core` | `ckl::Context`, the process wide default Context, the status names. |
| `ckl::api` | Descriptor validation, the dispatcher, and the whole C ABI. |
| `ckl::gemm` | The GEMM ladder and the cuBLAS oracles. |
| `ckl::gemv`, `ckl::sparse`, `ckl::trsm`, `ckl::solver` | The supporting families. |
| `ckl::telemetry` | `ckl::NvmlMonitor`. |
| `ckl::ckl` | All of the above in one target. Use this unless you have a reason not to. |

The install prefix is exercised by `tests/consume/`, a standalone project that
does nothing but `find_package(CKL REQUIRED)`, link `ckl::ckl`, call a kernel
from two different libraries and check the arithmetic. `examples/` builds the
same way; both are proof that the exported package works from outside the source
tree, not just inside it.

## Consuming the source tree with `add_subdirectory`

```cmake
add_subdirectory(third_party/cuda-kernel-lab)
target_link_libraries(my_app PRIVATE ckl::ckl)
```

Turn off what you do not want built before the `add_subdirectory` line, because
the defaults build the tests, the benchmarks, the tools and the examples:

```cmake
set(CKL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CKL_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(CKL_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(CKL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/cuda-kernel-lab)
```

Leaving `CKL_BUILD_TESTS` on makes your configure step fetch GoogleTest, which
is almost never what an embedding project wants.

Two things to know. The library sets `CMAKE_CUDA_ARCHITECTURES` to `120` if you
have not set it yourself, and it does that before `project()`, so define it in
your own cache first if you target something else. And `CKL_LINEINFO` is a
private per target flag rather than a `CMAKE_CUDA_FLAGS` append, so it does not
follow the compile flags into your targets.

## Consuming with pkg-config

The install writes `lib/pkgconfig/ckl.pc`. This is the route a Fortran or plain
C project usually takes:

```sh
export PKG_CONFIG_PATH=/opt/ckl/lib/pkgconfig
cc -std=c99 -o my_app main.c $(pkg-config --cflags --libs ckl)
```

The `Libs:` line is written in static link order (`ckl_api`, then the family
archives, then `ckl_core` underneath them all) and it carries `-lstdc++`,
because the library is C++ underneath and a C or gfortran driver does not add
that on its own.

One portability note that costs nothing to obey: a static install on a compiler
that honors `__declspec` needs `-DCKL_STATIC_DEFINE` on the compile line, or the
generated `ckl_export.h` decorates the declarations with `dllimport` and the
link fails. Under CMake the `ckl::headers` target defines it for you; under
pkg-config it is yours to add. On GCC and Clang it makes no difference.

## A complete first program

This is `examples/01_gemm_minimal.cpp`. A is all ones and B is all ones, so
every element of C comes out equal to k, which is a result I can check without a
reference kernel.

```cpp
#include <cmath>
#include <cstdio>
#include <vector>

#include "ckl/ckl.hpp"

namespace {

constexpr int kM = 512;
constexpr int kN = 512;
constexpr int kK = 512;

}  // namespace

int main() {
    try {
        ckl::Context ctx;

        std::vector<float> host_a(static_cast<std::size_t>(kM) * kK, 1.0F);
        std::vector<float> host_b(static_cast<std::size_t>(kK) * kN, 1.0F);

        ckl::DeviceBuffer<float> a(host_a.size());
        ckl::DeviceBuffer<float> b(host_b.size());
        ckl::DeviceBuffer<float> c(static_cast<std::size_t>(kM) * kN);
        a.copy_from_host(host_a);
        b.copy_from_host(host_b);
        c.zero();

        ckl::GemmDesc desc;
        desc.layout = ckl::Layout::kRowMajor;
        desc.m = kM;
        desc.n = kN;
        desc.k = kK;
        // Row major and packed: the leading dimension is the row length.
        desc.lda = kK;
        desc.ldb = kN;
        desc.ldc = kN;
        desc.algo = ckl::Algo::kAuto;

        const float alpha = 1.0F;
        const float beta = 0.0F;
        ckl::Algo chosen = ckl::Algo::kAuto;

        const ckl::Status s =
            ckl::gemm(ctx, desc, &alpha, a.data(), b.data(), &beta, c.data(), &chosen);
        if (s != ckl::Status::kSuccess) {
            std::printf("FAIL: gemm returned %s\n", ckl::status_string(s));
            return 1;
        }

        CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

        const std::vector<float> result = c.to_host();
        const float want = static_cast<float>(kK);
        for (std::size_t i = 0; i < result.size(); ++i) {
            if (std::fabs(result[i] - want) > 1e-2F) {
                std::printf("FAIL: c[%zu] = %f, expected %f\n", i,
                            static_cast<double>(result[i]), static_cast<double>(want));
                return 1;
            }
        }

        std::printf("PASS: %dx%dx%d fp32 gemm via %s, every element %.0f\n", kM, kN, kK,
                    ckl::algo_name(chosen), static_cast<double>(want));
        return 0;
    } catch (const ckl::Error& e) {
        std::printf("FAIL: ckl::Error(%s): %s\n", ckl::status_string(e.status()), e.what());
        return 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
```

On my RTX 5070 that prints:

```
PASS: 512x512x512 fp32 gemm via cp_async, every element 512
```

`cp_async` is the interesting part of the output. `Algo::kAuto` looked at a
512 by 512 by 512 FP32 shape, saw that it divides the 128 by 128 by 8 block
factors and is large enough for the double buffered kernel's staging to pay for
itself, and picked the hand written rung over cuBLAS. Change the shape to
something the hand kernels cannot take and `chosen` reads `cublas` instead. It
is never silent about which one ran.

The other three examples cover strided batched GEMM
(`02_gemm_strided_batched.cpp`), the C ABI (`03_c_api.c`), and a non default
stream with a workspace (`04_custom_stream_and_workspace.cpp`).

## The C ABI

`ckl/ckl.h` is C99 and includes `<stddef.h>`, `<stdint.h>` and the generated
`ckl_export.h`. No CUDA header, at all. Device pointers and streams cross as
`void*`, and the four memory helpers (`ckl_device_malloc`, `ckl_device_free`,
`ckl_memcpy`, `ckl_device_synchronize`) mean a C program can allocate and fill
its operands without the toolkit installed. `examples/03_c_api.c` compiles with
`cc -std=c99` and never sees a CUDA include directory it needs.

Seventeen entry points, grouped:

| Group | Entry points |
|---|---|
| Version and status | `ckl_get_version`, `ckl_get_version_string`, `ckl_status_string`, `ckl_last_error` |
| Device memory | `ckl_device_malloc`, `ckl_device_free`, `ckl_memcpy`, `ckl_device_synchronize` |
| Handle | `ckl_create`, `ckl_destroy`, `ckl_set_stream`, `ckl_set_workspace` |
| Planning | `ckl_gemm_workspace_size`, `ckl_gemm_query` |
| Compute | `ckl_sgemm`, `ckl_gemm_ex`, `ckl_gemm_strided_batched_ex` |

Every one of them returns a `ckl_status_t` and none of them can throw: each is
wrapped in a catch-all that turns anything escaping the C++ side into a status
code. The enums (`ckl_operation_t`, `ckl_layout_t`, `ckl_datatype_t`,
`ckl_algo_t`) mirror the C++ ones in the same order, and the shim range checks
every one, because a C caller can pass any `int`.

Fortran gets the same surface through a `bind(C)` module; see
[`fortran.md`](fortran.md).

## Contracts

These are the parts that are load bearing and easy to get wrong. Nothing here is
aspirational: each item describes 1.1.0 as it is built.

### Synchronous or asynchronous, per entry point

| Entry point | Behavior |
|---|---|
| `ckl::gemm`, `ckl_sgemm`, `ckl_gemm_ex`, `ckl_gemm_strided_batched_ex` | **Asynchronous.** Work is enqueued on the Context's stream and the call returns. Synchronize on that stream before reading C. |
| Every ladder rung free function (`gemm_naive` through `gemm_mma_opt`, `gemv_*`, `spmv_csr_*`, `trsm_*`) | **Asynchronous** on the stream argument. |
| The vendor oracles (`gemm_cublas`, `gemm_cublas_fp16`, `gemm_cublas_bf16`, `gemv_cublas`, `spmv_cusparse`, `trsm_cublas`) | **Asynchronous** on the stream argument, and they take a process wide lock for the duration of the enqueue. |
| `ckl::gemm_query`, `ckl::gemm_workspace_size`, `ckl_gemm_query`, `ckl_gemm_workspace_size` | Launch nothing and touch no operand. Pure planning. |
| `ckl::DenseSolver::solve_lu`, `solve_cholesky` | **Synchronous.** Each step reads the cuSOLVER device info word back with a blocking copy, so the call cannot return before the factorization has finished. |
| `ckl_memcpy` | **Synchronous** with respect to the host, on the default stream. It does not observe a handle's stream. |
| `ckl_device_synchronize` | Blocks until every stream on the current device is idle. |
| `ckl::DeviceBuffer` transfers | **Synchronous**, on the default stream. This is setup and teardown scaffolding; do not put it inside a timed region. |
| `Context::set_stream`, `ckl_set_stream` | Does not synchronize. Work already enqueued stays on the stream it was enqueued on. |

An asynchronous call that returns `Status::kSuccess` means the launch was
accepted, not that the math is right. A device side fault surfaces at the next
synchronization point.

### Thread safety

`ckl::Context` is **externally synchronized**. It owns mutable state (the
current stream, the workspace, the lazily created vendor handles), and none of
it is guarded. Two threads calling into one Context concurrently is a data race.
The same applies to `ckl_handle_t`, which is a `ckl::Context` behind a pointer,
and to `ckl::DenseSolver` and `ckl::NvmlMonitor`.

One Context per thread is the intended shape. Create it on the thread that will
use it, give it that thread's stream, and never share it. If you must share one,
hold a lock of your own across every call that touches it, including
`set_stream` and `set_workspace`.

Two pieces of shared state exist regardless:

- The **process wide default Context** the v1 free functions route through. The
  library serializes the set-stream plus call pair on it with an internal mutex,
  so `gemm_cublas` from two threads is safe but not concurrent. A Context of
  your own avoids that lock entirely. That default Context is deliberately never
  destroyed, because releasing vendor handles during static destruction, after
  the CUDA runtime has started tearing the primary context down, is worse than
  leaking them.
- `ckl_last_error` reads a **thread local** buffer. Each thread sees the detail
  of its own last failure, and every C entry point clears it on entry, so read
  it before making another call on that thread.

The library also does not create or set the CUDA device for you. A Context
records whichever device was current when it was constructed, and does not
`cudaSetDevice` on later calls. In a multi GPU program, set the device yourself
before constructing the Context, and keep that Context on that device.

### Alignment

Every vectorized fast path uses 128 bit loads: `float4` in `gemm_register`,
`gemm_cp_async` and `gemv_vectorized`, and `ldmatrix` in the tensor rungs. Those
instructions fault on a misaligned address, so:

- **Operands must be 16 byte aligned.** `cudaMalloc` and `ckl_device_malloc`
  return memory aligned to at least 256 bytes, so a base pointer is always fine.
  A pointer you formed by offsetting into a larger buffer is your
  responsibility: keep the offset a multiple of 4 floats, or 8 halves.
- **The shape has to divide the block factors**, and this the library does check.
  `gemm_register` and `gemm_cp_async` need m and n divisible by 128 and k
  divisible by 8; the WMMA and `mma.sync` rungs need m and n divisible by 64 and
  k divisible by 16; `gemm_mma_opt` needs m and n divisible by 128 and k
  divisible by 32. A shape that does not divide falls back to the boundary safe
  tiled kernel (for the free functions) or is refused with
  `Status::kNotSupported` (for a descriptor naming that algorithm explicitly).

Note what is *not* checked: the library validates shapes and leading dimensions,
never pointer alignment. A 16 byte misaligned operand on a fast path produces
`cudaErrorMisalignedAddress` at the next synchronization, which maps to
`Status::kExecutionFailed`.

### Aliasing

C must not overlap A or B. Nothing in the library checks it and no kernel here
is written to tolerate it: the register and tensor rungs stage tiles of A and B
into shared memory and write C in a separate epilogue pass, so an overlapping C
would be read after part of it had already been overwritten. A and B may alias
each other, or be the same pointer, since both are read only.

Within one call, the ranges the descriptor implies must be distinct
allocations or non overlapping regions of one. For a strided batched call that
means the batches themselves must not overlap: `stride_c` at least `m * ldc` for
row major, or `n * ldc` for column major.

The `beta` rule that goes with this: when `beta` is zero, C is not read. An
uninitialized or NaN C is legal input in that case and every kernel honors it,
which is why `beta == 0` clears C rather than multiplying it (`0 * NaN` is still
NaN).

### Errors

Two layers, deliberately different.

**C++ throws.** `ckl::Error` derives from `std::runtime_error` and carries a
`ckl::Status` you can read with `.status()`. The ladder rung free functions,
`ckl::Context`'s constructor, `ckl::DenseSolver` and `ckl::DeviceBuffer` all
report failure this way. `ckl::gemm` is the exception to the exception: it
returns a `Status` and catches internally, because it is the descriptor entry
point the C ABI sits directly on top of.

**C returns codes.** Every C entry point returns `ckl_status_t` and is wrapped
in a catch-all, so nothing propagates across the ABI boundary. The status tells
you the category; `ckl_last_error` gives you the sentence:

```c
ckl_status_t s = ckl_gemm_ex(h, /* ... */);
if (s != CKL_STATUS_SUCCESS) {
    char detail[512];
    ckl_last_error(detail, sizeof detail);
    fprintf(stderr, "%s: %s\n", ckl_status_string(s), detail);
}
```

The categories:

| Status | Means |
|---|---|
| `kInvalidValue` | Your descriptor is malformed: a negative dimension, a leading dimension too small for the shape, a null buffer for a non empty operand, an out of range enum. Nothing was launched. |
| `kNotSupported` | The shape or the type combination is outside what this path takes. An explicitly named algorithm that cannot take the shape lands here, and so do `kTileFamily`, `kSplitK`, `kStreamK` and `kCutlass`, which are declared for ABI stability and not implemented in 1.1.0. |
| `kArchMismatch` | The running device is below the compute capability the selected path needs, or the fatbin has no SASS for it. `Algo::kAuto` reroutes to cuBLAS instead of failing here; an explicitly named algorithm is never rerouted. |
| `kNotInitialized` | No usable device, or a vendor handle would not initialize. |
| `kAllocFailed` | A device or host allocation failed. |
| `kExecutionFailed` | A launch or a vendor call failed at run time. A misaligned or out of bounds access lands here. |
| `kInternal` | A defect in the library. The message says what happened; please open an issue. |

The rule underneath all of it: a silent fallback is a bug. Either the fast path
runs, or a status comes back, or `chosen` reports the path that was actually
taken. `Algo::kAuto` may reroute, and when it does, `chosen` says so.

### Layout and leading dimensions

Both conventions are supported and they mean exactly what they mean in BLAS.

**Row major** (`Layout::kRowMajor`, `CKL_ROW_MAJOR`): consecutive elements of a
row are adjacent. The leading dimension is the distance in elements between the
start of one row and the start of the next, so it counts columns and the minimum
legal value is the number of columns of the matrix *as stored*. Element (i, j)
of A lives at `a[i * lda + j]`.

**Column major** (`Layout::kColMajor`, `CKL_COL_MAJOR`): consecutive elements of
a column are adjacent, the leading dimension counts rows, and element (i, j)
lives at `a[i + j * lda]`. This is the LAPACK and Fortran convention, and it is
what `ckl::DenseSolver` uses unconditionally.

The stored shape is the shape *before* the op is applied. For `op_a == Op::kN`,
A is stored m by k; for `op_a == Op::kT` it is stored k by m, and the minimum
`lda` follows the stored shape, not the logical one. The validator computes that
minimum and puts it in the message when it rejects a call:

```
gemm: leading dimensions too small for the described shape, need lda at least 64,
ldb at least 128, ldc at least 128
```

"Packed" means each leading dimension is exactly at its minimum. The hand
written kernels take **row major, no transpose, packed, unbatched** only.
Anything else goes to cuBLAS, and `chosen` reports `cublas`. That is a
dispatch decision, not a limitation of the descriptor: a column major
transposed strided batched call is perfectly legal, it just runs on the vendor
path.

How the row major call reaches column major cuBLAS is one identity worth
knowing, because it explains why no transpose or copy is ever synthesized: a row
major buffer read column major *is* its own transpose. So a row major
`C = op(A) * op(B)` becomes the column major `C_t = op(B)_t * op(A)_t`, which
means swapping the operands, swapping m and n, and letting each operand keep its
own op flag. The bytes never move.

### Types

`DType::kR32F`, `kR16F` and `kR16BF` for inputs. The output type is
`DType::kR32F` and nothing else in 1.1.0; anything else returns
`Status::kNotSupported`. The two input types must match each other: mixed input
types are refused, not promoted. The compute type is always 32 bit float, which
is why `alpha` and `beta` are always host pointers to `float` even in
`ckl_gemm_ex`, whose signature takes them as `const void*`.

`alpha` and `beta` are **host** values. A device pointer mode is a documented
non goal until CUDA graph capture support lands.

### Workspace

`ckl::gemm_workspace_size` and `ckl_gemm_workspace_size` answer zero for every
path shipped in 1.1.0; every kernel here runs out of registers and shared memory
alone. The plumbing exists so a later split-K or stream-K rung can ask for
scratch without an ABI break, and the call pattern to write today is:

```cpp
const std::size_t need = ckl::gemm_workspace_size(ctx, desc);
if (need > 0) {
    CKL_CUDA_CHECK(cudaMalloc(&workspace, need));
}
ctx.set_workspace(workspace, need);
```

The Context does not own the buffer and never frees it. That is
`examples/04_custom_stream_and_workspace.cpp`.
