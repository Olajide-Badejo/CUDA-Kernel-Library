// The C ABI shim. Nothing but C crosses this boundary: the handle is an opaque
// pointer, streams and device pointers are void*, and every entry point is
// wrapped so no exception escapes. A failure becomes a ckl_status_t, and the
// detail goes into a thread local buffer the caller reads with ckl_last_error.

#include "ckl/ckl.h"

#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <string>

#include <cuda_runtime.h>

#include "ckl/context.hpp"
#include "ckl/gemm.hpp"
#include "ckl/sparse.hpp"
#include "ckl/status.hpp"
#include "ckl/types.hpp"
#include "ckl/version.hpp"

#include "detail/last_error.hpp"

namespace ckl {
namespace detail {

namespace {

// 512 bytes is enough for the longest message the library builds and small
// enough to sit on every thread without thought.
constexpr std::size_t kLastErrorCapacity = 512;
thread_local char g_last_error[kLastErrorCapacity] = {0};

}  // namespace

void set_last_error(const std::string& message) {
    const std::size_t n =
        message.size() < kLastErrorCapacity - 1 ? message.size() : kLastErrorCapacity - 1;
    std::memcpy(g_last_error, message.data(), n);
    g_last_error[n] = '\0';
}

void clear_last_error() {
    g_last_error[0] = '\0';
}

const char* last_error_message() {
    return g_last_error;
}

}  // namespace detail
}  // namespace ckl

namespace {

ckl_status_t to_c(ckl::Status s) {
    switch (s) {
        case ckl::Status::kSuccess:
            return CKL_STATUS_SUCCESS;
        case ckl::Status::kNotInitialized:
            return CKL_STATUS_NOT_INITIALIZED;
        case ckl::Status::kInvalidValue:
            return CKL_STATUS_INVALID_VALUE;
        case ckl::Status::kArchMismatch:
            return CKL_STATUS_ARCH_MISMATCH;
        case ckl::Status::kNotSupported:
            return CKL_STATUS_NOT_SUPPORTED;
        case ckl::Status::kAllocFailed:
            return CKL_STATUS_ALLOC_FAILED;
        case ckl::Status::kExecutionFailed:
            return CKL_STATUS_EXECUTION_FAILED;
        case ckl::Status::kInternal:
            return CKL_STATUS_INTERNAL;
    }
    return CKL_STATUS_INTERNAL;
}

// The enums are declared in the same order on both sides, so the mapping is a
// range check plus a cast. The range check is the part that matters: a C caller
// can pass any int.
bool to_layout(ckl_layout_t v, ckl::Layout* out) {
    if (v != CKL_ROW_MAJOR && v != CKL_COL_MAJOR) {
        return false;
    }
    *out = static_cast<ckl::Layout>(v);
    return true;
}

bool to_op(ckl_operation_t v, ckl::Op* out) {
    if (v < CKL_OP_N || v > CKL_OP_C) {
        return false;
    }
    *out = static_cast<ckl::Op>(v);
    return true;
}

bool to_dtype(ckl_datatype_t v, ckl::DType* out) {
    if (v < CKL_R_32F || v > CKL_R_16BF) {
        return false;
    }
    *out = static_cast<ckl::DType>(v);
    return true;
}

bool to_algo(ckl_algo_t v, ckl::Algo* out) {
    if (v < CKL_ALGO_AUTO || v > CKL_ALGO_CUBLAS) {
        return false;
    }
    *out = static_cast<ckl::Algo>(v);
    return true;
}

ckl_algo_t from_algo(ckl::Algo a) {
    return static_cast<ckl_algo_t>(a);
}

bool to_spmv_algo(ckl_spmv_algo_t v, ckl::SpmvAlgo* out) {
    if (v < CKL_SPMV_AUTO || v > CKL_SPMV_CUSPARSE_ALG2) {
        return false;
    }
    *out = static_cast<ckl::SpmvAlgo>(v);
    return true;
}

ckl_spmv_algo_t from_spmv_algo(ckl::SpmvAlgo a) {
    return static_cast<ckl_spmv_algo_t>(a);
}

ckl::Context* as_context(ckl_handle_t h) {
    return reinterpret_cast<ckl::Context*>(h);
}

// The C face of the SpMV family has no plan handle: adding one would be a second
// opaque type in an ABI that has exactly one. So the shim keeps the plan instead,
// one entry, keyed on the three device buffers and the shape. A repeated call on
// the same matrix reuses the descriptors, both workspaces and both converted
// layouts, which is the whole point of ground rule 4; alternating between two
// matrices rebuilds on every call, and the header says so and points a caller
// who cares at ckl::SpmvPlan.
//
// The cache is released at process exit rather than at destruction time, for the
// same reason the default Context is: destroying cuSPARSE descriptors after the
// CUDA runtime has begun tearing down is worse than holding them.
struct SpmvCacheKey {
    const int* row_ptr = nullptr;
    const int* col_idx = nullptr;
    const float* values = nullptr;
    int m = 0;
    int n = 0;
    int nnz = 0;

    bool operator==(const SpmvCacheKey& o) const {
        return row_ptr == o.row_ptr && col_idx == o.col_idx && values == o.values && m == o.m &&
               n == o.n && nnz == o.nnz;
    }
};

ckl::SpmvPlan& spmv_plan_cache(const int* row_ptr, const int* col_idx, const float* values, int m,
                               int n, int nnz) {
    static std::mutex guard;
    static ckl::SpmvPlan* cached = nullptr;
    static SpmvCacheKey cached_key;
    const SpmvCacheKey key{row_ptr, col_idx, values, m, n, nnz};
    const std::lock_guard<std::mutex> lock(guard);
    if (cached == nullptr || !(cached_key == key)) {
        ckl::SpmvCsr a;
        a.row_ptr = row_ptr;
        a.col_idx = col_idx;
        a.values = values;
        a.m = m;
        a.n = n;
        a.nnz = nnz;
        // Built before the old one is released, so a failed rebuild leaves the
        // previous plan usable.
        auto* fresh = new ckl::SpmvPlan(a);
        delete cached;
        cached = fresh;
        cached_key = key;
    }
    return *cached;
}

// Every entry point ends in this pair. A catch-all that returns kInternal is
// the last line of defence; anything the library throws itself carries a status.
ckl_status_t record(const ckl::Error& e) {
    ckl::detail::set_last_error(e.what());
    return to_c(e.status());
}

ckl_status_t record(const std::exception& e) {
    ckl::detail::set_last_error(e.what());
    return CKL_STATUS_INTERNAL;
}

ckl_status_t invalid(const char* message) {
    ckl::detail::set_last_error(message);
    return CKL_STATUS_INVALID_VALUE;
}

// A runtime failure carries no ckl::Error, so the detail has to be built here.
// An allocation failure is reported as such; everything else is an execution
// failure, which is what a caller can act on.
ckl_status_t record_cuda(const char* where, cudaError_t e) {
    ckl::detail::set_last_error(std::string(where) + ": " + cudaGetErrorString(e));
    if (e == cudaErrorMemoryAllocation) {
        return CKL_STATUS_ALLOC_FAILED;
    }
    return CKL_STATUS_EXECUTION_FAILED;
}

bool to_memcpy_kind(ckl_memcpy_kind_t v, cudaMemcpyKind* out) {
    switch (v) {
        case CKL_MEMCPY_H2D:
            *out = cudaMemcpyHostToDevice;
            return true;
        case CKL_MEMCPY_D2H:
            *out = cudaMemcpyDeviceToHost;
            return true;
        case CKL_MEMCPY_D2D:
            *out = cudaMemcpyDeviceToDevice;
            return true;
    }
    return false;
}

// Fills a descriptor from the flat argument list every entry point carries.
bool build_desc(ckl_layout_t layout, ckl_operation_t opa, ckl_operation_t opb, int64_t m, int64_t n,
                int64_t k, ckl_datatype_t dta, int64_t lda, ckl_datatype_t dtb, int64_t ldb,
                ckl_datatype_t dtc, int64_t ldc, ckl_algo_t algo, ckl::GemmDesc* out) {
    ckl::GemmDesc d;
    if (!to_layout(layout, &d.layout) || !to_op(opa, &d.op_a) || !to_op(opb, &d.op_b) ||
        !to_dtype(dta, &d.dt_a) || !to_dtype(dtb, &d.dt_b) || !to_dtype(dtc, &d.dt_c) ||
        !to_algo(algo, &d.algo)) {
        return false;
    }
    d.m = m;
    d.n = n;
    d.k = k;
    d.lda = lda;
    d.ldb = ldb;
    d.ldc = ldc;
    d.batch_count = 1;
    *out = d;
    return true;
}

}  // namespace

extern "C" {

int ckl_get_version(void) {
    return CKL_VERSION;
}

const char* ckl_get_version_string(void) {
    return CKL_VERSION_STRING;
}

const char* ckl_status_string(ckl_status_t s) {
    switch (s) {
        case CKL_STATUS_SUCCESS:
            return "CKL_STATUS_SUCCESS";
        case CKL_STATUS_NOT_INITIALIZED:
            return "CKL_STATUS_NOT_INITIALIZED";
        case CKL_STATUS_INVALID_VALUE:
            return "CKL_STATUS_INVALID_VALUE";
        case CKL_STATUS_ARCH_MISMATCH:
            return "CKL_STATUS_ARCH_MISMATCH";
        case CKL_STATUS_NOT_SUPPORTED:
            return "CKL_STATUS_NOT_SUPPORTED";
        case CKL_STATUS_ALLOC_FAILED:
            return "CKL_STATUS_ALLOC_FAILED";
        case CKL_STATUS_EXECUTION_FAILED:
            return "CKL_STATUS_EXECUTION_FAILED";
        case CKL_STATUS_INTERNAL:
            return "CKL_STATUS_INTERNAL";
    }
    return "CKL_STATUS_UNKNOWN";
}

ckl_status_t ckl_last_error(char* buf, size_t len) {
    if (buf == nullptr || len == 0) {
        return CKL_STATUS_INVALID_VALUE;
    }
    const char* src = ckl::detail::last_error_message();
    const size_t n = std::strlen(src) < len - 1 ? std::strlen(src) : len - 1;
    std::memcpy(buf, src, n);
    buf[n] = '\0';
    return CKL_STATUS_SUCCESS;
}

ckl_status_t ckl_device_malloc(void** dptr, size_t bytes) {
    ckl::detail::clear_last_error();
    if (dptr == nullptr) {
        return invalid("ckl_device_malloc: out-param is null");
    }
    *dptr = nullptr;
    if (bytes == 0) {
        return CKL_STATUS_SUCCESS;
    }
    try {
        const cudaError_t e = cudaMalloc(dptr, bytes);
        if (e != cudaSuccess) {
            *dptr = nullptr;
            return record_cuda("ckl_device_malloc", e);
        }
        return CKL_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_device_free(void* dptr) {
    ckl::detail::clear_last_error();
    if (dptr == nullptr) {
        return CKL_STATUS_SUCCESS;
    }
    try {
        const cudaError_t e = cudaFree(dptr);
        if (e != cudaSuccess) {
            return record_cuda("ckl_device_free", e);
        }
        return CKL_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_memcpy(void* dst, const void* src, size_t bytes, ckl_memcpy_kind_t kind) {
    ckl::detail::clear_last_error();
    if (bytes == 0) {
        return CKL_STATUS_SUCCESS;
    }
    if (dst == nullptr || src == nullptr) {
        return invalid("ckl_memcpy: source or destination is null");
    }
    cudaMemcpyKind k = cudaMemcpyHostToDevice;
    if (!to_memcpy_kind(kind, &k)) {
        return invalid("ckl_memcpy: kind is out of range");
    }
    try {
        const cudaError_t e = cudaMemcpy(dst, src, bytes, k);
        if (e != cudaSuccess) {
            return record_cuda("ckl_memcpy", e);
        }
        return CKL_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_device_synchronize(void) {
    ckl::detail::clear_last_error();
    try {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            return record_cuda("ckl_device_synchronize", e);
        }
        return CKL_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_create(ckl_handle_t* h) {
    ckl::detail::clear_last_error();
    if (h == nullptr) {
        return invalid("ckl_create: handle out-param is null");
    }
    *h = nullptr;
    try {
        *h = reinterpret_cast<ckl_handle_t>(new ckl::Context());
        return CKL_STATUS_SUCCESS;
    } catch (const ckl::Error& e) {
        return record(e);
    } catch (const std::bad_alloc& e) {
        ckl::detail::set_last_error(e.what());
        return CKL_STATUS_ALLOC_FAILED;
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_destroy(ckl_handle_t h) {
    ckl::detail::clear_last_error();
    if (h == nullptr) {
        return invalid("ckl_destroy: handle is null");
    }
    try {
        delete as_context(h);
        return CKL_STATUS_SUCCESS;
    } catch (const ckl::Error& e) {
        return record(e);
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_set_stream(ckl_handle_t h, void* cuda_stream) {
    ckl::detail::clear_last_error();
    if (h == nullptr) {
        return invalid("ckl_set_stream: handle is null");
    }
    try {
        as_context(h)->set_stream(static_cast<cudaStream_t>(cuda_stream));
        return CKL_STATUS_SUCCESS;
    } catch (const ckl::Error& e) {
        return record(e);
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_set_workspace(ckl_handle_t h, void* ws, size_t bytes) {
    ckl::detail::clear_last_error();
    if (h == nullptr) {
        return invalid("ckl_set_workspace: handle is null");
    }
    try {
        as_context(h)->set_workspace(ws, bytes);
        return CKL_STATUS_SUCCESS;
    } catch (const ckl::Error& e) {
        return record(e);
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_gemm_workspace_size(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                                     ckl_operation_t opb, int64_t m, int64_t n, int64_t k,
                                     ckl_datatype_t dta, int64_t lda, ckl_datatype_t dtb,
                                     int64_t ldb, ckl_datatype_t dtc, int64_t ldc, ckl_algo_t algo,
                                     size_t* bytes) {
    ckl::detail::clear_last_error();
    if (h == nullptr || bytes == nullptr) {
        return invalid("ckl_gemm_workspace_size: handle or out-param is null");
    }
    try {
        ckl::GemmDesc d;
        if (!build_desc(layout, opa, opb, m, n, k, dta, lda, dtb, ldb, dtc, ldc, algo, &d)) {
            return invalid("ckl_gemm_workspace_size: an enum argument is out of range");
        }
        *bytes = ckl::gemm_workspace_size(*as_context(h), d);
        return CKL_STATUS_SUCCESS;
    } catch (const ckl::Error& e) {
        return record(e);
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_gemm_query(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                            ckl_operation_t opb, int64_t m, int64_t n, int64_t k,
                            ckl_datatype_t dta, int64_t lda, ckl_datatype_t dtb, int64_t ldb,
                            ckl_datatype_t dtc, int64_t ldc, ckl_algo_t* chosen) {
    ckl::detail::clear_last_error();
    if (h == nullptr || chosen == nullptr) {
        return invalid("ckl_gemm_query: handle or out-param is null");
    }
    try {
        ckl::GemmDesc d;
        if (!build_desc(layout, opa, opb, m, n, k, dta, lda, dtb, ldb, dtc, ldc, CKL_ALGO_AUTO,
                        &d)) {
            return invalid("ckl_gemm_query: an enum argument is out of range");
        }
        *chosen = from_algo(ckl::gemm_query(*as_context(h), d));
        return CKL_STATUS_SUCCESS;
    } catch (const ckl::Error& e) {
        return record(e);
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_sgemm(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                       ckl_operation_t opb, int64_t m, int64_t n, int64_t k, float alpha,
                       const float* a, int64_t lda, const float* b, int64_t ldb, float beta,
                       float* c, int64_t ldc) {
    ckl::detail::clear_last_error();
    if (h == nullptr) {
        return invalid("ckl_sgemm: handle is null");
    }
    try {
        ckl::GemmDesc d;
        if (!build_desc(layout, opa, opb, m, n, k, CKL_R_32F, lda, CKL_R_32F, ldb, CKL_R_32F, ldc,
                        CKL_ALGO_AUTO, &d)) {
            return invalid("ckl_sgemm: an enum argument is out of range");
        }
        return to_c(ckl::gemm(*as_context(h), d, &alpha, a, b, &beta, c, nullptr));
    } catch (const ckl::Error& e) {
        return record(e);
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_gemm_ex(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                         ckl_operation_t opb, int64_t m, int64_t n, int64_t k, const void* alpha,
                         const void* a, ckl_datatype_t dta, int64_t lda, const void* b,
                         ckl_datatype_t dtb, int64_t ldb, const void* beta, void* c,
                         ckl_datatype_t dtc, int64_t ldc, ckl_algo_t algo, ckl_algo_t* chosen) {
    ckl::detail::clear_last_error();
    if (h == nullptr) {
        return invalid("ckl_gemm_ex: handle is null");
    }
    try {
        ckl::GemmDesc d;
        if (!build_desc(layout, opa, opb, m, n, k, dta, lda, dtb, ldb, dtc, ldc, algo, &d)) {
            return invalid("ckl_gemm_ex: an enum argument is out of range");
        }
        ckl::Algo taken = ckl::Algo::kAuto;
        const ckl_status_t s = to_c(ckl::gemm(*as_context(h), d, alpha, a, b, beta, c, &taken));
        if (chosen != nullptr) {
            *chosen = from_algo(taken);
        }
        return s;
    } catch (const ckl::Error& e) {
        return record(e);
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_gemm_strided_batched_ex(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                                         ckl_operation_t opb, int64_t m, int64_t n, int64_t k,
                                         const void* alpha, const void* a, ckl_datatype_t dta,
                                         int64_t lda, int64_t stride_a, const void* b,
                                         ckl_datatype_t dtb, int64_t ldb, int64_t stride_b,
                                         const void* beta, void* c, ckl_datatype_t dtc, int64_t ldc,
                                         int64_t stride_c, int32_t batch_count, ckl_algo_t algo) {
    ckl::detail::clear_last_error();
    if (h == nullptr) {
        return invalid("ckl_gemm_strided_batched_ex: handle is null");
    }
    try {
        ckl::GemmDesc d;
        if (!build_desc(layout, opa, opb, m, n, k, dta, lda, dtb, ldb, dtc, ldc, algo, &d)) {
            return invalid("ckl_gemm_strided_batched_ex: an enum argument is out of range");
        }
        d.stride_a = stride_a;
        d.stride_b = stride_b;
        d.stride_c = stride_c;
        d.batch_count = batch_count;
        return to_c(ckl::gemm(*as_context(h), d, alpha, a, b, beta, c, nullptr));
    } catch (const ckl::Error& e) {
        return record(e);
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

ckl_status_t ckl_spmv_csr(ckl_handle_t h, int64_t m, int64_t n, int64_t nnz, float alpha,
                          const int* row_ptr, const int* col_idx, const float* values,
                          const float* x, float beta, float* y, ckl_spmv_algo_t algo,
                          ckl_spmv_algo_t* chosen) {
    ckl::detail::clear_last_error();
    if (h == nullptr) {
        return invalid("ckl_spmv_csr: handle is null");
    }
    ckl::SpmvAlgo requested = ckl::SpmvAlgo::kAuto;
    if (!to_spmv_algo(algo, &requested)) {
        return invalid("ckl_spmv_csr: algo is out of range");
    }
    constexpr int64_t kIntMax = 2147483647;
    if (m < 0 || n < 0 || nnz < 0 || m > kIntMax || n > kIntMax || nnz > kIntMax) {
        return invalid("ckl_spmv_csr: m, n and nnz must be non negative and inside int range");
    }
    try {
        ckl::SpmvPlan& plan = spmv_plan_cache(row_ptr, col_idx, values, static_cast<int>(m),
                                              static_cast<int>(n), static_cast<int>(nnz));
        ckl::SpmvAlgo taken = ckl::SpmvAlgo::kAuto;
        const ckl_status_t s =
            to_c(ckl::spmv(plan, requested, alpha, x, beta, y, &taken, as_context(h)->stream()));
        if (chosen != nullptr) {
            *chosen = from_spmv_algo(taken);
        }
        return s;
    } catch (const ckl::Error& e) {
        return record(e);
    } catch (const std::exception& e) {
        return record(e);
    } catch (...) {
        return CKL_STATUS_INTERNAL;
    }
}

}  // extern "C"
