// Names for the status codes and the descriptor enums. One switch per enum with
// no default label, so adding a value to any of them is a compile error here
// until it is named.

#include "ckl/status.hpp"

#include "ckl/types.hpp"

namespace ckl {

const char* status_string(Status s) {
    switch (s) {
        case Status::kSuccess:
            return "kSuccess";
        case Status::kNotInitialized:
            return "kNotInitialized";
        case Status::kInvalidValue:
            return "kInvalidValue";
        case Status::kArchMismatch:
            return "kArchMismatch";
        case Status::kNotSupported:
            return "kNotSupported";
        case Status::kAllocFailed:
            return "kAllocFailed";
        case Status::kExecutionFailed:
            return "kExecutionFailed";
        case Status::kInternal:
            return "kInternal";
    }
    return "kUnknown";
}

const char* algo_name(Algo a) {
    switch (a) {
        case Algo::kAuto:
            return "auto";
        case Algo::kNaive:
            return "naive";
        case Algo::kTiled:
            return "tiled";
        case Algo::kRegister:
            return "register";
        case Algo::kCpAsync:
            return "cp_async";
        case Algo::kWmmaFp16:
            return "wmma_fp16";
        case Algo::kWmmaBf16:
            return "wmma_bf16";
        case Algo::kMmaPtx:
            return "mma_ptx";
        case Algo::kMmaLdm:
            return "mma_ldm";
        case Algo::kMmaOpt:
            return "mma_opt";
        case Algo::kTileFamily:
            return "tile_family";
        case Algo::kSplitK:
            return "split_k";
        case Algo::kStreamK:
            return "stream_k";
        case Algo::kCutlass:
            return "cutlass";
        case Algo::kCublas:
            return "cublas";
    }
    return "unknown";
}

const char* dtype_name(DType t) {
    switch (t) {
        case DType::kR32F:
            return "r32f";
        case DType::kR16F:
            return "r16f";
        case DType::kR16BF:
            return "r16bf";
    }
    return "unknown";
}

const char* op_name(Op o) {
    switch (o) {
        case Op::kN:
            return "n";
        case Op::kT:
            return "t";
        case Op::kC:
            return "c";
    }
    return "unknown";
}

const char* layout_name(Layout l) {
    switch (l) {
        case Layout::kRowMajor:
            return "row_major";
        case Layout::kColMajor:
            return "col_major";
    }
    return "unknown";
}

std::size_t dtype_size(DType t) {
    switch (t) {
        case DType::kR32F:
            return 4;
        case DType::kR16F:
        case DType::kR16BF:
            return 2;
    }
    return 0;
}

}  // namespace ckl
