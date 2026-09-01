#pragma once

// Gate S's allocation counter: a CUPTI runtime API callback that counts every
// cudaMalloc, cudaMallocAsync, cudaFree and cudaFreeAsync issued while the gate
// is armed.
//
// The claim the SpMV plan makes is that a timed call enqueues kernel launches
// and nothing else. That claim is exactly the one v1 got wrong, and it is not
// the kind of claim a code review settles: an allocation can hide inside a
// vendor call, inside a lazily built descriptor, or inside a helper that looked
// pure. So the benchmark arms a counter around its own timed lambda and exits
// non-zero if the counter moves. A gate that cannot fail is not a gate, so
// bench_spmv also carries --gate-red, which puts a malloc and a free back inside
// the timed region on purpose so the gate can be seen failing.
//
// CUPTI is linked into this one binary. It is a profiling dependency, not a
// library one, and nothing in libckl links it.

#include <atomic>
#include <string>

#include <cupti.h>

namespace ckl {
namespace bench {

/// Counts device allocations and frees while armed.
class AllocationGate {
public:
    AllocationGate() = default;

    AllocationGate(const AllocationGate&) = delete;
    AllocationGate& operator=(const AllocationGate&) = delete;

    ~AllocationGate() { shutdown(); }

    /**
     * Subscribes to the four runtime CBIDs.
     * @return An empty string on success, or why the gate could not be armed.
     */
    std::string start() {
        if (instance() != nullptr) {
            return "an allocation gate is already active in this process";
        }
        instance() = this;
        CUptiResult r = cuptiSubscribe(&subscriber_, &AllocationGate::on_callback, this);
        if (r != CUPTI_SUCCESS) {
            instance() = nullptr;
            return message("cuptiSubscribe", r);
        }
        const CUpti_runtime_api_trace_cbid cbids[] = {
            CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020,
            CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020,
            CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_v11020,
            CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020,
        };
        for (CUpti_runtime_api_trace_cbid cbid : cbids) {
            r = cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API, cbid);
            if (r != CUPTI_SUCCESS) {
                shutdown();
                return message("cuptiEnableCallback", r);
            }
        }
        return std::string();
    }

    /// Starts counting. Anything that allocates from here on is a gate failure.
    void arm() {
        count_.store(0, std::memory_order_relaxed);
        armed_.store(true, std::memory_order_relaxed);
    }

    /// Stops counting and returns what was counted.
    long long disarm() {
        armed_.store(false, std::memory_order_relaxed);
        return count_.load(std::memory_order_relaxed);
    }

    /// Allocations and frees seen since the last arm.
    long long count() const { return count_.load(std::memory_order_relaxed); }

    /// Whether the gate is running at all; a build without CUPTI reports false.
    bool active() const { return subscriber_ != nullptr; }

private:
    static AllocationGate*& instance() {
        static AllocationGate* current = nullptr;
        return current;
    }

    static std::string message(const char* what, CUptiResult r) {
        const char* detail = nullptr;
        cuptiGetResultString(r, &detail);
        return std::string(what) + " failed: " + (detail != nullptr ? detail : "unknown");
    }

    // Counted on entry only, so one cudaMalloc counts once rather than twice.
    static void CUPTIAPI on_callback(void* userdata, CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid, const void* cbdata) {
        (void)cbid;
        auto* self = static_cast<AllocationGate*>(userdata);
        if (self == nullptr || domain != CUPTI_CB_DOMAIN_RUNTIME_API) {
            return;
        }
        const auto* info = static_cast<const CUpti_CallbackData*>(cbdata);
        if (info->callbackSite != CUPTI_API_ENTER) {
            return;
        }
        if (self->armed_.load(std::memory_order_relaxed)) {
            self->count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void shutdown() {
        if (subscriber_ != nullptr) {
            cuptiUnsubscribe(subscriber_);
            subscriber_ = nullptr;
        }
        if (instance() == this) {
            instance() = nullptr;
        }
    }

    CUpti_SubscriberHandle subscriber_ = nullptr;
    std::atomic<bool> armed_{false};
    std::atomic<long long> count_{0};
};

}  // namespace bench
}  // namespace ckl
