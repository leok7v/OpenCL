#include "rt.h"
#include <CL/opencl.h>
#include <math.h>
#include <malloc.h>
#include <windows.h>
#include "ocl.h"
#include "gpu.h"

// Think about what is known in at compiler time for Parallel Reduction
// (e.g. sum of vector elements).
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

static size_t sizes[] = { sizeof(fp16_t), sizeof(fp32_t), sizeof(fp64_t) };
static_assert(gpu_fp16 == 0 && gpu_fp32 == 1 && gpu_fp64 == 2, "order");

static fp64_t dot(
        gpu_memory_t* v0, int64_t o0, int64_t s0,
        gpu_memory_t* v1, int64_t o1, int64_t s1, int64_t n,
        int precision) { // gpu_fp16, gpu_fp32, gpu_fp64
    fatal_if(v0->gpu != v1->gpu);
    gpu_t* g = v0->gpu;
    fatal_if(precision < gpu_fp16 || gpu_fp64 < precision,
            "precision: %d", precision);
    fp64_t s = 0;
    int64_t max_groups = ocl.devices[g->c.ix].max_groups;
    int64_t max_items  = ocl.devices[g->c.ix].max_items[0];
    #ifdef DEBUG // TODO: remove me after obtaining confidence in arithmetics
    #endif
    size_t bytes = sizes[precision];
    while (n > 0) {
        int64_t groups = min((n + max_items - 1) / max_items, max_groups);
        assertion(n >= (groups - 1) * max_items);
        int64_t total = groups == 1 ? n : groups * max_items;
        if (groups > 1 && total > n) { groups--; total -= max_items; }
        int64_t items = total / groups;
        assertion(items > 0 && groups > 0 && items * groups <= n);
        assertion(total == groups * items);
        gpu_memory_t mz = gpu.allocate(g, gpu_allocate_read, n * bytes);
// xxx TODO: int64_t offsets are not very expensive in kernel strides are more expensive
//           work out the change for large matrices
        ocl_arg_t dot_args[] =
           {{&v0->handle, sizeof(ocl_memory_t)},
            {&o0,         sizeof(int32_t)},
            {&s0,         sizeof(int32_t)},
            {&v1->handle, sizeof(ocl_memory_t)},
            {&o1,         sizeof(int32_t)},
            {&s1,         sizeof(int32_t)},
            {&mz.handle,  sizeof(ocl_memory_t)}
        };
//      traceln("    n: %lld (groups: %lld * items: %lld) %lld total: %lld", n, groups, items, groups * items, total);
        ocl_event_t dot_done = ocl.enqueue_range_kernel(&g->c,
            g->dot_os[gpu_fp32], groups, items, countof(dot_args), dot_args);
        int64_t m = total;
        int64_t k = m / 2;
        // Only (total / 2) elements are used for result. Single extra element
        // is added to avoid allocation of zero bytes when total = 1
        int64_t half = (total + 1) / 2 * bytes;
        gpu_memory_t ms = gpu.allocate(g, gpu_allocate_read, half);
        gpu_memory_t ma = mz;
        gpu_memory_t mb = ms;
        while (k >= 1) {
            ocl_arg_t sum_args[] =
               {{&ma.handle,  sizeof(ocl_memory_t)},
                {&mb.handle,  sizeof(ocl_memory_t)}
            };
            ocl_kernel_t sum = m % 2 == 0 ?
                g->sum_even[gpu_fp32] : g->sum_odd[gpu_fp32];
            if (groups > 1) {
                groups >>= 1;
            } else if (items  > 0) {
                items  >>= 1;
            }
            assertion(groups * items == k);
            ocl_event_t sum_done = ocl.enqueue_range_kernel(&g->c,
                sum, groups, items, countof(sum_args), sum_args);
            // TODO: cumulative EMA average performance
            ocl.dispose_event(sum_done);
            gpu_memory_t swap = ma; ma = mb; mb = swap;
            m = k;
            k /= 2;
        }
        ocl.finish(&g->c); // same as waiting for chain of events
        void* a = gpu.map(&ma, gpu_map_read, 0, bytes);
        switch (precision) {
            case gpu_fp16: s += fp16to32(*(fp16_t*)a); break;
            case gpu_fp32: s += *(fp32_t*)a; break;
            case gpu_fp64: s += *(fp64_t*)a; break;
            default: assert(false, "impossible"); break;
        }
        gpu.unmap(&ma);
        if (g->c.profile) {
        // xxx TODO move inside the loop and sum up
            ocl_profiling_t p = {0};
            ocl.profile(dot_done, &p, n);
            traceln("dot[%lldx%lld]: %.3f us (microseconds) Gops=%.3f",
                    groups, items, p.time * (1000 * 1000), p.gops);
            ocl.profile(dot_done, &p, n);
            traceln("sum[%lldx%lld]: %.3f us (microseconds) Gops=%.3f",
                    groups, items, p.time * (1000 * 1000), p.gops);
        }
        ocl.dispose_event(dot_done);
        gpu.deallocate(&mz);
        gpu.deallocate(&ms);
        n  -= total;
        o0 += total;
        o1 += total;
    }
    return s;
}

static fp16_t dot_fp16(gpu_memory_t* v0, int64_t o0, int64_t s0,
                       gpu_memory_t* v1, int64_t o1, int64_t s1, int64_t n) {
    return fp32to16((fp32_t)dot(v0, o0, s0, v1, o1, s1, n, gpu_fp16));
}

static fp32_t dot_fp32(gpu_memory_t* v0, int64_t o0, int64_t s0,
                       gpu_memory_t* v1, int64_t o1, int64_t s1, int64_t n) {
    return (fp32_t)dot(v0, o0, s0, v1, o1, s1, n, gpu_fp32);
}

static fp64_t dot_fp64(gpu_memory_t* v0, int64_t o0, int64_t s0,
                       gpu_memory_t* v1, int64_t o1, int64_t s1, int64_t n) {
    return dot(v0, o0, s0, v1, o1, s1, n, gpu_fp64);
}

void dot_init() {
    gpu.dot_fp16 = dot_fp16;
    gpu.dot_fp32 = dot_fp32;
    gpu.dot_fp64 = dot_fp64;
}