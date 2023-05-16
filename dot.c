#include "rt.h"
#include <CL/opencl.h>
#include <assert.h>
#include <math.h>
#include <malloc.h>
#include <windows.h>
#include "ocl.h"
#include "gpu.h"

// Think about what is known in at compiler time for Parallel Reduction
// (e.g. sum of vector elements).
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

static uint32_t seed = 1;

static fp32_t dot_f32(gpu_t* g,
        ocl_memory_t m0, int64_t offset0, int64_t stride0,
        ocl_memory_t m1, int64_t offset1, int64_t stride1,
        int64_t n) {
    assertion(n > 1, "to trivial for GPU");
    fp32_t s = 0;
    int64_t max_groups = ocl.devices[g->c.ix].max_groups;
    int64_t max_items  = ocl.devices[g->c.ix].max_items[0];
    #ifdef DEBUG // TODO: remove me after obtaining confidence in arithmetics
    #endif
    while (n > 0) {
        int64_t groups = min((n + max_items - 1) / max_items, max_groups);
        assertion(n >= (groups - 1) * max_items);
        int64_t total = groups == 1 ? n : groups * max_items;
        if (groups > 1 && total > n) { groups--; total -= max_items; }
        int64_t items = total / groups;
        assertion(items > 0 && groups > 0 && items * groups <= n);
        assertion(total == groups * items);
        ocl_memory_t mz = ocl.allocate(&g->c, ocl_allocate_read,
            n * sizeof(fp32_t));
        ocl_memory_t ms = ocl.allocate(&g->c, ocl_allocate_read,
            groups * items / 2 * sizeof(fp32_t));
        ocl_arg_t dot_args[] =
           {{&m0,      sizeof(ocl_memory_t)},
            {&offset0, sizeof(int32_t)},
            {&stride0, sizeof(int32_t)},
            {&m1,      sizeof(ocl_memory_t)},
            {&offset1, sizeof(int32_t)},
            {&stride1, sizeof(int32_t)},
            {&mz,      sizeof(ocl_memory_t)}
        };
//      traceln("    n: %lld (groups: %lld * items: %lld) %lld total: %lld", n, groups, items, groups * items, total);
        ocl_event_t dot_completion = ocl.enqueue_range_kernel(&g->c,
                        g->dot[gpu_fp32],
                        groups, items, countof(dot_args), dot_args);
        int64_t m = groups * items;
        int64_t k = m / 2;
        ocl_memory_t ma = mz;
        ocl_memory_t mb = ms;
        static int32_t offset_0 = 0;
        static int32_t stride_1 = 1;
        while (k >= 1) {
            ocl_arg_t sum_args[] =
               {{&ma,       sizeof(ocl_memory_t)},
                {&offset_0, sizeof(int32_t)},
                {&stride_1, sizeof(int32_t)},
                {&mb, sizeof(ocl_memory_t)}
            };
            ocl_kernel_t sum = m % 2 == 0 ?
                g->sum_even[gpu_fp32] : g->sum_odd[gpu_fp32];
            if (groups > 1) {
                groups >>= 1;
            } else if (items  > 0) {
                items  >>= 1;
            }
            assertion(groups * items == k);
            ocl_event_t sum_completion = ocl.enqueue_range_kernel(&g->c,
                sum, groups, items, countof(sum_args), sum_args);
            // TODO: cumulative EMA average performance
            ocl.dispose_event(sum_completion);
            ocl_memory_t swap = ma; ma = mb; mb = swap;
            m = k;
            k /= 2;
        }
        ocl.finish(&g->c); // same as waiting for chain of events
        fp32_t* a = (fp32_t*)ocl.map(&g->c, ocl_map_read, ma, 0, sizeof(fp32_t));
        s += a[0];
        ocl.unmap(&g->c, ma, a);
        if (g->c.profile) {
            ocl_profiling_t p = {0};
            ocl.profile(dot_completion, &p, n);
            traceln("dot[%lldx%lld]: %.3f us (microseconds) Gops=%.3f",
                    groups, items, p.time * (1000 * 1000), p.gops);
            ocl.profile(dot_completion, &p, n);
            traceln("sum[%lldx%lld]: %.3f us (microseconds) Gops=%.3f",
                    groups, items, p.time * (1000 * 1000), p.gops);
        }
        ocl.dispose_event(dot_completion);
        ocl.deallocate(mz);
        ocl.deallocate(ms);
        n -= total;
        offset0 += (int32_t)total;
        offset1 += (int32_t)total;
    }
    return s;
}

/* static */ fp32_t gpu_dot_f32(
        gpu_memory_t* v0, int64_t offset0, int64_t stride0,
        gpu_memory_t* v1, int64_t offset1, int64_t stride1,
        int64_t n) {
    fp32_t sum = 0;
    if (n > 1) {
        // TODO: for now should not be the same,
        //       but can be with a bit of juggling in the code
        fatal_if(v0 == v1 || v0->handle == v1->handle);
        fatal_if(v0->gpu != v1->gpu);
        ocl_memory_t m0 = (ocl_memory_t)v0->handle;
        ocl_memory_t m1 = (ocl_memory_t)v1->handle;
        // memory must be unmapped before kernel execution
        ocl_context_t* c = &v0->gpu->c;
        ocl.unmap(c, m0, v0->m);
        ocl.unmap(c, m1, v1->m);
        sum = dot_f32(v0->gpu, m0, offset0, stride0, m1, offset1, stride1, n);
        // map back (address will change)
        offset0 *= sizeof(fp32_t); // in bytes
        offset1 *= sizeof(fp32_t);
        v0->m = ocl.map(c, v0->map, m0, offset0, v0->bytes - offset0);
        v1->m = ocl.map(c, v1->map, m1, offset1, v1->bytes - offset1);
    } else {
        sum = ((fp32_t*)v0->m)[offset0] * ((fp32_t*)v1->m)[offset1];
    }
    return sum;
}

static void test(gpu_t* g) {
    double err = 0;
    for (int n = 2; n < 17; n++) {
        const int64_t bytes = n * sizeof(fp32_t);
        gpu_memory_t m0 = gpu.allocate(g, gpu_allocate_write, bytes);
        gpu_memory_t m1 = gpu.allocate(g, gpu_allocate_write, bytes);
        fp32_t* x = (fp32_t*)m0.m;
        fp32_t* y = (fp32_t*)m1.m;
        fp32_t sum = 0;
        for (int32_t i = 0; i < n; i++) {
//          x[i] = 1.0f + (i % 2 == 0 ? -1.0f : +1.f) / (i + 1);
//          y[i] = 1.0f - (i % 2 == 0 ? +1.0f : -1.f) / (i + 1);
x[i] = (fp32_t)(i + 1);
y[i] = (fp32_t)(i + 1);
//          traceln("%f %f %f", x[i], y[i], x[i] * y[i]);
            sum += x[i] * y[i];
        }
        fp32_t dot = gpu.dot_f32(&m0, 0, 1, &m1, 0, 1, n);
        gpu.deallocate(&m0);
        gpu.deallocate(&m1);
        double rse = sqrt(pow(dot - sum, 2));
        traceln("n: %d dot: %.7f  sum: %.7F rse: %.7f\n", n, dot, sum, rse);
        if (rse > err) {
//          traceln("n: %d dot: %.7f  sum: %.7F rse: %.7f\n", n, dot, sum, rse);
//          traceln("n: %d dot: %.7e  sum: %.7e rse: %.7e\n", n, dot, sum, rse);
            err = rse;
        }
//      assertion(fabs(dot - sum) <= CL_DBL_EPSILON, "dot_product():%.7e != %.7e\n", dot, sum);
    }
    traceln("max rse: %.7e %.17f\n", err, err);
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    bool profile = true;
    void* code = null;
    int64_t bytes = 0;
    int r = memmap_resource("gpu_cl", &code, &bytes);
    fatal_if(r != 0 || code == null || bytes == 0, "dot_cl is not in dot.rc");
    ocl.init();
    for (int cycles = 1; cycles > 0; cycles--) {
        for (int i = 0; i < ocl.count; i++) {
            gpu_t gp = { 0 };
            ocl_context_t c = ocl.open(i, profile);
            traceln("%s\n", ocl.devices[i].name);
            gpu.init(&gp, &c, code, (int32_t)bytes);
            test(&gp);
//          traceln("test: %s\n", result == 0 ? "OK" : "FAILED");
            gpu.fini(&gp);
            ocl.close(&c);
        }
    }
    return 0;
}
