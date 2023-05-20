#include "rt.h"
#include "blast.h"
#include <CL/opencl.h>
#include <math.h>
#include <malloc.h>

// Think about what is known in at compiler time for Parallel Reduction
// (e.g. sum of vector elements).
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

static size_t sizes[] = { sizeof(fp16_t), sizeof(fp32_t), sizeof(fp64_t) };
static_assert(gpu_fp16 == 0 && gpu_fp32 == 1 && gpu_fp64 == 2, "order");

// xxx TODO: call not _so kernel if offset == 0 stride == 1

static fp64_t blast_dot(blast_t* b,
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
            b->dot_os[gpu_fp32], groups, items, countof(dot_args), dot_args);
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
                b->sum_even[gpu_fp32] : b->sum_odd[gpu_fp32];
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

static fp64_t blast_dot_fp16(blast_t* b,
        gpu_memory_t* v0, int64_t o0, int64_t s0,
        gpu_memory_t* v1, int64_t o1, int64_t s1, int64_t n) {
    return blast_dot(b, v0, o0, s0, v1, o1, s1, n, gpu_fp16);
}

static fp64_t blast_dot_fp32(blast_t* b,
        gpu_memory_t* v0, int64_t o0, int64_t s0,
        gpu_memory_t* v1, int64_t o1, int64_t s1, int64_t n) {
    return blast_dot(b, v0, o0, s0, v1, o1, s1, n, gpu_fp32);
}

static fp64_t blast_dot_fp64(blast_t* b,
        gpu_memory_t* v0, int64_t o0, int64_t s0,
        gpu_memory_t* v1, int64_t o1, int64_t s1, int64_t n) {
    return blast_dot(b, v0, o0, s0, v1, o1, s1, n, gpu_fp64);
}

static const char* blast_program_options(gpu_t* g, int kind) {
    static const char* type_t[] = {"half", "float", "double"};
    static const char* suffix[] = {"fp16", "fp32", "fp64"};
    const char* fp_t = type_t[kind];
    // see https://man.opencl.org/clBuildProgram.html
    const ocl_device_t* d = &ocl.devices[g->c.ix];
    static char options[4096];
    char* p = options;
    #pragma push_macro("append")
    #define append(...) do {                                             \
        intptr_t k = options + countof(options) - p - 1;                 \
        fatal_if(k <= 0, "options[%d] overflow", (int)countof(options)); \
        p += snprintf(p, k, "" __VA_ARGS__);                             \
    } while (0)
    append("-D fp16_t=half -D fp32_t=float -D fp64_t=double ");
    append("-D int32_t=int -D int64_t=long ");
    append("-cl-std=CL%d.%d ", d->c_version_major, d->c_version_minor);
    append("-D fp_t=%s -D vec4=%s4 -D vec8=%s8 -D vec16=%s16 -D suffix=%s %s ",
           fp_t, fp_t,fp_t, fp_t, suffix[kind],
          (kind == gpu_fp16 ? "-D fp16_surrogate" : ""));
    #pragma pop_macro("append")
    *p = 0;
//  traceln("options: %s", options);
    return options;
}

static ocl_program_t blast_compile(gpu_t* g, int kind,
        const void* code, int bytes) {
    static const char* kinds[] = {"gpu_fp16", "gpu_fp32", "gpu_fp64"};
//  traceln("\nkind: %s\n%*.*s\n\n", kinds[kind], bytes, bytes, code);
    const char* opts = blast_program_options(g, kind);
    return ocl.compile_program(&g->c, code, bytes, opts);
}

static void blast_init(blast_t* b, gpu_t* g) {
    b->gpu = g;
    ocl_device_t* d = &ocl.devices[g->c.ix];
    void* code = null;
    int64_t bytes64 = 0;
    int r = memmap_resource("blast_cl", &code, &bytes64);
    fatal_if(r != 0 || code == null || bytes64 == 0, "blast.cl is not in blast.rc");
    fatal_if(bytes64 > INT_MAX, "blast.cl %lld bytes", bytes64);
    int bytes = (int)bytes64;
    static_assertion(gpu_fp16 == 0 && gpu_fp32 == 1 && gpu_fp64 == 2);
    ocl_program_t p[3] = {
        (d->fp_config & ocl_fp16) != 0 ?
            blast_compile(g, gpu_fp16, code, bytes) : null,
            blast_compile(g, gpu_fp32, code, bytes),
        d->double_fp_config != 0 ?
            blast_compile(g, gpu_fp64, code, bytes) : null
    };
    static const char* sum_odd[]     = {"sum_odd_fp16",     "sum_odd_fp32",     "sum_odd_fp64"};
    static const char* sum_odd_os[]  = {"sum_odd_os_fp16",  "sum_odd_os_fp32",  "sum_odd_os_fp64"};
    static const char* sum_even[]    = {"sum_even_fp16",    "sum_even_fp32",    "sum_even_fp64"};
    static const char* sum_even_os[] = {"sum_even_os_fp16", "sum_even_os_fp32", "sum_even_os_fp64"};
    static const char* dot[]         = {"dot_fp16",         "dot_fp32",         "dot_fp64"};
    static const char* dot_os[]      = {"dot_os_fp16",      "dot_os_fp32",      "dot_os_fp64"};
    static const char* gemv[]        = {"gemv_fp16",        "gemv_fp32",        "gemv_fp64"};
    static const char* gemv_os[]     = {"gemv_os_fp16",     "gemv_os_fp32",     "gemv_os_fp64"};
    for (int fp = gpu_fp16; fp <= gpu_fp64; fp++) {
        if (p[fp] != null) {
            b->sum_odd[fp]     = ocl.create_kernel(p[fp], sum_odd[fp]);
            b->sum_odd_os[fp]  = ocl.create_kernel(p[fp], sum_odd_os[fp]);
            b->sum_even[fp]    = ocl.create_kernel(p[fp], sum_even[fp]);
            b->sum_even_os[fp] = ocl.create_kernel(p[fp], sum_even_os[fp]);
            b->dot[fp]         = ocl.create_kernel(p[fp], dot[fp]);
            b->dot_os[fp]      = ocl.create_kernel(p[fp], dot_os[fp]);
            b->gemv[fp]        = ocl.create_kernel(p[fp], gemv[fp]);
            b->gemv_os[fp]     = ocl.create_kernel(p[fp], gemv_os[fp]);
            ocl.dispose_program(p[fp]);
        }
    }
    b->dot_fp16 = blast_dot_fp16;
    b->dot_fp32 = blast_dot_fp32;
    b->dot_fp64 = blast_dot_fp64;
}

static void blast_fini(blast_t* b) {
    ocl_device_t* d = &ocl.devices[b->gpu->c.ix];
    int from = (d->fp_config & ocl_fp16) != 0 ? gpu_fp16 : gpu_fp32;
    int to   =  d->double_fp_config != 0 ? gpu_fp64 : gpu_fp32;
    for (int fp = from; fp <= to; fp++) {
        ocl.dispose_kernel(b->sum_odd[fp]);
        ocl.dispose_kernel(b->sum_odd_os[fp]);
        ocl.dispose_kernel(b->sum_even[fp]);
        ocl.dispose_kernel(b->sum_even_os[fp]);
        ocl.dispose_kernel(b->dot[fp]);
        ocl.dispose_kernel(b->dot_os[fp]);
        ocl.dispose_kernel(b->gemv[fp]);
        ocl.dispose_kernel(b->gemv_os[fp]);
    }
}

blast_if blast = {
    .init = blast_init,
    .fini = blast_fini
};
