#include "rt.h"
#include <CL/opencl.h>
#include <assert.h>
#include <math.h>
#include <malloc.h>
#include <windows.h>
#include "ocl.h"

#define fp32_t float
#define fp64_t double

#define kernel_name "dot_product"

static uint32_t seed = 1;

enum { // .allocate() flags (matching OpenCL)
    gpu_allocate_read  = (1 << 2),
    gpu_allocate_write = (1 << 1),
    gpu_allocate_rw    = (1 << 0)
};

typedef struct gpu_s {
    ocl_context_t c;
    ocl_kernel_t kernel_dot_fp32;
    ocl_kernel_t kernel_sum_odd_fp32;
    ocl_kernel_t kernel_sum_even_fp32;
} gpu_t;

typedef struct gpu_memory_s { // treat as read only, will change don't cache
    void*  m; // mapped memory address in virtual memory.
    void*  handle;
    size_t bytes;
    gpu_t* gpu;
    uint32_t map; // mapping flags
} gpu_memory_t;

typedef struct gpu_if {
    // memory pointers returned by allocate() are "8 bytes alligned  "locked"
    // in the host memory and can be accessed by the host directly.
    gpu_memory_t (*allocate)(gpu_t* gpu, int flags, size_t bytes);
    void (*deallocate)(gpu_memory_t* gm);
    // Only allocated GPU shared memory can be used as arguments.
    // functions will unmap host memory and allow access by the GPU
    // and will remap it back when done with a hope address won't change.
    //
    // TODO: investigate if we unmapping and remapping does not move
    // the memory in virtual address space.
    // And if addresses of remapping will change we will need to rethink
    // the whole scheme and pass arguments in the form fp32_t *v[] instead
    // and have it modified on remapping.
    fp32_t (*dot_f32)(gpu_memory_t* x, gpu_memory_t* y, int64_t n,
                 int64_t stride0, int64_t stride1);
    void (*gemv_f32)(gpu_memory_t* vec, gpu_memory_t* matrixx,
                 gpu_memory_t* result,
                 int64_t n, int64_t m, // vec[n], matrix[m][n], result[m]
                 int64_t stride_v, int64_t stride_mx[2]);
} gpu_if;

static_assertion(gpu_allocate_read  == ocl_allocate_read );
static_assertion(gpu_allocate_write == ocl_allocate_write);
static_assertion(gpu_allocate_rw    == ocl_allocate_rw);

static gpu_memory_t gpu_allocate(gpu_t* gpu, int flags, size_t bytes) {
    gpu_memory_t gm;
    gm.gpu = gpu;
    gm.bytes = ((bytes + 7) & ~7); // aligned to 8 bytes
    ocl_memory_t handle = ocl.allocate(&gpu->c, flags, gm.bytes);
    gm.map = 0; // map flags
    switch (flags) {
        case gpu_allocate_read:  gm.map = ocl_map_read;  break;
        case gpu_allocate_write: gm.map = ocl_map_write; break;
        case gpu_allocate_rw:    gm.map = ocl_map_rw;    break;
        default: fatal_if(true, "invalid flags %d", flags);
    }
    gm.m = ocl.map(&gpu->c, gm.map, handle, /*offset:*/ 0, gm.bytes);
    gm.handle = handle;
    return gm;
}

static void gpu_deallocate(gpu_memory_t* gm) {
    ocl.unmap(&gm->gpu->c, (ocl_memory_t)gm->handle, gm->m);
    ocl.deallocate((ocl_memory_t)gm->handle);
}

static fp32_t dot_f32(gpu_t* gpu, ocl_memory_t mx, ocl_memory_t my,
        int64_t n, int64_t stride0, int64_t stride1) {
    assertion(n > 1, "to trivial for GPU");
    fp32_t sum = 0;
    int64_t max_groups = ocl.devices[gpu->c.ix].max_groups;
    int64_t max_items  = ocl.devices[gpu->c.ix].max_items[0];
    #ifdef DEBUG // TODO: remove me after obtaining confidence in arithmetics
    #endif
    int64_t offset = 0;
    while (n > 0) {
        int64_t groups = min((n + max_items - 1) / max_items, max_groups);
        assertion(n >= (groups - 1) * max_items);
        int64_t total = groups == 1 ? n : groups * max_items;
        if (groups > 1 && total > n) { groups--; total -= max_items; }
        int64_t items = total / groups;
        assertion(items > 0 && groups > 0 && items * groups <= n);
        assertion(total == groups * items);
        ocl_memory_t mz = ocl.allocate(&gpu->c, ocl_allocate_read,
            n * sizeof(fp32_t));
        ocl_memory_t ms = ocl.allocate(&gpu->c, ocl_allocate_read,
            groups * items / 2 * sizeof(fp32_t));
        ocl_arg_t dot_args[] =
            {{&mx,      sizeof(ocl_memory_t)},
            {&my,      sizeof(ocl_memory_t)},
            {&mz,      sizeof(ocl_memory_t)},
            {&offset,  sizeof(int32_t)},
            {&stride0, sizeof(int32_t)},
            {&stride1, sizeof(int32_t)}
        };
//      traceln("    n: %lld (groups: %lld * items: %lld) %lld total: %lld", n, groups, items, groups * items, total);
        ocl_event_t dot_completion = ocl.enqueue_range_kernel(&gpu->c,
                        gpu->kernel_dot_fp32,
                        groups, items, countof(dot_args), dot_args);
        int64_t m = groups * items;
        int64_t k = m / 2;
        ocl_memory_t ma = mz;
        ocl_memory_t mb = ms;
        while (k >= 1) {
            ocl_arg_t sum_args[] =
               {{&ma, sizeof(ocl_memory_t)},
                {&mb, sizeof(ocl_memory_t)}
            };
            ocl_kernel_t kernel = m % 2 == 0 ? gpu->kernel_sum_even_fp32 :
                                               gpu->kernel_sum_odd_fp32;
            if (groups > 1) {
                groups >>= 1;
            } else if (items  > 0) {
                items  >>= 1;
            }
            assertion(groups * items == k);
            ocl_event_t sum_completion = ocl.enqueue_range_kernel(&gpu->c,
                kernel, groups, items, countof(sum_args), sum_args);
            // TODO: cumulative EMA average performance
            ocl.dispose_event(sum_completion);
            ocl_memory_t swap = ma; ma = mb; mb = swap;
            m = k;
            k /= 2;
        }
        ocl.finish(&gpu->c); // same as waiting for chain of events
        fp32_t* a = (fp32_t*)ocl.map(&gpu->c, ocl_map_read, ma, 0, n / 2 * sizeof(fp32_t));
        sum += a[0];
        ocl.unmap(&gpu->c, ma, a);
        if (gpu->c.profile) {
            ocl_profiling_t p = {0};
            ocl.profile(dot_completion, &p, n);
            traceln("kernel dot[%lldx%lld]: %.3f us (microseconds) Gops=%.3f",
                    groups, items, p.time * (1000 * 1000), p.gops);
            ocl.profile(dot_completion, &p, n);
            traceln("kernel sum[%lldx%lld]: %.3f us (microseconds) Gops=%.3f",
                    groups, items, p.time * (1000 * 1000), p.gops);
        }
        ocl.dispose_event(dot_completion);
        ocl.deallocate(mz);
        ocl.deallocate(ms);
        n -= total;
        offset += total;
    }
    return sum;
}

static fp32_t gpu_dot_f32(gpu_memory_t* x, gpu_memory_t* y,
        int64_t n, int64_t stride0, int64_t stride1) {
    fp32_t sum = 0;
    if (n > 1) {
        fatal_if(x == y || x->handle == y->handle); // cannot be the same
        fatal_if(x->gpu != y->gpu);
        ocl_memory_t mx = (ocl_memory_t)x->handle;
        ocl_memory_t my = (ocl_memory_t)y->handle;
        // memory must be unmapped before kernel execution
        ocl_context_t* c = &x->gpu->c;
        ocl.unmap(c, mx, x->m);
        ocl.unmap(c, my, y->m);
        sum = dot_f32(x->gpu, mx, my, n, stride0, stride1);
        // map back (address will change)
        x->m = ocl.map(c, x->map, mx, /*offset:*/ 0, x->bytes);
        y->m = ocl.map(c, y->map, my, /*offset:*/ 0, y->bytes);
    } else {
        sum = *(fp32_t*)x->m * *(fp32_t*)y->m;
    }
    return sum;
}

gpu_if gpu = {
    .allocate = gpu_allocate,
    .deallocate = gpu_deallocate,
    .dot_f32 = gpu_dot_f32,
};

static ocl_program_t compile(gpu_t* gp, const void* code, int bytes) {
    int n = bytes + 1024;
    char* text = alloca(n);
    snprintf(text, n, "\n"
        "#define fp_t float\n"
        "#define name %s_float\n"
        "%*.*s", kernel_name, bytes, bytes, (const char*)code);
//  traceln("%s", text);
    return ocl.compile_program(&gp->c, text, strlen(text));
}

static void test(gpu_t* gp, const void* code, int bytes) {
    ocl_program_t p = compile(gp, code, bytes);
    gp->kernel_dot_fp32 = ocl.create_kernel(p, kernel_name "_float");
    gp->kernel_sum_odd_fp32  = ocl.create_kernel(p, "sum_odd_fp32");
    gp->kernel_sum_even_fp32 = ocl.create_kernel(p, "sum_even_fp32");
    ocl.dispose_program(p);
    double err = 0;
    for (int n = 2; n < 17; n++) {
        gpu_memory_t mx = gpu.allocate(gp, gpu_allocate_write, n * sizeof(fp32_t));
        gpu_memory_t my = gpu.allocate(gp, gpu_allocate_write, n * sizeof(fp32_t));
        fp32_t* x = (fp32_t*)mx.m;
        fp32_t* y = (fp32_t*)my.m;
        fp32_t sum = 0;
        for (int32_t i = 0; i < n; i++) {
//          x[i] = 1.0f + (i % 2 == 0 ? -1.0f : +1.f) / (i + 1);
//          y[i] = 1.0f - (i % 2 == 0 ? +1.0f : -1.f) / (i + 1);
x[i] = (fp32_t)(i + 1);
y[i] = (fp32_t)(i + 1);
//          traceln("%f %f %f", x[i], y[i], x[i] * y[i]);
            sum += x[i] * y[i];
        }
        fp32_t dot = gpu.dot_f32(&mx, &my, n, 1, 1);
        gpu.deallocate(&mx);
        gpu.deallocate(&my);
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
    ocl.dispose_kernel(gp->kernel_dot_fp32);
    ocl.dispose_kernel(gp->kernel_sum_odd_fp32);
    ocl.dispose_kernel(gp->kernel_sum_even_fp32);
}

int32_t _main_(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    bool profile = true;
    void* code = null;
    int64_t bytes = 0;
    int r = memmap_resource("dot_cl", &code, &bytes);
    fatal_if(r != 0 || code == null || bytes == 0, "dot_cl is not in dot.rc");
    ocl.init();
    for (int cycles = 1; cycles > 0; cycles--) {
        for (int i = 0; i < ocl.count; i++) {
            gpu_t gp = { 0 };
            ocl.open(&gp.c, i, profile);
            traceln("%s\n", ocl.devices[i].name);
            test(&gp, code, (int)bytes);
//          traceln("test: %s\n", result == 0 ? "OK" : "FAILED");
            ocl.close(&gp.c);
        }
    }
    return 0;
}

/*
   Intel: Broadwell, Skylake, Kaby Lake, Coffee Lake, Ice Lake, Tiger Lake
   https://en.wikipedia.org/wiki/List_of_Intel_CPU_microarchitectures
   support float16 kernels
*/

    // TODO: (on "as needed" basis)
    // Level 1 BLAS (14 subprograms):
    // asum
    // axpy
    // copy
    // dot
    // iamax
    // nrm2
    // rot
    // rotg
    // rotm
    // rotmg
    // scal
    // swap
    // sdsdot
    // dsdot
    //
    // Level 2 BLAS (6 subprograms):
    // gemv
    // gbmv
    // hemv
    // hbmv
    // symv
    // sbmv
    //
    // Level 3 BLAS (4 subprograms)
    // gemm
    // symm
    // hemm
    // syrk
    // herk
    // syr2k
    // her2k
    // trmm
    // trsm
